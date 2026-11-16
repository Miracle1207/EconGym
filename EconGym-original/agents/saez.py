import numpy as np


class SaezGovernment():
    def __init__(self,
                 period=100,
                 rate_min=0.0,
                 rate_max=1.0,
                 disable_taxes=False,
                 n_brackets=5,  # 税收模型中税率区间的数量
                 top_bracket_cutoff=4000000,  # 这个参数是用来设置最高税率区间的起始收入
                 usd_scaling=1000.0,
                 bracket_spacing="linear",
                 pareto_weight_type="inverse_income",
                 saez_fixed_elas=None
                 ):
        self.period = period
        self.disable_taxes = bool(disable_taxes)
        self.rate_min = 0.0 if self.disable_taxes else float(rate_min)
        # Maximum marginal bracket rate
        self.rate_max = 0.0 if self.disable_taxes else float(rate_max)
        assert 0 <= self.rate_min <= self.rate_max <= 1.0

        # === income bracket definitions ===
        self.n_brackets = int(n_brackets)
        assert self.n_brackets >= 2

        self.top_bracket_cutoff = float(top_bracket_cutoff)
        assert self.top_bracket_cutoff >= 10

        self.usd_scale = float(usd_scaling)
        assert self.usd_scale > 0

        self.bracket_spacing = bracket_spacing.lower()
        assert self.bracket_spacing in ["linear", "log", "us-federal"]

        if self.bracket_spacing == "linear":
            self.bracket_cutoffs = np.linspace(
                0, self.top_bracket_cutoff, self.n_brackets
            )

        elif self.bracket_spacing == "log":
            b0_max = self.top_bracket_cutoff / (2 ** (self.n_brackets - 2))
            self.bracket_cutoffs = np.concatenate(
                [
                    [0],
                    2
                    ** np.linspace(
                        np.log2(b0_max),
                        np.log2(self.top_bracket_cutoff),
                        n_brackets - 1,
                    ),
                ]
            )
        elif self.bracket_spacing == "us-federal":
            self.bracket_cutoffs = (
                    np.array([0, 9700, 39475, 84200, 160725, 204100, 510300])
                    / self.usd_scale
            )
            self.n_brackets = len(self.bracket_cutoffs)
            self.top_bracket_cutoff = float(self.bracket_cutoffs[-1])
        else:
            raise NotImplementedError

        self.bracket_edges = np.concatenate([self.bracket_cutoffs, [np.inf]])
        # print(self.bracket_edges)
        self.bracket_sizes = self.bracket_edges[1:] - self.bracket_edges[:-1]
        # print(self.bracket_sizes)
        assert self.bracket_cutoffs[0] == 0

        # === bracket tax rates ===
        self.curr_bracket_tax_rates = np.zeros_like(self.bracket_cutoffs)
        self.curr_rate_indices = [0 for _ in range(self.n_brackets)]

        # === Pareto weights, elasticity ===
        self.pareto_weight_type = pareto_weight_type
        self.elas_tm1 = 0.5
        self.elas_t = 0.5
        self.log_z0_tm1 = 0
        self.log_z0_t = 0

        self._saez_fixed_elas = saez_fixed_elas
        if self._saez_fixed_elas is not None:
            self._saez_fixed_elas = float(self._saez_fixed_elas)
            assert self._saez_fixed_elas >= 0

        # Size of the local buffer. In a distributed context, the global buffer size
        # will be capped at n_replicas * _buffer_size.
        # NOTE: Saez will use random taxes until it has self._buffer_size samples.
        self._buffer_size = 500
        self._reached_min_samples = False
        self.saez_buffer = []
        self._additions_this_episode = 0
        # Local buffer maintained by this replica.
        self._local_saez_buffer = []
        # "Global" buffer obtained by combining local buffers of individual replicas.
        self._global_saez_buffer = []

        self._saez_n_estimation_bins = 100
        self._saez_top_rate_cutoff = self.bracket_cutoffs[-1]
        self._saez_income_bin_edges = np.linspace(
            0, self._saez_top_rate_cutoff, self._saez_n_estimation_bins + 1
        )
        self._saez_income_bin_sizes = np.concatenate(
            [
                self._saez_income_bin_edges[1:] - self._saez_income_bin_edges[:-1],
                [np.inf],
            ]
        )
        self.running_avg_tax_rates = np.zeros_like(self.curr_bracket_tax_rates)
        self.tax_cycle_pos = 1

    @property
    def curr_rate_max(self):
        """Maximum allowable tax rate, given current progress of any tax annealing."""
        return self.rate_max

    def marginal_rate(self, income):
        """Return the marginal tax rate applied at this income level."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.curr_marginal_rates[np.argmax(bracket_bool)]

    @property
    def curr_marginal_rates(self):
        """The current set of marginal tax bracket rates."""
        marginal_tax_bracket_rates = np.minimum(
            self.curr_bracket_tax_rates, self.curr_rate_max
        )
        return marginal_tax_bracket_rates

    # ------- Saez formula
    def compute_and_set_new_period_rates_from_saez_formula(
            self, update_elas_tm1=True, update_log_z0_tm1=True
    ):
        """Estimates/sets optimal rates using adaptation of Saez formula

        See: https://www.nber.org/papers/w7628
        """
        # Until we reach the min sample number, keep checking if we have reached it.
        if not self._reached_min_samples:
            # Note: self.saez_buffer includes the global buffer (if applicable).
            if len(self.saez_buffer) >= self._buffer_size:
                self._reached_min_samples = True

        # If no enough samples, use random taxes.
        if not self._reached_min_samples:
            self.curr_bracket_tax_rates = np.random.uniform(
                low=self.rate_min,
                high=self.curr_rate_max,
                size=self.curr_bracket_tax_rates.shape,
            )
            return
        # print(self.saez_buffer)
        incomes_and_marginal_rates = np.array(self.saez_buffer)

        # Elasticity assumed constant for all incomes.
        # (Run this for the sake of tracking the estimate; will not actually use the
        # estimate if using fixed elasticity).
        if update_elas_tm1:
            self.elas_tm1 = float(self.elas_t)
        if update_log_z0_tm1:
            self.log_z0_tm1 = float(self.log_z0_t)

        elas_t, log_z0_t = self.estimate_uniform_income_elasticity(
            incomes_and_marginal_rates,
            elas_df=0.98,
            elas_tm1=self.elas_tm1,
            log_z0_tm1=self.log_z0_tm1,
            verbose=False,
        )

        if update_elas_tm1:
            self.elas_t = float(elas_t)
        if update_log_z0_tm1:
            self.log_z0_t = float(log_z0_t)

        # If a fixed estimate has been specified, use it in the formulas below.
        if self._saez_fixed_elas is not None:
            elas_t = float(self._saez_fixed_elas)

        # Get Saez parameters at each income bin
        # to compute a marginal tax rate schedule.
        binned_gzs, binned_azs = self.get_binned_saez_welfare_weight_and_pareto_params(
            population_incomes=incomes_and_marginal_rates[:, 0]
        )

        # Use the elasticity to compute this binned schedule using the Saez formula.
        binned_marginal_tax_rates = self.get_saez_marginal_rates(
            binned_gzs, binned_azs, elas_t
        )
        # print(binned_marginal_tax_rates)

        # Adapt the saez tax schedule to the tax brackets.
        self.curr_bracket_tax_rates = np.clip(
            self.bracketize_schedule(
                bin_marginal_rates=binned_marginal_tax_rates,
                bin_edges=self._saez_income_bin_edges,
                bin_sizes=self._saez_income_bin_sizes,
            ),
            self.rate_min,
            self.curr_rate_max,
        )
        # print(self.curr_bracket_tax_rates)
        self.running_avg_tax_rates = (self.running_avg_tax_rates * 0.99) + (
                self.curr_bracket_tax_rates * 0.01
        )
        # print(self.running_avg_tax_rates)

    '''
    def saez_buffer(self):
        if not self._global_saez_buffer:
            saez_buffer = self._local_saez_buffer
        elif self._additions_this_episode == 0:
            saez_buffer = self._global_saez_buffer
        else:
            saez_buffer = (
                self._global_saez_buffer
                + self._local_saez_buffer[-self._additions_this_episode :]
            )
        return saez_buffer

    def get_local_saez_buffer(self):
        return self._local_saez_buffer

    def set_global_saez_buffer(self, global_saez_buffer):
        assert isinstance(global_saez_buffer, list)
        assert len(global_saez_buffer) >= len(self._local_saez_buffer)
        self._global_saez_buffer = global_saez_buffer
    '''

    def update_saez_buffer(self, income):
        for i in range(len(income)):
            inc = income[i][0]
            margin = self.marginal_rate(inc)
            self.saez_buffer.append([inc, margin])
        while len(self.saez_buffer) > self._buffer_size:
            _ = self.saez_buffer.pop(0)

    def tax_due(self, income):
        tax_sum = []
        for i in range(len(income)):
            past_cutoff = np.maximum(0, income[i][0] - self.bracket_cutoffs)
            bin_income = np.minimum(self.bracket_sizes, past_cutoff)
            bin_taxes = self.curr_marginal_rates * bin_income
            tax_sum.append(np.sum(bin_taxes))
        return tax_sum

    def estimate_uniform_income_elasticity(
            self,
            observed_incomes_and_marginal_rates,
            elas_df=0.98,
            elas_tm1=0.5,
            log_z0_tm1=0.5,
            verbose=False,
    ):
        """Estimate elasticity using Ordinary Least Squares regression.
        OLS: https://en.wikipedia.org/wiki/Ordinary_least_squares
        Estimating elasticity: https://www.nber.org/papers/w7512
        """
        zs = []
        taus = []

        for z_t, tau_t in observed_incomes_and_marginal_rates:
            # If z_t is <=0 or tau_t is >=1, the operations below will give us nans
            if z_t > 0 and tau_t < 1:
                zs.append(z_t)
                taus.append(tau_t)

        if len(zs) < 10:
            return float(elas_tm1), float(log_z0_tm1)
        if np.std(taus) < 1e-6:
            return float(elas_tm1), float(log_z0_tm1)

        # Regressing log income against log 1-marginal_rate.
        x = np.log(np.maximum(1 - np.array(taus), 1e-9))
        # (bias term)
        b = np.ones_like(x)
        # Perform OLS.
        X = np.stack([x, b]).T  # Stack linear & bias terms
        Y = np.log(np.maximum(np.array(zs), 1e-9))  # Regression targets
        XXi = np.linalg.inv(X.T.dot(X))
        XY = X.T.dot(Y)
        elas, log_z0 = XXi.T.dot(XY)

        warn_less_than_0 = elas < 0
        instant_elas_t = np.maximum(elas, 0.0)

        elas_t = ((1 - elas_df) * instant_elas_t) + (elas_df * elas_tm1)

        if verbose:
            if warn_less_than_0:
                print("\nWARNING: Recent elasticity estimate is < 0.")
                print("Running elasticity estimate: {:.2f}\n".format(elas_t))
            else:
                print("\nRunning elasticity estimate: {:.2f}\n".format(elas_t))

        return elas_t, log_z0

    def get_binned_saez_welfare_weight_and_pareto_params(self, population_incomes):
        # print(population_incomes)

        def clip(x, lo=None, hi=None):
            if lo is not None:
                x = max(lo, x)
            if hi is not None:
                x = min(x, hi)
            return x

        def bin_z(left, right):
            return 0.5 * (left + right)

        def get_cumul(counts, incomes_below, incomes_above):
            n_below = len(incomes_below)
            n_above = len(incomes_above)
            n_total = np.sum(counts) + n_below + n_above

            def p(i, counts):
                return counts[i] / n_total

            # Probability that an income is below the taxable threshold.
            p_below = n_below / n_total

            # pz = p(z' = z): probability that [binned] income z' occurs in bin z.
            pz = [p(i, counts) for i in range(len(counts))] + [n_above / n_total]

            # Pz = p(z' <= z): Probability z' is less-than or equal to z.
            cum_pz = [pz[0] + p_below]
            for p in pz[1:]:
                cum_pz.append(clip(cum_pz[-1] + p, 0, 1.0))

            return np.array(pz), np.array(cum_pz)

        def compute_binned_g_distribution(counts, lefts, incomes):
            def pareto(z):
                if self.pareto_weight_type == "uniform":
                    pareto_weights = np.ones_like(z)
                elif self.pareto_weight_type == "inverse_income":
                    pareto_weights = 1.0 / np.maximum(1, z)
                else:
                    raise NotImplementedError
                return pareto_weights

            incomes_below = incomes[incomes < lefts[0]]
            incomes_above = incomes[incomes > lefts[-1]]

            # The total (unnormalized) Pareto weight of untaxable incomes.
            if len(incomes_below) > 0:
                pareto_weight_below = np.sum(pareto(np.maximum(incomes_below, 0)))
            else:
                pareto_weight_below = 0

            # The total (unnormalized) Pareto weight within each bin.
            if len(incomes_above) > 0:
                pareto_weight_above = np.sum(pareto(incomes_above))
            else:
                pareto_weight_above = 0

            # The total (unnormalized) Pareto weight within each bin.
            pareto_weight_per_bin = counts * pareto(bin_z(lefts[:-1], lefts[1:]))

            # The aggregate (unnormalized) Pareto weight of all incomes.
            cumulative_pareto_weights = pareto_weight_per_bin.sum()
            cumulative_pareto_weights += pareto_weight_below
            cumulative_pareto_weights += pareto_weight_above

            # Normalize so that the Pareto density sums to 1.
            pareto_norm = cumulative_pareto_weights + 1e-9
            unnormalized_pareto_density = np.concatenate(
                [pareto_weight_per_bin, [pareto_weight_above]]
            )
            normalized_pareto_density = unnormalized_pareto_density / pareto_norm

            # Aggregate Pareto weight of earners with income greater-than or equal to z.
            cumulative_pareto_density_geq_z = np.cumsum(
                normalized_pareto_density[::-1]
            )[::-1]

            # Probability that [binned] income z' is greather-than or equal to z.
            pz, _ = get_cumul(counts, incomes_below, incomes_above)
            cumulative_prob_geq_z = np.cumsum(pz[::-1])[::-1]

            # Average (normalized) Pareto weight of earners with income >= z.
            geq_z_norm = cumulative_prob_geq_z + 1e-9
            avg_pareto_weight_geq_z = cumulative_pareto_density_geq_z / geq_z_norm

            def interpolate_gzs(gz):
                # Assume incomes within a bin are evenly distributed within that bin
                # and re-compute accordingly.
                gz_at_left_edge = gz[:-1]
                gz_at_right_edge = gz[1:]

                avg_bin_gz = 0.5 * (gz_at_left_edge + gz_at_right_edge)
                # Re-attach the gz of the top tax rate (does not need to be
                # interpolated).
                gzs = np.concatenate([avg_bin_gz, [gz[-1]]])
                return gzs

            return interpolate_gzs(avg_pareto_weight_geq_z)

        def compute_binned_a_distribution(counts, lefts, incomes):
            incomes_below = incomes[incomes < lefts[0]]
            incomes_above = incomes[incomes > lefts[-1]]

            # z is defined as the MIDDLE point in a bin.
            # So for a bin [left, right] -> z = (left + right) / 2.
            Az = []

            # cum_pz = p(z' <= z): Probability z' is less-than or equal to z
            pz, cum_pz = get_cumul(counts, incomes_below, incomes_above)

            # Probability z' is greater-than or equal to z
            # Note: The "0.5" coefficient gives results more consistent with theory; it
            # accounts for the assumption that incomes within a particular bin are
            # uniformly spread between the left & right edges of that bin.
            p_geq_z = 1 - cum_pz + (0.5 * pz)

            T = len(lefts[:-1])

            for i in range(T):
                if pz[i] == 0:
                    Az.append(np.nan)
                else:
                    z = bin_z(lefts[i], lefts[i + 1])
                    # paz = z * pz[i] / (clip(1 - Pz[i], 0, 1) + 1e-9)
                    paz = z * pz[i] / (clip(p_geq_z[i], 0, 1) + 1e-9)  # defn of A(z)
                    paz = paz / (lefts[i + 1] - lefts[i])  # norm by bin width
                    Az.append(paz)

            # Az for the incomes past the top cutoff,
            # the bin is [left, infinity]: there is no "middle".
            # Hence, use the mean value in the last bin.
            if len(incomes_above) > 0:
                cutoff = lefts[-1]
                avg_income_above_cutoff = np.mean(incomes_above)
                # use a special formula to compute A(z)
                Az_above = avg_income_above_cutoff / (
                        avg_income_above_cutoff - cutoff + 1e-9
                )
            else:
                Az_above = 0.0

            return np.concatenate([Az, [Az_above]])

        counts, lefts = np.histogram(
            population_incomes, bins=self._saez_income_bin_edges
        )
        population_gz = compute_binned_g_distribution(counts, lefts, population_incomes)
        population_az = compute_binned_a_distribution(counts, lefts, population_incomes)

        # Return the binned stats used to create a schedule of marginal rates.
        return population_gz, population_az

    @staticmethod
    def get_saez_marginal_rates(binned_gz, binned_az, elas, interpolate=True):
        # Marginal rates within each income bin (last tau is the top tax rate).
        taus = (1.0 - binned_gz) / (1.0 - binned_gz + binned_az * elas + 1e-9)

        if interpolate:
            # In bins where there were no incomes found, tau is nan.
            # Interpolate to fill the gaps.
            last_real_rate = 0.0
            last_real_tidx = -1
            for i, tau in enumerate(taus):
                # The current tax rate is a real number.
                if not np.isnan(tau):
                    # This is the end of a gap. Interpolate.
                    if (i - last_real_tidx) > 1:
                        assert (
                                i != 0
                        )  # This should never trigger for the first tax bin.
                        gap_indices = list(range(last_real_tidx + 1, i))
                        intermediate_rates = np.linspace(
                            last_real_rate, tau, len(gap_indices) + 2
                        )[1:-1]
                        assert len(gap_indices) == len(intermediate_rates)
                        for gap_index, intermediate_rate in zip(
                                gap_indices, intermediate_rates
                        ):
                            taus[gap_index] = intermediate_rate
                    # Update the tracker.
                    last_real_rate = float(tau)
                    last_real_tidx = int(i)

                # The current tax rate is a nan. Continue without updating
                # the tracker (indicating the presence of a gap).
                else:
                    pass

        return taus

    def bracketize_schedule(self, bin_marginal_rates, bin_edges, bin_sizes):
        # Compute the amount of tax each bracket would collect
        # if income was >= the right edge.
        # Divide by the bracket size to get
        # the average marginal rate within that bracket.
        last_bracket_total = 0
        bracket_avg_marginal_rates = []
        for b_idx, income in enumerate(self.bracket_cutoffs[1:]):
            # How much income occurs within each bin
            # (including the open-ended, top "bin").
            past_cutoff = np.maximum(0, income - bin_edges)
            bin_income = np.minimum(bin_sizes, past_cutoff)

            # To get the total taxes due,
            # multiply the income within each bin by that bin's marginal rate.
            bin_taxes = bin_marginal_rates * bin_income
            taxes_due = np.maximum(0, np.sum(bin_taxes))

            bracket_tax_burden = taxes_due - last_bracket_total
            bracket_size = self.bracket_sizes[b_idx]

            bracket_avg_marginal_rates.append(bracket_tax_burden / bracket_size)
            last_bracket_total = taxes_due

        # The top bracket tax rate is computed directly already.
        bracket_avg_marginal_rates.append(bin_marginal_rates[-1])

        bracket_rates = np.array(bracket_avg_marginal_rates)
        assert len(bracket_rates) == self.n_brackets

        return bracket_rates

    def saez_step(self):
        if self.tax_cycle_pos == 1:
            self.compute_and_set_new_period_rates_from_saez_formula()

        # 2. On the last day of the tax period: Get $-taxes AND update agent endowments.
        if self.tax_cycle_pos >= self.period:
            # self.enact_taxes()
            self.tax_cycle_pos = 0

        # increment timestep.
        self.tax_cycle_pos += 1
