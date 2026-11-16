import numpy as np


class EconObservations:
    """Encapsulates observation settings for all economic agents in EconGym.

    This class defines observations for Individuals, Government, Bank, and Firm agents,
    including their variants. Users can extend or modify observation variables by
    editing the respective methods.
    """

    def __init__(self, society):
        """Initialize with agent objects to extract observation data."""
        for name, agent in society.agents.items():
            setattr(self, name, agent)
        self.society = society

    def get_global_obs(self):
        wealth = getattr(self.households, 'at_next', np.zeros(self.households.households_n))
        education = getattr(self.households, 'e', np.zeros(self.households.households_n))

        n_households = self.households.households_n

        # Sort indices based on wealth (instead of income) in descending order
        sorted_wealth_based_index = sorted(range(len(wealth)), key=lambda k: wealth[k], reverse=True)
        top_n = max(1, int(0.1 * n_households))
        bottom_n = max(1, int(0.5 * n_households))
        top10_wealth_based_index = sorted_wealth_based_index[:top_n]  # top 10% households
        bottom50_wealth_based_index = sorted_wealth_based_index[bottom_n:]  # bottom 50% households

        top10_e = education[top10_wealth_based_index]
        bot50_e = education[bottom50_wealth_based_index]

        top10_wealth = wealth[top10_wealth_based_index]
        bot50_wealth = wealth[bottom50_wealth_based_index]

        # Base global observations
        global_obs = np.array([
            np.mean(top10_wealth),  # Top 10 mean asset
            np.mean(top10_e),  # Top 10 mean education
            np.mean(bot50_wealth),  # Bottom 50 mean asset
            np.mean(bot50_e),  # Bottom 50 mean education
        ])

        # Add economic indicators
        wage_rate = getattr(self.market, 'WageRate', 0.0) if self.market else 0.0
        price_level = getattr(self.market, 'price', 1.0) if self.market else 1.0
        lending_rate = getattr(self.bank, 'lending_rate', 0.0345) if self.bank else 0.0345
        deposit_rate = getattr(self.bank, 'deposit_rate', 0.0345) if self.bank else 0.0345

        economic_indicators = np.concatenate([
            wage_rate.flatten(),
            price_level.flatten(),
            [lending_rate],
            [deposit_rate]
        ])
        global_obs = np.concatenate([global_obs, economic_indicators])
        return global_obs

    def get_individual_observations(self):
        """Generate observations for Individual agents (Ramsey or OLG models).

        Returns:
            np.array: Observation array of shape (n_households, obs_len) where
                     each row contains global_obs + private_obs for one household
        """
        if not self.households:
            raise ValueError("Household object is required for individual observations.")

        

        n_households = self.households.households_n

        # Private observations
        if 'OLG' in self.households.type:
            # age = getattr(self.households, 'age_write', np.zeros(n_households))
            # wealth = getattr(self.households, 'at_next_write', np.zeros(self.households.households_n))
            # education = getattr(self.households, 'e_write', np.zeros(self.households.households_n))

            wealth = getattr(self.households, 'at_next', np.zeros(self.households.households_n))
            education = getattr(self.households, 'e', np.zeros(self.households.households_n))
            age = getattr(self.households, 'age', np.zeros(n_households))
            # Each household's private obs: [education, wealth, age]
            private_obs_per_household = np.column_stack([education, wealth, age])
        elif "ramsey" in self.households.type:
            wealth = getattr(self.households, 'at_next', np.zeros(self.households.households_n))
            education = getattr(self.households, 'e', np.zeros(self.households.households_n))
            # Each household's private obs: [education, wealth]
            private_obs_per_household = np.column_stack([education, wealth])
        else:
            raise ValueError(f"AgentTypeError: household type {self.households.type} is not in type_list.")

        # Repeat global_obs for each household and concatenate with private_obs
        global_obs_repeated = np.tile(self.global_obs, (n_households, 1))
        observations = np.concatenate([global_obs_repeated, private_obs_per_household], axis=1)

        return observations

    def get_government_observations(self):
        """Generate observations for Government agents (Tax, Central Bank, Pension Authority).

        Returns:
            np.array: Government observations.

        Users can add/remove observation variables (e.g., public debt, inflation) below.
        """
        total_observations = {}
        for gov_type, gov_agent in self.government.items():
            common_obs = self.global_obs

            if gov_type == "tax":
                # Ensure the shapes match before concatenating
                observations = np.concatenate([common_obs, np.array([self.government[gov_type].Bt])])  # + debt

            elif gov_type == "central_bank":
                observations = np.concatenate([
                    common_obs,
                    np.array([getattr(self.society, 'inflation_rate', 0.02)]),
                    np.array([getattr(self.society, 'growth_rate', 0.05)]),
                ])

            elif gov_type == "pension":
                current_population = getattr(self.households, 'households_n', 0)
                observations = np.array([
                    np.sum(getattr(self.households, 'accumulated_pension_account', 0.)),
                    current_population,
                    getattr(self.households, 'old_n', 0.07 * current_population),
                    # 7% of the population aged 65 and above is the international standard for a society to enter aging.
                    getattr(self.government[gov_type], 'retire_age', 60.0),
                    getattr(self.government[gov_type], 'contribution_rate', 0.10),
                    getattr(self.government[gov_type], 'Bt', 0),
                    getattr(self.government[gov_type], 'GDP', 0)
                ])
            total_observations[gov_type] = observations

        return total_observations

    def get_bank_observations(self):
        """Generate observations for Bank agents (Non-Profit Platform or Commercial Bank).

        Returns:
            np.array: Bank observations.

        Users can add/remove observation variables (e.g., benchmark rate, deposits) below.
        """

        if self.bank.type.lower() == "non_profit":
            return np.array([])
        elif self.bank.type.lower() == "commercial":
            return np.array([
                getattr(self.bank, 'base_interest_rate', 0.03),
                getattr(self.bank, 'reserve_ratio', 0.0),
                getattr(self.bank, 'lending_rate', 0.0345),  # Lending rate of the previous period
                getattr(self.bank, 'deposit_rate', 0.0345),  # Deposit rate of the previous period
                getattr(self.bank, 'current_loans', 0.0),  # Total loans given out in the previous period
                getattr(self.bank, 'total_account', 0.0),  # Total deposits received in the previous period
            ])

        else:
            raise ValueError(f"BankTypeError: bank type {self.bank.type} is not supported.")

    def get_firm_observations(self):
        """Generate observations for Firm agents (Perfect Competition, Monopoly, Oligopoly, Monopolistic Competition).

        Returns:
            np.array: Firm observations.

        Users can add/remove observation variables (e.g., capital, productivity) below.
        """
    
        if self.market.type.lower() == "perfect":
            return np.array([])
        elif self.market.type.lower() in ["monopoly", "oligopoly", "monopolistic_competition"]:
            firm_n = getattr(self.market, 'firm_n', 0.0)
            firm_capital = getattr(self.market, 'Kt_next', 0.0)
            firm_productivity = getattr(self.market, 'Zt', 0.0)
            firm_rt = np.full((firm_n, 1), getattr(self.bank, 'lending_rate', 0.0))
            if firm_n > 0:
                firm_capital_2d = np.full((firm_n, 1), firm_capital)  # Shape (firm_n, 1)
                firm_productivity_2d = np.full((firm_n, 1), firm_productivity)  # Shape (firm_n, 1)
            else:
                # If firm_n = 0, avoid dimension errors
                firm_capital_2d = np.array([[firm_capital]])
                firm_productivity_2d = np.array([[firm_productivity]])
                firm_rt = np.array([[firm_rt]])  # Ensure firm_rt is also 2D
        
            # At this point, all arrays are 2D with shape (firm_n, 1), safe for horizontal stacking
            result = np.hstack([firm_capital_2d, firm_productivity_2d, firm_rt])
        
            return result
            # wage_rate = getattr(self.market, 'WageRate', 0.0) if self.market else 0.0    # Optionally add price and WageRate at the last timestep
            # price_level = getattr(self.market, 'price', 1.0) if self.market else 1.0
            # return np.hstack([firm_capital, firm_productivity, firm_rt])
    
        else:
            raise ValueError(f"FirmTypeError: market type {self.market.type} is not supported.")

    def get_obs(self):
        """Generate observations for all agents.

        Returns:
            dict: Dictionary with observations for each agent type.
        """
        self.global_obs = self.get_global_obs()

        GOV_NAME = "government"
        return {
            self.households.name: self.get_individual_observations(),
            GOV_NAME: self.get_government_observations(),
            self.bank.name: self.get_bank_observations(),
            self.market.name: self.get_firm_observations(),
        }
