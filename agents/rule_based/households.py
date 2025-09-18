import numpy as np
import torch

class HouseholdRules:
    """
    Rule set for household behavior based on demographic and country-specific patterns.
    Includes saving, risky investment, and 'advance consumption' (adv_consume) behavior.
    """

    # -------------------------------
    # China: Saving Rate Parameters
    # -------------------------------
    @staticmethod
    def _china_saving_params(age):
        """
        Return (mu, sd) of saving rate for Chinese households by age of household head.
        Based on empirical studies (CFPS/UHS) showing a U-shaped profile.
        """
        if age < 35:
            return 0.30, 0.10
        elif age < 45:
            return 0.27, 0.10
        elif age < 55:
            return 0.28, 0.09
        elif age < 65:
            return 0.32, 0.08
        else:
            return 0.34, 0.08

    # ------------------------------------
    # China: Risky Investment Parameters
    # ------------------------------------
    @staticmethod
    def _china_risky_invest_params(age):
        """
        Return (mu, sd) of risky investment share for Chinese households.
        Younger households invest more aggressively.
        """
        if age < 35:
            return 0.50, 0.12
        elif age < 45:
            return 0.40, 0.12
        elif age < 55:
            return 0.35, 0.10
        elif age < 65:
            return 0.25, 0.08
        else:
            return 0.15, 0.08

    # -----------------------------
    # US: Saving Rate Parameters
    # -----------------------------
    @staticmethod
    def _us_saving_params(age):
        """
        Return (mu, sd) of saving rate for US households by age of household head.
        Based on BLS-CE 2023 data. Note: values are approximate and smoothed.
        """
        if age < 35:
            return 0.16, 0.10  # ~16.2%
        elif age < 45:
            return 0.16, 0.09  # ~16.1%
        elif age < 55:
            return 0.12, 0.09  # ~12.1%
        elif age < 65:
            return 0.12, 0.08  # ~11.8%
        else:
            return 0.00, 0.10  # ~ -9.6%, clipped to 0

    # ----------------------------------
    # US: Risky Investment Parameters
    # ----------------------------------
    @staticmethod
    def _us_risky_invest_params(age):
        """
        Return (mu, sd) of risky investment share for US households.
        Generally higher than China; younger households invest more.
        """
        if age < 35:
            return 0.70, 0.10
        elif age < 45:
            return 0.60, 0.10
        elif age < 55:
            return 0.50, 0.10
        elif age < 65:
            return 0.35, 0.08
        else:
            return 0.25, 0.08

    # ----------------------------------------------------------
    # Advance Consumption: eats into planned savings
    # ----------------------------------------------------------
    @staticmethod
    def _china_consumption_params(age):
        """
        Return (mu, sd) of 'advance consumption' ratio for Chinese households.
        This is α_adv: the fraction of planned savings eaten by advance consumption.
        """
        if age < 35:
            return 0.50, 0.15
        elif age < 55:
            return 0.40, 0.12
        else:
            return 0.20, 0.08

    @staticmethod
    def _us_consumption_params(age):
        """
        Return (mu, sd) of 'advance consumption' ratio for US households.
        This is α_adv: the fraction of planned savings eaten by advance consumption.
        """
        if age < 35:
            return 0.60, 0.15
        elif age < 55:
            return 0.50, 0.12
        else:
            return 0.30, 0.10

    # -------------------------------------------
    # Routing function to select param functions
    # -------------------------------------------
    @staticmethod
    def _get_param_fn(country: str, kind: str):
        """
        Select the appropriate age-to-(mu, sd) mapping function based on country and kind.
        Args:
            country (str): "China" or "US"
            kind (str): "saving" | "risky" | "adv_consume"
        """
        if kind == "saving":
            if country == "US":
                return HouseholdRules._us_saving_params
            elif country == "China":
                return HouseholdRules._china_saving_params
        elif kind == "risky":
            if country == "US":
                return HouseholdRules._us_risky_invest_params
            elif country == "China":
                return HouseholdRules._china_risky_invest_params
        elif kind == "adv_consume":
            if country == "US":
                return HouseholdRules._us_consumption_params
            elif country == "China":
                return HouseholdRules._china_consumption_params
        raise ValueError(f"Unknown country={country} or kind={kind}.")

    # -------------------------------------------------
    # Sampling saving / risky / advance consumption
    # -------------------------------------------------
    @staticmethod
    def get_proportion(age, n, country, action_kind):
        """
        Generate proportion vector for 'saving' | 'risky' | 'adv_consume'.
        - 'saving'       -> baseline saving rate s in [0,1]
        - 'risky'        -> risky investment share in [0,1]
        - 'adv_consume'  -> α_adv in [0,1], i.e., fraction of planned savings eaten by advance consumption
        Args:
            age (array or None): age array for each agent, or None for global average
            n (int): number of agents
            country (str): "China" or "US"
            action_kind (str): "saving" | "risky" | "adv_consume"
        Returns:
            np.ndarray: clipped values in [0, 1]
        """
        if age is not None:
            age = np.asarray(age).reshape(-1)
            param_fn = HouseholdRules._get_param_fn(country, kind=action_kind)
            mus_sds = np.array([param_fn(a) for a in age], dtype=float)  # shape (N, 2)
            mus, sds = mus_sds[:, 0], mus_sds[:, 1]
            sp = np.random.normal(loc=mus, scale=sds)
        else:
            # Global priors if age is not given (e.g., Ramsey-type agent)
            if action_kind == "saving":
                mu, sd = (0.40, 0.08) if country == "US" else (0.50, 0.10)
            elif action_kind == "risky":
                mu, sd = (0.50, 0.10) if country == "US" else (0.30, 0.10)
            elif action_kind == "adv_consume":
                mu, sd = (0.50, 0.12) if country == "US" else (0.5, 0.15)
            else:
                raise ValueError(f"Unknown action_kind={action_kind}")
            sp = np.random.normal(mu, sd, size=n)

        return np.clip(sp, 0.0, 1.0)

    # ------------------------------------------
    # Main household action sampling interface
    # ------------------------------------------
    @staticmethod
    def get_action(type, obs, action_dim, firm_n):
        """
        Generate rule-based household action vector based on type and observation.
        Action layout:
            - Column 0: saving proportion in [0, 1] (after adv_consume adjustment if enabled)
            - Column 1: labor supply proportion in [0, 1]
            - Column 2: risky investment proportion in [0, 1] (if "risk_invest" in type and action_dim>=3)
            - Column -firm_n-1: normalized selected firm index in (0,1) when firm_n>1
            - Last firm_n columns: consumption shares across firms (sum to 1) when firm_n>1

        Enable 'advance consumption' by including the token "adv_consume" in `type`.
        Example: type="OLG_risk_invest_adv_consume"
        """
        N = len(obs)
        country = "China"  # Default; replace or infer from obs if needed
        action = np.random.randn(N, action_dim)

        if "OLG" in type:
            ages = obs[:, -1]  # age is assumed to be the last column
        else:
            ages = None

        # --- Baseline saving ---
        saving = HouseholdRules.get_proportion(ages, N, country, action_kind='saving')

        # --- Advance consumption adjustment (s' = s * (1 - alpha_adv)) ---
        if "adv_consume" in type:
            alpha_adv = HouseholdRules.get_proportion(ages, N, country, action_kind='adv_consume')
            saving = np.clip(saving * (1.0 - alpha_adv), 0.0, 1.0)

        action[:, 0] = saving.reshape(-1)

        # --- Labor supply (truncated normal) ---
        mean, std_dev, lower_bound, upper_bound = 0.5, 0.2, 0.0, 1.0
        action[:, 1] = np.clip(np.random.normal(loc=mean, scale=std_dev, size=N), lower_bound, upper_bound)

        # --- Risky investment (optional) ---
        if "risk_invest" in type and action_dim >= 3:
            risk = HouseholdRules.get_proportion(ages, N, country, action_kind='risky')
            action[:, 2] = risk.reshape(-1)

        # --- Firm choice & consumption shares across firms ---
        if firm_n > 1:
            # Extract wage rates and prices from observations
            wagerate = obs[:, 4:4 + firm_n]  # shape (N, firm_n)
            price = obs[:, 4 + firm_n: 4 + firm_n * 2]  # shape (N, firm_n)

            # Convert to torch for stable operations
            wage_t = torch.tensor(wagerate, dtype=torch.float32)
            wage_sum = wage_t.sum(dim=1, keepdim=True).clamp_min(1e-8)
            wage_probs = wage_t / wage_sum

            if torch.isnan(wage_probs).any() or torch.isinf(wage_probs).any():
                print("Warning: NaN or Inf in wagerate_probs")

            firm_index = torch.multinomial(wage_probs, 1)  # (N,1)
            action[:, -firm_n - 1] = firm_index.squeeze(1).numpy() / firm_n  # normalized index in (0,1)

            # Price-based choice probabilities (softmax on -price)
            price_t = torch.tensor(price, dtype=torch.float32)
            price_exp = torch.exp(-price_t)
            price_sum = price_exp.sum(dim=1, keepdim=True).clamp_min(1e-8)
            price_probs = price_exp / price_sum

            if torch.isnan(price_probs).any() or torch.isinf(price_probs).any():
                print("Warning: NaN or Inf in price_probs")

            action[:, -firm_n:] = price_probs.numpy()
            # Ensure exact row-wise normalization (numerical safety)
            row_sum = action[:, -firm_n:].sum(axis=1, keepdims=True)
            action[:, -firm_n:] = action[:, -firm_n:] / np.clip(row_sum, 1e-8, None)

        return action
