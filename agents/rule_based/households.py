import numpy as np
import torch
class HouseholdRules:
    """
    Rule set for household behavior based on demographic and country-specific patterns.
    This includes saving and risky investment behavior by age group and country.
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

    # -------------------------------------------
    # Routing function to select param functions
    # -------------------------------------------
    @staticmethod
    def _get_param_fn(country: str, kind: str):
        """
        Select the appropriate age-to-(mu, sd) mapping function based on country and kind.
        Args:
            country (str): "China" or "US"
            kind (str): "saving" or "risky"
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
        raise ValueError(f"Unknown country={country} or kind={kind}.")

    # -------------------------------------------------
    # Sampling saving or risky investment proportions
    # -------------------------------------------------
    @staticmethod
    def get_proportion(age, n, country, action_kind):
        """
        Generate saving or risky investment proportion.
        Args:
            age (array or None): age array for each agent, or None for global average
            n (int): number of agents
            country (str): "China" or "US"
            action_kind (str): "saving" or "risky"
        Returns:
            np.ndarray: clipped values in [0, 1]
        """
        param_fn = HouseholdRules._get_param_fn(country, kind=action_kind)
        if age is not None:
            age = np.asarray(age).reshape(-1)
            mus_sds = np.array([param_fn(a) for a in age], dtype=float)  # shape (N, 2)
            mus, sds = mus_sds[:, 0], mus_sds[:, 1]
            sp = np.random.normal(loc=mus, scale=sds)
        else:
            # Fall back to global priors if age is not given (e.g., Ramsey-type agent)
            if action_kind == "saving":
                mu, sd = (0.40, 0.08) if country == "US" else (0.50, 0.10)
            elif action_kind == "risky":
                mu, sd = (0.50, 0.10) if country == "US" else (0.30, 0.10)
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
            - Column 0: saving proportion in [0, 1]
            - Column 1: (optional custom action, clipped to [0, 1])
            - Column 2: risky investment proportion in [0, 1] (if applicable)

        Args:
            type (str): agent type ("OLG", "Ramsey", etc.)
            obs (np.ndarray): observation matrix, must include age if "OLG"
            action_dim (int): dimension of the action vector
        Returns:
            np.ndarray: (N, action_dim) action matrix
        """
        N = len(obs)
        country = "China"  # Default setting; can be replaced or inferred from obs
        action = np.random.randn(N, action_dim)

        if "OLG" in type:
            ages = obs[:, -1]  # age is assumed to be the last column
        else:
            ages = None

        # Generate saving proportion
        saving = HouseholdRules.get_proportion(ages, N, country, action_kind='saving')
        action[:, 0] = saving.reshape(-1)

        mean, std_dev, lower_bound, upper_bound = 0.5, 0.2, 0.0, 1.0
        # Generate labor supply proportion using truncated normal distribution
        action[:, 1] = np.clip(np.random.normal(loc=mean, scale=std_dev, size=N), lower_bound, upper_bound)

        # Optionally generate risky investment proportion
        if "risk_invest" in type:
            risk = HouseholdRules.get_proportion(ages, N, country, action_kind='risky')
            action[:, 2] = risk.reshape(-1)

        # If there are more than 1 firm
        if firm_n > 1:
            # Extract wage rates and prices from observations
            wagerate = obs[:, 4:4 + firm_n] # Assuming wage rates are located in this range
            price = obs[:, 4 + firm_n: 4 + firm_n * 2]  # Prices of the firms
    
            wagerate_probs = wagerate / torch.sum(wagerate, dim=1, keepdim=True)

            if torch.isnan(wagerate_probs).any() or torch.isinf(wagerate_probs).any():
                print("Warning: NaN or Inf in wagerate_probs")
                
            firm_index = torch.multinomial(wagerate_probs, 1)
            action[:, -firm_n - 1] = firm_index.squeeze()/firm_n  # Store the selected firm index in the action array
    
            # Now, calculate the probabilities of choosing each firm's goods based on price
            # Apply softmax to prices to get a probability distribution
            price_exp = torch.exp(-price)
            price_probs = price_exp / torch.sum(price_exp, dim=1, keepdim=True)
            if torch.isnan(price_probs).any() or torch.isinf(price_probs).any():
                print("Warning: NaN or Inf in price_probs")

            # Assign the probabilities to the last firm_n columns of the action
            action[:, -firm_n:] = price_probs
            # Ensure the probabilities sum to 1 across each row (household)
            action[:, -firm_n:] = action[:, -firm_n:] / np.sum(action[:, -firm_n:], axis=1, keepdims=True)

        # Now, 'action' should contain:
        # - action[:, 0] for saving proportion
        # - action[:, 1] for working hours proportion
        # - action[:, 2] for risky investment proportion (if applicable)
        # - action[:, -firm_n-1] for selected firm_index/firm_n (\in (0,1)) based on wage rates
        # - action[:, -firm_n:] for the proportions of consumption from each firm (sum to 1)

        return action
