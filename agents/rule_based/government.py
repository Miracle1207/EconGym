import numpy as np
import copy


class GovernmentRules:
    """
    Rule set for different government roles.
    Uses @staticmethod because rules do not depend on class instances.
    """


    @staticmethod
    def cb_rule_taylor(obs, *,
                       pi_star=0.02, g_star=0.05, r_star=0.02,
                       phi_pi=1.5, phi_g=0.5,
                       noise=0.002, reserve_ratio=0.08):
        """Taylor rule: r = r* + π + φπ(π-π*) + φg(g-g*)."""
        arr = np.ravel(obs)
        pi_t, g_t = float(arr[-2]), float(arr[-1])
        rate = r_star + pi_t + phi_pi * (pi_t - pi_star) + phi_g * (g_t - g_star)
        if noise is not None:
            rate += np.random.normal(0.0, noise)
        rate = max(rate, 0.0)
        return np.array([rate, np.clip(reserve_ratio, 0.0, 1.0)], dtype=float)

    @staticmethod
    def cb_rule_random(obs, *,
                       mean=0.02, std=0.01,
                       min_rate=0.0, max_rate=None,
                       noise=0.002, reserve_ratio=0.08):
        """Random baseline using obs as required input (not used in computation)."""
        rate = float(np.random.normal(mean, std))
        if noise is not None:
            rate += float(np.random.normal(0.0, noise))
        rate = max(min_rate, rate) if max_rate is None else float(np.clip(rate, min_rate, max_rate))
        return np.array([rate, np.clip(reserve_ratio, 0.0, 1.0)], dtype=float)

    @staticmethod
    def central_bank_action(obs, *, rule="taylor", **kwargs):
        """Dispatch central bank rules. You can design and add more rules here."""
        rules = {
            "taylor": lambda: GovernmentRules.cb_rule_taylor(obs, **kwargs),
            "random": lambda: GovernmentRules.cb_rule_random(obs, **kwargs),
        }
        try:
            return rules[rule]()
        except KeyError:
            raise ValueError(f"Unknown rule: {rule!r}")

    @staticmethod
    def pension_rule_imf(
            obs, *,
            # IMF-style adjustments
            debt_ratio_upper: float = 0.60,
            phi_RA: float = 0.20,
            phi_gamma: float = 0.10,
            # bounds
            retire_bounds: tuple = (55, 75),
            contrib_bounds: tuple = (0.05, 0.30),
    ) -> np.ndarray:
        """
        IMF-style pension rule: adjust retirement age & contribution when debt/GDP exceeds the threshold.
        obs can be:
          - dict with keys: ("Bt_next" or "Bt") and "GDP"
          - array-like: Bt at idx_Bt, GDP at idx_GDP
        """
        GDP = obs[-1]
        Bt = obs[-2]
        debt_GDP = Bt / max(GDP, 1e-8)

        base_retire_age = obs[-4]
        base_contrib_rate = obs[-3]
    
        retire = base_retire_age + max(debt_GDP - debt_ratio_upper, 0.0) * phi_RA
        contrib = base_contrib_rate + max(debt_GDP - debt_ratio_upper, 0.0) * phi_gamma
    
        retire = int(np.clip(retire, *retire_bounds))
        contrib = float(np.clip(contrib, *contrib_bounds))
        return np.array([retire, contrib], dtype=float)

    @staticmethod
    def pension_rule_fixed(
            obs, *,  # obs kept for a unified interface (unused)
            retire_age: int | float = 67,
            contribution_rate: float = 0.08
    ) -> np.ndarray:
        """Fixed rule: constant retirement age and contribution rate."""
        return np.array([int(retire_age), float(contribution_rate)], dtype=float)

    @staticmethod
    def pension_action(obs, *, rule: str = "imf", **kwargs) -> np.ndarray:
        """
        Dispatch pension rules. Supported: {"imf", "fixed"}.
        - "imf":   parses Bt/GDP from `obs` (dict or array-like) and applies IMF-style adjustment.
        - "fixed": ignores values in `obs`, returns constant (retire_age, contribution_rate).
        """
        rules = {
            "imf": lambda: GovernmentRules.pension_rule_imf(obs, **kwargs),
            "fixed": lambda: GovernmentRules.pension_rule_fixed(obs, **kwargs),
        }
        try:
            return rules[rule]()
        except KeyError:
            raise ValueError(f"Unknown pension rule: {rule!r}")

    @staticmethod
    def tax_rule_config(obs, tau=0.263, xi=0.049, tau_a=0., xi_a=0., Gt_prob=0.189):
        """Config policy: set parameters explicitly; optional subsidy vector appended."""
        base = np.array([tau, xi, tau_a, xi_a, Gt_prob], dtype=float)
        return np.concatenate([base])

    @staticmethod
    def tax_action(obs, action_dim):
        """
        Dispatch tax rules based on the selected rule.
        Supported rules: 'free_market', 'config'.

        Args:
            obs: The current observation data.
            action_dim: The expected dimensionality of the action space.

        Returns:
            output_actions: The calculated actions according to the selected tax rule.
        """
        # Define the tax rule (can be modified to allow dynamic rule selection)
        rule = "config"  # You can change this to a dynamic rule selection based on conditions
    
        # Define the available tax rules
        rules = {
            "free_market": lambda: GovernmentRules.tax_rule_config(obs, tau=0., xi=0., tau_a=0., xi_a=0., Gt_prob=0.),
            "config": lambda: GovernmentRules.tax_rule_config(obs),  # Default configuration-based rule
        }
    
        # Check if the selected rule is valid
        if rule not in rules:
            raise ValueError(f"Unknown tax rule: {rule}")
    
        # Call the tax rule function to generate the actions
        output_actions = rules[rule]()
    
        # If action_dim exceeds the length of output_actions, add extra actions for government spending allocations
        if action_dim >= len(output_actions):
            N = action_dim - len(output_actions)
            # Create additional actions (e.g., evenly distribute government spending)
            add_actions = np.ones(N) / N  # Example rule: evenly distribute the remaining action space
            output_actions = np.concatenate([output_actions, add_actions])  # Concatenate the new actions
        else:
            # Ensure the generated actions fit within the expected action_dim
            raise ValueError(f"Wrong tax rule: output action size {len(output_actions)} > expected action dim {action_dim}")

        return output_actions

    @staticmethod
    def get_action(type, obs, action_dim):
        """
        Define specific rules for each type of agent
        """
        if type == "central_bank":
            return GovernmentRules.central_bank_action(obs)
        elif type == "pension":
            return GovernmentRules.pension_action(obs)
        elif type == "tax":
            return GovernmentRules.tax_action(obs, action_dim)
        else:
            raise ValueError(f"Unknown government type: {type}")
