import numpy as np


class BankRules:
    """
    Example Rules for commercial bank actions.
    """

    @staticmethod
    def get_action(type, action_dim, obs):
        """
        Calculate the commercial bank's lending and deposit rates.

        Args:
            type (str): 'non_profit', 'commercial'. Specifies the bank type.
            action_dim (int): The dimensionality of the action space (expected to be 2 for lending and deposit rates).
            obs (list or np.ndarray): Observations containing the base interest rate, reserve rate,
                                      last lending rate, last deposit rate, total loans, and total deposits.

        Returns:
            np.ndarray: An array containing the calculated lending rate and deposit rate.

        Raises:
            ValueError: If the bank type is not recognized.
        """
        if type == 'non_profit':
            # Action for non-profit bank, setting action_dim to 0 as it is non-profit
            return np.random.randn(action_dim)  # action_dim of non_profit bank is 0.
        elif type == 'commercial':
            # Extract observations
            base_interest_rate = obs[0]  # Base interest rate (e.g., central bank rate)
            reserve_rate = obs[1]  # Reserve rate
            last_lending_rate = obs[2]  # Last lending rate
            last_deposit_rate = obs[3]  # Last deposit rate
            total_loans = obs[4]  # Total loans
            total_deposits = obs[5]  # Total deposits
        
            # Calculate lending and deposit rates considering the above factors
            lending_rate = BankRules._calculate_lending_rate(base_interest_rate, reserve_rate, total_loans)
            deposit_rate = BankRules._calculate_deposit_rate(base_interest_rate, reserve_rate, total_deposits)
        
            # Create the action array
            action = np.array([lending_rate, deposit_rate])
        
            # Ensure the action array has the correct dimension
            if action.shape[0] != action_dim:
                raise ValueError(f"Action dimension mismatch: expected {action_dim}, got {action.shape[0]}")
        
            return action
        else:
            raise ValueError(f"Invalid bank type: '{type}'. Expected 'non_profit' or 'commercial'.")

    @staticmethod
    def _calculate_lending_rate(base_interest_rate, reserve_rate, total_loans, risk_premium=0.03, operating_cost=0.05):
        """
        Calculate the lending rate based on base interest rate, reserve rate, total loans, risk premium, and operating cost.
        """
        # Adjusting the lending rate based on reserve rate and total loans
        liquidity_adjustment = 0.02 * reserve_rate  # If reserve_rate increases, liquidity decreases, increasing lending rate
        # Similarly, scale the loan demand adjustment based on the ratio of total loans relative to GDP or a similar baseline
        loan_demand_adjustment = 0 if total_loans < 0 else np.log10(total_loans + 1) * 0.005   # Using logarithmic scaling to prevent excessive increase in rate

        # Lending rate formula considering risk, operating cost, liquidity and loan demand
        lending_rate = base_interest_rate + risk_premium + operating_cost + liquidity_adjustment + loan_demand_adjustment
        return lending_rate

    @staticmethod
    def _calculate_deposit_rate(base_interest_rate, reserve_rate, total_deposits, operating_spread=0.02,
                                market_adjustment=0.01):
        """
        Calculate the deposit rate based on base interest rate, reserve rate, total deposits, operating spread, and market adjustment.
        """
        # Adjusting the deposit rate based on total deposits and reserve rate
        # Instead of directly scaling by total_deposits, consider a ratio to avoid very large numbers
        deposit_incentive = 0 if total_deposits < 0 else np.log10(total_deposits + 1) * 0.005  # Using logarithmic scaling to dampen the effect of large deposits
        liquidity_adjustment = 0.02 * reserve_rate  # If reserve_rate increases, liquidity decreases, increasing deposit rate
    
        # Deposit rate formula considering liquidity, deposits and market adjustments
        deposit_rate = base_interest_rate - operating_spread + market_adjustment + deposit_incentive + liquidity_adjustment
        return deposit_rate

