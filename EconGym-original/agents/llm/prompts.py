import numpy as np


def build_central_bank_prompt(central_bank_obs: np.ndarray) -> str:
    """Build prompt for central bank authority agent."""
    try:
        o = [float(x) for x in central_bank_obs]

        prompt = f"""
        You are the policy leader of the national central bank.
        Based on the macroeconomic indicators below, your task is to adjust **monetary policy parameters**
        for the next period in order to control inflation and maintain stable economic growth.
        
        General Principles:
        - Policies must be realistic and avoid extreme or impractical values.
        - Aim to stabilize inflation while supporting sustainable economic growth.
        - Avoid excessive interest rate changes that could destabilize consumption, investment, or financial markets.
        - Balance short-term impacts with long-term stability.
        - Recommendations should be reasonable, implementable, and consistent with economic logic.
        
        Current Macro Indicators:
        - Mean household wealth (Top 10%): {o[0]:.2f}
        - Mean household education (Top 10%): {o[1]:.2f}
        - Mean household assets (Bottom 50%): {o[2]:.2f}
        - Mean household education (Bottom 50%): {o[3]:.2f}
        - Market wage rate: {o[4]:.4f}
        - General price level: {o[5]:.4f}
        - Bank lending rate: {o[6]:.4f}
        - Bank deposit rate: {o[7]:.4f}
        - Previous inflation rate: {o[8]:.4f}
        - Previous GDP growth rate: {o[9]:.4f}
        
        Policy Parameters You Can Adjust:
        1. Base interest rate (`base_interest_rate`) ∈ [-0.02, 0.2]
           - Controls borrowing costs and savings incentives.
        2. Reserve requirement (`reserve_ratio`) ∈ [0.0, 0.5]
           - Affects banks’ lending capacity and money supply.
        
        Constraints:
        - All values must be floats.
        - Parameter bounds strictly enforced.
        - Maximum absolute change per parameter: ±0.1 per step.
        
        Output Format:
        Respond only with the decision in strict JSON:
        ```json
        {{
          "base_interest_rate": ...,
          "reserve_ratio": ...
        }}
        ```
        """
        return prompt
    except Exception:
        return "You need to check whether the prompt here is consistent with the obs set in get_obs() within env_core"


def build_pension_prompt(pension_obs: np.ndarray, objective: str) -> str:
    """Build prompt for pension authority agent."""
    objective_map = {
        "pension_gap": "improve the sustainability of the pension fund",
    }
    objective_text = objective_map.get(objective, objective)

    try:
        o = [float(x) for x in pension_obs]

        prompt = f"""
        You are the policy leader of the national pension system.
        Based on the macroeconomic indicators below, your task is to adjust the **retirement age** and **contribution rate**
        for the next period in order to {objective_text}.
        
        General Principles:
        - Policies must be realistic and avoid extreme or impractical values.
        - Ensure the sustainability of the pension system while considering household welfare.
        - Avoid imposing excessive work burdens or sharply reducing disposable income.
        - Balance short-term fiscal needs with long-term demographic trends.
        - Recommendations should be reasonable, implementable, and consistent with economic logic.
        
        Current Macro Indicators:
        - Total accumulated pension fund balance: {o[0]:.2f}
        - Total population: {o[1]:.0f}
        - Current share of retired population: {o[2]:.0f}
        - Previous retirement age: {o[3]:.1f}
        - Previous pension contribution rate: {o[4]:.2f}
        - Government debt (Bt): {o[5]:.2f}
        - Previous GDP: {o[6]:.2f}
        
        Policy Parameters You Can Adjust:
        1. Retirement age (`retire_age`) ∈ [60, 70]
           - Determines the average retirement age in the population.
        2. Contribution rate (`contribution_rate`) ∈ [0.05, 0.2]
           - Represents the fraction of income allocated to pension contributions.
        
        Constraints:
        - retire_age must be int, contribution_rate must be float.
        - Parameter bounds strictly enforced.
        - Maximum absolute change per parameter: ±0.1 per step.
        
        Output Format:
        Respond only with the decision in strict JSON:
        ```json
        {{
          "retire_age": ...,
          "contribution_rate": ...
        }}
        ```
        """

        return prompt
    except Exception:
        return "You need to check whether the prompt here is consistent with the obs set in get_obs() within env_core"


def build_tax_prompt(tax_obs: np.ndarray, objective) -> str:
    """Build prompt for government tax policy agent."""
    # Map objective to human-readable description
    objective_map = {
        "gdp": "maximize GDP growth",
        "gini": "reduce wealth inequality (lower Gini index)",
        "social_welfare": "maximize overall social welfare",
        "mean_welfare": "increase mean household welfare across the population",
        "gdp_gini": "balance GDP growth and wealth equality",
    }
    objective_text = objective_map.get(objective, objective)
    
    try:
        o = [float(x) for x in tax_obs]
            
        prompt = f"""
        You are the fiscal policy leader of the national taxation and budget department.
        Based on the macroeconomic indicators below, your task is to adjust tax policy parameters for the next period
        in order to {objective_text}.
        
        General Principles:
        - Policies must be realistic and avoid extreme or impractical values.
        - Economic stability should be maintained while pursuing the objective.
        - Trade-offs are natural: avoid maximizing one goal at the cost of ignoring others.
        - Balance short-term impact with long-term sustainability.
        - Recommendations should be reasonable, implementable, and consistent with economic logic.
        
        Current Macro Indicators:
        - Mean household wealth (Top 10%): {o[0]:.2f}
        - Mean household education (Top 10%): {o[1]:.2f}
        - Mean household assets (Bottom 50%): {o[2]:.2f}
        - Mean household education (Bottom 50%): {o[3]:.2f}
        - Market wage rate: {o[4]:.4f}
        - General price level: {o[5]:.4f}
        - Bank lending rate: {o[6]:.4f}
        - Bank deposit rate: {o[7]:.4f}
        - Government debt: {o[8]:.4f}
        
        Policy Parameters You Can Adjust:
        1. Government spending ratio (`Gt_prob`) ∈ [0.0, 0.6]
        2. Average marginal income tax rate (`tau`) ∈ [0.0, 0.6]
           - Parameter of the nonlinear HSV tax function, affects labor supply and incentives.
        3. Tax progressivity slope for income (`xi`) ∈ [0.0, 2.0]
           - HSV function parameter, determines how steeply income tax increases with income.
        4. Average marginal asset tax rate (`tau_a`) ∈ [0.0, 0.05]
        5. Tax progressivity slope for asset (`xi_a`) ∈ [0.0, 2.0]
        
        Constraints:
        - All values must be floats.
        - Parameter bounds must be strictly enforced.
        - Maximum absolute change per parameter: ±0.1 per step.
        
        Output Format:
        Respond only with the decision in strict JSON:
        ```json
        {{
          "Gt_prob": ...,
          "tau": ...,
          "xi": ...,
          "tau_a": ...,
          "xi_a": ...
        }}```
        """
        return prompt
    
    except Exception:
        return "You need to check whether the prompt here is consistent with the obs set in get_obs() within env_core"


def build_bank_prompt(bank_obs: np.ndarray) -> str:
    try:
        b = [float(x) for x in bank_obs]

        base_interest_rate = b[0] if len(b) > 0 else 0.03
        reserve_ratio = b[1] if len(b) > 1 else 0.1
        last_lending_rate = b[2] if len(b) > 2 else base_interest_rate + 0.02
        last_deposit_rate = b[3] if len(b) > 3 else base_interest_rate - 0.005
        last_total_loans = b[4] if len(b) > 4 else 0.0
        last_total_deposits = b[5] if len(b) > 5 else 0.0

        prompt = f'''You are the decision maker of a commercial bank. Based on the indicators below, 
        set the next period's lending and deposit rates to balance profitability and stability, 
        while respecting the central bank's benchmark rate and reserve requirements.
        
        ---
        Current indicators:
        - Central bank base interest rate (benchmark): {base_interest_rate:.4f}
        - Reserve requirement ratio: {reserve_ratio:.4f}
        - Last lending rate: {last_lending_rate:.4f}
        - Last deposit rate: {last_deposit_rate:.4f}
        - Total loans (previous): {last_total_loans:.2f}
        - Total deposits (previous): {last_total_deposits:.2f}
        ---
        Constraints:
        - lending_rate ∈ [base_interest_rate + 0.01, base_interest_rate + 0.03]
        - deposit_rate ∈ [base_interest_rate - 0.01, base_interest_rate]
        - All values must be floats.
        ---
        Output strict JSON only:
        ```json
        {{
        "lending_rate": ..., 
        "deposit_rate": ...
        }}
        ```
        '''
        return prompt
    except Exception:
        return "You need to check whether the prompt here is consistent with the bank_obs set in get_bank_observations() within set_observation"


def build_market_prompt(firm_obs: np.ndarray) -> str:
    """Build prompt for firm-level market decision-making."""
    try:
        # firm_obs shape: (firm_n, 3) -> [capital_next, productivity(Zt), interest_rate]
        firm_n = int(firm_obs.shape[0]) if firm_obs is not None and firm_obs.ndim == 2 else 1

        preview = []
        if firm_obs is not None and firm_obs.ndim == 2:
            for i in range(min(firm_n, 5)):
                K, Z, r = [float(x) for x in firm_obs[i].tolist()]
                preview.append(f"Firm {i}: capital={K:.2f}, productivity={Z:.2f}, borrowing_rate={r:.4f}")

        preview_text = "\n".join(preview)

        prompt = f"""
                You are responsible for pricing and wage-setting decisions for each firm in the market.
                Based on firm-level indicators (capital, productivity, borrowing rate), determine **price** and **wage** adjustments
                for the next period to maximize profit while keeping the market stable.
                
                General Principles:
                - Consumer demand decreases as prices rise (inverse relationship).
                - Labor supply increases as wages rise (positive relationship).
                - Unsold goods represent losses for firms and the economy.
                - Policies should avoid extreme or unrealistic values.
                - Ensure that firms remain competitive and sustainable in the long run.
                
                Firms (showing up to 5 for preview):
                {preview_text if preview_text else 'No preview available'}
                
                For each firm i in [0..{firm_n - 1}], output:
                - price ∈ [0., 100.]
                - wage ∈ [0., 100.]
                
                Constraints:
                - These are direct action values bounded by the environment.
                - Output exactly {firm_n} entries, one per firm, in order.
                
                Output Format:
                Respond only with the decision in strict JSON as an array of objects:
                ```json
                [
                  {{"price": ..., "wage": ...}},
                  {{"price": ..., "wage": ...}}
                ]
                ```
                """
        return prompt
    except Exception:
        return "You need to check whether the prompt here is consistent with the firm_obs set in get_firm_observations() within set_observation"
