import numpy as np


def build_central_bank_prompt(central_bank_obs: np.ndarray) -> str:
    try:
        o = [float(x) for x in central_bank_obs]

        prompt = f'''You are the policy leader of a national central bank and budget department. Based on the macro 
        indicators below,your task is to adjust **monetary policy parameters** for the next period to **maximize social 
        welfare**.
        ---
        **Your objectives**:
        - Maximize household **consumption** (more is better)
        - Minimize **work burden** (longer working years reduce welfare)
        - Reduce **wealth inequality** (lower Gini)
        - Maintain **stable GDP growth**
        ---
        **Macro indicators**:
        - Mean household wealth of the Top 10% (households): {o[0]:.2f}
        - Mean household education level of the Top 10% (households): {o[1]:.2f}
        - Mean household asset of the Bottom 50% (households): {o[2]:.2f}
        - Mean household education level of the Bottom 50% (households): {o[3]:.2f}
        - Market wage rate: {o[4]:.4f}
        - General price level: {o[5]:.4f}
        - Bank lending rate: {o[6]:.4f}
        - Bank deposit rate: {o[7]:.4f}
        - Society inflation rate: {o[8]:.4f}
        - Society economic growth rate: {o[9]:.4f}
        ---
        **Your reasoning should consider**:
        1. Does a higher base interest rate encourage savings at the expense of current consumption?
        2. Could low rates stimulate spending but threaten pension fund sustainability?
        3. Is the current inflation level justifying an increase in interest rates?
        4. How will changes affect borrowing costs and thus labor participation or retirement decisions?
        5. Will the new rate support stable GDP growth and lower inequality?
        ---
        **Policy Parameters You Can Adjust**:
        1. Base interest rate (`base_interest_rate`) ∈ [-0.02,0.2]
           → Controls borrowing costs and savings incentives
        2. Reserve requirement (`reserve_ratio`) ∈ [0,0.5]
           → Affects banks' lending capacity and money supply
        ---
        **Constraints**:
        - All values must be floats.
        - Parameter bounds strictly enforced.
        - Max absolute change per parameter: ±0.1
        ---
        **Output format**:
        Respond only with the decision in strict JSON:
        ```json
        {{
        "base_interest_rate": ...,
        "reserve_ratio": ...
        }}
        ```
        '''
        return prompt
    except Exception:
        return "You need to check whether the prompt here is consistent with the obs set in get_obs() within env_core"


def build_pension_prompt(pension_obs: np.ndarray) -> str:
    try:
        o = [float(x) for x in pension_obs]

        prompt = f'''You are the policy leader of a national pension system and budget department. Based on the macro 
        indicators below, your task is to adjust the **retirement age** and **contribution rate** for the next period 
        to **maximize social welfare**.
        ---
        **Your objectives**:
        - Maximize household **consumption** (more is better)
        - Minimize **work burden** (longer working years reduce welfare)
        - Reduce **wealth inequality** (lower Gini)
        - Maintain **stable GDP growth**
        ---
        **Macro indicators**:
        - Total accumulated pension account balance: {o[0]:.2f}
        - Current total population: {o[1]:.0f}
        - Elderly population (aged 65 and above, 7% of total population): {o[2]:.0f}
        - Statutory retirement age: {o[3]:.1f}
        - Pension contribution rate: {o[4]:.2f}
        - Government transfer/policy factor (Bt): {o[5]:.2f}
        - Gross Domestic Product (GDP): {o[6]:.2f}
        ---
        **Your reasoning should consider**:
        1. Does increasing the retirement age help sustain the pension system without overly burdening workers?
        2. Could higher contribution rates improve pension fund sustainability but reduce disposable income and consumption?
        3. How will changes affect labor market participation and overall economic productivity?
        4. Will adjustments contribute to reducing wealth inequality?
        5. Are there any inflationary pressures that might influence these decisions?
        ---
        **Policy Parameters You Can Adjust**:
        1. Retirement age (`retire_age`) ∈ [50, 75]
           → Determines the average retirement age in the population.
        2. Contribution rate (`contribution_rate`) ∈ [0.05, 0.2]
           → Represents the percentage of income that goes towards pension contributions.
        **Constraints**:
        - All values must be floats.
        - Parameter bounds strictly enforced.
        - Max absolute change per parameter: ±0.1
        ---
        **Output format**:
        Respond only with the decision in strict JSON:
        ```json
        {{
        "retire_age": ..., 
        "contribution_rate": ... 
        }}
        ```
        '''
        return prompt
    except Exception:
        return "You need to check whether the prompt here is consistent with the obs set in get_obs() within env_core"


def build_tax_prompt(tax_obs: np.ndarray) -> str:
    try:
        o = [float(x) for x in tax_obs]

        prompt = f'''You are the fiscal policy leader of a national taxation and budget department.Based on the macro 
        indicators below,  your task is to adjust **tax policy parameters** for the next period to **maximize 
        social welfare**.
        ---
        **Your objectives**:
        - Maximize household **consumption** (more is better)
        - Minimize **wealth inequality** (lower Gini index)
        - Maintain **stable GDP growth**
        - Ensure **sustainable government debt** levels
        - Improve **public service quality** through efficient spending
        ---
        **Macro indicators**:
        - Mean household wealth of the Top 10% (households): {o[0]:.2f}
        - Mean household education level of the Top 10% (households): {o[1]:.2f}
        - Mean household asset of the Bottom 50% (households): {o[2]:.2f}
        - Mean household education level of the Bottom 50% (households): {o[3]:.2f}
        - Market wage rate: {o[4]:.4f}
        - General price level: {o[5]:.4f}
        - Bank lending rate: {o[6]:.4f}
        - Bank deposit rate: {o[7]:.4f}
        - Government debt: {o[8]:.4f}
        ---
        **Policy Parameters You Can Adjust**:
        1. Government spending ratio (`Gt_prob`) ∈ [0.01, 0.1]
           → Controls government consumption as a share of GDP
        2. Average marginal income tax rate (`τ`) ∈ [0.01, 1]
            → Affects labor supply, disposable income, and work incentives
        3. Tax progressivity slope for income (`ξ`) ∈ [0.0, 2.0]
            → Determines how steeply income tax increases with income
        ---
        **Your reasoning should consider**:
        1. Does increasing τ discourage labor participation or increase inequality?
        2. Could raising τ_a reduce capital formation or disproportionately affect middle-class savers?
        3. Is current inequality high enough to justify steeper tax slopes (ξ)?
        4. Are government revenues sufficient to support Gt_prob at its desired level?
        5. Will changes help stabilize GDP growth and improve household welfare?
        ---
        **Constraints**:
        - All values must be floats.
        - Parameter bounds strictly enforced.
        - Max absolute change per parameter: ±0.1
        ---
        **Output format**:
        Respond only with the decision in strict JSON:
        ```json
        {{
        "Gt_prob": ..., 
        "tau": ...,    
        "xi": ...      
        }}
        ```
        '''
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
    try:
        # firm_obs shape: (firm_n, 3) -> [capital_next, productivity(Zt), interest_rate]
        firm_n = int(firm_obs.shape[0]) if firm_obs is not None and firm_obs.ndim == 2 else 1

        preview = []
        if firm_obs is not None and firm_obs.ndim == 2:
            for i in range(min(firm_n, 5)):
                K, Z, r = [float(x) for x in firm_obs[i].tolist()]
                preview.append(f"Firm {i}: capital={K:.2f}, productivity={Z:.2f}, interest_rate={r:.4f}")

        preview_text = "\n".join(preview)

        prompt = f'''You control pricing and wage-setting for each firm in a product market. Given each firm's capital, 
        productivity, and borrowing rate, decide a modest adjustment for price and wage for the next period to maximize 
        profit while maintaining stability.
        ---
        Firms (showing up to 5 for preview):
        {preview_text if preview_text else 'No preview available'}
        ---
        For each firm i in [0..{firm_n - 1}], output:
        - price: a small adjustment in [-0.2, 0.2]
        - wage: a small adjustment in [-0.2, 0.2]
        Notes:
        - These are direct action values constrained by the environment's action space.
        - Return exactly {firm_n} entries, one per firm, in order.
        ---
        Output strict JSON only as an array of objects:
        ```json
        {[
            {"price": ..., "wage": ...},
            {"price": ..., "wage": ...}
        ]}
        ```
        '''
        return prompt
    except Exception:
        return "You need to check whether the prompt here is consistent with the firm_obs set in get_firm_observations() within set_observation"
