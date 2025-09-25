# Q5: How to coordinate monetary and fiscal policies?

## 1. Introduction

#### 1.1 Fundamental Functions of the Treasury and Central Bank

The Treasury Department primarily **influences economic activity through taxation, government spending, and debt management**, fulfilling roles in resource allocation, income redistribution, and macroeconomic stabilization. The central bank, by contrast, **implements monetary policy and ensures financial stability—adjusting interest rates, money supply**, and conducting open-market operations to control inflation, promote employment, and safeguard the financial system.

#### 1.2 Necessity and Context for Policy Coordination

Although fiscal and monetary authorities have distinct mandates, their objectives are closely aligned. Operating in isolation can weaken overall effectiveness or even produce **“policy offset” (e.g., fiscal expansion counteracted by tight monetary policy).** Since the 2008 financial crisis and the COVID-19 shock in 2020, many countries have pursued coordinated fiscal–monetary packages (such as combining fiscal stimulus with quantitative easing) to improve transmission, stabilize expectations, and boost aggregate demand.

### 1.3 Research Questions

This study uses an economic simulation platform to investigate the economic impacts of fiscal–monetary policy coordination, specifically examining:

* **GDP**​**​ Effects: ​**How does coordinated policy intervention affect short-term recovery and long-term economic growth compared to uncoordinated actions?
* **Wealth Distribution: ​**What are the distributional consequences of policy coordination, particularly in terms of asset ownership and intergenerational inequality?
* **Household Consumption: ​**How do combined fiscal transfers and low interest rates influence aggregate and heterogeneous household spending behavior?

#### 1.4 Research Significance

* **Deepening Systemic Understanding of Macro-Policy Interactions:**  Explore the feedback loops between fiscal and monetary measures to help researchers and policymakers build a coordinated, multi-agency stabilization framework.
* **Optimizing Transmission Paths:**  Analyze how coordinated policy affects micro-level decisions and macro outcomes, avoiding “policy clashes” or inefficient pass-through.
* **Advancing Complex Policy-Mix Design:**  Leverage RL Agents and similar methods to learn multi-objective optimal control paths within the policy space and to explore AI-driven policy design solutions.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role               | Selected Type        | Role Description                                                                                                             | Observation                                                                                                  | Action                                                                                 | Reward                                              |
| ------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Individual**                | Ramsey Model         | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.                     | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Government(Tax)**          | Fiscal Authority     | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.                         |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
| **Government(Central Bank)** | Central Bank         | Central Bank adjusts nominal interest rates and reserve requirements, transmitting monetary policy to households and firms. |\$\$o\_t^g = (\\mathcal{A}\_{t}, \\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, \\pi\_{t-1}, g\_{t-1})\$\$ <br>Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, inflation rate, growth rate. | $a_t^{\text{cb}} = ( \phi_t, \iota_t )$<br>Reserve ratio, benchmark rate | Inflation/GDP stabilization                                  |
| **Firm**                     | Perfect Competition  | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                       | /                                                                                                            | /                                                                                    | Zero (long-run)                                     |
| **Bank**                     | Commercial Bank     | Commercial Banks strategically set deposit and lending rates to maximize profits, subject to central bank constraints.      | $o_t^{\text{bank}} = ( \iota_t, \phi_t, r^l_{t-1}, r^d_{t-1}, loan, F_{t-1} )$<br>Benchmark rate, reserve ratio, last lending rate, last deposit_rate, loans, pension fund. | $a_t^{\text{bank}} = ( r^d_t, r^l_t )$<br>Deposit, lending decisions | $r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$<br>Interest margin |



---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
Households optimize their labor supply, savings, and consumption decisions based on life-cycle optimization principles. As the microfoundation of policy transmission, their behaviors provide crucial feedback to both fiscal and monetary policies.

**Government → Fiscal Authority & Central Bank**  
**Fiscal Authority :** Responsible for designing tax and spending policies, adjusting aggregate demand and income distribution, and managing public debt to ensure fiscal sustainability. Its decisions directly affect households’ disposable income and government funding allocations, making it a key instrument for influencing growth and equity.
**Central Bank:** Controls inflation, stabilizes prices, and maintains financial-system liquidity by adjusting interest rates and money supply. Its policies have broad but indirect impacts on consumption, investment, and credit behavior, positioning it as a central actor in macroeconomic stabilization.

**Firm → Perfect Competition**  
Wages and goods prices are determined by supply and demand, acting as the intermediary mechanism through which fiscal and monetary policies influence household and firm behavior.

**Bank → Commercial Bank**  
Simulate the formation of deposit and lending rates, reflecting how central-bank policies transmit to investment, interest rates, and liquidity.

---

## 3. Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent        | Imitates real-world behavior by training on empirical data. Enables realistic micro-level behavior.                                         |
| Government             | RL Agent         | Use reinforcement learning to jointly adjust fiscal and monetary tools, achieving optimal growth–stability coordination.                    |
| Firm                 | Rule-Based Agent | Wages and prices are set by supply and demand; rules capture rapid market feedback to policy shocks.                                         |
| Bank  | Rule-Based Agent | Interest rates and investment returns feed back savings and policy changes via rules, maintaining capital‐market equilibrium and liquidity. |

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "dbl_government"
```

This command loads the configuration file `cfg/dbl_government.yaml`, which defines the setup for the "dbl_government" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `dbl_government.yaml`

```yaml
device_num: 0                      # GPU device number (0 = first GPU)

Environment:                       # Parameters for the environment
  env_core:
    problem_scene: "dbl_government"
    consumption_tax_rate: 0.0      # Initial consumption tax rate
    estate_tax_rate: 0.0           # Initial estate/inheritance tax rate
    estate_tax_exemption: 13610000 # Estate tax exemption threshold (USD)
    screen: False                  # Whether to render the environment (currently unavailable)
    episode_length: 300            # Number of steps in one episode
    step_cnt: 0                    # Initial step counter

  Entities:
    - entity_name: 'government'    # Government agent parameters
      entity_args:
        params:
          type: "central_bank"     # government type ("tax", "pension", or "central_bank")
          type_list: [ 'tax', 'pension', 'central_bank' ]   # Supported government roles
          tau: 0.263               # Initial HSV tax parameter: average level of marginal income tax. If fiscal government ("tax") is active, actions will override this value.
          xi: 0.049                # Initial HSV tax parameter: slope (progressivity) of income tax schedule. Larger xi → more progressive taxation.
          tau_a: 0.0               # Initial HSV parameter: average level of marginal asset tax
          xi_a: 0.0                # Initial HSV parameter: slope of marginal asset tax schedule
          Gt_prob: 0.189           # Initial ratio of government spending to GDP
          retire_age: 65           # Default statutory retirement age (overridden if pension government is active)
          contribution_rate: 0.08  # Pension contribution rate (employee share of wage)
          pension_growth_rate: 0.01 # Annual growth rate of pension benefits
          base_interest_rate: 0.03 # Central bank base interest rate (overridden if central bank is active)
          reserve_ratio: 0.08      # Central bank reserve requirement ratio
          gov_task: "gdp"          # Optimization objective ("gdp", "gini", "social_welfare", "mean_welfare", "gdp_gini", "pension_gap"). For central bank: inflation stabilization.
          tax_type: "ai_agent"     # Tax agent type ("ai_agent", "us_federal", "saez")
          real_gdp: 254746e8       # Initial real GDP (USD)
          real_debt_rate: 1.2129   # Initial government debt-to-GDP ratio
          real_population: 333428e3 # Initial population size
        central_bank:
          type: "central_bank"     # Central bank policy agent
          action_dim: 2            # Dimension of action space
          action_space:
            low: -1                # Lower bound for normalized monetary actions
            high: 1                # Upper bound for normalized monetary actions
            dtype: float32
          initial_action:          # Initial monetary policy parameters
            base_interest_rate: 0.03
            reserve_ratio: 0.08
          real_action_max: [0.1, 0.2]   # Max values for (base interest rate, reserve ratio)
          real_action_min: [-0.02, 0.0] # Min values for (base interest rate, reserve ratio)

    - entity_name: 'government'    # Government agent parameters
      entity_args:
        params:
          type: "tax"     # government type ("tax", "pension", or "central_bank")
          type_list: [ 'tax', 'pension', 'central_bank' ]   # Supported government roles
          tau: 0.263               # Initial HSV tax parameter: average level of marginal income tax. If fiscal government ("tax") is active, actions will override this value.
          xi: 0.049                # Initial HSV tax parameter: slope (progressivity) of income tax schedule. Larger xi → more progressive taxation.
          tau_a: 0.0               # Initial HSV parameter: average level of marginal asset tax
          xi_a: 0.0                # Initial HSV parameter: slope of marginal asset tax schedule
          Gt_prob: 0.189           # Initial ratio of government spending to GDP
          retire_age: 65           # Default statutory retirement age (overridden if pension government is active)
          contribution_rate: 0.08  # Pension contribution rate (employee share of wage)
          pension_growth_rate: 0.01 # Annual growth rate of pension benefits
          base_interest_rate: 0.03 # Central bank base interest rate (overridden if central bank is active)
          reserve_ratio: 0.08      # Central bank reserve requirement ratio
          gov_task: "gdp"          # Optimization objective ("gdp", "gini", "social_welfare", "mean_welfare", "gdp_gini", "pension_gap"). For central bank: inflation stabilization.
          tax_type: "ai_agent"     # Tax agent type ("ai_agent", "us_federal", "saez")
          real_gdp: 254746e8       # Initial real GDP (USD)
          real_debt_rate: 1.2129   # Initial government debt-to-GDP ratio
          real_population: 333428e3 # Initial population size

        tax:
          type: "tax"              # Fiscal authority agent executing tax policy
          action_dim: 5            # Dimension of action space. If firm_n > 1, action_dim expands (gov spending allocated by firm).
          action_space:
            low: -1                # Lower bound for normalized actions. In RL training, constraining policy outputs to (-1, 1) helps stabilize learning.
            high: 1                # Upper bound for normalized actions
            dtype: float32         # Action data type
          initial_action: # Initial tax policy parameters
            tau: 0.263
            xi: 0.049
            tau_a: 0.0
            xi_a: 0.0
            Gt_prob: 0.189
          real_action_max: [ 0.6, 2, 0.05, 2.0, 0.6 ]  # Max real-world values for scaling. Each agent's (real_action_max, real_action_min) defines the meaningful range of actions in real-world terms.
          # After actions enter env.step(), they are automatically scaled and clipped to this range.
          real_action_min: [ 0.0, 0.0, 0.0, 0.0, 0.0 ] # Min real-world values for scaling.


    - entity_name: 'households'         # Household agents
      entity_args:
        params:
          type: 'ramsey'                # Household type ('ramsey', 'OLG', 'OLG_risk_invest', 'ramsey_risk_invest')
          type_list: [ 'ramsey', 'OLG', 'OLG_risk_invest', 'ramsey_risk_invest' ]  # Supported household models
          households_n: 100             # Number of households
          CRRA: 1                       # Coefficient of relative risk aversion
          IFE: 1                        # Inverse Frisch elasticity
          stock_alpha: 0.1              # Sensitivity coefficient: controls stock price response to buy/sell imbalance
          action_dim: 2                 # Household action dimension
          e_p: 2.2e-6                   # Probability of transitioning from normal to superstar state
          e_q: 0.990                    # Probability of remaining in the current state
          rho_e: 0.982                  # Persistence in state transitions (normal/superstar)
          sigma_e: 0.200                # Volatility of standard normal shocks
          super_e: 504.3                # Labor productivity in superstar state
          at_min: -1e6                  # Minimum asset holdings
          h_max: 2512                   # Maximum annual working hours
          real_action_max: [ 1.0, 1.0 ]   # Max values for (savings share of income, labor share of h_max)
          real_action_min: [ -0.5, 0.0 ]  # Min values for (savings share of income, labor share of h_max)
          action_space:
            low: -1                     # Lower bound for normalized actions
            high: 1                     # Upper bound for normalized actions
            dtype: float32
        OLG:
          birth_rate: 0.011             # Birth rate (e.g., US demographics)
          initial_working_age: 24       # Starting age for new workers

    - entity_name: 'market'             # Firm agents in the market
      entity_args:
        params:
          type: "perfect"               # Market type: 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly'
          type_list: [ 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly' ] # Supported market types
          alpha: 0.36                   # Capital share in Cobb-Douglas production
          Z: 1.0                        # Initial productivity level
          sigma_z: 0.0038               # Std. deviation of productivity shocks
          epsilon: 0.5                  # Demand elasticity parameter in CES utility
          real_action_max: [ 1e2, 1e2 ]   # Max values for firm actions (price, wage rate)
          real_action_min: [ 0.0001, 0.0001 ] # Min values for firm actions

        perfect:
          type: "perfect"               # Perfect competition market
          firm_n: 1                     # Number of firms
          action_dim: 0                 # No decision variables in perfect competition
          action_space:
            low: -1.
            high: 1.
            dtype: float32
          initial_action: null
        monopoly:
          type: "monopoly"              # Monopoly market
          action_dim: 2                 # Firm sets price and wage rate
          firm_n: 1                     # One monopolist
          action_space:
            low: -1.
            high: 1.
            dtype: float32
          initial_action:
            price: 1                    # Initial product price
            WageRate: 1                 # Initial wage rate
        monopolistic_competition:
          type: "monopolistic_competition" # Monopolistic competition
          action_dim: 2
          firm_n: 10                    # Number of firms (user-defined)
          action_space:
            low: -1.
            high: 1.
            dtype: float32
          initial_action:
            price: 1
            WageRate: 1
        oligopoly:
          type: "oligopoly"             # Oligopoly market
          action_dim: 2
          firm_n: 2                     # Number of firms (user-defined)
          action_space:
            low: -1.
            high: 1.
            dtype: float32
          initial_action:
            price: 1
            WageRate: 1

    - entity_name: 'bank'               # Bank agent
      entity_args:
        params:
          type: 'non_profit'            # Bank type ('non_profit', 'commercial')
          type_list: [ 'non_profit', 'commercial' ]  # Supported bank types
          n: 1                          # Number of banks
          lending_rate: 0.0345          # Initial loan interest rate
          deposit_rate: 0.0345          # Initial deposit interest rate
          reserve_ratio: 0.1            # Reserve requirement ratio
          base_interest_rate: 0.0345    # Base policy interest rate
          depreciation_rate: 0.06       # Capital depreciation rate
          real_action_max: [ 0.2, 0.1 ]   # Max values for (loan rate, deposit rate)
          real_action_min: [ 0.03, -0.001 ] # Min values for (loan rate, deposit rate)
          action_space:
            low: -1
            high: 1
            dtype: float32
        non_profit:
          type: "non_profit"            # Non-profit bank (fixed rates)
          action_dim: 0
          initial_action: null
        commercial:
          type: "commercial"            # Commercial bank (chooses lending/deposit rates)
          action_dim: 2
          initial_action: [ 0.03, 0.03 ]  # Initial (lending rate, deposit rate)


Trainer:                                # Store parameters related to policy training/testing
  log_std_min: -20                      # Minimum log standard deviation (policy net)
  log_std_max: 2                        # Maximum log standard deviation
  hidden_size: 128                      # Hidden layer size of neural nets
  cuda: False                           # Whether to use CUDA GPU
  q_lr: 3e-4                            # Learning rate for Q-network
  p_lr: 3e-4                            # Learning rate for policy network
  buffer_size: 1e6                      # Replay buffer size
  n_epochs: 1000                       # Number of training epochs
  update_cycles: 100                    # Training updates per epoch
  epoch_length: 300                     # Sample Steps per epoch
  display_interval: 1                   # Logging/printing interval
  batch_size: 64                        # Batch size for training
  gamma: 0.975                          # Discount factor for RL
  tau: 0.95                             # Soft update coefficient of DDPG
  eval_episodes: 5                      # Number of evaluation episodes
  init_exploration_steps: 1000          # Steps before training starts (exploration)
  ppo_tau: 0.95                         # PPO smoothing coefficient (GAE λ)
  ppo_gamma: 0.99                       # PPO discount factor
  eps: 1e-5                             # Small epsilon for numerical stability
  update_epoch: 20                      # PPO update epochs per iteration
  clip: 0.1                             # PPO clipping ratio
  vloss_coef: 0.5                       # Value loss coefficient
  ent_coef: 0.01                        # Entropy regularization coefficient
  max_grad_norm: 0.5                    # Gradient clipping norm
  update_freq: 10                       # Frequency of policy updates
  initial_train: 10                     # Initial training steps
  noise_rate: 0.01                      # Exploration noise rate
  epsilon: 0.1                          # Epsilon-greedy exploration rate
  save_interval: 10                     # Model save interval
  house_alg: "bc"
  gov_alg: "rule_based"
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  central_bank_gov_alg: 'ppo'
  tax_gov_alg: 'ppo'
  update_each_epoch: 20                 # Updates per epoch
  seed: 1                               # Random seed
  wandb: False                          # Whether to log with W&B/Swanlab
  test: False                           # Test mode flag
  bc_test: True                         # in agents/behavior_cloning/bc_agent.py. if False, get actions from real data, and trained via BC; if True, get action from trained BC policy.
```
---

## 5. Illustrative Experiments

### Experiment 1: Analysis of Fiscal–Monetary Policy Coordination Effects

* **Experiment Description:**
  
  In the simulated economy, allow the Treasury and the Central Bank to learn optimal coordination strategies via RL Agents. Under different objective functions (e.g., “stability priority” vs. “growth priority”), evaluate the impact of coordinated policies on key macro indicators such as GDP, inflation, and the Gini coefficient.
* **Core Experimental Variables:**
  
  * The fiscal and monetary policy departments, calibrated through economic modeling, were designed as follows: the Treasury Department implemented the **Saez Tax** system, while the Central Bank adopted the **Taylor Rule** as its behavioral logic.
  * Scale of fiscal spending
  * Income-tax rate & government-debt ceiling
  * Nominal interest rate or money-supply growth rate
  * Macro outcomes: GDP, inflation rate, wealth Gini coefficient
* **Baselines：**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **​OLG\_tax(blue line):​**The households use the **Behavior Cloning Agent** and the goverment modeled as ​**RL-Agent**​.The Goverment represent **Treasury Department only.**
  * **​OLG\_CenBank(green line):​**The households use the **Behavior Cloning Agent** and the goverment modeled as ​**RL-Agent**​.The Goverment represent **Central Bank**​**​ only.**
  * **​OLG\_tax\_CenBank(yellow line):​**The households use the **Behavior Cloning Agent** and the goverment modeled as ​**RL-Agent**​.The Goverment represent both **the Treasury Department and The ​**​​**Central Bank.**

![Monetary Q5 P1](../img/Moneraty%20Q5%20P1.png)

**Figure 1:** Compared the effects of separate operations by the Treasury Department or the Central Bank with the coordination of their policies . Despite the fact that, in the short term, single-department operations achieve higher GDP growth rates, coordinated policy implementation leads to a longer-lasting simulation economy and results in better long-term GDP growth.


![Monetary Q5 P2](../img/Moneraty%20Q5%20P2.png)

**​Figure 2:​**Under the coordinated policies of the Treasury and the Central Bank, the wealth Gini coefficient in the short term is higher compared to single-sector policies. However, after the 60th year, thanks to more stable long-term economic growth, the wealth disparity under policy coordination is smaller than under single-sector policies.

* **Baselines：**
  * **Left panel: ​**Different bar colors represent **age cohorts** (e.g., <24, 25–34, 35–44, 45–54, 55–64, 65–74, 75–84, 85+, total).
  * **Right panel:** Different bar colors represent ​**income classes ​**​(rich, middle, poor, and mean).

![Monetary Q5 P3](../img/Moneraty%20Q5%20P3.png)

​**​ Figure 3**​: When examining a specific year (e.g., The Year 25), the coordinated policies result in wealth being more concentrated among the middle-aged and younger population (green and orange-yellow lines), whereas single-sector policies lead to a relatively equal distribution of wealth across youth to middle-aged groups.

![Monetary Q5 P4](../img/Moneraty%20Q5%20P4.png)

​**Figure 4**​: The coordination between the Treasury and the Central Bank has no significant short-term impact on the income Gini coefficient. In the long term, the income disparity under policy coordination is significantly smaller than under the Treasury-only policy, but still notably higher than under the Central Bank-only policy.

![Monetary Q5 P5](../img/Moneraty%20Q5%20P5.png)

​**Figure 5**​: The collaboration between the Treasury Department and the Central Bank significantly reduces the overall social welfare.

* The coordination between fiscal policy (executed by the Treasury) and monetary policy (executed by the Central Bank) produces complex macroeconomic effects. However, in the long run, policy coordination enhances the long-term growth vitality of the simulated economy, with no significant difference in wealth inequality compared to the single-policy scenarios. Analyzing wealth distribution supports this conclusion (e.g.,wealth is more concentrated among middle-aged and younger generations).
* Moreover, policy coordination leads to a noticeable decline in social welfare, which may be due to the higher efficiency of market mechanisms under coordinated policies.



