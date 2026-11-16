# Q3: How does asset allocation affect wealth?

## **​1. Introduction**

### **1.1 Definition of the Issue**

This study examines how individuals allocate their**​ ​**​**disposable income** between **savings** (low risk, low return) and **risky investments** (e.g., stocks, mutual funds, cryptocurrencies) characterized by high volatility. Specifically, it investigates how to determine the savings–investment mix based on market conditions, personal risk preferences, and life-cycle stage, and explores how different allocation behaviors influence long-term wealth-accumulation trajectories and immediate utility levels.

### **1.2 Research Questions**

This study uses an economic simulation platform to investigate the **economic impacts of asset‐allocation behaviors,** specifically examining:

* **GDP**​**​ Effects:** How do different savings–investment mixes influence aggregate economic growth?
* **Household Wealth:** How do asset‐allocation strategies affect long‐term household wealth accumulation and distribution?
* **Household Consumption:** How does asset allocation shape household consumption patterns across different risk preferences and life‐cycle stages?


### **1.3 Research Significance**

* **Intertemporal Wealth Allocation and Financial Guidance:**  In modern economies, personal finance behaviors critically affect individual economic well‐being. While savings offer stability, their returns are limited; risk‐taking investments can boost wealth but carry uncertainty and downside risk. As financial markets grow increasingly complex, different socio‐economic groups exhibit varying risk tolerances and financial literacies. A simulation platform is therefore essential for testing strategies and uncovering behavioral mechanisms.
* **Knowledge Base for Financial Education:**  Through this simulation experiment, the government can gain deeper insights into citizens’ asset allocation behaviors and their long‐term societal impacts. These findings will help guide financial literacy programs and inform policies that encourage saving or investment.

---

## **​2. Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                                                                                    | Observation                                                                                                  | Action                                                                                 | Reward                                              |
| ----------- | ------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Individual**  | Ramsey Model        | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.              | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Firm**       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.               | /                                                                                                            | /                                                                                      | Zero (long-run)                                     |
| **Bank**       | Commercial Bank   | Commercial Banks strategically set deposit and lending rates to maximize profits, subject to central bank constraints. | $o_t^{\text{bank}} = ( \iota_t, \phi_t, r^l_{t-1}, r^d_{t-1}, loan, F_{t-1} )$<br>Benchmark rate, reserve ratio, last lending rate, last deposit_rate, loans, pension fund.| $$a_t^{\text{bank}} = \{ r^d_t, r^l_t \}$$<br>Deposit, lending decisions(Commercial Banks)            | $$r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$$<br>Interest margin (Commercial Banks)  |
| **Bank**        | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives. | /                                                            | No rate control                                              | No profit                                |


---

### Rationale for Selected Roles

**Individual →Ramsey Model**  
The Ramsey Model analyzes **individuals’ continuous saving and investment decisions over their life spans**, emphasizing intertemporal optimization and utility maximization. It thus captures personal financial behavior and its impact on long-term wealth accumulation.

**Government → Any type**  
In this experiment, we focus on the interaction between households and financial institutions (providing liquidity/lending or modeling high-risk investments) to assess how different financial behaviors affect individual utility, wealth trajectories, and, by extension, broader economic outcomes. **We do not assign a dedicated government department.**

**Firm → Perfect Competition**  
A perfectly competitive market provides a baseline backdrop for both economic growth and investment returns. Firm profits influence market yields, and this setting realistically simulates overall growth rates and capital returns, underpinning the valuation of high-risk investments.

**Bank → Commercial Bank / Non-Profit Platform**  
**Commercial Banks:** Act as the core conduit for savings, offering stable returns and liquidity guarantees. Changes in bank deposit rates directly influence asset-allocation decisions.

**Non-Profit Platform:** Represent high-risk investment channels in the market, modeling the uncertain returns of volatile assets. They determine the potential returns and volatility faced by investors in the risk-taking segment.

---

## **​3.Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual                            | RL Agent          | Optimizes the saving–investment mix, balancing risk and return to maximize wealth and utility.            |
| Commercial Banks                      | Rule‑Based Agent | Supplies stable savings returns and models the interest‑rate mechanism.                                   |
| No-Arbitrage Platform | Rule‑Based Agent | Generates uncertain investment returns, creating a realistic market‑risk environment.                     |
| Market                                | Rule‑Based Agent | Delivers background capital‑return rates and macroeconomic conditions that influence investment behavior. |

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "asset_allocation"
```

This command loads the configuration file `cfg/asset_allocation.yaml`, which defines the setup for the "asset_allocation" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `asset_allocation.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "asset_allocation"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "tax" # central_bank gov

    - entity_name: 'households'
      entity_args:
        params:
          type: 'ramsey_risk_invest' #The OLG Model can also be chosen in this experiment.

    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"   # ['perfect', 'monopoly', 'monopolistic_competition', 'oligopoly']


    - entity_name: 'bank'
      entity_args:
        params:
          type: 'commercial'


Trainer:
  house_alg: "ppo" #The Behavior Cloning Agent can also be chosen in this experiment.
  gov_alg: "saez"
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  seed: 1
#  n_epochs: 1000
  wandb: True
```
---

## **​5.​**​**Illustrative Experiments**

### Experiment  1: Impact of Asset Allocation on Individual Wealth

* **Experiment Description:**
  
  Compare households that allocate funds to **risky assets** (stocks, crypto) versus **risk‑free assets** (deposits, government bonds) and track long‑run wealth paths.
* **Experimental Variables:**
  
  * Risky vs. risk‑free allocation
  * Return‑distribution parameters
  * Final net wealth, consumption level, utility trajectory

* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **​bc\_saez\_100\_OLG\_risk\_invest (Blue bar/line):​**Households are modeled as ​**Behavior Cloning Agents**​, and the government remains a **Rule-based Agent** applying the ​**Saez tax formula**​.Households operate within the **OLG Model** with **100 total households** but are allowed to engage in **risky investment behavior.**
  * **​bc\_saez\_100\_OLG (Green bar/line):​**Households are modeled as ​**Behavior Cloning Agents**​, while the government is a **Rule-based Agent** applying the **Saez tax formula** from optimal taxation theory.Households operate within the **OLG Model** with**​ 100 total households** and follow **standard investment behavior.**
* **Visualized Experimental Results：**

![Individual Q3P1 asset allocation](../img/Individual%20Q3P1%20asset%20allocation.png)

**Figure 1: ​**Long-run individual wealth accumulation under risky-asset allocations versus risk-free allocations.  The left panel reports wealth by age cohorts, while the right panel reports wealth by income classes.On average, households that avoid risky investments accumulate greater wealth over time, consistent with empirical findings that many investors underperform when exposed to high-volatility assets.

![Individual Q3P2 asset allocation](../img/Individual%20Q3P2%20asset%20allocation.png)

**Figure 2:** Households that engage in risky investments exhibit lower average working hours.

![Individual Q3P3 asset allocation](../img/Individual%20Q3P3%20asset%20allocation.png)

**Figure 3:** Households with risky investments achieve higher long-term cumulative rewards.

* Households allocating wealth to risky assets accumulate lower long-term personal wealth compared to those that avoid risky investments, consistent with empirical evidence that most investors underperform over time.
* Interestingly, households taking on risky investments work fewer hours on average, which in part contributes to their higher long-term personal utility.

---

### Experiment  2 : Macroeconomic Impact of Asset‑Allocation Behavior

* **Experiment Description:**
  
  Divide households into two groups—those engaging in risky investments (e.g., equities, cryptocurrencies) and those holding only risk-free assets (e.g., bank deposits, government bonds)—and model societies composed of these different household types. Compare the long-term GDP growth trajectories of each society to simulate how heterogeneous asset-allocation behaviors affect aggregate production over time.
* **Experimental Variables:**
  
  * Risky Investment vs. Risk-Free Investment
  * Long-Term GDP Trajectory
* **​ Visualized Experimental Results：**

![Individual Q3P4 asset allocation](../img/Individual%20Q3P4%20asset%20allocation.png)

**Figure 4:** The blue line represents a society composed of households engaging in risky investments, while the green line represents a society of households holding only risk-free assets. Over the long run, the society with risky investments exhibits relatively slower GDP growth.

* Over a 300-year observation period, both societies display nearly identical GDP levels and growth trends for the first 100 years. From year 100 to 150, the risky-investment society’s GDP notably exceeds that of the risk-free society. However, for t > 150, the risk-free society’s GDP clearly surpasses the risky-investment society’s, and the gap widens over time.
* Nevertheless, the risky-investment society’s GDP continues to exhibit a sustained positive growth trend over the long term.



