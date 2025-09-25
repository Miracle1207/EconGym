# Q2: How to control inflation via monetary policy?

## 1. Introduction

### 1.1 Inflation Phenomenon

Inflation refers to a sustained and significant rise in the overall price level of an economy, typically measured by the Consumer Price Index (CPI) or the GDP deflator. Since 2021, the United States has experienced severe inflationary pressure, with the year-over-year CPI increase exceeding 9% at its peak. To combat high inflation, the Federal Reserve has raised the federal funds rate continuously since 2022—from 0.25% to above 5%—and has reduced its balance sheet to tighten liquidity, suppress demand, and curb price growth.

### 1.2 Research Questions

This study uses an economic simulation platform to investigate the economic impacts of **monetary policy in controlling inflation,** specifically examining:

* **Inflation**​**​ Control Effectiveness:** How effective are interest rate hikes in reducing inflation dynamics?
* **Household Consumption: ​**How does monetary tightening affect aggregate and age-differentiated household consumption?
* **GDP**​**​ Effects:** What is the impact of inflation-control policies on real economic output and growth?


### 1.3 Research Significance

* **Exploring the Long-Run Transmission Mechanisms of Monetary Policy:**  By tracing the multi-period feedback effects of interest-rate adjustments on consumption, investment, and employment, simulation experiments can illuminate the process by which monetary policy evolves from a short-term stabilizer into a long-term economic shaping force.
* **Policy Guidance:**  This research can provide central banks with a simulation-based framework to balance price stability against growth, helping to avoid “over-tightening” or “delayed response” in policy implementation.

---

## ​2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type        | Role Description                                                                                                             | Observation                                                                                                  | Action                                                             | Reward                         |
| ----------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------ |
| **Individual**  | Ramsey Model         | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.                    | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Government**  | Central Bank         | Central Bank adjusts nominal interest rates and reserve requirements, transmitting monetary policy to households and firms. |\$\$o\_t^g = (\\mathcal{A}\_{t}, \\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, \\pi\_{t-1}, g\_{t-1})\$\$ <br>Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, inflation rate, growth rate. | $a_t^{\text{cb}} = ( \phi_t, \iota_t )$<br>Reserve ratio, benchmark rate | Inflation/GDP stabilization                                  |
| **Firm**       | Perfect Competition  | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                       | /                                                                                                            | /                                                                | Zero (long-run)                |
| **Bank**       | Commercial Bank     | Commercial Bank strategically set deposit and lending rates to maximize profits, subject to central bank constraints.      | $o_t^{\text{bank}} = ( \iota_t, \phi_t, r^l_{t-1}, r^d_{t-1}, loan, F_{t-1} )$<br>Benchmark rate, reserve ratio, last lending rate, last deposit_rate, loans, pension fund. | $a_t^{\text{bank}} = ( r^d_t, r^l_t )$<br>Deposit, lending decisions | $r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$<br>Interest margin |


---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
The Ramsey model effectively captures **how households adjust their behavior based on expected prices, wage changes, and interest rates,** making it a key theoretical tool for analyzing consumption and labor‐supply responses to inflation control.

**Government → Central Bank**  
The central bank designs and implements monetary policy, **directly controlling key variables such as interest rates and money supply**. Tools for inflation control—like rate hikes and balance‐sheet reduction—are led by the central bank, so this role accurately simulates policy transmission and its macroeconomic effects.

**Firm → Perfect Competition**  
A perfectly competitive market reflects **the basic mechanism of prices set by supply and demand**, helping to clearly observe how inflation‐control policies (e.g., rate increases) affect labor markets, goods prices, and investment returns. It avoids distortions from oligopoly or monopoly‐induced price rigidity, making it well suited to identify policy effects.

**Bank → Commercial Bank**  
During inflation control, commercial bank typically raise lending and deposit rates, which in turn affect firms’ financing costs and households’ saving returns. Compared with a no-arbitrage framework, commercial banks also involve microbehavior such as risk assessment, credit rationing, and asset‐liability management, enabling a more realistic simulation of credit contraction or expansion under policy shifts.

---

## ​3.​ Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | RL Agent  | Households modeled as RL agents will make decisions by seeking to maximize long-term utility.  |
| Government             | RL Agent               | Inflation control requires dynamic policy adjustments (e.g., incremental rate hikes or quantitative tightening); the RL Agent learns optimal strategies through environment feedback. |
| Firm                 | Rule-Based Agent       | A rule-based agent can directly simulate the “cost increase → price increase” transmission chain.                                                                                  |
| Bank | RL Agent       | Learns through trial-and-error to optimize long-term cumulative rewards. Well-suited for solving dynamic decision-making problems.                                                                                                       |

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "inflation_control"
```

This command loads the configuration file `cfg/inflation_control.yaml`, which defines the setup for the "inflation_control" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `inflation_control.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "inflation_control"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "central_bank"  # Focus on pension policy. type_list: ['tax', 'pension', 'central_bank']

    - entity_name: 'households'
      entity_args:
        params:
          type: 'ramsey'
          type_list: ['ramsey', 'OLG', 'OLG_risk_invest', 'ramsey_risk_invest']
          households_n: 100
          action_dim: 2


    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"   #  type_list: [ 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly' ]

    - entity_name: 'bank'
      entity_args:
        params:
          type: 'commercial'   # [ 'non_profit', 'commercial' ]
          n: 1
          lending_rate: 0.0345
          deposit_rate: 0.0345
          reserve_ratio: 0.1
          base_interest_rate: 0.0345

Trainer:
  house_alg: "ddpg"
  gov_alg: "ddpg"
  firm_alg: "rule_based"
  bank_alg: "ddpg"
  seed: 1
  epoch_length: 300
  cuda: False
  n_epochs: 300
  test: False
  wandb: False
```
---

## **​5.​**​**Illustrative Experiment**

### Experiment 1: Evaluating Optimal Monetary Policy via Reinforcement Learning

* **Experiment Description: ​**

  The central bank dynamically adjusts interest rates through a Reinforcement-Learning Agent aiming to minimize CPI inflation while maintaining stable GDP growth. Monitor macroeconomic indicators to assess policy effectiveness.
* **Experimental Variables:**
  
  * Comparison of RL Agent vs. Rule-Based Agent in exploring optimal monetary policy
  * Aggregate GDP level
  * Gini coefficient


* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **bc\_ppo\_100\_ramsey (blue line):** Households are **Behavior Cloning Agents** under the**​ Ramsey model ​**with ​**100 households**​, and the government use **Reinforcement Learning**​**​ ​**​**PPO**​​**​ Agent**​.
  * **bc\_rule\_based\_100\_ramsey (green line):** Households are **Behavior Cloning Agents** under the **Ramsey model** with ​**100 households**​, and the government use ​**Rule-Based Agent**​.
* **Visualized Experimental Results：**

![Monetary Q2 P1](../img/Monetary%20Q2%20P1.png)

**Figure 1:** The simulated economy under an inflation‐control regime shows a lower long-run GDP level compared with the economy governed by a predefined rule-based policy , indicating that monetary policy aimed at curbing inflation does indeed dampen GDP growth.

![Monetary Q2 P2](../img/Monetary%20Q2%20P2.png)

**Figure 2:** The inflation‐control policy increases income inequality.

* During an overheating episode, the central bank employs an RL approach to learn an inflation-targeted policy. This policy reduces growth relative to the overheated scenario, and it also aggravates income inequality. The RL-derived tightening disproportionately lowers incomes of low-income households due to falling employment rates, whereas high-income households are less impacted, thus widening the wealth gap.
* The simulation platform enables quantification of different policy mixes’ effectiveness in suppressing inflation, offering decision support to balance economic growth against price stability.


