# Q4: How to design optimal tax policies?

## 1. Introduction

### 1.1 Optimal Tax Policy

* Taxation is the government’s core tool for **resource allocation, income redistribution, and macroeconomic stabilization.** Different tax structures (e.g., labor‐income taxes, capital‐gains taxes, consumption taxes) have varied impacts on economic efficiency, income distribution, and fiscal sustainability. Designing an “optimal tax policy” means maximizing social welfare—balancing growth incentives and equity—while ensuring fiscal balance.
* Traditional optimal‐tax theory minimizes distortionary effects without harming efficiency. However, real economies feature heterogeneous agents, income inequality, and political constraints. Reinforcement Learning (RL) offers a powerful “policy search” approach to dynamically explore the optimal strategy space.
  
### 1.2 Research Questions

Using an economic-simulation platform, this study explores how governments can employ RL algorithms to design optimal tax policies, specifically examining:

* ​**Household Income**​: How do different tax policies affect household disposable income across income groups?
* ​**Labor Supply**​: What is the impact of tax rate adjustments on household working hours and labor participation?
* **GDP**​​**​ Effects**​: How do changes in the tax mix influence long-term GDP growth and fiscal sustainability?
* ​**Social Equity**​: To what extent can optimized tax policies reduce inequality and improve overall fairness in the economy?

### 1.3 Research Significance

* **Building an Intelligent Policy‐Evaluation Tool:** Deploy RL to create a “simulate–feedback–optimize” loop, equipping policymakers with advanced tools for experimental policy design and institutional assessment.
* **Achieving Dynamic Growth–Equity Balance:** Use multi‐objective optimization to finely tune tax systems for efficiency and fairness, enhancing the responsiveness and adaptability of fiscal frameworks.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                                                                               | Observation                                                                                               | Action                                                                                 | Reward                                   |
| ----------- | ------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------- |
| **Individual**  | Ramsey Model        | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.         | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Government**  | Fiscal Authority    | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.              |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
| **Firm**       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.           | /                                                                                                         | /                                                                                    | Zero (long-run)                          |
| **Bank**       | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives. | /                                                                                                         | No rate control                                                                      | No profit                                |

---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
Households, as rational economic agents, make decisions on labor supply, consumption, and savings based on utility‐maximization principles. Changes in tax policy alter their marginal choices, providing essential micro‐level feedback that underpins government policy optimization.

**Government → Fiscal Authority**  
The government, as the architect and executor of tax policy, adjusts tax‐rate structures in line with economic conditions and macro objectives. Its core function is to achieve a dynamic balance among growth, equity, and fiscal sustainability.

**Firm → Perfect Competition**   
Wages and prices are determined by market mechanisms, reflecting the transmission channels of tax policy through labor and goods markets.

**Bank → Non-Profit Platform**  
Tax policies influence saving and investment behaviors; the financial system, via interest‐rate mechanisms, feeds back changes that restore economic equilibrium.

---

## 3. Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | Use historical behavioral rules to create a stable simulated environment.          |
| Government             | RL Agent         | Employ reinforcement learning to explore the tax‐policy space, dynamically optimizing GDP, income distribution, and fiscal balance. |
| Firm                 | Rule-Based Agent | Model wages and employment reacting to tax changes via supply–demand mechanism rules.                                               |
| Bank | Rule-Based Agent | Adjust interest rates and capital returns based on rules governing savings behavior and tax burdens.                                 |

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "optimal_tax"
```

This command loads the configuration file `cfg/optimal_tax.yaml`, which defines the setup for the "optimal_tax" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `optimal_tax.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "optimal_tax"
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "tax"  # Focus on pension policy. type_list: ['tax', 'pension', 'central_bank']
          gov_task: "gdp"
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
          alpha: 0.36    # 0.25;  0.36 in Aiyagari
          Z: 1.     # 10
          sigma_z: 0.0038
          epsilon: 0.5

    - entity_name: 'bank'
      entity_args:
        params:
          type: 'non_profit'   # [ 'non_profit', 'commercial' ]
          n: 1
          lending_rate: 0.0345
          deposit_rate: 0.0345
          reserve_ratio: 0.1
          base_interest_rate: 0.0345
          depreciation_rate: 0.06


Trainer:
  house_alg: "bc"
  gov_alg: "ddpg"
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  seed: 1
  cuda: False
#  n_epochs: 1000
  wandb: True
```
---

## 5.Illustrative Experiments

### Experiment : RL-Based Optimal Tax-Structure Policy Trial

* **Experiment Description:**

  In the simulated economic environment, the government can use a reinforcement learning (RL) agent to automatically learn the optimal mix and rates of labor-income, consumption, and capital taxes. In this experiment, we compare the government’s use of reinforcement learning methods (DDPG), economic rule-based methods (Seaz Tax), and the real tax rates set by the U.S. federal government (2022), and discuss the impact of different tax rate settings and tax structures on the macroeconomy.
* **Experimental Variables:**
  
  * Different government department agents and their corresponding tax structures.
  * Macro indicators: GDP, wealth Gini coefficient, average household wealth
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.The bar charts show household wealth distributions under different tax policies.
  * **Left group (optimal\_tax\_ramsey\_100\_bc\_tax\_saez):** Households are modeled under the **Ramsey Model** with **Behavior Cloning Agents**, while the government adopts the **Saez rule-based tax formula** to determine optimal taxation.
    * Blue bar: Rich households
    * Green bar: Middle-class households
    * Yellow bar: Poor households
    * Red bar: Overall average
  * **Middle group (optimal\_tax\_ramsey\_100\_bc\_tax\_ddpg):** Households follow the **Ramsey Model** with **Behavior Cloning Agents**, and the government employs a **DDPG-based RL algorithm** to dynamically adjust tax rates over time.
    * Blue bar: Rich households
    * Green bar: Middle-class households
    * Yellow bar: Poor households
    * Red bar: Overall average
  * **Right group (optimal\_tax\_ramsey\_100\_bc\_tax\_us\_federal):** Households are modeled with the **Ramsey Model** and **Behavior Cloning Agents**, while the government applies **the U.S. federal tax system** as the benchmark baseline.
    * Blue bar: Rich households
    * Green bar: Middle-class households
    * Yellow bar: Poor households
    * Red bar: Overall average
* **Visualized Experimental Results：**

![Fiscal Q4 P1](../img/Fiscal%20Q4%20P1.png)

​**Figure 1**​: Comparison of household wealth under different tax policies at T=192 years. The tax system trained by the RL-Agent results in higher average household wealth, with the average wealth of the wealthier households significantly higher than the other two tax systems. The simulated economy using the Seaz rule has the second highest average household wealth, while the simulated economy using the real U.S. tax system shows the lowest average household wealth.

![Fiscal Q4 P2](../img/Fiscal%20Q4%20P2.png)

​**Figure 2**​:At T=192 years, the phenomenon reflected in household wealth is identical, where the tax system trained by the RL-Agent maximizes consumption across different wealth tiers of households.

* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  * **optimal\_tax\_ramsey\_100\_bc\_tax\_saez (Blue line):** Households are modeled under the **Ramsey Model** with **Behavior Cloning Agents**, while the government adopts the **Saez rule-based tax formula** to determine optimal taxation.
  * **optimal\_tax\_ramsey\_100\_bc\_tax\_ddpg (Green line):** Households are modeled under the **Ramsey Model** with **Behavior Cloning Agents**, and the government employs a **DDPG-based RL algorithm** to dynamically adjust tax rates over time.
  * **optimal\_tax\_ramsey\_100\_bc\_tax\_us\_federal (Yellow line):** Households are modeled under the **Ramsey Model** and **Behavior Cloning Agents**, while the government applies the **U.S. federal tax system** as the benchmark baseline.

![Fiscal Q4 P3](../img/Fiscal%20Q4%20P3.png)

​**Figure 3**​: Comparison of long-term GDP growth levels under different tax policies. The RL-Agent economy has the fastest GDP growth, followed by the economy with the Seaz rule. The simulated economy using the real U.S. tax system experiences the lowest GDP growth.

![Fiscal Q4 P4](../img/Fiscal%20Q4%20P4.png)

​**Figure 4**​: As time progresses, all tax strategies significantly reduce the wealth gap. However, when the government uses the RL-Agent , the long-term wealth disparity remains relatively higher.

* Through the learning process of the RL-Agent, the simulated economy’s government is able to design a tax system that maximizes aggregate wealth accumulation, household income growth, and total social consumption.
* All tax strategies effectively reduce long-term income inequality in the simulated economy, though the RL-Agent–led government tends to exhibit relatively higher long-run inequality.

