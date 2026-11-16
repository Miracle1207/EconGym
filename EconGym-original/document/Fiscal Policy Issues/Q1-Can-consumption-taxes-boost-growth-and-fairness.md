# Q1: Can consumption taxes boost growth and fairness?

## 1. Introduction

### 1.1 Social Context of Increasing Consumption Tax

Increasing consumption tax refers to the imposition of taxes on the purchase of goods and services. Amid global economic slowdown and rising income inequality, governments around the world seek new revenue sources to fund public services and **social welfare**​**​ ​**programs. As a potential policy tool, the effects of increasing consumption tax require in-depth research.

Taking the United States as an example, consumption tax typically consists of a comprehensive sales tax, composed of fixed State Sales Tax and varying Local Sales Taxes. The top three U.S. states by combined sales tax rates (state + local taxes) are:

* Louisiana: 9.56%
* Tennessee: 9.55%
* Arkansas: 9.45%

### 1.2 Research Questions

Using an economic simulation platform, this study examines the effects of increasing consumption tax on growth and fairness, focusing on:

* **GDP**​**​ Effects:** Does raising consumption tax promote or hinder long-term economic growth?
* **Social Fairness:** How does higher consumption tax influence social equity, such as reducing inequality or redistributing resources across groups?
* **Household Consumption:** How does consumption tax change household spending behavior, and what are its implications for welfare and living standards?


### 1.3 Research Significance

* **Evaluating dual impacts on economy and distribution:**
  As an indirect tax, consumption tax has advantages such as a broad tax base and high collection efficiency but may disproportionately burden low-income groups. Assessing its effects on household wealth, consumption, and utility through simulation platforms helps fully understand its ​**redistributive effects**​, guiding more rational tax policy designs.
* **Providing policy ​**​**insights**​**​ for balancing fiscal revenue and social equity:**
  Given increasing fiscal expenditure demands and worsening income inequality, increasing consumption tax becomes a policy option. This research reveals trade-offs between economic growth and social equity under different tax rates, offering quantitative support for achieving fiscal sustainability and social justice.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                                                                               | Observation                                                                                               | Action                                                                                 | Reward                                   |
| ----------- | ------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------- |
| **Individual**  | Ramsey Model        | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.         | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
 | **Government**  | Fiscal Authority    | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.            |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
| **Firm**       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.           | /                                                                                                         | /                                                                                    | Zero (long-run)                          |
| **Bank**       | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives. | /                                                                                                         | No rate control                                                                      | No profit                                |


### Rationale for Selected Roles

**Individual → Ramsey Model**  
 Ramsey Model.The Ramsey Model analyzes aggregate macroeconomic responses from representative households’ optimal intertemporal decisions, ideal for studying long-term equilibrium trends,while the OLG Model captures heterogeneity across age groups in income, consumption, and tax burdens, enabling analysis of the intergenerational fairness effects of consumption taxes.

**Government → Fiscal Authority**  
The Tax Policy Department directly formulates and implements consumption tax policies, fully simulating tax collection, income redistribution, and fiscal expenditure responses. Compared with pension and central bank departments, the Treasury more accurately reflects the impact of consumption tax on tax structures, government budgets, and social equity.
Pension and monetary policy are not core variables in this study and thus are not used.

**Firm → Perfect Competition**  
Selecting perfectly competitive markets helps eliminate distortions, making the impact of consumption tax policies on supply-demand dynamics, pricing, and distribution clearer.
Monopolistic markets have non-market-determined prices and complex corporate strategies, potentially obscuring the economic effects of consumption taxes and reducing experimental clarity.

**Bank→ Non-Profit Platform**  
No-Arbitrage Platform are more suitable for analyzing long-term wealth accumulation and asset allocation responses, without active participation in credit expansion. They clearly reflect savings and investment behavior under policy changes.
Commercial banks involve complex behaviors like lending, interest spreads, and financial risks, less suited for focused macroeconomic policy analysis.

---

## 3. Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                                                  
| ------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------- |
| Individual            | Behavior Cloning Agent | Learn behavioral patterns from empirical data via behavior cloning.                               |
| Government             | Rule-Based Agent       | Predict changes in public finance after implementing a consumption tax using Seaz Tax Framework. |
| Firm                 | Rule-Based Agent       | Encode supply–demand rules to simulate consumer behavior under a consumption tax.                |
| Bank | Rule-Based Agent       | Define financial-market operations based on macroeconomic variables.                              |

---
## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "consumption_tax"
```

This command loads the configuration file `cfg/consumption_tax.yaml`, which defines the setup for the "consumption_tax" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `consumption_tax.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "consumption_tax"
    consumption_tax_rate: 0.07   # todo: set consumption_tax_rate!!
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "tax"  # Focus on pension policy. type_list: ['tax', 'pension', 'central_bank']
          gov_task: "gdp"
    - entity_name: 'households'
      entity_args:
        params:
          type: 'ramsey' #The OLG Model can also be chosen in this experiment.
          type_list: [ 'ramsey', 'OLG', 'OLG_risk_invest', 'ramsey_risk_invest' ]
          households_n: 100
          action_dim: 2

    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"   #  type_list: [ 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly' ]
          alpha: 0.36
          Z: 1.
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
  gov_alg: "saez"
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  seed: 1
  epoch_length: 300
  cuda: False
#  n_epochs: 300
```
---

## 5. **Illustrative Experiment**

### Experiment 1: Impact of Increased Consumption Tax on Macroeconomy and Social Welfare

* **Experiment Description:**
  
  Comparing macroeconomic indicators and welfare levels across different consumption tax rates.
* **Experimental Variables:**
  
  * Different consumption tax rates (0%, 7%, 9%)
  * Simulated GDP growth trends
  * Simulated social welfare levels
  * Simulated household income inequality (Gini coefficient)
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **consumption\_tax\_0%\_bc\_saez\_100\_OLG (Blue line):** Households are modeled using the OLG model with **Behavior Cloning Agents** following **Saez’s rule-based taxation framework**, with **100 households** and a **consumption tax rate of 0%**.
  * ​**consumption\_tax\_7%\_bc\_saez\_100\_OLG (Light green line):** Households are modeled using the OLG model with **Behavior Cloning Agents** following **Saez’s rule-based taxation framework**, with **100 households** and a **consumption tax rate of 7%**.
  * ​**consumption\_tax\_9%\_bc\_saez\_100\_OLG (Yellow line)​:** Households are modeled using the OLG model with **Behavior Cloning Agents** following **Saez’s rule-based taxation framework**, with **100 households** and a **consumption tax rate of 9%**.
* **Visualized Experimental Results:**

![Fiscal Q3 P1](../img/Fiscal%20Q3%20P1.png)

**Figure 1: ​**Blue, green, and yellow lines represent GDP under 0%, 7%, and 9% consumption tax rates, respectively. Higher taxes slightly increase GDP but show minimal difference compared to no tax.

![Fiscal Q3 P2](../img/Fiscal%20Q3%20P2.png)

**Figure 2: ​**Higher consumption taxes have almost no long-term effect on total social welfare.

![Fiscal Q3 P3](../img/Fiscal%20Q3%20P3.png)

**Figure 3:** Increased consumption tax reduces the income Gini coefficient, indicating it effectively lowers income inequality.

* Increasing the consumption tax can slightly raise GDP in the simulated economy while effectively narrowing income disparities. Moreover, total social welfare remains unchanged under a higher consumption tax regime. Thus, from the perspective of promoting social equity, raising the consumption tax allows revenue to be redistributed—in some form—to lower-income groups more effectively, making it a reasonable tax policy choice.

---

### Experiment 2: Impact of Increased Consumption Tax on Household Wealth and Individual Utility

* **Experiment Description:**
  
  Comparing household wealth and individual utility across different consumption tax rates.
* **Experimental Variables:**
  
  * Different consumption tax rates (0%, 7%, 9%)
  * Household structure stratified by income levels
  * Household wealth levels
  * Household utility
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **Agent Settings:**
    * **consumption\_tax\_0%\_bc\_saez\_100\_OLG (Blue bar):** Households are modeled using the OLG model with **Behavior Cloning Agents** following **Saez’s rule-based taxation framework**, with **100 households** and a **consumption tax rate of 0%**.
    * ​**consumption\_tax\_7%\_bc\_saez\_100\_OLG (Green bar):** Households are modeled using the OLG model with **Behavior Cloning Agents** following **Saez’s rule-based taxation framework**, with **100 households** and a **consumption tax rate of 7%**.
    * ​**consumption\_tax\_9%\_bc\_saez\_100\_OLG (Yellow bar)​:** Households are modeled using the OLG model with **Behavior Cloning Agents** following **Saez’s rule-based taxation framework**, with **100 households** and a **consumption tax rate of 9%**.
  * **Panel Interpretation:**
    * **Left panel:** Different bar colors represent household wealth by **age cohorts** (e.g., <24, 25–34, 35–44, 45–54, 55–64, 65–74, 75–84, 85+, total).
    * **Right panel:** Different bar colors represent household wealth by **income/wealth classes** (rich, middle, poor, and mean).

* **Visualized Experimental Results:**

![Fiscal Q3 P4](../img/Fiscal%20Q3%20P4.png)

**Figure 4: ​**Different consumption taxes and household income levels. Higher consumption taxes result in higher income levels across different age groups and economic conditions, especially benefiting young individuals (25-34 years).

![Fiscal Q3 P5](../img/Fiscal%20Q3%20P5.png)

**Figure 5: ​**Consumption tax effects on household utility are not significant overall, but higher taxes notably reduce utility for individuals aged 25-34 and 45-54 years.

* Higher consumption taxes increase household income, notably benefiting young individuals, but simultaneously reduce utility for certain age groups. This indicates income gains do not fully offset welfare losses from higher consumption costs.




