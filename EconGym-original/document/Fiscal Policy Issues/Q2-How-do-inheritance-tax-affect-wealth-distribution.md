# Q2: How does inheritance tax affect wealth distribution?

## 1. Introduction

### 1.1 Introduction to Estate Tax

Estate Tax is a tax levied on assets inherited upon an individual's death, aiming to regulate intergenerational wealth transfer. As **global wealth inequality** grows, particularly the issue of intergenerational wealth accumulation within families, estate tax has become a crucial tool for reducing social inequality and promoting wealth redistribution. It not only helps diminish the concentration of "hereditary wealth" but also generates fiscal resources for public services and social welfare.

Implementation forms and rates of estate tax vary widely among developed countries. For example, Japan's maximum estate tax rate is 55%, Korea's is 50%, and France's is 60%. The U.S. has an estate tax at the federal level, with an exemption threshold as high as \$11.6 million, though states differ significantly. Connecticut applies a flat rate of 12%, whereas Iowa plans to abolish its estate tax entirely by 2025.

In contrast, China has **yet to implement** an estate tax. However, with increasing population aging and wealth concentration, discussions about introducing an estate tax to achieve fairer wealth redistribution have become more prevalent.

### 1.2 Research Questions

Based on an economic simulation platform, this study investigates how increasing estate tax impacts social wealth accumulation and distribution from both macro and micro perspectives. Specific questions include:

* **Social ​and GDP**: Does estate tax enhance societal output?
* **Individual working hours:** Does a higher estate tax encourage people to work more flexibly?
* **Household wealth and consumption:** Does a higher estate tax encourage quicker consumption of wealth during individuals' lifetimes?

### 1.3 Research Significance

* **Policy Guidance:**  By evaluating estate tax impacts on social output, household behaviors, and wealth distribution through simulation platforms, this study provides quantitative policy foundations for countries like China, which currently lack estate tax mechanisms, assisting in scientifically designing taxation systems and exemption thresholds.
* **Understanding Intergenerational Incentives and Behavioral Mechanisms:**  This study explores its long-term effects on household savings, labor supply, and consumption decisions, shedding light on the dynamic relationship between household behavior and wealth transfers, thereby offering empirical evidence for behavioral economics and public finance research.

---

## ​2.​ Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                             | Observation                                                  | Action                                                       | Reward                                   |
| ----------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| **Individual**  | OLG Model           | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals.   | $$o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$$<br/>Private: assets, education, age<br/>Global: distributional statistics                                  | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment <br/>*OLG*: old agents $$\lambda_t^i = 0$$                               |$r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)<br/>OLG includes pension if retired |
| **Government**  | Fiscal Authority    | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.                     |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
| **Firm**       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                   | /                                                                                                                                                    | /                                                                                          | Zero (long-run)                        |
| **Bank**       | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives.     | /                                                                                                                                                    | No rate control                                                                            | No profit                              |


---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
The OLG framework tracks lifecycle patterns of income, saving, and estate transfers, making it an ideal tool to study how inheritance tax influences intergenerational wealth transmission, labor incentives, and consumption decisions.

**Government → Fiscal Authority**  
As the authority that establishes and collects inheritance tax, the Tax Policy Department shapes policy features—such as exemption thresholds and redistribution rules—and directly affects fiscal revenue and aggregate demand. Modeling this actor enables simulation of policy adjustments’ transmission through the public budget.

**Firm → Perfect Competition**  
A perfectly competitive setting ensures efficient price formation, allowing clear observation of how an inheritance tax shifts supply and demand.

**Bank → Non-Profit Platform**  
No-Arbitrage Platform reliably reflect household asset allocation and returns across different life stages without introducing leverage dynamics or risk-preference distortions, making them well-suited for analyzing the medium- and long-term effects of inheritance tax on saving behavior and wealth accumulation paths.

---

## ​3.​ Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             |Behavior Cloning Agent | Use a rule-based agent to model household decision processes.Employ behavior cloning to learn patterns from empirical data.           |
| Government             | Rule-Based Agent                           | Reproduce changes in public finances after inheritance-tax implementation within the simulation environment using defined rules. |
| Firm                 | Rule-Based Agent                          | Encode market supply–demand rules to simulate consumer behavior under inheritance tax.                                                |
| Bank | Rule-Based Agent                          | Configure financial-market operations based on macroeconomic variables.                                                                |


---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "estate_tax"
```

This command loads the configuration file `cfg/estate_tax.yaml`, which defines the setup for the "estate_tax" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `estate_tax.yaml`

```yaml

Environment:
  env_core:
    problem_scene: "estate_tax"
    estate_tax_rate: 0.0   # todo: set estate_tax_rate!!
    estate_tax_exemption: 13610000
    episode_length: 300

  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "pension"  # Focus on pension policy. type_list: ['tax', 'pension', 'central_bank']

    - entity_name: 'households'
      entity_args:
        params:
          type: 'OLG'
          type_list: ['ramsey', 'OLG', 'OLG_risk_invest', 'ramsey_risk_invest']
          households_n: 100
          action_dim: 2
          real_action_max: [1.0, 1.0]
          real_action_min: [-0.5, 0.0]
        OLG:
          birth_rate: 0.011
          initial_working_age: 24

Trainer:
  house_alg: "bc"
  gov_alg: "rule_based"
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  seed: 1
#  epoch_length: 300
  cuda: False
```
---

## **​5.​**​**Illustrative Experiment**


### Experiment 1: Macroeconomic Impact of Estate Tax

* **Experiment Description: ​**
  
  Compare macroeconomic indicators under different estate tax rates.
* **Experimental Variables:**
  
  * Estate tax rates (0%, 10%, 15%)
  * Social output (GDP)
* **Baselines:**

  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **estate\_15%\_bc\_saez\_100\_OLG (Blue line):** Households are modeled as Behavior Cloning (BC) Agents under the OLG framework, with the estate tax rate set to 15%.
  * **estate\_0%\_bc\_saez\_100\_OLG (Green line):** Households are modeled as Behavior Cloning (BC) Agents under the OLG framework, with the estate tax rate set to 0%.
  * **estate\_10%\_bc\_saez\_100\_OLG (Yellow line):** Households are modeled as Behavior Cloning (BC) Agents under the OLG framework, with the estate tax rate set to 10%.
* **Visualized Experimental Results：**

![Fiscal Q2P1 inherit](../img/Fiscal%20Q2P1%20inherit.png)
  
  **Figure 1:**  Imposing an inheritance tax clearly raises aggregate output, but increasing the rate from 10% to 15% yields little additional benefit.
* Compared to a zero–tax scenario, introducing an inheritance tax improves the efficiency of wealth circulation and thus boosts GDP. The inheritance-tax policy also carries a signaling effect, but further rate increases have diminishing returns for GDP growth.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Experiment 2: Household Impact of Estate Tax

* **Experiment Description: ​**
  
  Compare household income, consumption, and labor behaviors under different estate tax rates.
* **Experimental Variables:**
  
  * Estate tax rates (0%, 10%, 15%)
  * Simulated households stratified by age and income
  * Household wealth
  * Household working hours
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.The bar charts show household wealth distributions under different estate tax policies.
  
  * **Agent Settings:**
    
    * **estate\_0%\_bc\_saez\_100\_OLG (Green bars):** Estate tax rate set to 0%.
    * **estate\_10%\_bc\_saez\_100\_OLG (Yellow bars):** Estate tax rate set to 10%.
    * **estate\_15%\_bc\_saez\_100\_OLG (Blue bars):** Estate tax rate set to 15%.
  * **Panel Interpretation:**
    
    * **Left panel:** Different bar colors represent household wealth by **age cohorts** (e.g., <24, 25–34, 35–44, 45–54, 55–64, 65–74, 75–84, 85+, total).
    * **Right panel:** Different bar colors represent household wealth by **income/wealth classes** (rich, middle, poor, and mean).
* **Visualized Experimental Results：**
  
![Fiscal Q2P2 inherit](../img/Fiscal%20Q2P2%20inherit.png)
  
  **Figure 2: ​**Higher estate taxes consistently reduce household wealth across different age groups and income levels.
  
![Fiscal Q2P3 inherit](../img/Fiscal%20Q2P3%20inherit.png)
  
  **​ Figure 3:** Higher estate taxes enhance labor incentives for low-income households but discourage labor among middle-income groups; minimal age-based effects observed.
* Estate taxes decrease household wealth.
* Estate taxes exhibit clear labor incentives differences based on income: enhancing labor incentives among low-income groups while discouraging middle-income groups.


