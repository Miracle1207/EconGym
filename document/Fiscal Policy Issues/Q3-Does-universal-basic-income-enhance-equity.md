# Q3: Does universal basic income enhance equity?

## 1.​ Introduction

### 1.1 Definition

Universal Basic Income (UBI) is a form of social security that provides every citizen with an unconditional, regular cash transfer. This study examines the extent to which UBI fosters social equity. The idea draws on the National Bureau of Economic Research (NBER) article *“Universal Basic Income in the United States and Advanced Countries.”* Key features are:

* **Adequate generosity** – the grant is large enough to cover basic living expenses, even when recipients have no other income;
* **Non-means-tested or only gradually phased-out** – payments do not immediately cease as personal income rises;
* **Near-universal coverage** – benefits are delivered to almost the entire population rather than to narrowly defined groups (e.g., single mothers).

In short, UBI can be viewed as a perpetual transfer that provides every resident with sufficient resources for a minimal standard of living.

### 1.2 Limitations and Policy Inspiration

UBI faces three major hurdles: (i) ​**fiscal cost**​, (ii) ​**potential work-incentive distortions**​, and (iii) ​**integration with existing welfare programs**​. Nevertheless, rapid advances in automation and AI are reshaping labour markets and eroding traditional jobs. In this context, UBI has been proposed as a mechanism to alleviate poverty, curb inequality, and forestall social unrest. The principles of universality and dignity embedded in UBI still offer valuable guidance for modern welfare reform.

### 1.3 Research Questions

Using an economic-simulation platform, we explore how UBI affects:

* ​**Income inequality**​: Does UBI narrow the gap between rich and poor?
* ​**Household wealth**​: How does UBI influence the long-term accumulation and distribution of household assets?
* ​**Household labor supply**​: How will total working hours change across different income groups, and does UBI reduce incentives to work?

### 1.4 Research Significance

* **Welfare-system reform:**
  The universal and unconditional nature of UBI suggests remedies for incomplete coverage and high eligibility thresholds in current schemes, informing more inclusive policy design.
* **Social protection in the ​AI era:**
  As AI and large models increasingly substitute routine work, a UBI-style safety net can provide reliable income for displaced or transitioning workers. Assessing UBI in this technological setting helps build future-proof social insurance.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                             | Observation                                                  | Action                                                       | Reward                                   |
| ----------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| **Individual**  | OLG Model           | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals.   | $o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$<br/>Private: assets, education, age<br/>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment <br/>*OLG*: old agents $\lambda_t^i = 0$    | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)   <br/>*OLG includes pension if retired*      |
| **Government**  | Fiscal Authority    | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.                     |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
| **Firm**       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                   | /                                                                                                                                                    | /                                                                                          | Zero (long-run)                        |
| **Bank**       | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives.     | /                                                                                                                                                    | No rate control                                                                            | No profit                              |


---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
The OLG framework captures age-specific income, consumption, and saving behaviors, making it well-suited to simulate UBI’s dynamic effects on intergenerational resource allocation. It is preferable to an infinitely heterogeneous agent model for this purpose.

**Government → Fiscal Authority**  
The Fiscal Authority is directly responsible for funding and administering UBI, making it the core institution for assessing fiscal impacts. It aligns more closely with UBI financing duties than a central bank.

**Firm → Perfect Competition**  
A perfectly competitive market eliminates distortions from market power, allowing a clear assessment of UBI’s influence on labor supply.

**Bank → Non-Profit Platform**  
UBI may alter saving rates and investment patterns; a no-arbitrage framework is ideal for analyzing these financial dynamics.

---

## 3.​ Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent |  Employ behavior cloning to learn patterns from empirical data. |
| Government             | Rule-Based Agent               | Forecast changes in public finances following UBI implementation using defined rules.                               |
| Firm                 | Rule-Based Agent                          | Encode supply–demand rules to simulate labor-market responses under UBI.                                                    |
| Bank | Rule-Based Agent                          | Define financial-market operations based on macroeconomic variables.                                                         |


---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "universal_basic_income"
```

This command loads the configuration file `cfg/universal_basic_income.yaml`, which defines the setup for the "universal_basic_income" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `universal_basic_income.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "universal_basic_income"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "tax"

    - entity_name: 'households'
      entity_args:
        params:
          type: 'OLG'
          households_n: 100
          action_dim: 2

        OLG:
          birth_rate: 0.011

          initial_working_age: 24
    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"
          sigma_z: 0.0038
          epsilon: 0.5

    - entity_name: 'bank'
      entity_args:
        params:
          type: 'non_profit'
          n: 1
          lending_rate: 0.0345
          deposit_rate: 0.0345
          reserve_ratio: 0.1
          base_interest_rate: 0.0345
          depreciation_rate: 0.06
          real_action_max: [ 1.0, 0.20 ]
          real_action_min: [ 0.0, -1e-3 ]

Trainer:
  house_alg: "bc"
  gov_alg: "rule_based"
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  seed: 1
  epoch_length: 300
  cuda: False
#  n_epochs: 300
```
---
## **​5.​**​**Illustrative Experiment**

### Experiment 1: Impact of UBI on Social Equity

* **Experiment Description: ​**

    Compare the effects of two UBI levels on income distribution.
* **Experimental Variables:**
  * UBI amount (UBI = 0 or UBI = 50% of the base wage)
  * Income inequality (measured by the Gini coefficient of income)

* ​**Baselines**​:

  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **benchmark\_100\_house\_bc\_gov\_rule\_firm\_rule\_bank\_rule (Blue line)** : Households are modeled as **BC Agent ​**with​**​ 100 households**​,while the government adopts the ​**Rule-Based Agent**​.Company and bank are modeled as ​**Rule-Based Agent**​.**Without ​**​**UBI**​**​ in the environment, as the benchmark situation.**
  * ​**universal\_basic\_income\_benchmark\_100\_house\_bc\_gov\_rule\_firm\_rule\_bank\_rule (Green line) ​**​: Households are modeled as **BC Agent ​**with​**​ 100 households**​,while the government adopts the ​**Rule-Based Agent**​.Company and bank are modeled as ​**Rule-Based Agent**​.**With ​**​**UBI**​**​ introduced into the environment, a higher level of welfare is reflected.**
    
* **Visualized Experimental Results：**

![Fiscal Q2 P1+](../img/Fiscal%20Q2%20P1+.png)

**Figure 1: ​**In the simulation with UBI, the income Gini coefficient is lower than in the economy without UBI, indicating that the UBI policy reduces wealth inequality.

* The UBI policy effectively reduces the gap between rich and poor.

---

### Experiment 2: UBI’s Effect on Household Labor Supply

* **Experiment Description: ​**
  
  Assess how varying UBI levels influence average working hours across income deciles.
* **Experimental Variables:**
  
  * UBI level (UBI = 0 or UBI = 50% of the base wage)
  * Average working hours of households by income tier
* ​**Baselines**​:

  We constructed the simulated economic environment using ​**Individuals modeled as Behavior Cloning Agents under the OLG (Overlapping Generations) framework and the Government modeled as a PPO-based RL Agent**​. The bar charts illustrate average household working hours under two different policy settings:
  
  * **Left group (benchmark\_100\_house\_bc\_gov\_rule\_firm\_rule\_bank\_rule)** : Represents the baseline policy without Universal Basic Income (UBI).
    * Blue bar: Rich households
    * Green bar: Middle-class households
    * Yellow bar: Poor households
    * Red bar: Overall average
  * **Right group (universal\_basic\_income\_benchmark\_100\_house\_bc\_gov\_rule\_firm\_rule\_bank\_rule)** : Represents the Universal Basic Income (UBI) policy, where households receive unconditional transfers.
    * Blue bar: Rich households
    * Green bar: Middle-class households
    * Yellow bar: Poor households
    * Red bar: Overall average
* **Visualized Experimental Results：**

![Fiscal Q2 P2+](../img/Fiscal%20Q2%20P2+.png)

**Figure 2: ​**When society adopts Universal Basic Income, the average working hours are slightly higher than in an economy without UBI. Moreover, the average working hours of wealthy households increase, while those of poor households decrease.

* The impact of Universal Basic Income on household working hours is relatively small, but the increase in labor among wealthy groups is significant. Under UBI, wealthy groups may need to work longer hours in order to achieve high income.

