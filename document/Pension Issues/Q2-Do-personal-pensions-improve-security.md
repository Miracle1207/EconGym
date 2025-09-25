# Q2: Do personal pensions improve security?

## ​1.​ Introduction

### 1.1 Definition of the Issue

The **Individual Pension Policy** refers to government-sponsored personal retirement saving schemes designed to supplement the public pension system. These schemes typically include tax incentives, investment incentives, and mandatory contributions.

### 1.2 Social Practice of Individual Pensions

* **United States:** A mature individual pension framework has existed since the 1970s with the establishment of the 401(k) plan, which encourages citizens to save for retirement on a tax-deferred basis.
* **China:** The individual pension system was formally established in 2022 and rolled out nationwide in 2024. As the “third pillar” of the retirement framework, it plays a key role in reinforcing the overall pension structure.

### 1.3 Research Questions

This study uses an economic simulation platform to investigate the ​**economic impacts of introducing and expanding personal pension schemes**​, specifically examining:

* ​**Household Savings**​: How do personal pensions affect households’ savings, consumption, and investment decisions across different income groups?
* ​**Income Security**​: Do personal pensions improve financial security for retirees, especially in vulnerable groups (e.g., low-income, middle class)?
* **GDP**​​**​ Effects**​: What is the impact of mandatory pension contributions on aggregate output and long-term GDP growth?
* ​**Wealth Distribution**​: How do personal pensions reshape income and wealth distribution across different cohorts (young vs. old) and social classes (poor, middle, rich)?
* ​**Fiscal Sustainability**​: How do personal pension schemes interact with the public pension system, and do they reduce fiscal pressure on the government?

### 1.4 Research Significance

* **Policy Reference for an Aging Society:**  Under demographic aging, public pension funds face growing fiscal strain. By promoting individual pension plans, governments can bolster personal retirement reserves and ease the burden on public finances.
* **Guidance for Personal Retirement Investment:**  Individuals must decide whether to participate in personal pension schemes and how much to contribute. This study helps households make more informed decisions when engaging with these plans.

---

## ​2. ​Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                                                                                       | Observation                                                                                                                                          | Action                                                       | Reward                                               |
| ----------- | ------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **Individual**  | OLG Model           | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals. | $o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$<br/>Private: assets, education, age<br/>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment <br/>*OLG*: old agents $\lambda_t^i = 0$    | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)   <br/>*OLG includes pension if retired*      |
| **Government**  | Pension Authority   | Pension Authority manages intergenerational transfers by setting retirement age, contribution rates, and pension payouts. | \$\$o\_t^g = ( F\_{t-1}, N\_{t}, N^{old}\_{t}, \\text{age}^r\_{t-1}, \\tau^p\_{t-1}, B\_{t-1}, Y\_{t-1}) \$\$ <br>Pension fund, current population, old individuals number, last retirement age, last contribution rate, debt, GDP | $a_t^{\text{pension}} = ( \text{age}^r_t, \tau^p_t, k )$<br>Retirement age, contribution rate, growth rate | Pension fund sustainability                                  |
| **Firm**       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                 | /                                                                                                                                                    | /                                                            | Zero (long-run)                                      |
| **Bank**       | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives.   | /                                                                                                                                                    | No rate control                                              | No profit                                            |


---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
Captures how pension policies affect savings, investment, and retirement planning across age cohorts. The OLG framework yields precise insights into intergenerational behavior.

**Government →  Pension Authority**  
Designs and adjusts individual pension regulations, directly influencing personal saving, consumption, and investment. Focuses specifically on pension rules, whereas the Ministry of Finance oversees the broader government budget.

**Firm → Perfect Competiton**  
Firms’ production, investment, and wage decisions respond to changes in personal pension incentives. Perfectly Competitive Market ensures that increased households' savings are fully reflected in capital-market prices.

**Bank → Non-Profit Platform**  
Channel pension contributions into various financial investments and determine capital allocation. Arbitrage-Free Financial Institutions simulate the impact of different investment strategies, evaluating how pension reforms shock capital markets.

---

## 3. Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | Imitates real-world behavior by training on empirical data. Enables realistic micro-level behavior |
| Government             | Rule-Based Agent       | Formulate individual pension policies based on policy parameters    |
| Firm                 | Rule-Based Agent                          | Simulate firms’ investment and wage‐setting decisions                                       |
| Bank | Rule-Based Agent                          | Assess how pension savings influence capital markets                                          |

---
## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "personal_pension"
```

This command loads the configuration file `cfg/personal_pension.yaml`, which defines the setup for the "personal_pension" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `personal_pension.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "personal_pension"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "pension"
          personal_contribution_rate: 0.1   # todo: set personal_contribution_rate
    - entity_name: 'households'
      entity_args:
        params:
          type: 'OLG'
          households_n: 1000
          action_dim: 2

        OLG:
          birth_rate: 0.011

          initial_working_age: 24
    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"

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

### Experiment 1: How do tax incentives for personal pension savings impact social GDP?

* **Experiment Description:**
  
  Providing tax exemptions for a portion of personal pensions can increase households’ enthusiasm for saving in personal pension accounts. This experiment simulates the impact of different pension tax incentives (tax-exemption levels) on GDP in the simulated economy.
* **Experimental Variables:**
  * Tax-exemption levels (e.g., +10%, +15%, +20%)
  * Long-term GDP performance under each scenario
* **Baselines：**

  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.

  * **rule\_based\_rule\_based\_1000\_personal\_pension\_0.15 (Blue line):** Both households and the government are modeled as ​**Rule-Based Agents**​, with **1000 households** and a tax-exemption level of 15%.
  * **rule\_based\_rule\_based\_1000\_personal\_pension\_0.1 (Light green line):** Both households and the government are modeled as ​**Rule-Based Agents**​, with **1000 households** and a tax-exemption level of 10%.
  * **rule\_based\_rule\_based\_1000\_personal\_pension\_0.2 (Yellow line):** Both households and the government are modeled as ​**Rule-Based Agents**​, with **1000 households** and a tax-exemption level of 20%.
  * **rule\_based\_rule\_based\_100\_personal\_pension\_0.15 (Red line):** Both households and the government are modeled as ​**Rule-Based Agents**​, with **100 households** and a tax-exemption level of 15%.
  * **​rule\_based\_rule\_based\_100\_personal\_pension\_0.1 (​**​**Cyan**​**​ line):** Both households and the government are modeled as ​**Rule-Based Agents**​, with **100 households** and a tax-exemption level of 10%.
  * **rule\_based\_rule\_based\_100\_personal\_pension\_0.2 (Dark green line):** Both households and the government are modeled as **Rule-Based Agents**​, with **100 households** and a tax-exemption level of 20%.

![Pension Q3 P1](../img/Pension%20Q3%20P1.png)
  
  **Figure 1:** GDP trajectories for household populations of 1,000 and 100 under differing exemption levels. Higher exemptions correlate with lower GDP.
* Higher tax incentives for individual pension savings are associated with lower aggregate GDP, possibly because such incentives prompt households to sacrifice some consumption in favor of increased pension contributions, thereby reducing overall demand for goods and leading to a decline in GDP.

---

### Experiment 2: How do tax incentives for pensions affect household income?

* **Experiment Description:**
  
  Simulate effects of varying tax-exemption levels on long-term household incomes.
* **Experimental Variables:**
  
  * Tax-exemption levels (+10%, +15%, +20%)
  * Long-term income comparisons across age and income deciles
* **Visualized Experimental Results：**
![Pension Q3 P2](../img/Pension%20Q3%20P2.png)
  
  **Figure 2:** Bar charts showing income distributions by age group (left) and income bracket (right) under different exemption levels. Lower exemptions (the green bar) lead to larger income declines.
* Generous tax incentives for personal pension contributions are associated with higher long-term household incomes: by increasing households’ willingness to participate in pension schemes, these exemptions channel savings into banks and financial markets, which in turn support greater income growth over time.

---

### Experiment 3: Training Curves for Pension Problems with RL-Agent Government 

* **Experiment Description:**

    In certain pension-related problems, the government is trained using the PPO reinforcement learning algorithm, and we observe the relevant economic variables of households and the government.
* **Experimental Variables:**
  
  * Gov\_reward
  * Pension\_gov\_reward
  * Years
  * House\_work\_hours
  * House\_age
  * House\_pension
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  * ​**pension\_gap\_1000\_house\_bc\_gov\_ppo\_firm\_rule\_bank\_rule\_seed=1**​: Households are modeled as **Behavior Cloning (BC) Agents** , using the OLG model with ​**1000 households**​, while the government is trained using the **PPO**​​**​ algorithm**​.Bank and firm are modeled as **Rule-Based Agent.**

![Pension Q1 PP](../img/Pension%20Q1%20PP.png)

Figure 2: After the government is trained with a PPO Agent, both household working hours and wages show a significant long-term increase. The duration of the simulated economy extends as the number of steps increases. In addition, government rewards rise steadily with training steps.


