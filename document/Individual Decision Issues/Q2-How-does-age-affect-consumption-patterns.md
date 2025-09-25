# Q2: How does age affect consumption patterns?

## 1.Introduction

### 1.1 **Consumption‑Propensity Differences Across Age Cohorts**

Consumption Propensity Differences refer to systematic variations in spending behavior, patterns, and preferences across different age cohorts. Specifically, this encompasses differences in:

* **Marginal Propensity to Consume:** ​​​ (MPC) the proportion of disposable income devoted to consumption
* **Determinants of Consumption Decisions:** such as income, wealth, expected lifespan, and health status
* **Spending Habits and Priority of Expenditures**

Dividing adults into four groups:Youth (18–30), Early Middle Age (31–45), Late Middle Age (46–65), and Senior (> 65).We can reasonably infer the following patterns based on economic theory and observed social phenomena:

* **Youth (18–30):**  Young adults prioritize immediate gratification and experiential spending. They often depend on short-term income streams and credit-financed consumption, resulting in a high marginal propensity to consume.Driven by quality‐of‐life demands, they allocate a larger share of disposable income to consumption.
* **Early-middle-aged (31–45):**  With growing family responsibilities, this cohort’s MPC declines. They begin saving for the future, particularly for education, housing, and retirement. Consumption is more grounded in stable income and long‐term household planning.
* **Late-middle-aged (46–65):**  Individuals in this stage are typically at their career peak and focus increasingly on retirement and health security. Their MPC decreases further as they prioritize savings for retirement and future needs, leading to more stable spending and reduced non‐essential expenditures.
* **Elderly (65 and older):**  Seniors typically live on fixed pensions or accumulated savings, concentrating consumption on daily necessities and healthcare. With lower income, their MPC is low, and spending centers on essential goods and medical support.

### 1.2 Research Questions

This study uses an economic simulation platform to investigate the ​**economic impacts of age-dependent consumption patterns**​, specifically examining:

* **Household Consumption:** How do different age cohorts affect aggregate household consumption levels and structures?
* **Household Utility:** How does age influence individual utility derived from consumption and saving decisions?
* ​**Social Welfare:** What are the overall welfare implications of heterogeneous consumption behaviors across age groups?


### 1.3 **Research Significance**

Population‐structure shifts are a critical determinant of both macroeconomic outcomes and household‐level decisions. With global population aging intensifying, understanding consumption‐behavior differences across age cohorts is of paramount importance:

* **Consumption Behavior in an Aging Society:**  As the world’s population ages rapidly, this experiment uses consumption‐propensity differences to shed light on how demographic changes influence aggregate demand and economic growth.
* **Corporate Marketing Strategies:**  Technological transformations are profoundly reshaping consumption habits across age groups. Insights from this study can help firms tailor product and marketing strategies to better align with the preferences of each cohort.

## 2.Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:


| Social Role | Selected Type       | Role Description                                                        | Observation                                                  | Action                                                       | Reward                                   |
| ----------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| **Individual**  | OLG Model           | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals. | $o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$<br/>Private: assets, education, age<br/>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment <br/>*OLG*: old agents $\lambda_t^i = 0$    | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)   <br/>*OLG includes pension if retired*      |
| **Firm**        | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses. | /                                                            | /                                                            | Zero (long-run)                          |
| **Bank**       | Commercial Banks   | Commercial Banks strategically set deposit and lending rates to maximize profits, subject to central bank constraints. | $o_t^{\text{bank}} = ( \iota_t, \phi_t, r^l_{t-1}, r^d_{t-1}, loan, F_{t-1} )$<br>Benchmark rate, reserve ratio, last lending rate, last deposit_rate, loans, pension fund.| $$a_t^{\text{bank}} = \{ r^d_t, r^l_t \}$$<br>Deposit, lending decisions(Commercial Banks)    | $$r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$$<br>Interest margin (Commercial Banks) |

---

### Rationale for Selected Roles

**Individual →  Overlapping Generations (OLG) Model /  Ramsey Model**  
From the OLG perspective, differences in consumption propensities are fundamentally an intergenerational issue.The OLG framework explicitly **distinguishes consumption decisions, saving behaviors, and budget constraints across age cohorts.**
In contrast, while the infinite‐horizon Ramsey model assumes perpetual individual lifespans, it can still differentiate age‐related variations in consumption propensity.

**Government →  No Specific Role**  
This study focuses on the endogenous differences in consumption and saving behaviors across age cohorts, driven by life-cycle characteristics such as income, health status, and household composition. It **deliberately excludes any form of fiscal intervention (e.g., taxation, subsidies, or social transfers) in order to isolate and observe the pure behavioral dynamics of different age groups.** The absence of government policies ensures that observed consumption-propensity differences stem solely from individual decision-making under varying demographic and economic conditions.

**Firm →Perfect Competition**  
To study consumption‐propensity differences, we need to **examine how markets respond to varying demand across age cohorts**. A perfectly competitive market most directly reflects the impact of demand shifts on prices and supply, making it ideal for analyzing age‐specific consumption effects. In contrast, monopolistic or oligopolistic structures can distort price signals and mask true consumer behavior, rendering them unsuitable for this research.

**Bank → Commercial Banks**  
Commercial banks, by offering deposit and lending services, directly shape consumption–saving decisions across age groups. Younger cohorts may rely more on borrowing (e.g., student loans, mortgages), whereas seniors focus on savings yields and pension management. By comparison, arbitrage‐free financial intermediaries are more concerned with overall capital‐market efficiency and interact less directly with individual consumers, making them less suitable for this study.

---

## 3.Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent          | Each household maximizes utility by using real world data to optimize age‑specific consumption‑saving decisions and preferences. |
| Government             | Rule‑Based Agent | Executes taxation and transfer policies according to explicit fiscal rules.                                                               |
| Firm                 | Rule‑Based Agent | Adjusts prices and supply under predefined rules to match demand shifts and keep the market in equilibrium.                               |
| Bank | Rule‑Based Agent | Delivers standardized financial services—uniform risk assessment and product pricing—for all age cohorts.                               |


---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "age_consumption"
```

This command loads the configuration file `cfg/age_consumption.yaml`, which defines the setup for the "age_consumption" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `age_consumption.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "age_consumption"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "pension" # central_bank gov

    - entity_name: 'households'
      entity_args:
        params:
          type: 'OLG'

    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"   # ['perfect', 'monopoly', 'monopolistic_competition', 'oligopoly']


    - entity_name: 'bank'
      entity_args:
        params:
          type: 'commercial'


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

## 5.Illustrative Experiment

### **Experiment : Baseline Consumption Propensity Analysis**

* **Experiment Description:**
  
  Build benchmark models of consumption propensity for each age group based on statistical survey data, and record their consumption outcomes when run on the simulated economic platform.Compare the differences between the rule-based government  and the government using the PPO algorithm。
* **Experimental Variables:**

  * Government agent: Rule-Based Agent vs. PPO Agent
  * Rich,Middle-class,Poor and Overall people consumption level
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each group of bars in the visualization to help readers better understand the results. The bar charts show household consumption distributions under different tax policies at year 50.
  *   ​**age\_consumption\_100\_house\_bc\_gov\_rule\_firm\_rule\_bank\_rule(left panel)**​:Households are modeled as **BC Agent** with ​**100 households**​,while the government adopts the ​**Rule-Based Agent**​.Company and bank are modeled as ​**Rule-Based Agent**​.
      * Blue bar: Rich households
      * Green bar: Middle-class households
      * Yellow bar: Poor households
      * Red bar: Overall average
  * ​**age\_consumption\_100\_house\_bc\_gov\_ppo\_firm\_rule\_bank\_rule(right panel)**​:Households are modeled as **BC Agent** with ​**100 households**​,while the government adopts the **PPO**​​**​ RL Agent**​.Company and bank are modeled as ​**Rule-Based Agent**​.
      * Blue bar: Rich households
      * Green bar: Middle-class households
      * Yellow bar: Poor households
      * Red bar: Overall average



* **​ Visualized Experimental Results：**

![Individual Q3 P1+](../img/Individual%20Q3%20P1+.png)

**Figure 1:** Comparison of household consumption at year t = 31 under two simulation rules. Compared with a government using a Rule-Based Agent, a government adopting an RL Agent achieves higher per capita consumption, reflecting a better overall consumption level. At the same time, the lower consumption of wealthy groups indicates a more equal consumption pattern.

* Incorporating age layering alongside the platform’s existing income stratification enables a more accurate characterization of household groups and yields a more complete model.

