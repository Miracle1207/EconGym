# Q1: How does delayed retirement affect the economy?

## 1. Introduction

### 1.1 Delayed Retirement Policy

The **Delayed Retirement Policy** refers to raising the statutory retirement age so that workers remain active in the labor market for a longer period. This directly affects the economic behavior of individuals, firms, governments, and financial markets.

### 1.2 Background of the Study

Globally, population aging has become increasingly severe, placing substantial fiscal pressure on **pension systems**. Many countries are considering or have already implemented delayed retirement policies to relieve pension payment burdens, boost labor supply, and promote economic growth.

### 1.3 Research Questions

This study uses an economic simulation platform to investigate the **economic impacts of delayed retirement policies**, specifically examining:

* **Labor Supply:** How does delayed retirement affect total labor supply?  
* **GDP Effects:** Does delaying retirement contribute to economic growth?  
* **Pension System Sustainability:** How does delayed retirement influence government finances and pension disbursements?  
* **Aging Pressure:** What is the impact of delayed retirement on the overall aging burden in the economy?  
* **Welfare Effects:** How does delayed retirement affect individual benefits and overall social welfare?  

### 1.4 Research Significance

* **Policy Guidance:** Considering that delayed retirement may yield diverse economic effects—such as impacts on labor markets, wage levels, capital markets, and social equity—this research provides strong guidance for policymaking.  
* **Balancing Equity and Efficiency:** The study clarifies how delayed retirement redistributes wealth and opportunities across generations, supporting policy designs that reconcile social fairness with economic efficiency.  

---

## 2. Selected Economic Roles

As an example, the following table shows the economic roles most relevant to the delayed retirement question, along with their role descriptions, observations, actions, and reward functions. For further details, please refer to [our paper](https://arxiv.org/pdf/2506.12110).

| Social Role | Selected Type       | Role Description                                             | Observation                                                  | Action                                                       | Reward                                   |
| ----------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| **Individual**  | OLG Model           | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals. | $o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$<br/>Private: assets, education, age<br/>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment <br/>*OLG*: old agents $\lambda_t^i = 0$    | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)   <br/>*OLG includes pension if retired*      |
| **Government**  | Pension Authority   | Pension Authority manages intergenerational transfers by setting retirement age, contribution rates, and pension payouts. | \$\$o\_t^g = ( F\_{t-1}, N\_{t}, N^{old}\_{t}, \\text{age}^r\_{t-1}, \\tau^p\_{t-1}, B\_{t-1}, Y\_{t-1}) \$\$ <br>Pension fund, current population, old individuals number, last retirement age, last contribution rate, debt, GDP | $a_t^{\text{pension}} = ( \text{age}^r_t, \tau^p_t, k )$<br>Retirement age, contribution rate, growth rate | Pension fund sustainability                                  |
| **Firm**        | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses. | /                                                            | /                                                            | Zero (long-run)                          |
| **Bank**        | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives. | /                                                            | No rate control                                              | No profit                                |

---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
In aging-related questions, it is essential to choose the OLG model, since it explicitly models the **age feature**. The Ramsey model instead treats individuals as infinitely-lived households, which is more suitable for questions that are not age-sensitive.  

**Government → Pension Authority**  
For aging and retirement questions, the **Pension Authority** is necessary because it directly controls retirement age and pension transfers. Other government agents (e.g., Fiscal Authority, Central Bank) may also be included in extended experiments.  

**Firm → Perfect Competition**  
When market competition details are not the focus, firms can be modeled as perfectly competitive. This assumes market clearing and eliminates strategic complexity, providing a simplified baseline.  

**Bank → Non-Profit Platform**  
When banking interest-rate dynamics are not the focus, the Non-Profit Platform serves as a neutral financial intermediary, applying uniform rates without profit-seeking behavior.  

---

## 3. Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual    | Behavior Cloning Agent   | Imitates real-world behavior by training on empirical data. Enables realistic micro-level behavior. |
| Government    | RL Agent       | Learns through trial-and-error to optimize long-term cumulative rewards. Well-suited for solving dynamic decision-making problems.  |
| Firm          | Rule-Based Agent       | Perfect competition implies market clearing and first-order optimality conditions, consistent with rule-based methods. |
| Bank          | Rule-Based Agent       | Non-Profit Platforms have no interest-rate control authority, and thus can be modeled as rule-based intermediaries. |


---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "delayed_retirement"
```

This command loads the configuration file `cfg/delayed_retirement.yaml`, which defines the setup for the "delayed_retirement" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `delayed_retirement.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "delayed_retirement"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "pension"  # Type of government task: ['tax', 'pension', 'central_bank']
          gov_task: "gdp"
    - entity_name: 'households'
      entity_args:
        params:
          type: 'OLG'  # Household model type: ['ramsey', 'OLG', 'OLG_risk_invest', 'ramsey_risk_invest']
          households_n: 1000
        OLG:
          birth_rate: 0.011
          initial_working_age: 24
    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"   # Market type: ['perfect', 'monopoly', 'monopolistic_competition', 'oligopoly']
    - entity_name: 'bank'
      entity_args:
        params:
          type: 'non_profit'   # Bank type: ['non_profit', 'commercial']
          lending_rate: 0.0345
          deposit_rate: 0.0345
          reserve_ratio: 0.1
          base_interest_rate: 0.0345
          depreciation_rate: 0.06

Trainer:
  house_alg: "bc"  # Household algorithm (e.g., bc = Behavior Cloning)
  gov_alg: "ppo"   # Government algorithm (e.g., ppo = Proximal Policy Optimization)
  firm_alg: "rule_based"  # Firm algorithm 
  bank_alg: "rule_based"  # Bank algorithm 
  seed: 3  # Random seed for reproducibility
  epoch_length: 300  # Length of each training episode
  n_epochs: 1000  # Number of training epochs
  test: False  # Flag to indicate if this is a test run
```
---



## 5.​Illustrative Experiments

### Experiment 1: Impact of Different Retirement Ages on Economic Growth

* **Experiment Description:**
  
  We tested the economic effects corresponding to different retirement ages (RA = 60, 63, 65, 67, 70).

* **Experimental Variables:**

  * Different retirement ages
  * Long-term GDP performance under each scenario

* **Baselines:**

  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * ​**rule\_based\_rule\_based\_1000\_OLG\_60.0 (Blue line)**: Both households and the government are modeled as ​**Rule-Based Agents**​, with a retirement age of 60 and 1000 households.
  * ​**rule\_based\_rule\_based\_1000\_OLG\_65.0 (Light green line)**: Both households and the government are modeled as ​**Rule-Based Agents**​, with a retirement age of 65 and 1000 households.
  * ​**rule\_based\_rule\_based\_1000\_OLG\_70.0 (Yellow line)**: Both households and the government are modeled as ​**Rule-Based Agents**​, with a retirement age of 70 and 1000 households.
  * ​**rule\_based\_rule\_based\_100\_OLG\_60.0 (Red line)**: Both households and the government are modeled as ​**Rule-Based Agents**​, with a retirement age of 60 and 100 households.
  * ​**rule\_based\_rule\_based\_100\_OLG\_65.0 (Cyan line)**: Both households and the government are modeled as ​**Rule-Based Agents**​, with a retirement age of 65 and 100 households.
  * ​**rule\_based\_rule\_based\_100\_OLG\_70.0 (Dark green line)**: Both households and the government are modeled as ​**Rule-Based Agents**​, with a retirement age of 70 and 100 households.

* **Visualized Experimental Results：**

![Pension Q2 P1](../img/Pension%20Q2%20P1.png)

**Figure 1:** The experiment observes that economies with earlier retirement ages exhibit higher aggregate GDP, although this difference is less evident when the household count is 100.

* Delaying retirement does not raise aggregate output in the long run. One reason may be that extended working years reduce households’ time and willingness to consume, interrupting their life-cycle consumption and saving plans.

---

### **Experiment 2: Training Curves for Pension Problems with RL-Agent Government ​**

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


