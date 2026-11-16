# Q1: How does technology drive long-term growth?

 ## 1. Introduction

### **1.1 Introduction to Technological Progress**

Technological progress refers to improvements in production techniques, management methods, or tools that enable higher output from the same level of input (e.g., capital and labor). In economics, it is commonly measured by the **Solow ​**​​**Residual**​, which captures the portion of economic growth not accounted for by increases in capital or labor.

### **1.2 ​**​**Primary**​**​ Drivers of Technological Progress**

According to ​**endogenous growth theory**​, technological progress originates from within the economic system, driven primarily by innovation, R&D investment, and knowledge accumulation. Improvements in education and human capital further enhance productivity and foster innovation, while capital deepening strengthens this process by enabling the adoption of advanced technologies such as automation and artificial intelligence.

### **1.3 Research Questions**

This study leverages an economic simulation platform to investigate the long-term effects of technological progress on economic and social dynamics, focusing on:

* ​**Wage Levels**​: Does technological progress raise or lower average wages?
* ​**Income Inequality**​: Does it widen or narrow the wealth gap between individuals?
* **GDP**​​**​ Effect**​: How does technological progress impact GDP?

### **1.4 Research Significance**

* **Understanding Growth Mechanisms:**  Analyzing how technological progress affects the labor market helps reveal the underlying drivers of economic growth, offering policy insights to guide sustainable development and improve social welfare.
* **Optimizing Income Distribution:**  As technological progress may either exacerbate or alleviate income inequality, studying its effects can inform the design of equitable tax and social security policies to ensure inclusive economic development.

---

## **2. Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                                                                                                                     | Observation                                                                                                                                          | Action                                                                                   | Reward                                 |
| ----------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------- |
| **Individual**  | OLG Model           | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals.    | $o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$<br/>Private: assets, education, age<br/>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment <br/>*OLG*: old agents $\lambda_t^i = 0$    | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)   <br/>*OLG includes pension if retired*      |
| **Government**  | Fiscal Authority    | Design and adjust inheritance-tax policy and assess its impact on public finances.                                           |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
| **Firm**       | Perfect Competition | Observe how shifts in consumer demand affect firms’ production and pricing strategies.                                                              | /                                                                                                                                                    | /                                                                                          | Zero (long-run)                        |
| **Bank**       | Non-Profit Platform | Study capital-market reactions to inheritance-tax policy, particularly changes in saving rates and investment behavior.                             | /                                                                                                                                                    | No rate control                                                                            | No profit                              |


---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
The OLG model is particularly suitable for **studying the direct impact of technological progress on the labor market**. It allows for the simulation of labor-related issues associated with demographic structure changes across different generations.

**Government → Fiscal Authority**  
Technological progress may lead to **rising income inequality or increased unemployment**. The government, through the Treasury Department , is responsible for formulating policy responses—such as taxation or subsidies—to mitigate these effects.

**Firm → Perfect Competition**  
In a perfectly competitive market, wages are determined by supply and demand. This structure allows the model to **more directly reflect the impact of technological progress** on firm-level wage setting and labor demand.

**Bank → Non-Profit Platform**  
Technological progress can influence investment returns and capital market dynamics. Therefore, arbitrage-free financial institutions are selected to simulate impacts on variables such as interest rates. Commercial banks are not selected in this context, as their focus is primarily on deposit and lending activities, which have a more limited scope of influence in this setting.

---

## **3. Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | Imitates real-world behavior by training on empirical data. Enables realistic micro-level behavior.      |
| Government             | Rule-Based Agent         | Regulate government behavior based on the US-government rules.                  |
| Firm                 | Rule-Based Agent | Define how firms adjust wages, production scale, and hiring decisions in response to technological change.   |
| Bank  | Rule-Based Agent | Set interest rates and investment returns to assess the impact of technological progress on capital markets. |

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "technology"
```

This command loads the configuration file `cfg/technology.yaml`, which defines the setup for the "technology" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `technology.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "technology"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "tax"  # Focus on pension policy. type_list: ['tax', 'pension', 'central_bank']
    - entity_name: 'households'
      entity_args:
        params:
          type: 'OLG'
          type_list: ['ramsey', 'OLG', 'OLG_risk_invest', 'ramsey_risk_invest']
          households_n: 1000
          action_dim: 2


    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"   #  type_list: [ 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly' ]
          Z: 1.0                        # todo: set different productivity level


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
  house_alg: "bc" #The rule-based Agent can also be chosen in this experiment.
  gov_alg: "us_federal" #The PPO Agent can also be chosen in this experiment.
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  seed: 1
  epoch_length: 300
  cuda: False
#  n_epochs: 300
#  wandb: True
```
---

## **5.Illustrative Experiment**

### **Experiment 1: The Impact of Technological Progress on Average Wages**

* ​**Experiment Description**​:

  Analyze how technological progress influences workers' wages.
* ​**Experimental Variables**​:
  
  * Rate of technological progress (the parameters representing tech growth)
  * Social wage level
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.The following experiments have the same agent settings as this experiment, so we will give all explaination here.
  
  * **base\_rule\_based\_ppo\_100\_OLG (blue line):** Households are modeled as **Rule\_based Agents** under the **OLG model ​**with ​**100 households**​, while the government is modeled as **Reinforcement Learning**​**​ ​**​**PPO**​**​​ Agent.​**The firm is accompanied by a higher rate of technological growth.
  * **TechGrowth\_rule\_based\_ppo\_100\_OLG (green line):** Households are modeled as **Rule\_based Agents** under the **OLG model ​**with ​**100 households**​, while the government is modeled as **Reinforcement Learning**​**​ ​**​**PPO**​**​​ Agent.​**The firm has a normal technomlogical growth rate.
* **​ Visualized Experimental Results：**
  
![Market Q1 P1](../img/Market%20Q1%20P1.png)

**Figure 1:** Under accelerated technological progress, the social wage rate rises steadily, and the gap with the baseline scenario of normal progress expands over time.

* Technological progress has significantly driven the increase in the average wage level in society, and this gap has become increasingly pronounced over time, sufficient to demonstrate that technological progress will raise the overall wage rate in society.

---

### **Experiment 2: The Impact of Technological Progress on Income Inequality**

* ​**Experiment Description**​:

  Does technological progress promote employment? What are its short-term and long-term utility effects?
* ​**Experimental Variables**​:
  * Speed of technological progress (the parameters representing tech growth)
  * Income inequality (measured by the Gini coefficient)
* **​ Visualized Experimental Results：**

![Market Q1 P2](../img/Market%20Q1%20P2.png)

​**Figure 2**​: When technological progress accelerates, income inequality increases compared to the case of normal progress, with a higher Gini coefficient indicating more severe social inequality.

* Technological progress has led to an increase in income inequality in society (as indicated by the rise in the ​Gini coefficient), suggesting that the wage growth rate of high-skilled workers will outpace that of general workers, thereby creating a larger income ​disparity​.

---

### **Experiment 3: The Impact of Technological Progress on Total Social ​**​**Output**

* ​**Experiment Description**​:

  Does technological progress lead to an increase in total social output?
* ​**Experimental Variables**​:
  * Speed of technological progress (or parameters representing tech growth)
  * Total social output (GDP)
* **Visualized Experimental Results：**

![Market Q1 P3](../img/Market%20Q1%20P3.png)

**Figure 3: ​**Technological progress leads to a rapid increase in total social output.

* Technological progress has significantly increased the ​level of social output, and while the disparity in social output was not particularly noticeable at first, it has become increasingly pronounced over time.

