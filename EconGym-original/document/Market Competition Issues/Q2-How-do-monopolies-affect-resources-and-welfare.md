# Q2: How do monopolies affect resources and welfare?

## **1. Introduction**

### **1.1 Introduction to Monopoly Markets**

A**​ monopoly market ​**refers to a market structure in which the supply of a particular good or service is entirely controlled by a single firm. In such markets, firms possess significant pricing power due to the absence of effective competition, allowing them to freely set prices to maximize profits. Compared to perfectly competitive markets, monopolies often result in reduced allocative efficiency and diminished consumer surplus, though they may also incentivize innovation.

In reality, monopolistic firms exert strong control over pricing and market access, enabling them to extract substantial monopoly profits. Notable examples include:

* ​**U.S. Defense Contractors (e.g., Lockheed Martin)** ​: The U.S. defense industry is highly concentrated, with a few firms handling the majority of defense contracts. This de facto monopoly enables high pricing and bundling strategies in international arms sales.
* ​**Samsung in South Korea**​: Samsung holds a dominant position in South Korea's electronics sector, with monopolistic influence over semiconductors, smartphones, and display panels, significantly shaping pricing and supply chain structures.

### **1.2 How Monopoly Markets Affect Resource Allocation**

* **Price ​Distortion and Resource Misallocation:**  Monopoly firms tend to set prices above marginal cost, suppressing demand and creating deadweight loss, leading to inefficient resource allocation in the economy.
* **Innovation Incentives and Government Intervention:**  Government price regulation may alleviate welfare losses but can also dampen firms' incentives for innovation. This creates a trade-off between efficiency and equity, affecting long-term resource allocation dynamics.

### **1.3 Research Questions**

This study uses an economic simulation platform to investigate the economic impacts of monopoly market structures, specifically examining:

* **Household Consumption: ​**How does monopoly pricing reduce household purchasing power and overall consumption levels?
* **Wealth Distribution: ​**How do monopoly profits influence the concentration of income and wealth across households?
* ​**Social Welfare: ​**What is the net effect of monopolies on social welfare, considering losses in consumer surplus and deadweight loss?

### **1.4 Research Significance**

* **Policy Guidance:**  Many countries have implemented antitrust laws to regulate monopolies. A clear understanding of monopolies' impacts at both macro and micro levels is essential for designing more effective and targeted regulatory policies.
* **Equity and Distributional Analysis:**  The study also investigates how different social groups are affected by monopolistic pricing, and whether such pricing contributes to widening income inequality.

---

## **2. Selected Economic Roles**

Select the following roles from the social role classification of the economic simulation platform:

| Social Role | Selected Type        | Role Description                                                                                                    | Observation                                                                                                  | Action                                                                                 | Reward                                              |
| ----------- | -------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Individual**  | Ramsey Model         | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.           | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Government**  | Fiscal Authority     | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.                 |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
| **Firm**       | Monopoly             | Monopoly Firms set prices and wages to maximize profits under aggregate demand constraints.                        | $o_t^{\text{mono}} = ( K_t, Z_t, r_{t-1}^l )$<br>Production capital, productivity, lending rate | $a_t^{\text{mono}} = ( p_t, W_t )$<br>Price and wage decisions | $r_t^{\text{mono}} = p_t Y_t - W_t L_t - R_t K_t$<br>Profits = Revenue – costs |
| **Bank**       | Commercial Bank   | Commercial Banks strategically set deposit and lending rates to maximize profits, subject to central bank constraints. | $o_t^{\text{bank}} = ( \iota_t, \phi_t, r^l_{t-1}, r^d_{t-1}, loan, F_{t-1} )$<br>Benchmark rate, reserve ratio, last lending rate, last deposit_rate, loans, pension fund.| $$a_t^{\text{bank}} = \{ r^d_t, r^l_t \}$$<br>Deposit, lending decisions(Commercial Banks)            | $$r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$$<br>Interest margin (Commercial Banks)  |


---

### Rationale for Selected Roles

**Individual →Ramsey Model**  
Ramsey Model is used to capture how price changes in a monopoly market influence household consumption decisions, focusing on differences across individuals rather than across age groups.
Although the Overlapping Generations (OLG) model accounts for intergenerational decision-making—e.g., younger individuals tend to save more for the future—the focus here is not on age-specific behavior, but rather on heterogeneity unrelated to the life stage.

**Government → Fiscal Authority**  
The Treasury Department may implement price control policies to ensure market fairness and balance social welfare under monopolistic conditions.

**Firm → Monopoly**  
The firm sets a monopoly price, thereby impacting households and further influencing society as a whole.

**Bank → Commercial Bank**  
Within the research framework of monopoly problems, commercial bank can more realistically simulate real-world situations, reflecting the game between banks and firms.

---

## **3. Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent    | Learns consumer behavior from historical data, including price sensitivity, consumption patterns, and responses to monopoly pricing.                                |
| Government             | Rule-Based Agent | The government should be able to execute regulatory functions through predefined rules,like Seaz tax framework. |
| Firm                 |  RL Agent                  |Monopoly firms learn to obtain the optimal pricing rules in order to ensure the maximization of their own profits.                                                                         |
| Bank| RL Agent          | As profit-oriented institutions, commercial banks need to maximize their own profits, and reinforcement learning agents serve as an efficient means of learning.               |

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "monopoly"
```

This command loads the configuration file `monopoly.yaml`, which defines the setup for the "monopoly" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `monopoly.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "monopoly"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "tax"  # Focus on pension policy. type_list: ['tax', 'pension', 'central_bank']
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
          type: "monopoly"   #  type_list: [ 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly' ]
          alpha: 0.36
          Z: 1.0
          sigma_z: 0.0038
          epsilon: 0.5

    - entity_name: 'bank'
      entity_args:
        params:
          type: 'commercial'   # [ 'non_profit', 'commercial' ]
          n: 1
          lending_rate: 0.0345
          deposit_rate: 0.0345
          reserve_ratio: 0.1
          base_interest_rate: 0.0345
          depreciation_rate: 0.06

Trainer:
  house_alg: "bc"
  gov_alg: "saez"
  firm_alg: "ppo"
  bank_alg: "ppo"
  seed: 1
  epoch_length: 300
  cuda: False
#  wandb: True
#  n_epochs: 1000
```

---

## **​5.​**​**Illustrative Experiment**


### **Experiment 1: The Impact of Monopoly Market on Income Inequality**

* **Experiment Description:**
  
  Investigate the impact of monopolies on income inequality.
* **Experimental Variables:**
  
  * Monopoly Market vs. Perfectly Competitive Market
  * Change in income inequality (measured by the Gini coefficient)
* **Baselines:**

  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  * **baseline\_real\_rule\_based\_100\_OLG (Blue line):**​​**Perfectly competitive market**​.The households modeled as ​**Behavior-Cloning agents,100 households**​.The government modeled as a **Rule-Based Agent.**
    
  * **baseline\_real\_ppo\_100\_OLG:**​**​Perfectly competitive market.​**The households modeled as ​**Behavior-Cloning agents,100 households**​.The government modeled as a **PPO**​**​ Agent.**
  * **monopoly\_real\_rule\_based\_100\_OLG (Yellow line):**​**​Monopoly market.​**The households modeled as ​**Behavior-Cloning agents,100 households**​.The government modeled as a **Rule-Based Agent.**
  * **monopoly\_real\_ppo\_100\_OLG:**​​**Monopoly market**​.The households modeled as ​**Behavior-Cloning agents,100 households**​.The government modeled as a **PPO**​**​ Agent.**
* **​ Visualized Experimental Results：**

![Market Q2 P1](../img/Market%20Q2%20P1.png)

**Figure 1:** In the first 50 years, under the monopoly market, income inequality is similar to that in a perfectly competitive market. However, over time, income inequality under the monopoly market becomes significantly higher compared to the perfectly competitive market.

* The monopoly market increases the wealth gap between households in the medium to long term.

---

### **Experiment 2: The Impact of Monopoly Market on Household Consumption and Income**

* **Experiment Description:**

  Simulate changes in household consumption and income under a monopoly market to observe the effects of monopolies on micro-level individuals.
* **Experimental Variables:**
  
  * Monopoly Market vs. Perfectly Competitive Market
  * Household income and consumption levels (divided by income tiers)
* **Baselines:**

  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  * **Groups from left to right:**
    * **baseline\_real\_rule\_based\_100\_OLG :**​​**Perfectly competitive market**​.The households modeled as ​**Behavior-Cloning agents,100 households**​.The government modeled as a **Rule-Based Agent.**
    * **baseline\_real\_ppo\_100\_OLG:**​**​Perfectly competitive market.​**The households modeled as ​**Behavior-Cloning agents,100 households**​.The government modeled as a **PPO**​**​ Agent.**
    * **monopoly\_real\_rule\_based\_100\_OLG :**​**​Monopoly market.​**The households modeled as ​**Behavior-Cloning agents,100 households**​.The government modeled as a **Rule-Based Agent.**
    * **monopoly\_real\_ppo\_100\_OLG:**​​**Monopoly market**​.The households modeled as ​**Behavior-Cloning agents,100 households**​.The government modeled as a **PPO**​**​ Agent.**
  * **Bar description:**
    * **Blue bar:** Rich households
    * **Green bar:** Middle-class households
    * **Yellow bar:** Poor households
    * **Red bar:** Overall average
* **Visualized Experimental Results：**

![Market Q2 P2](../img/Market%20Q2%20P2.png)

![Market Q2 P3](../img/Market%20Q2%20P3.png)

![Market Q2 P4](../img/Market%20Q2%20P4.png)

**Figure 2-4:** The blue, green, yellow, and red bars represent the average income levels of different income groups. From left to right, we show the income levels under a perfectly competitive market with government using a rule-based agent or RL agent, and under a monopoly market with both agents.Under the monopoly market, households' exhibit higher income levels in the medium to long term.

![Market Q2 P5](../img/Market%20Q2%20P5.png)

​**Figure 5**​**​：​**Under the monopoly market, households' consumption levels are significantly lower than those in the perfectly competitive market.

* Under the monopoly market, although residents' income levels are higher than those in a perfectly competitive market, the rise in commodity prices and the decline in effective social demand lead to a significant reduction in household consumption levels compared to the perfectly competitive scenario.

### **Experiment 3: The Impact of Banking Behavior on Simulated Economies under a Monopoly Market**

* **Experiment Description:**

   This experiment simulates how different banking models influence the stability and longevity of a monopoly-driven economy. Specifically, it compares the effects of **non-profit banks** versus **commercial banks** on household consumption, firm profits, and overall system sustainability.
* **Experimental Variables:**
  
  * ​**Monopoly market ​**​(single dominant firm).
  * Banking model: **non-profit / rule-based** vs. ​**commercial / PPO-optimized**​.
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **monopoly\_100\_house\_bc\_gov\_seaz\_firm\_ppo\_bank\_ppo(dark red):** Households are modeled as **BC Agent with**​​**100 households**​,while the government adopts the **Saez rule-based tax formula** to determine optimal taxation.Company and bank are modeled using the**​ ​**​**PPO**​**​​ (​**​​**Proximal Policy Optimization)** reinforcement learning algorithm.
  * **monopoly\_100\_house\_bc\_gov\_seaz\_firm\_ppo\_bank\_rule(green):** Households are modeled as **BC Agent with**​​**100 households**​,while the government adopts the **Saez rule-based tax formula** to determine optimal taxation.Company is modeled using the **PPO (​**​​**Proximal Policy Optimization) ​**algorithm, while bank is modeled as a **Rule-Based Agent**​.
* **Visualized Experimental Results：**

![Market Q2 P6](../img/Market%20Q2%20P6.png)

![Market Q2 P7](../img/Market%20Q2%20P7.png)

**​Figure 6–7:​**We observed the changes in key indicators under ​**different banking settings**​, including market prices, wage rates, household consumption, firm rewards, bank rewards, the duration of the simulated economy, household rewards, social welfare, and household wealth.In the ​**non-profit banking setting**​, monopolistic firms rapidly extract ​**consumer surplus**​, leading to a sharp decline in households' ability to consume and save. As a result, financial intermediaries run out of funds and firms face financing shortages. After several iterations, ​**the duration of economic operation (measured in years) converges toward 1**​.

In contrast, under the ​**commercial banking setting**​, banks actively adjust deposit and lending rates to attract savings and build up a capital pool. Funds flow back to the corporate sector,**​ ​**boosting production and profits. Consequently, firms’ rewards increase, and ​**the overall economy sustains a longer lifespan**​. At the same time, the involvement of **commercial banks delays the collapse of the ​**​**social welfare**​​**​ system**​, allowing aggregate output and financial circulation to remain at relatively high levels.

* Under ​**monopoly**​, the extraction of consumer surplus by firms is the primary cause of economic collapse.
* **Non-profit banks** cannot buffer financial breakdowns, leading to rapid systemic failure.
* ​**Commercial banks**​, through interest-rate optimization, can “extend the life” of the system to some extent, but cannot fundamentally reverse the downward trend in consumer welfare.


