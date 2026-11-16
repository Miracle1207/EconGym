# Q3: What is algorithmic collusion in oligopolies?

## **1. Introduction**

### **1.1 Introduction**

An oligopoly market is a market structure dominated by a few large firms, each holding significant market share. These firms are highly interdependent, with competition and cooperation occurring simultaneously. Due to the limited number of market participants, decisions made by one firm can significantly affect others, leading to frequent strategic interactions regarding prices, output, and technology.

### **1.2 Algorithmic Collusion in ​**​**Oligopoly**​**​ Markets**

In an oligopoly market, firms make strategic pricing decisions, anticipating and responding to competitors’ actions to avoid price wars and maximize profits. With the development of AI technologies, an increasing number of firms are adopting pricing systems based on reinforcement learning or machine learning. These systems may evolve into **algorithmic collusion** through trial-and-error optimization, leading to cartel-like pricing strategies that reduce market competition and consumer welfare. **Real-world examples include:**

* Algorithmic price synchronization among airline companies in ticket pricing.
* E-commerce platforms maintaining price stability using dynamic pricing algorithms.
* Ongoing antitrust investigations into algorithmic collusion by the **U.S. FTC** and the ​**European Union**​.

### **1.3 Research Questions**

This study uses an economic simulation platform to explore the issue of **algorithmic collusion** in oligopoly markets and its potential social and individual impacts, specifically focusing on:

* ​**Macroeconomic Impacts**​: Algorithmic collusion may cause market prices to deviate persistently from competitive equilibrium, reducing allocative efficiency. This distortion can suppress aggregate output and dampen investment incentives, while also exacerbating entry barriers and hindering structural adjustments within industries.
* ​**Household Welfare**​: Higher prices and weakened competition may lead to a decline in consumer surplus, particularly affecting price-sensitive groups such as low-income households. This, in turn, could intensify income and consumption inequality, thereby undermining overall social welfare.


### **1.4 Research Significance**

Studying algorithmic collusion in oligopoly markets has the following significance:

* **Revealing the Strategic Dynamics of AI Systems:**  The collusion issues arising from algorithmic pricing in oligopoly markets shed light on how AI systems engage in strategic interactions.
* **Promoting Ethical Algorithm Design:**  The misuse of reinforcement learning technologies in pricing can lead to losses in individual utility and social welfare. Understanding potential algorithmic collusion among oligopolies can guide the design of “regulatable and constrained” AI mechanisms, thus preventing the abuse of reinforcement learning technologies.

---

 ## **2. Selected Economic Roles**

Select the following roles from the social role classification of the economic simulation platform:

| Social Role | Selected Type        | Role Description                                                                                                                                            | Observation                                                                                                                                                                                   | Action                                                                                                  | Reward                                           |
| ----------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Individual**  | Ramsey Model         | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.                                                  | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Government**  | Fiscal Authority     | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.                                                         |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
| **Firm**       | Oligopoly             | Oligopoly Firms engage in strategic competition, anticipating household responses and rival actions.                | $o_t^{\text{olig}} = ( K_t^j,  Z_t^j, r_{t-1}^l)$<br>Production capital, productivity, lending rate | $a_t^{\text{olig}} = ( p_t^j, W_t^j )$<br>Price and wage decisions for firm $j$ | $r_t^{\text{olig}} = p_t^j y_t^j - W_t^j L_t^j - R_t K_t^j$<br>Profits = Revenue – costs for firm $j$ | 
| **Bank**       | Non-Profit Platform   | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives.                                        | /                                                                                                                                                                                             | No rate control                                                                                         | No profit                                         |


---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
As representative agents, households optimize**​ intertemporal utility** under dynamically changing prices set by oligopolistic firms. The Ramsey model assumes a forward-looking, infinitely-lived household, suitable for analyzing aggregate consumption responses to price dynamics without incorporating agent-level heterogeneity.

**Government → Fiscal Authority**  
The Tax Policy Department focuses on market competition and consumer welfare. When price manipulation or collusion is detected in the market, the government intervenes through ​**antitrust laws**​, price controls, or regulatory measures.

**Firm → ​Oligopoly Market**  
Firms engage in **tacit coordination** through algorithms, leading to price consistency behavior. The oligopoly market structure provides fertile ground for ​**algorithmic collusion**​, which serves as the core behavior in this study.

**Bank →Non-Profit Platform**

In markets where **firm profits** and **stock prices** are closely linked, financial institutions adjust their investment portfolios, influencing capital allocation. The study aims to evaluate whether **oligopoly**​**​ collusion** distorts investment signals and impacts market efficiency.

---

## **3.Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | Can learn consumer behavior patterns from historical data, including price sensitivity, consumption tendencies, and reactions to pricing by oligopolistic firms.                            |
| Government             | Rule-Based Agent       | The government performs specific regulatory functions in the market, such as antitrust policies, tax adjustments, and price interventions, which can be implemented using predefined rules. |
| Firm                 | RL Agent               | Oligopolistic firms have pricing authority and typically follow profit maximization principles, making RL Agent suitable for multi-agent game simulations.                                  |
| Bank | Rule-Based Agent       | Financial markets follow the no-arbitrage principle, and investment and pricing rely on market rule adjustments. Rule-Based Agent ensures stability and liquidity in the market simulation. |

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "oligopoly"
```

This command loads the configuration file `cfg/oligopoly.yaml`, which defines the setup for the "oligopoly" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `oligopoly.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "oligopoly"
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


    - entity_name: 'market'
      entity_args:
        params:
          type: "oligopoly"   #  type_list: [ 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly' ]
          alpha: 0.36
          Z: 1.0
          sigma_z: 0.0038
          epsilon: 0.5

    - entity_name: 'bank'
      entity_args:
        params:
          type: 'non_profit'   # [ 'non_profit', 'commercial' ]


Trainer:
  house_alg: "bc" #The Rule-Based Agent can also be chosen in this experiment.
  gov_alg: "saez"
  firm_alg: "ppo"
  bank_alg: "rule_based"
  seed: 1
  cuda: False
#  n_epochs: 1000
  wandb: True
```
---

## **​5.​**​**Illustrative Experiment**

### **Experiment 1: Pricing Behavior of Firms in an ​**​**Oligopoly**​**​ Market**

* **Experiment Description:**
  In an oligopoly market consisting of multiple firms, introduce RL-based pricing algorithms and observe the pricing strategies that emerge under reinforcement learning as well as their impact on consumers.
* **Experimental Variables:**
  * Market Type (Oligopoly Firms&Perfect competition)
  * Market Price (P)
* **Baselines:** Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  * **Groups from left to right:**
    * **perfect\_rule\_based\_rule\_based\_100\_ramsey :**​**Perfectly competitive market ​**as the benchmark.Both households and the government are modeled as ​**Rule-Based Agents**​, with **100 households** and **Ramsey Model households.**
    * **algorithmic\_collusion\_rule\_based\_rule\_based\_100\_ramsey:**​​**Algorithmic collusion market**​.Both households and the government are modeled as ​**Rule-Based Agents**​, with **100 households** and **Ramsey Model households.**
  * **Bar description:**
    * **Blue bar:** Rich households
    * **Green bar:** Middle-class households
    * **Yellow bar:** Poor households
    * **Red bar:** Overall average

![Market Q3 P1](../img/Market%20Q3%20P1.png)

​**Figure 1**​: Comparison of household income under perfect competition and oligopoly collusion algorithms. Under oligopolistic collusion, the average income of households increases significantly (right chart), particularly evident among the lower-income groups. The income level of the impoverished population under oligopoly is about three times that in the perfect competition market.

![Market Q3 P2](../img/Market%20Q3%20P2.png)

​**Figure 2**​: Comparison of household consumption under the two market structures. The oligopolistic market (right chart) exhibits a clear consumption-suppressing effect, particularly on wealthy and middle-income households.

* In the oligopolistic market, firms adopt reinforcement learning strategies to "collude" on pricing in pursuit of maximizing profits and market share. As a result, in the simulated economy, this pricing behavior leads to a noticeable suppression of consumption among wealthy and middle-class households, while simultaneously increasing the average income of low-income households.


