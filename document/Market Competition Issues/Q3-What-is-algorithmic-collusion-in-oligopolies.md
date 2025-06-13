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
* ​**Welfare Impacts on Households**​: Higher prices and weakened competition may lead to a decline in consumer surplus, particularly affecting price-sensitive groups such as low-income households. This, in turn, could intensify income and consumption inequality, thereby undermining overall social welfare.

### **1.4 Research Significance**

Studying algorithmic collusion in oligopoly markets has the following significance:

* **Revealing the Strategic Dynamics of AI Systems:**  The collusion issues arising from algorithmic pricing in oligopoly markets shed light on how AI systems engage in strategic interactions.
* **Promoting Ethical Algorithm Design:**  The misuse of reinforcement learning technologies in pricing can lead to losses in individual utility and social welfare. Understanding potential algorithmic collusion among oligopolies can guide the design of “regulatable and constrained” AI mechanisms, thus preventing the abuse of reinforcement learning technologies.

---

 ## **2. Selected Economic Roles**

Select the following roles from the social role classification of the economic simulation platform:

| Social Role            | Selected Type                         | Role Description                                                                                                                                                             |
| ------------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Individual             | Ramsey Model                          | Price changes affect household consumption decisions without involving firm or algorithm behavior, reflecting the heterogeneous responses of the group.                      |
| Government             | Fiscal Authority                 | The government may need to adopt dynamic policies to counter algorithmic collusion, such as introducing antitrust measures or algorithm auditing rules.                      |
| Firm                 | Oligopoly                       | Strategic interactions exist between firms, and algorithms may achieve price coordination without explicit communication, simulating a non-collaborative game environment.   |
| Bank | No-Arbitrage Platform | Financial markets follow the no-arbitrage principle, with investment and risk assessment relying on market price fluctuations, indirectly influenced by algorithmic pricing. |

### **Individual → Ramsey Model**

* As representative agents, households optimize**​ intertemporal utility** under dynamically changing prices set by oligopolistic firms. The Ramsey model assumes a forward-looking, infinitely-lived household, suitable for analyzing aggregate consumption responses to price dynamics without incorporating agent-level heterogeneity.

### **Government → Fiscal Authority**

* The Tax Policy Department focuses on market competition and consumer welfare. When price manipulation or collusion is detected in the market, the government intervenes through ​**antitrust laws**​, price controls, or regulatory measures.

### **Firm → ​Oligopoly Market**

* Firms engage in **tacit coordination** through algorithms, leading to price consistency behavior. The oligopoly market structure provides fertile ground for ​**algorithmic collusion**​, which serves as the core behavior in this study.

### **Bank → No-Arbitrage Platform**

* In markets where **firm profits** and **stock prices** are closely linked, financial institutions adjust their investment portfolios, influencing capital allocation. The study aims to evaluate whether **oligopoly**​**​ collusion** distorts investment signals and impacts market efficiency.

---

## **3.Selected Agent Algorithms**

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type          | Role Description                                                                                                                                                                            |
| ------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Behavior Cloning Agent | Can learn consumer behavior patterns from historical data, including price sensitivity, consumption tendencies, and reactions to pricing by oligopolistic firms.                            |
| Government             | Rule-Based Agent       | The government performs specific regulatory functions in the market, such as antitrust policies, tax adjustments, and price interventions, which can be implemented using predefined rules. |
| Firm                 | RL Agent               | Oligopolistic firms have pricing authority and typically follow profit maximization principles, making RL Agent suitable for multi-agent game simulations.                                  |
| Bank | Rule-Based Agent       | Financial markets follow the no-arbitrage principle, and investment and pricing rely on market rule adjustments. Rule-Based Agent ensures stability and liquidity in the market simulation. |

### **Individual  → BC Agent**

* Household consumption behavior can be learned from historical data. A BC Agent can simulate consumption decisions at different income levels, accurately reflecting response patterns to price changes.

### **Government→ Rule-Based Agent**

* The government should maintain a predetermined strategy, which allows for better observation of the social impacts brought by oligopoly markets.

### **Firm → RL Agent**

* Firms introduce **RL-based pricing algorithms** to maximize long-term profits. With competitors also using RL, the agent learns strategies through interaction, potentially evolving collusive behavior.

### **Bank → Rule-Based Agent**

* Financial institutions make non-arbitrage decisions based on market rules. A **Rule-Based Agent** ensures system stability and is used to monitor the impact of firm behavior on capital markets.

---

## **4.Illustrative Experiment**

```python
# Simulating the impact of algorithmic collusion on price, output, and profit
# Key variables: P (Price), Q (Quantity), Profit

# Scenario 1: Perfectly Competitive Market (Baseline)
At each time step t:
    For each firm i:
        1. Observe the market's average production cost
        2. Set the product price P_i,t close to marginal cost
        3. Decide output Q_i,t based on the market price
        4. Compute profit: Profit_i,t = P_i,t × Q_i,t - cost

# Scenario 2: Algorithmic Collusion Market (Simulating tacit collusion)
At each time step t:
    For each firm i:
        1. Observe past prices of other firms
        2. Adjust price P_i,t to align with competitors (avoid undercutting)
        3. Keep output Q_i,t relatively stable
        4. Profit increases due to coordinated higher pricing
```

### **Experiment 1: Pricing Behavior of Firms in an ​**​**Oligopoly**​**​ Market**
* **Experiment Description:**
  In an oligopoly market consisting of multiple firms, introduce RL-based pricing algorithms and observe the pricing strategies that emerge under reinforcement learning as well as their impact on consumers.
* **Involved Social Roles:**
  * ​*Market*​**:** Oligopoly Firms
  * ​*Households*​**:** Immortal Heterogeneous Model
* **AI**​**​ Agents:**
  * *Market: ​*RL Agent
  * *Households: ​*BC Agent
* **Experimental Variables:**
  * Market Type (Oligopoly Firms&Perfect competition)
  * Market Price (P)
* **​ Visualized Experimental Results：**

![](https://yqq94nlzjex.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2UyYjRiZGVlNmMwMmJkYzgyYjkxY2NlYTUyNjUwMzlfS3VEZ0lRY2Y2Sm0xUnR2angxaTB1M3gzVGtaZzJjN0xfVG9rZW46VFBEeGJrR0Vlb2VnMkF4azNMamNuVWg2bmtiXzE3NDk4MTc5MTM6MTc0OTgyMTUxM19WNA)

​**Figure 1**​: Comparison of household income under perfect competition and oligopoly collusion algorithms. Under oligopolistic collusion, the average income of households increases significantly (right chart), particularly evident among the lower-income groups (yellow bar). The income level of the impoverished population under oligopoly is about three times that in the perfect competition market.

![](https://yqq94nlzjex.feishu.cn/space/api/box/stream/download/asynccode/?code=ODM2MTAzNjEyNzQzYmM2ZTU1MmI0OTFmNTJhMmIzMmVfU2pta3JmQndUSzlVUjZDV0NpRHpXTUZHTU85a0xlVHpfVG9rZW46V0JNM2I1S2FHb3JWcnp4aktUYWNpaWg3bm1oXzE3NDk4MTc5MTM6MTc0OTgyMTUxM19WNA)

​**Figure 2**​: Comparison of household consumption under the two market structures. The oligopolistic market (right chart) exhibits a clear consumption-suppressing effect, particularly on wealthy and middle-income households.

* In the oligopolistic market, firms adopt reinforcement learning strategies to "collude" on pricing in pursuit of maximizing profits and market share. As a result, in the simulated economy, this pricing behavior leads to a noticeable suppression of consumption among wealthy and middle-class households, while simultaneously increasing the average income of low-income households.


