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

This study uses an economic simulation platform to examine how monopolistic market structures affect resource allocation and household welfare, focusing on:

* ​**Macroeconomic outcomes and inequality**​: How do monopoly markets influence overall output and wealth distribution?
* ​**Household income**​: Do household earnings increase or decrease under monopoly conditions?

### **1.4 Research Significance**

* **Policy Guidance:**  Many countries have implemented antitrust laws to regulate monopolies. A clear understanding of monopolies' impacts at both macro and micro levels is essential for designing more effective and targeted regulatory policies.
* **Equity and Distributional Analysis:**  The study also investigates how different social groups are affected by monopolistic pricing, and whether such pricing contributes to widening income inequality.

---

## **2. Selected Economic Roles**

Select the following roles from the social role classification of the economic simulation platform:

| Social Role            | Selected Type                         | Role Description                                                                                                                |
| ------------------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Ramsey Model                          | Household consumption decisions respond to price change.                                                                        |
| Government             | Fiscal Authority                  | The government may implement specific fiscal policies, such as price regulation.                                                |
| Firm                 | Monopoly                        | Monopolistic pricing follows fixed market rules, e.g., profit maximization, cost-plus pricing, or government-regulated pricing. |
| Bank  | No-Arbitrage Platform | Financial markets follow the no-arbitrage principle; investment and pricing adjust based on market rules.                       |

### **Individual →Ramsey Model**

* **​ Ramsey Model** is used to capture how price changes in a monopoly market influence household consumption decisions, focusing on differences across individuals rather than across age groups.
* Although the Overlapping Generations (OLG) model accounts for intergenerational decision-making—e.g., younger individuals tend to save more for the future—the focus here is not on age-specific behavior, but rather on heterogeneity unrelated to the life stage.

### **Government → Fiscal Authority**

* The Treasury Department may implement price control policies to ensure market fairness and balance social welfare under monopolistic conditions.

### **Firm → Monopoly**

* The model studies how monopolistic pricing behavior influences market competition and broader socioeconomic outcomes.

### **Bank → No-Arbitrage Platform**

* These sets ensure the efficiency of capital markets and help analyze the indirect effects of monopoly structures on the financial system.

---

## **3. Selected Agent Algorithms**

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type             | Role Description                                                                                                                                                    |
| ------------------------ | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Behavior Cloning Agent    | Learns consumer behavior from historical data, including price sensitivity, consumption patterns, and responses to monopoly pricing.                                |
| Government             | Rule-Based Agent/RL Agent | The government should be able to execute regulatory functions through predefined rules, or interact with the market to implement more targeted regulatory policies. |
| Firm                 | RL Agent                  | Possesses pricing power and typically sets prices based on a profit-maximization strategy.                                                                          |
| Bank| Rule-Based Agent          | Operates under the no-arbitrage principle; uses rule-based mechanisms to maintain market stability and analyze capital flows under monopoly pricing.                |

### **Individual → BC Agent**

* Since households are modeled using an **immortal heterogeneous agent** framework, there is no need to account for intergenerational behavioral links. A BC agent can learn consumer behavior patterns—such as ​**price sensitivity**​, consumption preferences, and responses to monopoly pricing—from historical data.
* During the simulation, the BC agent can directly **imitate observed consumption behavior** without complex policy optimization, thus improving ​**computational efficiency**​.

### **Government → Rule-Based Agent / RL Agent**

* Basic government pricing regulations—such as **price ceilings** and ​**tax policies**​—are typically based on fixed rules, making them suitable for modeling with a rule-based agent.
* However, if the government needs to **interact more effectively with the market** and respond more precisely to monopoly behavior—such as implementing **antitrust** or ​**wage policies**​—a reinforcement learning (RL) agent may be required to learn adaptive strategies through interaction.

### **Firm → RL Agent**

* Monopoly firms have **pricing power** and typically pursue ​**profit maximization**​, which aligns well with the use of an RL agent.
* An RL agent allows for analysis of **optimal pricing strategies** and their **societal impacts** under dynamic market conditions.

### **Bank → Rule-Based Agent**

* As financial institutions follow the ​**no-arbitrage principle**​, their **pricing and investment strategies** are usually rule-based rather than learned through complex models.
* A rule-based agent can ensure **market stability** and simulate capital flows in response to different pricing scenarios.
* This setup enables the analysis of how **monopolistic pricing strategies** affect the capital market—for example, through changes in **financing costs** or ​**investment returns**​.

---

## **4. Illustrative Experiment**

```python
# Scenario setup related to Monopoly Market
If firm_type == "monopoly":
    At each time step:
        1. Observe current state s_t
        2. Choose price a_t based on policy π(s_t)
        3. Receive reward r_t and update policy π via RL algorithm
```

### **Experiment 1: The Impact of Monopoly Market on Income Inequality**

* **Experiment Description:**
  Investigate the impact of monopolies on income inequality.
* **Involved Social Roles:**
  * ​*Firm*​: Monopoly
  * ​*Individual*​: OLG Model
  * ​*Government*​: Fiscal Authority
* **AI**​**​ Agents:**
  * ​*Firm*​: RL Agent
  * ​*Individual*​: BC Agent
  * ​*Government*​：Rule-Based Agent
* **Experimental Variables:**
  * Monopoly Market vs. Perfectly Competitive Market
  * Change in income inequality (measured by the Gini coefficient)
* **​ Visualized Experimental Results：**

![Market Q2 P1](../img/Market%20Q2%20P1.png)

**Figure 1:** In the first 50 years, under the monopoly market (yellow line), income inequality is similar to that in a perfectly competitive market (blue line). However, over time, income inequality under the monopoly market becomes significantly higher compared to the perfectly competitive market.

* The monopoly market increases the wealth gap between households in the medium to long term.

---

### **Experiment 2: The Impact of Monopoly Market on Household Consumption and Income**

* **Experiment Description:**  Simulate changes in household consumption and income under a monopoly market to observe the effects of monopolies on micro-level individuals.
* **Involved Social Roles:**
  * ​*Firm*​: Monopoly 
  * ​*Individual*​: Ramsey Model
  * ​*Government*​: Fiscal Authority
* **AI**​**​ Agents:**
  * ​*Firm*​: RL Agent
  * ​*Individual*​: BC Agent
  * ​*Government*​: Rule-Based Agent / RL Agent
* **Experimental Variables:**
  * Monopoly Market vs. Perfectly Competitive Market
  * Household income and consumption levels (divided by income tiers)
* **Visualized Experimental Results：**

![Market Q2 P2](../img/Market%20Q2%20P2.png)

![Market Q2 P3](../img/Market%20Q2%20P3.png)

![Market Q2 P4](../img/Market%20Q2%20P4.png)

**Figure 2-4:** The blue, green, yellow, and red bars represent the average income levels of different income groups (Top 10%, Mid 40%, Bottom 50%,Average in 100%). From left to right, we show the income levels under a perfectly competitive market with government using a rule-based agent and RL agent, and under a monopoly market with both agents.Under the monopoly market, households' exhibit higher income levels in the medium to long term.

![Market Q2 P5](../img/Market%20Q2%20P5.png)

​**Figure 5**​**​：​**Under the monopoly market, households' consumption levels are significantly lower than those in the perfectly competitive market.

* Under the monopoly market, although residents' income levels are higher than those in a perfectly competitive market, the rise in commodity prices and the decline in effective social demand lead to a significant reduction in household consumption levels compared to the perfectly competitive scenario.

