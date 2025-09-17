# Q4: How does product diversity affect welfare?

## 1. Introduction

### 1.1 Concerns over Product Variety in Monopolistic Competition Markets

* In monopolistic competition, firms offer differentiated products to niche segments, expanding consumer choice but potentially causing redundant resource allocation and declining scale efficiencies. Theoretically, **excessive product variety can undermine aggregate welfare, making the trade-off between “freedom of choice” and “efficiency optimality” a critical research question.**
* New Keynesian and New Trade theories suggest that **moderate variety boosts consumer surplus and market dynamism**. However, if firms over-segment products, duplicate production capacity, or shift focus from innovation to marketing, social resources may be used inefficiently. Hence, we need to quantify the net welfare effect of product variety across economic environments.

### 1.3 Research Questions

Using an economic-simulation platform, this study examines the mechanisms by which changes in product variety affect social welfare, focusing on:

* Whether RL-driven firms naturally evolve toward “excessive variety.”
* How pursuit of variety impacts resource-allocation efficiency at different levels of market concentration.

### 1.4 Research Significance

* **Microfoundation for Competition Policy and Product Regulation:** Quantify the marginal welfare effects of variety to inform “anti-over-differentiation” regulatory frameworks.
* **Enhancing Realism in Market Simulations:** Incorporate AI agents into firm behavior models to explore how product-strategy evolution balances personalization with profit maximization.
* **Highlighting the Tension Between Choice and Efficiency:** Shed light on achieving a dynamic balance between individualized consumption and system-wide efficiency.

---

## **2. Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                            | Role Description                                                                                                                          |
| ------------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Ramsey Model                             | Rationally choose preferred goods, reflecting the marginal utility of variety and price sensitivity.                                      |
| Government             | Fiscal Authority                     | Oversee market competition and product‐classification policies, assessing when policy intervention is needed to limit excessive variety. |
| Firm                 | Monopolistic Competition Market          | Firms employ product‐differentiation strategies to compete for market share, serving as the direct source of changes in variety.         |
| Bank  | No-Arbitrage Platform | Provide capital support and feedback on long‐term investment efficiency and macroeconomic return paths.                                  |

---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
Households maximize utility when choosing products, showing sensitivity to prices, variety, and brand preferences; they are the core agents for measuring welfare changes.

**Government → Fiscal Authority**  
As the regulatory authority, the government may set standards or implement incentives/restrictions on firms’ diversification behaviors; its policy‐response mechanisms should be incorporated into the model.

**Firm → Monopolistic Competition Market**  
Firms hold pricing power and engage in non‐price competition through differentiation strategies, forming the key structural foundation of the simulation.

**Bank → No-Arbitrage Platform**  
Provide business loans and household‐savings platforms, reflecting the efficiency of resource allocation over long‐run paths and aiding analysis of macro‐efficiency changes.

---

## **3. Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | Reproduce consumers’ reactions to prices and product variety based on real preference data.                                             |
| Government             | Rule-Based Agent       | Monitor product-differentiation trends under fixed policy rules, identifying intervention thresholds and boundaries.                     |
| Firm                 | RL Agent               | Firms dynamically adjust product strategies via reinforcement learning to optimize profits and market share.                             |
| Bank | Rule-Based Agent       | Reflect resource-allocation efficiency in the financial system and simulate how interest-rate changes affect firms’ expansion capacity. |


## 4. Illustrative Experiments

### Experiment 1: Impact of Product Variety on Consumer Welfare

* **Experiment Description:**
  Construct a monopolistic competition market in which firms use reinforcement learning to adjust product prices and variety in order to maximize profits. Compare different combinations of price levels and product diversity, and measure their effects on consumer utility.
* **Involved Social Roles:**
  * *Households: ​*Ramsey Model
  * *Market: ​*Monopolistic Competition Market
* **AI**​**​ Agents:**
  * *Households: ​*Behavior Cloning Agent
  * *Government:* Rule-Based Agent
  * *Market:* RL Agent
  * *Financial Institutions:* Rule-Based Agent
* **Experimental Variables:**
  * A consumer utility function consistent with economic theory, where **epsilon** represents the elasticity of substitution between goods.
  * Number and identity of firms in the monopolistic competition market
  * Product prices and range of product varieties
  * Household utility in the simulated economy (stratified by age and income; used as the reward function)

```python
# Consumer utility is modeled using the CES utility function, where epsilon represents the elasticity of substitution between goods

function calculate_CES_utility(consumption_list, epsilon):
    # Step 1: Initialize the sum_term to 0
    sum_term = 0
    
    # Step 2: For each good in the consumption list, calculate the weighted consumption (c_j^(epsilon - 1))
    for each consumption in consumption_list:
        sum_term = sum_term + (consumption ^ (epsilon - 1))  # c_j^(epsilon - 1)

    # Step 3: Calculate the final utility using the CES utility formula
    # U = (sum_term) ^ (epsilon / (epsilon - 1))
    utility = sum_term ^ (epsilon / (epsilon - 1))

    return utility
```

![Market Q4 P1](../img/Market%20Q4%20P1.png)

![Market Q4 P2](../img/Market%20Q4%20P2.png)

​**Figure 1 and Figure 2**​: Comparison of consumer wealth levels across different numbers of firms in the market (with different colors representing households from various wealth tiers). As the number of firms increases from two (rightmost bar) to eight (middle bar), there is a noticeable increase in consumer wealth. However, as the number of firms continues to rise, such as when there are ten firms (second bar from the left), the average household wealth actually declines.

![Market Q4 P3](../img/Market%20Q4%20P3.png)

**​​ Figure 3：​**Comparison of social welfare levels, with the following ranking: firm number (N = 8) > (N = 10) >( N = 6) > (N = 4) > (N = 2), indicating that social welfare is highest when there are eight firms and lowest when there are two firms.

![Market Q4 P4](../img/Market%20Q4%20P4.png)

​**Figure 4**​: Long-term GDP growth trends under different numbers of firms in the market. When the number of firms is 10 (green line) or 8 (yellow line), the long-term GDP growth rate is significantly higher.

* An increase in the number of firms represents a higher product variety. Overall, a greater variety of products tends to promote long-term growth in social output.
* As the number of firms increases, household wealth follows a pattern of initial growth followed by a decline. Therefore, having more firms does not necessarily result in higher average consumer wealth.

### **Experiment 2: The Impact of Product Substitutability on Consumer Behavior**

* **Experiment Description:**  Based on the CES function characterizing consumer utility, different product substitution elasticities (epsilon) are introduced to examine their impact on consumer behavior patterns. The experiment fixes the number of firms in the market for a specific epsilon value and discusses consumer behavior characteristics under different product substitution elasticities.
* **Involved Social Roles:**
  * Households: Ramsey Model
  * Market: Monopolistic Competition Market
* **AI**​**​ Agents:**
  * Households: Behavior Cloning Agent
  * Government: Rule-Based Agent
  * Market: RL Agent
  * Financial Institutions: Rule-Based Agent
* **Experimental Variables:**
  * Different product substitution elasticities (epsilon)
  * Product prices and range of product varieties
  * Household reward and consumption in the simulated economy

```python
# Market scenarios and the impact of epsilon (ε)
scenarios = [
    {"scenario": "Moderate Differentiation", "epsilon": 4, "explanation": "Standard value (Dixit-Stiglitz)"},
    {"scenario": "High Substitutability", "epsilon": 10, "explanation": "Low diversity utility"},
    {"scenario": "Low Substitutability", "epsilon": 2, "explanation": "High value on diversity, like luxury market"}
]

# Epsilon's effect on market competition
effects = [
    {"epsilon_change": "ε ↑", "effect": "Increased Substitutability", "impact": "More competition, lower value of diversity"},
    {"epsilon_change": "ε ↓", "effect": "Decreased Substitutability", "impact": "Stronger brand power, higher value of diversity"}
]
```

![Market Q4 P5](../img/Market%20Q4%20P5.png)

​**Figure 5**​: The impact of product substitution elasticity on the consumption of households from different wealth tiers, with Firm = 8 fixed. When ϵ=2, the substitution elasticity is at its lowest, indicating that consumers are less willing to change their purchasing behavior due to price factors, leading to the highest consumption levels (right chart). When ϵ=8, consumers show the lowest consumption levels (left chart).

![Market Q4 P6](../img/Market%20Q4%20P6.png)

​**Figure 6**​: Comparison of household utility across different wealth tiers under different consumption elasticities. The effect of consumption elasticity on household utility is not significant. When ϵ=2 (right chart), the average utility of households across different income levels is slightly higher.

![Market Q4 P7](../img/Market%20Q4%20P7.png)

​**Figure 7**​: Changes in social welfare levels under different consumption elasticities. When ϵ=2(yellow line), social welfare is noticeably higher and continues to improve over time.

* As the product substitution elasticity decreases, consumers become less sensitive to price changes, resulting in more active consumption behavior in the simulated economy, which leads to a slight increase in individual utility.
* Lower product substitution elasticity can effectively enhance overall social welfare.

