# Q4: How does product diversity affect welfare?

## 1. Introduction

### 1.1 Concerns over Product Variety in Monopolistic Competition Markets

* In monopolistic competition, firms offer differentiated products to niche segments, expanding consumer choice but potentially causing redundant resource allocation and declining scale efficiencies. Theoretically, **excessive product variety can undermine aggregate welfare, making the trade-off between “freedom of choice” and “efficiency optimality” a critical research question.**
* New Keynesian and New Trade theories suggest that **moderate variety boosts consumer surplus and market dynamism**. However, if firms over-segment products, duplicate production capacity, or shift focus from innovation to marketing, social resources may be used inefficiently. Hence, we need to quantify the net welfare effect of product variety across economic environments.

### 1.3 Research Questions

This study uses an economic simulation platform to investigate the ​**economic impacts of product variety in the market**​, specifically examining:

* ​**Household Consumption**​: How does greater product diversity influence household spending patterns and brand-switching behavior?
* ​**Household Utility**​: How does increased choice affect individual and aggregate consumer satisfaction?
* **GDP**​​**​ Effects**​: What is the impact of product variety on aggregate output, considering trade-offs between market expansion and production inefficiency?

### 1.4 Research Significance

* **Microfoundation for Competition Policy and Product Regulation:** Quantify the marginal welfare effects of variety to inform “anti-over-differentiation” regulatory frameworks.
* **Enhancing Realism in Market Simulations:** Incorporate AI agents into firm behavior models to explore how product-strategy evolution balances personalization with profit maximization.
* **Highlighting the Tension Between Choice and Efficiency:** Shed light on achieving a dynamic balance between individualized consumption and system-wide efficiency.

---

## **2. Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type            | Role Description                                                                                                                                                           | Observation                                                                                                                                                                                                 | Action                                                                                                   | Reward                                                                                     |
| ----------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Individual**  | Ramsey Model             | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.                                                                  | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Government**  | Fiscal Authority         | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.                                                                        |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
| **Firm**       | Monopolistic Competition | Monopolistic Competitors offer differentiated products with CES demand and endogenous entry, supporting studies of consumer preference and market variety.                 | $o_t^{\text{mono-comp}} = ( K_t^j,  Z_t^j, r_{t-1}^l )$<br> Production capital, productivity, lending rate. Here, $j$ denotes the firm index. | $a_t^{\text{mono-comp}} = ( p_t^j, W_t^j )$<br>Price and wage decisions for firm $j$ | $r_t^{\text{mono-comp}} = p_t^j y_t^j - W_t^j L_t^j - R_t K_t^j$<br>Profits = Revenue – costs for firm $j$ |
| **Bank**       | Non-Profit Platform      | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives.                                                        | /                                                                                                                                                                                                           | No rate control                                                                                          | No profit                                                                                  |

---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
Households maximize utility when choosing products, showing sensitivity to prices, variety, and brand preferences; they are the core agents for measuring welfare changes.

**Government → Fiscal Authority**  
As the regulatory authority, the government may set standards or implement incentives/restrictions on firms’ diversification behaviors; its policy‐response mechanisms should be incorporated into the model.

**Firm → Monopolistic Competition Market**  
Firms hold pricing power and engage in non‐price competition through differentiation strategies, forming the key structural foundation of the simulation.

**Bank → Non-Profit Platform**  
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

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "monopolistic_competition"
```

This command loads the configuration file `cfg/monopolistic_competition.yaml`, which defines the setup for the "monopolistic_competition" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `monopolistic_competition.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "monopolistic_competition"
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
          type: "monopolistic_competition"   #  type_list: [ 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly' ]

        monopolistic_competition:
          type: "monopolistic_competition"
          action_dim: 2
          firm_n: 10

    - entity_name: 'bank'
      entity_args:
        params:
          type: 'non_profit'   # [ 'non_profit', 'commercial' ]

Trainer:
  house_alg: "bc"
  gov_alg: "us_federal"
  firm_alg: "ppo"
  bank_alg: "rule_based"
  seed: 1
  cuda: False
#  n_epochs: 300
  wandb: True
```
---

## 5.Illustrative Experiments

### Experiment 1: Impact of Product Variety on Consumer Welfare

* **Experiment Description:**
  
  Construct a monopolistic competition market in which firms use reinforcement learning to adjust product prices and variety in order to maximize profits. Compare different combinations of price levels and product diversity, and measure their effects on consumer utility.
* **Experimental Variables:**
  
  * A consumer utility function consistent with economic theory, **where epsilon represents the elasticity of substitution between goods**.
  * Number and identity of firms in the monopolistic competition market
  * Product prices and range of product varieties
  * Household utility in the simulated economy (stratified by age and income; used as the reward function)

* **Baselines:**

  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  * **Groups description:**
    * ​**mp\_f2\_e4 **​: A monopolistic competition market with ​**2 firms**​, households modeled as ​**Ramsey Model and Behavior Cloning Agent**​, government modeled as a ​**rule-based agent**​, elasticity of substitution parameter ​**ε=4**​.
    * ​**mp\_f4\_e4 **​: A monopolistic competition market with ​**4 firm**​,the other settings are the same.
    * ​**mp\_f6\_e4 **​: A monopolistic competition market with ​**6 firms**​, the other settings are the same.
    * ​**mp\_f8\_e4 **​: A monopolistic competition market with ​**8 firms**​, the other settings are the same.
    * ​**mp\_f10\_e4 **​: A monopolistic competition market with ​**10 firms**​, the other settings are the same.
  * **Bar description:**
    * **Blue bar:** Rich households
    * **Green bar:** Middle-class households
    * **Yellow bar:** Poor households
    * **Red bar:** Overall average


![Market Q4 P1](../img/Market%20Q4%20P1.png)

![Market Q4 P2](../img/Market%20Q4%20P2.png)

​**Figure 1 and Figure 2**​: Comparison of consumer wealth levels across different numbers of firms in the market. As the number of firms increases from two to eight, there is a noticeable increase in consumer wealth. However, as the number of firms continues to rise, such as when there are ten firms, the average household wealth actually declines.

* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.

    * ​**mp\_f2\_e4 (Light blue line)**​: A monopolistic competition market with ​**2 firms**​, households modeled as ​**Ramsey Model and Behavior Cloning Agent**​, government modeled as a ​**rule-based agent**​, elasticity of substitution parameter ​**ε=4**​.
    * ​**mp\_f4\_e4 (Red line)**​: A monopolistic competition market with ​**4 firm**​,the other settings are the same.
    * ​**mp\_f6\_e4 (Dark blue line)**​: A monopolistic competition market with ​**6 firms**​, the other settings are the same.
    * ​**mp\_f8\_e4 (Yellow line)**​: A monopolistic competition market with ​**8 firms**​, the other settings are the same.
    * ​**mp\_f10\_e4 (Green line)**​: A monopolistic competition market with ​**10 firms**​, the other settings are the same.


![Market Q4 P3](../img/Market%20Q4%20P3.png)

**​​ Figure 3：​**Comparison of social welfare levels, with the following ranking: firm number (N = 8) > (N = 10) >( N = 6) > (N = 4) > (N = 2), indicating that social welfare is highest when there are eight firms and lowest when there are two firms.

![Market Q4 P4](../img/Market%20Q4%20P4.png)

​**Figure 4**​: Long-term GDP growth trends under different numbers of firms in the market. When the number of firms is 10 (green line) or 8 (yellow line), the long-term GDP growth rate is significantly higher.

* An increase in the number of firms represents a higher product variety. Overall, a greater variety of products tends to promote long-term growth in social output.
* As the number of firms increases, household wealth follows a pattern of initial growth followed by a decline. Therefore, having more firms does not necessarily result in higher average consumer wealth.

### **Experiment 2: The Impact of Product Substitutability on Consumer Behavior**

* **Experiment Description:**

  Based on the CES function characterizing consumer utility, different product substitution elasticities (epsilon) are introduced to examine their impact on consumer behavior patterns. The experiment fixes the number of firms in the market for a specific epsilon value and discusses consumer behavior characteristics under different product substitution elasticities.
* **Experimental Variables:**
  
  * Different product substitution elasticities (epsilon)
  * Product prices and range of product varieties
  * Household reward and consumption in the simulated economy
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * ​**mp\_f8\_e10 (left group)**​: A monopolistic competition market with ​**8 firms**​, households modeled as ​**Ramsey Model and Behavior Cloning Agent**​, government modeled as a ​**rule-based agent**​, elasticity of substitution parameter ​**ε=10**​.
  * ​**mp\_f8\_e4 (middle group)**​: A monopolistic competition market with ​**8 firms**​, households modeled as ​**Ramsey Model and Behavior Cloning Agent**​, government modeled as a ​**rule-based agent**​, elasticity of substitution parameter ​**ε=4**​.
  * ​**mp\_f8\_e2 (right group)**​: A monopolistic competition market with ​**8 firms**​, households modeled as ​**Ramsey Model and Behavior Cloning Agent**​, government modeled as a ​**rule-based agent**​, elasticity of substitution parameter ​**ε=2**​.


![Market Q4 P5](../img/Market%20Q4%20P5.png)

​**Figure 5**​: The impact of product substitution elasticity on the consumption of households from different wealth tiers, with Firm = 8 fixed. When ϵ=2, the substitution elasticity is at its lowest, indicating that consumers are less willing to change their purchasing behavior due to price factors, leading to the highest consumption levels. When ϵ=8, consumers show the lowest consumption levels.

![Market Q4 P6](../img/Market%20Q4%20P6.png)

​**Figure 6**​: Comparison of household utility across different wealth tiers under different consumption elasticities. The effect of consumption elasticity on household utility is not significant. When ϵ=2, the average utility of households across different income levels is slightly higher.

![Market Q4 P7](../img/Market%20Q4%20P7.png)

​**Figure 7**​: Changes in social welfare levels under different consumption elasticities. When ϵ=2(yellow line), social welfare is noticeably higher and continues to improve over time.

* As the product substitution elasticity decreases, consumers become less sensitive to price changes, resulting in more active consumption behavior in the simulated economy, which leads to a slight increase in individual utility.
* Lower product substitution elasticity can effectively enhance overall social welfare.



