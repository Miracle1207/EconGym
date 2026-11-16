# Q6: How do market structures shape consumer behavior?

## 1. Introduction

### 1.1 Definitions and Characteristics of Different Market Structures

This model examines four market structures, providing definitions, key features, and real‐world examples for each:

* **Perfectly Competitive Market**
  * **Definition:** Many firms and consumers, no single firm can influence the market price; prices are determined by supply and demand.
  * **Features:** Homogeneous products, complete information, free entry and exit.
  * **Example:** Agricultural commodities such as wheat or rice.
* **Monopoly**
  * **Definition:** A single firm controls the entire industry’s production and sales and sets the market price.
  * **Features:** No close substitutes, significant pricing power, high entry barriers.
  * **Example:** Defense contractors (e.g., Lockheed Martin), large conglomerates (e.g., Samsung in Korea).
* **Oligopoly**
  * **Definition:** A few large firms dominate the market and mutually influence pricing and output decisions.
  * **Features:** Strategic interactions among firms, potential for price‐fixing alliances or competitive behaviors.
  * **Example:** The commercial aircraft industry (e.g., Boeing and Airbus).
* **Monopolistic Competition**
  * **Definition:** Many firms produce similar but differentiated products and have some degree of pricing power.
  * **Features:** Product differentiation via branding or marketing, but profits tend toward normal in the long run.
  * **Example:** The restaurant or coffee‐shop industry.

### 1.2 Research Questions

This study investigates how wage growth affects consumer‐goods prices under different market structures, specifically:

* **Wage‐Change Effects:** Do wage increases have differing impacts across market structures?
* **Price Dynamics:** Does higher wage growth lead to price changes, and if so, in which direction and over what time horizon?

### 1.3 Research Significance

* **Policy Guidance for Income and Antitrust Regulation:** If certain market structures translate wage increases rapidly into higher prices, they may exacerbate living‐cost pressures and inflation. This research aids in assessing how industry regulation or antitrust measures indirectly stabilize prices.
* **Enhancing Structural Realism in Macroeconomic Modeling:** Standard macro‐policy analysis often assumes an “average” market. Incorporating market‐structure differences into the wage–price linkage improves the accuracy and practicability of macro‐simulation and policy evaluation.

---

## **2.Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                                                                                    | Observation                                                                                                  | Action                                                                                 | Reward                                              |
| ----------- | ------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Individual**  | **Ramsey Model**        | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.              | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Firm**       | **Perfect Competition**      |Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses. | /                                                            | /                                                            | Zero (long-run)                                              |
|                | **Monopoly**                 |	Monopoly Firms set prices and wages to maximize profits under aggregate demand constraints.| $o_t^{\text{mono}} = ( K_t, Z_t, r_{t-1}^l )$<br>Production capital, productivity, lending rate | $a_t^{\text{mono}} = ( p_t, W_t )$<br>Price and wage decisions | $r_t^{\text{mono}} = p_t Y_t - W_t L_t - R_t K_t$<br>Profits = Revenue – costs |
|                | **Oligopoly**                |Oligopoly Firms engage in strategic competition, anticipating household responses and rival actions. | $o_t^{\text{olig}} = ( K_t^j,  Z_t^j, r_{t-1}^l)$<br>Production capital, productivity, lending rate | $a_t^{\text{olig}} = ( p_t^j, W_t^j )$<br>Price and wage decisions for firm $j$ | $r_t^{\text{olig}} = p_t^j y_t^j - W_t^j L_t^j - R_t K_t^j$<br>Profits = Revenue – costs for firm $j$ |
|                | **Monopolistic Competition** |Monopolistic Competitors offer differentiated products with CES demand and endogenous entry, supporting studies of consumer preference and market variety. | $o_t^{\text{mono-comp}} = ( K_t^j,  Z_t^j, r_{t-1}^l )$<br> Production capital, productivity, lending rate. Here, $j$ denotes the firm index. | $a_t^{\text{mono-comp}} = ( p_t^j, W_t^j )$<br>Price and wage decisions for firm $j$ | $r_t^{\text{mono-comp}} = p_t^j y_t^j - W_t^j L_t^j - R_t K_t^j$<br>Profits = Revenue – costs for firm $j$ |
| **Bank**       | **Non-Profit Platform** | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives.   | /                                                                                                                                                    | No rate control                                              | No profit                                            |

---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
Since the focus is on how wage changes under different market structures affect individual consumption decisions—rather than age effects—we model households using a representative infinite-horizon heterogeneous-agent framework.

**Government → Not Applicable**  
This study examines how wage changes in different market structures influence individual consumption behavior and price formation mechanisms. It focuses solely on market dynamics and household responses; policy‐making or institutional interventions (fiscal, monetary, regulatory) are excluded, so government does not act as an active decision-maker in the model.

**Firm → Multiple Market Structures**  
We consider four market structures:
**Perfectly Competitive Market:** firms freely compete; wages respond to labor supply.
**Monopoly:** a single firm sets the price.
**Oligopoly:** a few firms compete strategically.
**Monopolistic Competition:** many firms with differentiated products and some pricing power.

**Bank → Non-Profit Platform**  
We abstract from profit-seeking bank behavior and instead focus on how capital markets adjust to technological and wage shocks under no-arbitrage conditions.

---

## **3.Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | Trained on real‐world data to replicate households’ consumption, saving, and labor decisions in response to wage changes, enhancing realism.                      |
| Government             | Rule‐Based Agent      | Maintains consistent policy settings across market structures; uses fixed rules to observe market feedback, avoiding optimization‐driven strategy shifts.          |
| Firm                 | Rule‐Based Agent      | Focuses on how market structure shapes wage‐ and price‐formation; employs rules to model firm pricing and wage behavior, sidestepping strategic game simulations. |
| Bank | Rule‐Based Agent      | Simulates interest‐rate impacts on prices and wages via preset rules to capture macro feedback channels, without modeling profit‐maximizing bank strategies.      |

---
## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "market_type"
```

This command loads the configuration file `cfg/market_type.yaml`, which defines the setup for the "market_type" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `market_type.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "market_type"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "tax"

    - entity_name: 'households'
      entity_args:
        params:
          type: 'ramsey'

    - entity_name: 'market'
      entity_args:
        params:
          type: "oligopoly"     # todo: select from ['perfect', 'monopoly', 'monopolistic_competition', 'oligopoly']

    - entity_name: 'bank'
      entity_args:
        params:
          type: 'non_profit'


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

## 5.Illustrative Experiments

### Experiment : Impact of Market Structure on Households

* **Experiment Description:**

   Conduct a direct comparison of simulated societies under four market structures to analyze the effects of different market structures on key household indicators.
* **Experimental Variables:**
  
  * Market structure design (choice among the four regimes)
  * Household wealth level
  * Household utility
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **(market_type)\_ramsey\_100\_bc\_pension\_data\_based:** Households are modeled as **Behavior Cloning Agents** operating under the **Ramsey model** with **100** total households, while the government is represented as a**​ Rule-Based Agent** applying pension policies based on empirical data, and the overall market structure follows a ​**selected type**​.**(In the figure,from left to right are Monopolistic Competition Market , Oligopoly markets , Perfect Competition Market, Monopoly Market​----with all other agent settings kept unchanged)**
  * **Blue bars:** Represent the average wealth of ​**rich households**​.
  * **Green bars:** Represent the average wealth of ​**middle-class households**​.
  * **Yellow bars:** Represent the average wealth of ​**poor households**​.
  * **Red bars:** Represent the **overall mean** across all households.

![Individual Q6 P1](../img/Individual%20Q6%20P1.png)

**Figure 1:** The impact of different market structures on household wealth at year 50. From left to right are Monopolistic Competition Market , Oligopoly markets , Perfect Competition Market, Monopoly Market​. In the Monopolistic Competition Market, households have the lowest average wealth level, while under Oligopoly markets, household average wealth is higher. In Perfectly Competitive and Monopoly markets, household average wealth levels are similar; however, the Monopoly market shows a significantly more unequal distribution of wealth, with higher wealth among the rich and lower overall household average wealth. Moreover, in Oligopoly markets, the average wealth of the wealthy far surpasses that in other market types.
![Individual Q6 P2](../img/Individual%20Q6%20P2.png)

**Figure 2:** The impact of different market structures on household utility at year 50. From left to right are Monopolistic Competition Market , Oligopoly markets , Perfect Competition Market, Monopoly Market​. Different colored bars represent households from different income groups.In the Monopolistic Competition Market, households have the lowest average wealth level. Compared to the large disparities in household wealth, the differences in household utility across the four market types are relatively small. Under Oligopoly markets, household average utility is the highest, whereas in Monopoly markets, household average utility is the lowest. Although the differences in household utility across different wealth groups are relatively small, the wealthy exhibit noticeably higher average household utility.

* Household wealth varies significantly across different market structures, whereas differences in household utility are less pronounced. Overall, the Oligopoly markets achieves the highest average household wealth as well as the highest household utility.
