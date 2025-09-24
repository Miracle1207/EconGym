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

| Social Role            | Selected Type                            | Role Description                                                                                                              |
| ------------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Ramsey Model                             | Account for non–age-related factors influencing household decisions under various market structures.                         |
| Firm                 | Multiple Market Structures               | Simulate how firms adjust labor demand and production decisions in response to technological advancements.                    |
| Bank | No-Arbitrage Platform | Measure the impact of technological progress on financial markets, such as shifts in borrowing demand and investment returns. |

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

**Bank → No-Arbitrage Platform**  
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
  
  * **monopolistic\_ramsey\_100\_bc\_pension\_data\_based:** Households are modeled as **Behavior Cloning Agents** operating under the **Ramsey model** with **100** total households, while the government is represented as a**​ Rule-Based Agent** applying pension policies based on empirical data, and the overall market structure follows a ​**monopolistic setting**​.**(In the figure,from left to right are Monopoly, Monopolistic Competition, Perfect Competition, and ​**​**Oligopoly**​**​ markets----with all other agent settings kept unchanged)**
  * **Blue bars:** Represent the average wealth of ​**rich households**​.
  * **Green bars:** Represent the average wealth of ​**middle-class households**​.
  * **Yellow bars:** Represent the average wealth of ​**poor households**​.
  * **Red bars:** Represent the **overall mean** across all households.

![Individual Q6 P1](../img/Individual%20Q6%20P1.png)

**Figure 1:** The impact of different market structures on household wealth. From left to right are Monopoly, Monopolistic Competition, Perfect Competition, and Oligopoly markets. Different colored bars represent households from different income groups. At year 50, households in the Monopolistic Competition market show significantly higher average wealth, especially among the rich group. In contrast, households in the Monopoly market have the lowest average wealth.

![Individual Q6 P2](../img/Individual%20Q6%20P2.png)

**Figure 2:** The impact of different market structures on household utility. From left to right are Monopoly, Monopolistic Competition, Perfect Competition, and Oligopoly markets. Different colored bars represent households from different income groups. At year 50, household utility is highest in the Monopolistic Competition market, though the difference compared to other market structures is relatively small.

* Household wealth varies significantly across different market structures, whereas differences in household utility are less pronounced. Overall, the monopolistic competition market achieves the highest average household wealth as well as the highest household utility.

