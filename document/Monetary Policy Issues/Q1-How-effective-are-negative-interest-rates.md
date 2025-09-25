# Q1: How effective are negative interest rates?

## 1. Introduction

### 1.1 Negative Interest Rate Policy

A Negative Interest Rate Policy (NIRP) occurs when a central bank sets certain policy rates below zero to incentivize bank lending and corporate investment, thereby stimulating economic growth. This measure is typically employed during periods of deflationary pressure, insufficient demand, or prolonged economic stagnation.

For example, in January 2016 the Bank of Japan introduced a negative rate on part of its excess reserves, lowering the rate to –0.1%. This policy aimed to counteract long-term stagnation and shrinking domestic demand arising from an aging population by promoting credit expansion and investment.

### 1.2 Research Questions

This study uses an economic simulation platform to investigate the economic impacts of negative interest rate policies, specifically examining:

* **Wealth Distribution: ​**How do prolonged negative rates influence the distribution of household wealth and income inequality?
* **GDP**​**​ Effects:** Does implementing negative interest rates stimulate aggregate output and economic growth?
* **Household Consumption:** How do negative interest rates affect household saving and consumption behavior across different income groups?

### 1.3 Research Significance

* **Assessing Monetary Policy Effectiveness at Low Rates:**  With many economies at or near the zero lower bound, NIRP has become a final policy tool. Evaluating its impact on output and investment helps delineate its limits and risks.
* **Insights**​**​ into Long-Term Growth Dynamics:**  Countries like Japan have paired NIRP with demographic aging and growth stagnation. Understanding this experience aids in balancing growth and distribution during structural transitions.

---

## ​2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type        | Role Description                                                                                                             | Observation                                                                                                  | Action                                                             | Reward                         |
| ----------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------ |
| **Individual**  | Ramsey Model         | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.                   | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Government**  | Central Bank         | Central Bank adjusts nominal interest rates and reserve requirements, transmitting monetary policy to households and firms. |\$\$o\_t^g = (\\mathcal{A}\_{t}, \\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, \\pi\_{t-1}, g\_{t-1})\$\$ <br>Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, inflation rate, growth rate. | $a_t^{\text{cb}} = ( \phi_t, \iota_t )$<br>Reserve ratio, benchmark rate | Inflation/GDP stabilization                                  |
| **Firm**       | Perfect Competition  | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                       | /                                                            | /                                                            | Zero (long-run)                                              |
| **Bank**        | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives. | /                                                            | No rate control                                              | No profit                                |

---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
The Ramsey model assumes **infinitely lived agents** with rational expectations who optimize consumption and saving in response to interest-rate changes. This framework is well suited to analyze how a negative-rate policy influences households’ propensities to consume and save, thereby affecting macroeconomic dynamics.
Since this experiment focuses on the aggregate impact of negative rates rather than intergenerational decision-making, we employ the Ramsey model instead of an OLG setup.

**Government → Central Bank**  
Negative interest-rate policy is a standard monetary-policy tool set and executed by the central bank, encompassing rate setting, reserve requirements, and asset-purchase operations. Compared with the treasury, the central bank is the appropriate authority for simulating systemic effects of rate changes on the economy.

**Firm → Perfect Competition**  
In a perfectly competitive market, prices are determined by supply and demand. This setting helps clearly identify the transmission channels through which negative rates affect real output, labor supply, and capital investment, avoiding distortions from market power.

**Bank → Non-Profit Bank**  
In this problem, banks need to cooperate with the central bank’s negative interest rate policy in order to fulfill the task of stimulating the economy; therefore, adopting non-profit banks is more appropriate. 

---

## 3. Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | The BC Agent learns saving and consumption patterns from real household data to simulate behavior under changing interest rates.            |
| Government             | Rule-Based Agent       | Central-bank policy adjustments follow fixed rules.                                                                                         |
| Firm                 | Rule-Based Agent       | The market responds according to supply–demand rules (e.g., lower rates boost investment); the simple rule-based approach ensures control. |
| Bank | Rule-Based Agent       | Commercial banks respond to base-rate changes by expanding loans or cutting deposit rates.                                                  |

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "negative_interest"
```

This command loads the configuration file `cfg/negative_interest.yaml`, which defines the setup for the "negative_interest" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `negative_interest.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "negative_interest"
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "central_bank"  # Focus on pension policy. type_list: ['tax', 'pension', 'central_bank']
          base_interest_rate: 0.03  # todo: set monetary policy
          reserve_ratio: 0.08
    - entity_name: 'households'
      entity_args:
        params:
          type: 'ramsey'
          type_list: ['ramsey', 'OLG', 'OLG_risk_invest', 'ramsey_risk_invest']
          households_n: 100
          action_dim: 2
        OLG:
          birth_rate: 0.04
          die_rate: 0.05
          initial_working_age: 24
    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"   #  type_list: [ 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly' ]
          alpha: 0.25
          Z: 10.0
          sigma_z: 0.0038
          epsilon: 0.5

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
          real_action_max: [ 1.0, 0.20 ]
          real_action_min: [ 0.0, -1e-3 ]

Trainer:
  house_alg: "bc"
  gov_alg: "rule_based"
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  seed: 1
  epoch_length: 300
  cuda: False
#  n_epochs: 1000
```
---
## **​5.​**​**Illustrative Experiment**

### Experiment 1: Evaluation of the Economic Effects of a Negative-Rate Policy

* **Experiment Description: ​**

  Compare a baseline economy with one that falls into a crisis in year 10 but implements a negative-interest-rate policy to assess how such a policy aids recovery.
* **Experimental Variables:**
  * Crisis onset in year 10 vs. a normally operating economy
  * Negative-rate policy (r = –1%) vs. Normal policy
  * GDP Level
  * Income-inequality indicator (e.g., Gini coefficient)
    

* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **pension\_bc\_rule\_based\_100\_ramsey (red line):** Households are modeled as **Behavior Cloning Agents ​**under the **Ramsey model ​**with ​**100 households**​, while the government is represented as a **Rule-Based Agent.**
  * **bc\_rule\_based\_100\_ramsey\_int\_-0.01 (orange line):** Households follow the Ramsey model as **Behavior Cloning Agents ​**with ​**100 households**​, and the government is a ​**Rule-Based Agent**​, with an additional setting introducing a **negative interest rate**​​**​ of –0.01**​.
* **Visualized Experimental Results：**

![Monetary Q1 P1](../img/Monetary%20Q1%20P1.png)

**Figure 1:** At Year 10, the orange line represents an economy that has entered a crisis (slowed GDP growth). After the government implements a negative‐rate policy, the crisis economy narrows its GDP gap with the normally growing economy over the medium and long term, demonstrating the policy’s supportive role in recovery.

![Monetary Q1 P2](../img/Monetary%20Q1%20P2.png)

**Figure 2:** Before the crisis, income inequality in both simulated economies is roughly the same. However, after implementing negative rates, the crisis economy experiences a steadily rising Gini coefficient, increasingly diverging from the normal economy.

* A **negative‐rate policy** can help restore economic vitality and stabilize GDP growth during a crisis. However, it also widens income inequality, indicating that complementary, more equitable fiscal measures are necessary to prevent the policy from exacerbating the wealth gap.


