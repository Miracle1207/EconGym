# Q3: What are the long-term effects of quantitative easing?

## ​1. Introduction

### 1.1 ​Quantitative Easing​ and The Phases

**Quantitative Easing (QE)** is a monetary policy in which a central bank purchases long-term assets—such as government bonds and mortgage-backed securities—to inject liquidity into the economy and stimulate growth. Since the 2008 financial crisis, the Federal Reserve conducted three rounds of QE (QE1 through QE3) to counteract credit tightening and collapsing demand. Its balance sheet expanded from roughly \$0.9 trillion before the crisis to \$4.5 trillion by the end of QE3 in 2014, representing an unprecedented scale in modern monetary policy. **The effects of QE extend beyond short-term stimulus to reshape long-run financial market structure, wealth distribution, and international capital flows**, making it indispensable for understanding contemporary monetary policy.

### **1.2 Research Questions**

This study uses an economic simulation platform to investigate the economic impacts of quantitative easing (QE), specifically examining:

* **GDP**​**​ Effects:** How does prolonged quantitative easing affect short-term recovery and long-term economic growth?
* **Wealth Distribution:** How does QE influence the distribution of household wealth through asset price appreciation?
* **Household Consumption:** How do liquidity injections and asset revaluation under QE alter consumption behavior across income and age groups?
* ​**Social Welfare**​: What are the overall welfare implications of QE, considering trade-offs between macroeconomic stabilization and rising inequality?

### 1.3 Research Significance

* **Quantitative Guidance for Policymakers:**  Simulating different magnitudes of QE and observing feedback on GDP and market indicators provides data-driven input for designing QE operations during downturns, supporting risk assessment.
* **Evaluation of Aggressive Monetary Tools:**  At the zero lower bound, QE becomes the principal policy lever. Assessing its effects on output, consumption, and employment clarifies its operational limits and the efficacy of liquidity injections under trap conditions.

---

## ​2.​ Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type        | Role Description                                                                                                             | Observation                                                                                                  | Action                                                             | Reward                         |
| ----------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------ |
| **Individual**  | Ramsey Model         | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.                    | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Government**  | Central Bank         | Central Bank adjusts nominal interest rates and reserve requirements, transmitting monetary policy to households and firms.  |\$\$o\_t^g = (\\mathcal{A}\_{t}, \\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, \\pi\_{t-1}, g\_{t-1})\$\$ <br>Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, inflation rate, growth rate. | $a_t^{\text{cb}} = ( \phi_t, \iota_t )$<br>Reserve ratio, benchmark rate | Inflation/GDP stabilization                                  |
| **Firm**       | Perfect Competition  | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                       | /                                                                                                            | /                                                                | Zero (long-run)                |
| **Bank**       | Commercial Bank     | Commercial Bank strategically set deposit and lending rates to maximize profits, subject to central bank constraints.      | $o_t^{\text{bank}} = ( \iota_t, \phi_t, r^l_{t-1}, r^d_{t-1}, loan, F_{t-1} )$<br>Benchmark rate, reserve ratio, last lending rate, last deposit_rate, loans, pension fund. | $a_t^{\text{bank}} = ( r^d_t, r^l_t )$<br>Deposit, lending decisions | $r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$<br>Interest margin |

---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
The Ramsey model is well suited to simulate a representative household’s optimal consumption and saving decisions in response to long-term interest-rate changes. Under QE, it captures how households adjust savings based on expected returns and prices, allowing evaluation of policy effects on asset accumulation and intertemporal welfare—especially for analyzing wealth‐distribution shifts and utility distortions induced by QE.

**Government → Central Bank**  
Within the QE framework, the central bank is the sole policy implementer, responsible for setting asset‐purchase volumes, adjusting the policy rate, and guiding market expectations.

**Firm → Perfect Competition**  
In a perfectly competitive market, price, capital, and labor allocations respond directly to policy shocks, facilitating analysis of QE’s transmission to real economic activity (e.g., investment, output, wages). Its transparent structure also aids in measuring how financial‐variable changes pass through to the real economy.

**Bank → Commercial Bank**  
Commercial bank are the core nodes in QE’s transmission mechanism: their funding sources, lending behavior, and rate‐setting directly affect credit supply. After asset purchases depress long‐term rates, banks act as intermediaries to expand lending and adjust risk preferences, thereby altering corporate financing capacity and household consumption choices—making them the critical bridge between central‐bank policy and microeconomic responses.

---

## ​3.​ Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | Learns actual household consumption and investment behaviors under changing interest rates to simulate response differences across income groups. |
| Government             | Rule-Based Agent       | Sets interest-rate and asset-purchase rules, ensuring interpretability and adherence to central-bank protocols.                                   |
| Firm                 | Rule-Based Agent       | Models firms adjusting output and investment according to marginal principles, faithfully reproducing supply–demand dynamics.                    |
| Bank | Rule-Based Agent       | Captures banks’ lending adjustments based on rate rules, providing a stable and controllable representation of financial transmission channels.  |

---

## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "QE"
```

This command loads the configuration file `cfg/QE.yaml`, which defines the setup for the "QE" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `QE.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "QE"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "central_bank"  # Focus on pension policy. type_list: ['tax', 'pension', 'central_bank']
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
          type: "perfect"   #  type_list: [ 'perfect', 'monopoly', 'monopolistic_competition', 'oligopoly' ]
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
  gov_alg: "rule_based"
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  seed: 1
  epoch_length: 300
  cuda: False
#  n_epochs: 300
```
---

## **​5.​**​**Illustrative Experiment**

### Experiment : Socioeconomic Impacts of Quantitative Easing

* **Experiment Description: ​**
  
  Study the effect of a quantitative-easing policy on an economy’s GDP growth rate.
* **Experimental Variables:**
  
  * Quantitative Easing vs. Standard monetary policy vs. cutting lending-deposit interest-rate spread
  * Transmission intensity represented by an increase in government-bond purchases or a reduction in the lending-deposit interest-rate spread
  * Long-term GDP levels across different simulated economies

```SQL
# Step 1: Define the QE policy scenario
This experiment assumes a predefined QE policy environment  
Set the total QE purchase volume to a certain percentage of the simulated economy’s GDP

# Step 2: Central bank conducts asset purchases
for each period during QE:
    - Central bank purchases long-term government bonds (from market or banks)
    - Injects base money into the economy (M0 increases)
    - Expands commercial banks’s reserves

# Step 3: Financial system transmission
Banks increase lending capacity due to excess reserves  
Loan interest rates fall → firms face lower financing costs  
Asset prices rise → households experience wealth effect → higher consumption

# Step 4: Macroeconomic variable updates
Update key indicators: GDP, CPI, consumption, investment  
Track both short-term and long-term effects of QE policy
```

* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **bc\_rule\_based\_100\_ramsey\_int\_margin\_0.03 (blue line):** Households are modeled as **Behavior Cloning Agents** under the **Ramsey model ​**with ​**100 households**​, while the government is a ​**Rule-Based Agent**​,with **reduces the lending–deposit interest-rate margin to 0.03**.
  * **bc\_rule\_based\_100\_ramsey (yellow line):** Households are modeled as **Behavior Cloning Agents** under the **Ramsey model ​**with ​**100 households**​, while the government is a ​**Rule-Based Agent**​.**Baseline scenario.**
  * **bc\_rule\_based\_100\_ramsey\_gt\_0.15 (red line):** Households are modeled as **Behavior Cloning Agents** under the **Ramsey model ​**with ​**100 households**​, while the government is a **​Rule-Based Agent,​** **with the extra goverment purchase volume set to 15% of GDP**.
* **Visualized Experimental Results：**

![Monetary Q3 P1](../img/Monetary%20Q3%20P1.png)

![Monetary Q3 P2](../img/Monetary%20Q3%20P2.png)

**Figure 1 & Figure 2:** Increasing government-bond purchases or narrowing the bank lending–deposit spread represent different transmission channels of quantitative easing. Both implementations of QE produce clear GDP gains in the simulated economy, in both the short run (Year < 20) and the medium-to-long run (Years 20–80).

* Quantitative easing is an effective tool for stimulating economic growth. Modeling QE via increased bond purchases yields a higher aggregate GDP in the simulation.

