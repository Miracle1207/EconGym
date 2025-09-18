# Q5: How to coordinate monetary and fiscal policies?

## 1. Introduction

#### 1.1 Fundamental Functions of the Treasury and Central Bank

The Treasury Department primarily **influences economic activity through taxation, government spending, and debt management**, fulfilling roles in resource allocation, income redistribution, and macroeconomic stabilization. The central bank, by contrast, **implements monetary policy and ensures financial stability—adjusting interest rates, money supply**, and conducting open-market operations to control inflation, promote employment, and safeguard the financial system.

#### 1.2 Necessity and Context for Policy Coordination

Although fiscal and monetary authorities have distinct mandates, their objectives are closely aligned. Operating in isolation can weaken overall effectiveness or even produce **“policy offset” (e.g., fiscal expansion counteracted by tight monetary policy).** Since the 2008 financial crisis and the COVID-19 shock in 2020, many countries have pursued coordinated fiscal–monetary packages (such as combining fiscal stimulus with quantitative easing) to improve transmission, stabilize expectations, and boost aggregate demand.

### 1.3 Research Questions

This study uses an economic simulation platform to investigate the economic impacts of fiscal–monetary policy coordination, specifically examining:

* **GDP**​**​ Effects: ​**How does coordinated policy intervention affect short-term recovery and long-term economic growth compared to uncoordinated actions?
* **Wealth Distribution: ​**What are the distributional consequences of policy coordination, particularly in terms of asset ownership and intergenerational inequality?
* **Household Consumption: ​**How do combined fiscal transfers and low interest rates influence aggregate and heterogeneous household spending behavior?

#### 1.4 Research Significance

* **Deepening Systemic Understanding of Macro-Policy Interactions:**  Explore the feedback loops between fiscal and monetary measures to help researchers and policymakers build a coordinated, multi-agency stabilization framework.
* **Optimizing Transmission Paths:**  Analyze how coordinated policy affects micro-level decisions and macro outcomes, avoiding “policy clashes” or inefficient pass-through.
* **Advancing Complex Policy-Mix Design:**  Leverage RL Agents and similar methods to learn multi-objective optimal control paths within the policy space and to explore AI-driven policy design solutions.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role               | Selected Type        | Role Description                                                                                                             | Observation                                                                                                  | Action                                                                                 | Reward                                              |
| ------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Individual**                | Ramsey Model         | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.                    | $$o_t^i = (a_t^i, e_t^i)$$<br>Private: assets, education<br>Global: distributional statistics                | $$a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$$<br>Asset allocation, labor, investment | $$r_t^i = U(c_t^i, h_t^i)$$ (CRRA utility)          |
| **Government(Tax)**          | Fiscal Authority     | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.                         | $$o_t^g = \{ B_{t-1}, W_{t-1}, P_{t-1}, \pi_{t-1}, Y_{t-1}, \mathcal{I}_t \}$$<br>Public debt, wage, price level, inflation, GDP, income dist. | $$a_t^{\text{fiscal}} = \{ \boldsymbol{\tau}, G_t \}$$<br>Tax rates, spending          | GDP growth, equality, welfare                       |
| **Government(Central Bank)** | Central Bank         | Central Bank adjusts nominal interest rates and reserve requirements, transmitting monetary policy to households and firms. | $o_t^g = \{ B_{t-1}, W_{t-1}, P_{t-1}, \pi_{t-1}, Y_{t-1}, \mathcal{I}_t \}$<br>Public debt, wage, price level, inflation, GDP, income dist.                                                                                           | $$a_t^{\text{cb}} = \{ \phi_t, \iota_t \}$$<br>Reserve ratio, benchmark rate           | Inflation/GDP stabilization                         |
| **Firm**                     | Perfect Competition  | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                       | /                                                                                                            | /                                                                                    | Zero (long-run)                                     |
| **Bank**                     | Commercial Banks     | Commercial Banks strategically set deposit and lending rates to maximize profits, subject to central bank constraints.      | $$o_t^{\text{bank}} = \{ \iota_t, \phi_t, A_{t-1}, K_{t-1}, B_{t-1} \}$$<br>Benchmark rate, reserve ratio, deposits, loans, debts | $$a_t^{\text{bank}} = \{ r^d_t, r^l_t \}$$<br>Deposit, lending decisions               | $$r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$$<br>Interest margin |


---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
Households optimize their labor supply, savings, and consumption decisions based on life-cycle optimization principles. As the microfoundation of policy transmission, their behaviors provide crucial feedback to both fiscal and monetary policies.

**Government → Fiscal Authority & Central Bank**  
**Fiscal Authority :** Responsible for designing tax and spending policies, adjusting aggregate demand and income distribution, and managing public debt to ensure fiscal sustainability. Its decisions directly affect households’ disposable income and government funding allocations, making it a key instrument for influencing growth and equity.**Central Bank:** Controls inflation, stabilizes prices, and maintains financial-system liquidity by adjusting interest rates and money supply. Its policies have broad but indirect impacts on consumption, investment, and credit behavior, positioning it as a central actor in macroeconomic stabilization.

**Firm → Perfect Competition**  
Wages and goods prices are determined by supply and demand, acting as the intermediary mechanism through which fiscal and monetary policies influence household and firm behavior.

**Bank → Commercial Banks**  
Simulate the formation of deposit and lending rates, reflecting how central-bank policies transmit to investment, interest rates, and liquidity.

---

## 3. Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | RL Agent         | Learn optimal labor, consumption, and saving strategies in response to a dynamic policy environment.                                         |
| Government             | RL Agent         | Use reinforcement learning to jointly adjust fiscal and monetary tools, achieving optimal growth–stability coordination.                    |
| Firm                 | Rule-Based Agent | Wages and prices are set by supply and demand; rules capture rapid market feedback to policy shocks.                                         |
| Bank  | Rule-Based Agent | Interest rates and investment returns feed back savings and policy changes via rules, maintaining capital‐market equilibrium and liquidity. |

## **4. Running the Experiment**

### **4.1 Quick Start**

To run the simulation with a specific problem scene, use the following command:

```Bash
python main.py --problem_scene ""
```

This command loads the configuration file `cfg/`, which defines the setup for the "" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### **4.2 Problem Scene Configuration**

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### **Example ​**​**YAML**​**​ Configuration: ​**

---

## 5. Illustrative Experiments

### Experiment 1: Analysis of Fiscal–Monetary Policy Coordination Effects

* **Experiment Description:**
  
  In the simulated economy, allow the Treasury and the Central Bank to learn optimal coordination strategies via RL Agents. Under different objective functions (e.g., “stability priority” vs. “growth priority”), evaluate the impact of coordinated policies on key macro indicators such as GDP, inflation, and the Gini coefficient.
* **Core Experimental Variables:**
  
  * The fiscal and monetary policy departments, calibrated through economic modeling, were designed as follows: the Treasury Department implemented the **Saez Tax** system, while the Central Bank adopted the **Taylor Rule** as its behavioral logic.
  * Scale of fiscal spending
  * Income-tax rate & government-debt ceiling
  * Nominal interest rate or money-supply growth rate
  * Macro outcomes: GDP, inflation rate, wealth Gini coefficient
* **Baselines：**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **​OLG\_tax(blue line):​**The households and the goverment modeled as ​**RL-Agent**​.The Goverment represent **Treasury Department only.**
  * **​OLG\_CenBank(green line):​**The households and the goverment modeled as ​**RL-Agent**​.The Goverment represent **Central Bank**​**​ only.**
  * **​OLG\_tax\_CenBank(yellow line):​**The households and the goverment modeled as ​**RL-Agent**​.The Goverment represent both **the Treasury Department and The ​**​​**Central Bank.**

![Monetary Q5 P1](../img/Moneraty%20Q5%20P1.png)

**Figure 1:** Compared the effects of separate operations by the Treasury Department or the Central Bank with the coordination of their policies . Despite the fact that, in the short term, single-department operations achieve higher GDP growth rates, coordinated policy implementation leads to a longer-lasting simulation economy and results in better long-term GDP growth.


![Monetary Q5 P2](../img/Moneraty%20Q5%20P2.png)

**​Figure 2:​**Under the coordinated policies of the Treasury and the Central Bank, the wealth Gini coefficient in the short term is higher compared to single-sector policies. However, after the 60th year, thanks to more stable long-term economic growth, the wealth disparity under policy coordination is smaller than under single-sector policies.

* **Baselines：**
  * **Left panel: ​**Different bar colors represent **age cohorts** (e.g., <24, 25–34, 35–44, 45–54, 55–64, 65–74, 75–84, 85+, total).
  * **Right panel:** Different bar colors represent ​**income classes ​**​(rich, middle, poor, and mean).

![Monetary Q5 P3](../img/Moneraty%20Q5%20P3.png)

​**​ Figure 3**​: When examining a specific year (e.g., The Year 25), the coordinated policies result in wealth being more concentrated among the middle-aged and younger population (green and orange-yellow lines), whereas single-sector policies lead to a relatively equal distribution of wealth across youth to middle-aged groups.

![Monetary Q5 P4](../img/Moneraty%20Q5%20P4.png)

​**Figure 4**​: The coordination between the Treasury and the Central Bank has no significant short-term impact on the income Gini coefficient. In the long term, the income disparity under policy coordination is significantly smaller than under the Treasury-only policy, but still notably higher than under the Central Bank-only policy.

![Monetary Q5 P5](../img/Moneraty%20Q5%20P5.png)

​**Figure 5**​: The collaboration between the Treasury Department and the Central Bank significantly reduces the overall social welfare.

* The coordination between fiscal policy (executed by the Treasury) and monetary policy (executed by the Central Bank) produces complex macroeconomic effects. However, in the long run, policy coordination enhances the long-term growth vitality of the simulated economy, with no significant difference in wealth inequality compared to the single-policy scenarios. Analyzing wealth distribution supports this conclusion (e.g.,wealth is more concentrated among middle-aged and younger generations).
* Moreover, policy coordination leads to a noticeable decline in social welfare, which may be due to the higher efficiency of market mechanisms under coordinated policies.



