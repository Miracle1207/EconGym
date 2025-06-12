# Q5: How to coordinate monetary and fiscal policies?

## 1. Introduction

#### 1.1 Fundamental Functions of the Treasury and Central Bank

The Treasury Department primarily **influences economic activity through taxation, government spending, and debt management**, fulfilling roles in resource allocation, income redistribution, and macroeconomic stabilization. The central bank, by contrast, **implements monetary policy and ensures financial stability—adjusting interest rates, money supply**, and conducting open-market operations to control inflation, promote employment, and safeguard the financial system.

#### 1.2 Necessity and Context for Policy Coordination

Although fiscal and monetary authorities have distinct mandates, their objectives are closely aligned. Operating in isolation can weaken overall effectiveness or even produce **“policy offset” (e.g., fiscal expansion counteracted by tight monetary policy).** Since the 2008 financial crisis and the COVID-19 shock in 2020, many countries have pursued coordinated fiscal–monetary packages (such as combining fiscal stimulus with quantitative easing) to improve transmission, stabilize expectations, and boost aggregate demand.

#### 1.3 Research Questions

Using an economic-simulation platform, this study investigates the mechanisms and effects of fiscal–monetary policy coordination on:

* Inflation, output, unemployment, and wealth distribution in both the short and long run.
* The dynamic trade-offs among policy goals (growth vs. stability vs. equity).

#### 1.4 Research Significance

* **Deepening Systemic Understanding of Macro-Policy Interactions:**  Explore the feedback loops between fiscal and monetary measures to help researchers and policymakers build a coordinated, multi-agency stabilization framework.
* **Optimizing Transmission Paths:**  Analyze how coordinated policy affects micro-level decisions and macro outcomes, avoiding “policy clashes” or inefficient pass-through.
* **Advancing Complex Policy-Mix Design:**  Leverage RL Agents and similar methods to learn multi-objective optimal control paths within the policy space and to explore AI-driven policy design solutions.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role               | Selected Type                | Role Description                                                                                                                                                                                 |
| --------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual                | Ramsey Model                 | Respond to fiscal and monetary policies by adjusting consumption, saving, and labor decisions, generating micro‐level feedback.                                                                 |
| Government (Tax)     | Fiscal Authority          | This department manages taxation and spending policies, regulates aggregate demand, and implements redistribution mechanisms.                                                                       |
| Government (Central Bank) | Central Bank                 | The central bank controls money supply and interest rates to manage inflation and stabilize financial markets. Both authorities must be co‐modeled to capture the full macro‐policy framework. |
| Firm                  | Perfect Competition | Wage and price adjustments reflect policy shock transmission paths, affecting firm hiring and household incomes.                                                                                 |
| Bank    | Commercial Banks             | Profit‐seeking commercial banks provide a realistic channel to simulate the effects of coordinated fiscal and monetary policies.                                                                |

### Individual → Ramsey Model

* Households optimize their labor supply, savings, and consumption decisions based on life-cycle optimization principles. As the microfoundation of policy transmission, their behaviors provide crucial feedback to both fiscal and monetary policies.

### Government → Fiscal Authority & Central Bank

* **Fiscal Authority :** Responsible for designing tax and spending policies, adjusting aggregate demand and income distribution, and managing public debt to ensure fiscal sustainability. Its decisions directly affect households’ disposable income and government funding allocations, making it a key instrument for influencing growth and equity.
* **Central Bank:** Controls inflation, stabilizes prices, and maintains financial-system liquidity by adjusting interest rates and money supply. Its policies have broad but indirect impacts on consumption, investment, and credit behavior, positioning it as a central actor in macroeconomic stabilization.

### Firm → Perfect Competition

* Wages and goods prices are determined by supply and demand, acting as the intermediary mechanism through which fiscal and monetary policies influence household and firm behavior.

### Bank → Commercial Banks

* Simulate the formation of deposit and lending rates, reflecting how central-bank policies transmit to investment, interest rates, and liquidity.

---

## 3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type    | Role Description                                                                                                                             |
| ------------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | RL Agent         | Learn optimal labor, consumption, and saving strategies in response to a dynamic policy environment.                                         |
| Government             | RL Agent         | Use reinforcement learning to jointly adjust fiscal and monetary tools, achieving optimal growth–stability coordination.                    |
| Firm                 | Rule-Based Agent | Wages and prices are set by supply and demand; rules capture rapid market feedback to policy shocks.                                         |
| Bank  | Rule-Based Agent | Interest rates and investment returns feed back savings and policy changes via rules, maintaining capital‐market equilibrium and liquidity. |

### Individual → RL Agent

* In a complex policy environment (taxes, interest rates, prices), households must dynamically adjust labor, consumption, and saving strategies. An RL Agent learns optimal behavioral patterns, enhancing the flexibility and adaptiveness of micro‐level responses in the simulation.

### Government → RL Agent

* Modeled as a joint Treasury–Central Bank agent, the government uses reinforcement learning to explore dynamic, coordinated policy paths under multiple objectives (e.g., maximizing GDP, controlling inflation, minimizing inequality).

### Firm → Rule-Based Agent

* Wages and goods prices adjust according to standard supply–demand rules, reflecting how macro policies transmit through market mechanisms.

### Bank  → Rule-Based Agent

* Simulate capital‐market interest‐rate dynamics; rule‐based adjustments to investment returns capture the transmission channels between monetary policy and saving behavior.

---

## 4. Illustrative Experiments

### Experiment 1: Analysis of Fiscal–Monetary Policy Coordination Effects

* **Experiment Description:**
  In the simulated economy, allow the Treasury and the Central Bank to learn optimal coordination strategies via RL Agents. Under different objective functions (e.g., “stability priority” vs. “growth priority”), evaluate the impact of coordinated policies on key macro indicators such as GDP, inflation, and the Gini coefficient.
* **Involved Social Roles:**
  * *Households:* Ramsey Model
  * *Government:* Treasury Department & Central Bank
* **AI**​**​ Agents:**
  * *Households:* RL Agent
  * *Government:* RL Agent
  * *Market: ​*Rule-Based Agent
  * *Financial Institutions: ​*Rule-Based Agent
* **Core Experimental Variables:**
  * The fiscal and monetary policy departments, calibrated through economic modeling, were designed as follows: the Treasury Department implemented the **Saez Tax** system, while the Central Bank adopted the **Taylor Rule** as its behavioral logic.
  * Scale of fiscal spending
  * Income-tax rate & government-debt ceiling
  * Nominal interest rate or money-supply growth rate
  * Macro outcomes: GDP, inflation rate, wealth Gini coefficient

![Monetary Q5 P1](../img/Monetary%20Q5%20P1.png)

**Figure 1:** Compared the effects of separate operations by the Treasury Department or the Central Bank with the coordination of their policies (blue line, green line, and yellow line represent Treasury only, Central Bank only, and coordinated policy respectively). Despite the fact that, in the short term, single-department operations achieve higher GDP growth rates (blue line, green line), coordinated policy implementation leads to a longer-lasting simulation economy and results in better long-term GDP growth.

![Monetary Q5 P2](../img/Monetary%20Q5%20P2.png)

![Monetary Q5 P3](../img/Monetary%20Q5%20P3.png)

Figure 2 and Figure 3: Under the coordinated policies of the Treasury and the Central Bank, the wealth Gini coefficient in the short term is higher compared to single-sector policies (yellow line). However, after the 60th year, thanks to more stable long-term economic growth, the wealth disparity under policy coordination is smaller than under single-sector policies. When examining a specific year (e.g., Year 25), the coordinated policies result in wealth being more concentrated among the middle-aged and younger population (green and orange-yellow lines), whereas single-sector policies lead to a relatively equal distribution of wealth across youth to middle-aged groups.

![Monetary Q5 P4](../img/Monetary%20Q5%20P4.png)

Figure 4: The coordination between the Treasury and the Central Bank has no significant short-term impact on the income Gini coefficient (yellow line). In the long term, the income disparity under policy coordination is significantly smaller than under the Treasury-only policy, but still notably higher than under the Central Bank-only policy.

![Monetary Q5 P5](../img/Monetary%20Q5%20P5.png)

Figure 5: The collaboration between the Treasury Department and the Central Bank (yellow line) significantly reduces the overall social welfare.

* The coordination between fiscal policy (executed by the Treasury) and monetary policy (executed by the Central Bank) produces complex macroeconomic effects. However, in the long run, policy coordination enhances the long-term growth vitality of the simulated economy, with no significant difference in wealth inequality compared to the single-policy scenarios. Analyzing wealth distribution supports this conclusion (e.g., the wealth is more concentrated among middle-aged and younger generations).
* Moreover, policy coordination leads to a noticeable decline in social welfare, which may be due to the higher efficiency of market mechanisms under the coordinated policies.

