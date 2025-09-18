# Q2: How to control inflation via monetary policy?

## 1. Introduction

### 1.1 Inflation Phenomenon

Inflation refers to a sustained and significant rise in the overall price level of an economy, typically measured by the Consumer Price Index (CPI) or the GDP deflator. Since 2021, the United States has experienced severe inflationary pressure, with the year-over-year CPI increase exceeding 9% at its peak. To combat high inflation, the Federal Reserve has raised the federal funds rate continuously since 2022—from 0.25% to above 5%—and has reduced its balance sheet to tighten liquidity, suppress demand, and curb price growth.

### 1.2 Research Questions

Using an economic-simulation platform, this study investigates the control of inflation via reinforcement-learning methods and evaluates the macroeconomic consequences of such control, focusing on:

* **GDP level** (how inflation control affects GDP)
* **Income inequality** (whether inflation control widens or narrows the wealth gap)

### 1.3 Research Significance

* **Exploring the Long-Run Transmission Mechanisms of Monetary Policy:**  By tracing the multi-period feedback effects of interest-rate adjustments on consumption, investment, and employment, simulation experiments can illuminate the process by which monetary policy evolves from a short-term stabilizer into a long-term economic shaping force.
* **Policy Guidance:**  This research can provide central banks with a simulation-based framework to balance price stability against growth, helping to avoid “over-tightening” or “delayed response” in policy implementation.

---

## ​2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type        | Role Description                                                                                                             | Observation                                                                                                  | Action                                                             | Reward                         |
| ----------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------ |
| **Individual**  | Ramsey Model         | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.                    | $$o_t^i = (a_t^i, e_t^i)$$<br>Private: assets, education<br>Global: distributional statistics                | $$a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$$<br>Asset allocation, labor, investment | $$r_t^i = U(c_t^i, h_t^i)$$ (CRRA utility) |
| **Government**  | Central Bank         | Central Bank adjusts nominal interest rates and reserve requirements, transmitting monetary policy to households and firms. | $o_t^g = \{ B_{t-1}, W_{t-1}, P_{t-1}, \pi_{t-1}, Y_{t-1}, \mathcal{I}_t \}$<br>Public debt, wage, price level, inflation, GDP, income dist.                                                                                           | $$a_t^{\text{cb}} = \{ \phi_t, \iota_t \}$$<br>Reserve ratio, benchmark rate           | Inflation/GDP stabilization    |
| **Firm**       | Perfect Competition  | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                       | /                                                                                                            | /                                                                | Zero (long-run)                |
| **Bank**       | Commercial Banks     | Commercial Banks strategically set deposit and lending rates to maximize profits, subject to central bank constraints.      | $$o_t^{\text{bank}} = \{ \iota_t, \phi_t, A_{t-1}, K_{t-1}, B_{t-1} \}$$<br>Benchmark rate, reserve ratio, deposits, loans, debts | $$a_t^{\text{bank}} = \{ r^d_t, r^l_t \}$$<br>Deposit, lending decisions               | $$r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$$<br>Interest margin |


---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
The Ramsey model effectively captures **how households adjust their behavior based on expected prices, wage changes, and interest rates,** making it a key theoretical tool for analyzing consumption and labor‐supply responses to inflation control.

**Government → Central Bank**  
The central bank designs and implements monetary policy, **directly controlling key variables such as interest rates and money supply**. Tools for inflation control—like rate hikes and balance‐sheet reduction—are led by the central bank, so this role accurately simulates policy transmission and its macroeconomic effects.

**Firm → Perfect Competition**  
A perfectly competitive market reflects **the basic mechanism of prices set by supply and demand**, helping to clearly observe how inflation‐control policies (e.g., rate increases) affect labor markets, goods prices, and investment returns. It avoids distortions from oligopoly or monopoly‐induced price rigidity, making it well suited to identify policy effects.

**Bank → Commercial Banks**  
During inflation control, commercial banks typically raise lending and deposit rates, which in turn affect firms’ financing costs and households’ saving returns. Compared with a no-arbitrage framework, commercial banks also involve microbehavior such as risk assessment, credit rationing, and asset‐liability management, enabling a more realistic simulation of credit contraction or expansion under policy shifts.

---

## ​3.​ Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | The BC Agent can learn consumption patterns of households at different income levels from historical data (e.g., high-income households use investments to hedge against inflation).  |
| Government             | RL Agent               | Inflation control requires dynamic policy adjustments (e.g., incremental rate hikes or quantitative tightening); the RL Agent learns optimal strategies through environment feedback. |
| Firm                 | Rule-Based Agent       | A rule-based agent can directly simulate the “cost increase → price increase” transmission chain.                                                                                  |
| Bank | Rule-Based Agent       | Commercial bank behavior can be modeled directly through rule-based logic.                                                                                                            |

## 4.​ Illustrative Experiment

### Experiment 1: Evaluating Optimal Monetary Policy via Reinforcement Learning

* **Experiment Description: ​** The central bank dynamically adjusts interest rates through a Reinforcement-Learning Agent aiming to minimize CPI inflation while maintaining stable GDP growth. Monitor macroeconomic indicators to assess policy effectiveness.
* **Involved Social Roles:**
  * *Government:* Central Bank
  * *Firm:* Perfectly Competitive Market
  * *Banks: ​*Commercial Banks
* **AI**​**​ Agents:**
  * *Government: ​*RL Agent
  * *Firm:* Rule-Based Agent
  * *Banks:* ​Rule-Based Agent
* **Experimental Variables:**
  * Comparison of RL Agent vs. Rule-Based Agent in exploring optimal monetary policy
  * Aggregate GDP level
  * Gini coefficient

```Python
# Observe macroeconomic variables: CPI, GDP growth, interest rate, etc.

for each time step t:

    # Step 1: The central bank observes macroeconomic conditions and sets policy rate
    Observe current CPI_t and GDP_t  
    Feed these into the reinforcement learning–based policy module,  
    which outputs the period-specific optimal interest rate policy_rate_t  
    aiming to balance price stability and economic growth
    
    # Step 2: Rule-based commercial banks transmit the policy rate to the economy
    Commercial banks, modeled as rule-based agents, adjust deposit and lending rates according to policy_rate_t and predefined spread rules  
    Firms respond by updating investment and employment plans  
    Households adjust consumption and savings based on observed interest rate changes

    # Step 3: Macroeconomic indicators are updated and feedback is recorded
    The system generates new CPI_t+1 and GDP_t+1  
    The central bank collects feedback and uses it to update the RL policy model
```

* **Visualized Experimental Results：**

![Monetary Q2 P1](../img/Monetary%20Q2%20P1.png)

**Figure 1:** The simulated economy under an inflation‐control regime (blue line) shows a lower long-run GDP level compared with the economy governed by a predefined rule-based policy (green line), indicating that monetary policy aimed at curbing inflation does indeed dampen GDP growth.

![Monetary Q2 P2](../img/Monetary%20Q2%20P2.png)

**Figure 2:** The inflation‐control policy (blue line) increases income inequality.

* During an overheating episode, the central bank employs an RL approach to learn an inflation-targeted policy. While this policy reduces growth relative to the overheated scenario, it also aggravates income inequality. The RL-derived tightening disproportionately lowers incomes of low-income households due to falling employment rates, whereas high-income households are less impacted, thus widening the wealth gap.
* **Summary:** The simulation platform enables quantification of different policy mixes’ effectiveness in suppressing inflation, offering decision support to balance economic growth against price stability.

