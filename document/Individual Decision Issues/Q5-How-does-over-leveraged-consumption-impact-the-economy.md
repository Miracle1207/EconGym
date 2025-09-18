# Q5: How does over-leveraged consumption impact the economy?

## 1.Introduction

### 1.1 Definition of the Issue

**​ over-leveraged consumption** refer to behaviors in which individuals borrow against future income to finance present spending. The two concepts differ subtly:

* **Front‐loading consumption:** Planned borrowing based on a clear repayment capacity and schedule (e.g., mortgages for housing, student loans for education).
* **Overdraft consumption:** Unsustainable borrowing that exceeds repayment ability, leading to long‐term debt accumulation (e.g., excessive credit‐card use, unplanned online payday loans).

Both involve intertemporal decision‐making, but differ in motivation, scale, and sustainability.

### 1.2 Research Background

* With the rise of modern financial systems and consumer‐credit products, front‐loading and overdraft consumption have become pervasive economic phenomena.
* Digital finance and internet lending platforms have drastically lowered access barriers to consumer credit, making borrowing easier than ever. Meanwhile, younger generations embrace **“buy now, pay later.”** Post‐pandemic recovery has accentuated the complex interplay between consumption and debt, and widening inequality has driven some groups to rely heavily on credit, creating class‐based consumption patterns.

### 1.3 Research Questions

Using an economic‐simulation platform, this study explores the societal impacts of front‐loading and overdraft consumption, focusing on:

* **Macroeconomic stability:** How do these consumption modes affect GDP volatility and social welfare?
* **Income distribution:** How do consumption‐borrowing patterns differ across income groups, and what is their effect on inequality?

### 1.4 Research Significance

* **​Theoretical significance:** Studying front‐loading and overdraft consumption enriches consumption theory, tests the life‐cycle hypothesis and permanent‐income hypothesis, and deepens understanding of intertemporal choice and time preference. It also illuminates the link between financial deepening and macro stability, showing how credit expansion shapes business‐cycle volatility and systemic risk accumulation.
* **​Practical significance:** Findings can guide policymakers in macroprudential regulation, help financial supervisors refine credit oversight to contain household‐debt risks, and support consumer‐education initiatives that bolster financial literacy and resilience.

---

## 2.**Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                                                                                       | Observation                                                                                                                                          | Action                                                       | Reward                                               |
| ----------- | ------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **Individual**  | OLG Model           | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals.   | $$o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$$<br/>Private: assets, education, age<br/>Global: distributional statistics                                  | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment <br/>*OLG*: old agents $$\lambda_t^i = 0$$                               |$r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)<br/>OLG includes pension if retired |
| **Firm**       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                 | /                                                                                                                                                    | /                                                            | Zero (long-run)                                      |
| **Bank**       | Commercial Banks    | Commercial Banks strategically set deposit and lending rates to maximize profits, subject to central bank constraints. | $$o_t^{\text{bank}} = \{ \iota_t, \phi_t, A_{t-1}, K_{t-1}, B_{t-1} \}$$<br>Benchmark rate, reserve ratio, deposits, loans, debts                    | $$a_t^{\text{bank}} = \{ r^d_t, r^l_t \}$$<br>Deposit, lending decisions | $$r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$$<br>Interest margin |

---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
Front‐loading and overdraft consumption are fundamentally **intertemporal decision problems involving trade‐offs between present and future consumption.** The OLG framework clearly captures age‐specific borrowing and spending choices, reflecting life‐cycle liquidity constraints, income variations, and time preferences. By contrast, an infinite‐horizon Ramsey model assumes perpetual life and cannot represent stage‐specific borrowing motives and constraints, making it unsuitable for studying front‐loading consumption.

**Government → Not Applicable**  
This study focuses on **endogenous mechanisms by which individual borrowing behaviors (front‐loading/overdraft) affect aggregate demand, financial‐system stability, and wealth‐accumulation paths.** To avoid confounding policy interventions, no active government agent is included—no fiscal subsidies, tax incentives, or regulatory constraints—so that government only provides a passive backdrop.

**Bank → Commercial Banks**  
Commercial banks are the primary intermediaries for consumer credit, determining loan accessibility, costs, and terms. Through credit assessment, interest‐rate setting, and lending‐limit policies, they shape the feasible boundary of front‐loading consumption. In contrast, arbitrage‐free intermediaries focus on capital‐market efficiency and are less relevant to consumer‐credit behaviors.

**Firm → Perfect Competition**  
Front‐loading and overdraft consumption directly alter demand for goods and services, impacting production and prices. A perfectly competitive market model accurately **reflects firms’ responses to demand shifts and the transmission of borrowing effects through price mechanisms**. Monopoly or oligopoly structures may distort price responses, hindering precise assessment of front‐loading consumption’s true impact, and are therefore not appropriate for this study.

---

## 3.**Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Rule-Based Agent | Use predefined rules to determine household consumption, saving, and labor decisions under technological progress. |
| Government             | Rule-Based Agent | Government follows predefined policy objectives in response to technological change.                               |
| Firm                 | Rule-Based Agent | Specify how firms adjust wages, production scale, and hiring decisions in response to technological progress.      |
| Bank | Rule-Based Agent | Set interest-rate and investment-return rules to measure technological impacts on capital markets.                 |



## 4.Illustrative Experiments

### Experiment 1: Macroeconomic Impact of Front-Loading Consumption Spread

* **Experiment Description:**
  Create two simulated economies: one permits households to engage in front-loading consumption via credit overdrafts, the other restricts consumption to current income. Compare macro-indicators (GDP, aggregate wealth, saving rate, financial-system stability) to assess how borrowing-enabled consumption trades off short-term stimulus against long-term sustainability.
* **Involved Social Roles:**
  * *Market:* Perfectly Competitive Market
  * *Individual: ​*Overlapping Generations (OLG) Model
* **AI**​**​ Agents:**
  * *Market: ​*Rule-Based Agent
  * *Individual:* Rule-Based Agent
* **Experimental Variables:**
  * Degree of front-loading propensity in the population
  * Wealth levels, consumption, and working hours of households across different age and wealth groups under the influence of front-loading consumption
  * GDP level
* **Visualized Experimental Results:**
![Individual Q5 P1](../img/Individual%20Q5%20P1.png)

​**Figure 1**​: Comparison of household consumption distribution between front-loading (overdraft-enabled) and normal consumption groups. From the age perspective (left chart), young households in the front-loading group show significantly higher consumption (green bar); from the income perspective (right chart), front-loading consumption notably increases the average consumption of poor households (yellow bar).

![Individual Q5 P2](../img/Individual%20Q5%20P2.png)

​**Figure 2**​: Comparison of household working hours between front-loading (overdraft-enabled) and normal consumption groups. Front-loading consumption does not significantly alter household labor supply.

![Individual Q5 P3](../img/Individual%20Q5%20P3.png)

​**Figure 3**​: Comparison of individual utility between front-loading (overdraft-enabled) and normal consumption groups. The overall impact of front-loading consumption on individual utility is not significant; however, from an age-based perspective, it leads to a slight increase in utility for younger households (left chart, green bar).

![Individual Q5 P4](../img/Individual%20Q5%20P4.png)

​**Figure 4**​: Comparison of GDP trends between the two simulated economies. Overall, front-loading consumption promotes higher long-term GDP growth (blue line).

* In the simulated economy under the front-loading consumption assumption, household consumption, individual utility, and aggregate output differ from those in the normal economy. While front-loading significantly increases consumption among younger households, the corresponding improvement in their utility is less pronounced. Under moderate front-loading behavior, the overall economy exhibits relatively higher GDP growth.


