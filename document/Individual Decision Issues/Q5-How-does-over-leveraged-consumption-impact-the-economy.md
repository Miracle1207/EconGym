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

| Social Role            | Selected Type                | Role Description                                                                                                                         |
| ------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Individual             | OLG Model                    | The Overlapping Generations model captures intertemporal consumption and borrowing decisions across age cohorts.                         |
| Firm                 | Perfect Competition | A perfectly competitive market reflects firm responses to changing consumer demand and transmits borrowing effects via price mechanisms. |
| Bank | Commercial Banks             | Commercial banks act as the primary intermediaries providing consumer credit.                                                            |

### Individual → Overlapping Generations (OLG) Model

* Front‐loading and overdraft consumption are fundamentally **intertemporal decision problems involving trade‐offs between present and future consumption.** The OLG framework clearly captures age‐specific borrowing and spending choices, reflecting life‐cycle liquidity constraints, income variations, and time preferences. By contrast, an infinite‐horizon Ramsey model assumes perpetual life and cannot represent stage‐specific borrowing motives and constraints, making it unsuitable for studying front‐loading consumption.

### Government → Not Applicable

* This study focuses on **endogenous mechanisms by which individual borrowing behaviors (front‐loading/overdraft) affect aggregate demand, financial‐system stability, and wealth‐accumulation paths.** To avoid confounding policy interventions, no active government agent is included—no fiscal subsidies, tax incentives, or regulatory constraints—so that government only provides a passive backdrop.

### Bank → Commercial Banks

* Commercial banks are the primary intermediaries for consumer credit, determining loan accessibility, costs, and terms. Through credit assessment, interest‐rate setting, and lending‐limit policies, they shape the feasible boundary of front‐loading consumption. In contrast, arbitrage‐free intermediaries focus on capital‐market efficiency and are less relevant to consumer‐credit behaviors.

### Firm → Perfect Competition

* Front‐loading and overdraft consumption directly alter demand for goods and services, impacting production and prices. A perfectly competitive market model accurately **reflects firms’ responses to demand shifts and the transmission of borrowing effects through price mechanisms**. Monopoly or oligopoly structures may distort price responses, hindering precise assessment of front‐loading consumption’s true impact, and are therefore not appropriate for this study.

---

## 3.**Selected Agent Algorithms**

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | Selected Type    | Role Description                                                                                                   |
| ------------------------ | ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Individual             | Rule-Based Agent | Use predefined rules to determine household consumption, saving, and labor decisions under technological progress. |
| Government             | Rule-Based Agent | Government follows predefined policy objectives in response to technological change.                               |
| Firm                 | Rule-Based Agent | Specify how firms adjust wages, production scale, and hiring decisions in response to technological progress.      |
| Bank | Rule-Based Agent | Set interest-rate and investment-return rules to measure technological impacts on capital markets.                 |

### **Individual → Rule-Based Agent**

* When studying the societal impacts of front-loading and overdraft consumption, a Rule-Based Agent lets us precisely control and analyze distinct borrowing-and-spending patterns. By encoding clear decision rules (e.g., “if disposable income falls below X% of target consumption, borrow Y amount”; “debt ratio capped at Z”; “minimum repayment behavior”), we can systematically simulate varying degrees of front-loading bias and trace long-term outcomes. Rule-based design decomposes complex credit behaviors into quantifiable parameters, facilitating counterfactual and sensitivity analyses. In contrast, large-language–model approaches obscure causality, and purely data-driven methods may lack flexibility in novel credit environments.

### **Government → Rule-Based Agent**

* Regulators overseeing front-loading consumption typically follow explicit policy rules and response functions.This approach clearly delineates intervention thresholds and intensities, aiding evaluation of how different regulatory frameworks modulate borrowing behaviors and macro-stability.

### **Bank → Rule-Based Agent**

* Commercial banks make lending decisions based on risk-management rules and profit targets. Rule-Based Agents can model how banks set loan terms from borrower attributes (income, credit score, existing debt).Unlike black-box methods, rule-based design transparently illustrates banks’ prudential decision processes and their impact on access to front-loading credit.

### **Firm → Rule-Based Agent**

* Market responses to consumption-driven demand shifts follow fundamental economic laws, well suited to explicit rules.Rule-Based Agent can precisely captures how credit-driven consumption propagates through price mechanisms to the wider economy and its resource-allocation effects.

---

## 4.Illustrative Experiments

### Experiment 1: Macroeconomic Impact of Front-Loading Consumption Spread

* **Experiment Description:**
  Create two simulated economies: one permits households to engage in front-loading consumption via credit overdrafts, the other restricts consumption to current income. Compare macro-indicators (GDP, aggregate wealth, saving rate, financial-system stability) to assess how borrowing-enabled consumption trades off short-term stimulus against long-term sustainability.
* **Involved Social Roles:**
  * *Market:* Perfectly Competitive Market
  * *Households: ​*Overlapping Generations (OLG) Model
* **AI**​**​ Agents:**
  * *Market: ​*Rule-Based Agent
  * *Households:* Rule-Based Agent
* **Experimental Variables:**
  * Degree of front-loading propensity in the population
  * Wealth levels, consumption, and working hours of households across different age and wealth groups under the influence of front-loading consumption
  * GDP level

![](https://yqq94nlzjex.feishu.cn/space/api/box/stream/download/asynccode/?code=MjA5MDQzNDNiOGM3OTQxYTU0ZWU0YzgzOWE5YjljMzNfMEYzVkJQMWh6Z0VsM3JzVkhmTDFFZHlydnJvV3BRWjVfVG9rZW46RE9BY2JieXB2b0o4MmF4UE9TemNkYmNUbk5jXzE3NDk4MTg5Nzg6MTc0OTgyMjU3OF9WNA)

​**Figure 1**​: Comparison of household consumption distribution between front-loading (overdraft-enabled) and normal consumption groups. From the age perspective (left chart), young households in the front-loading group show significantly higher consumption (green bar); from the income perspective (right chart), front-loading consumption notably increases the average consumption of poor households (yellow bar).

![](https://yqq94nlzjex.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTY2MjM4MjI1ZDRkMTU1YWM1MTQ3OWRkMGQ5Y2EyOWFfV0Uxc1I5RDFEYVhkdm5hTGlmVG5pZnJBd2d2d3luMTJfVG9rZW46SjRiSWJJVExXb1M4NWJ4cWE2YmNaTjVwbnlmXzE3NDk4MTg5Nzg6MTc0OTgyMjU3OF9WNA)

​**Figure 2**​: Comparison of household working hours between front-loading (overdraft-enabled) and normal consumption groups. Front-loading consumption does not significantly alter household labor supply.

![](https://yqq94nlzjex.feishu.cn/space/api/box/stream/download/asynccode/?code=OTI2YjA0NjQ5MTBkMWVlNmEyNmYyM2UyMzcwNTZiY2ZfOFhEZVpHZHk4SnhiTDU2eExaa2Z5Rlc1bkZTaXBLVUJfVG9rZW46RFhSQmJucUg0b2IzbHF4SjN1YmNhcTZkbmZiXzE3NDk4MTg5Nzg6MTc0OTgyMjU3OF9WNA)

​**Figure 3**​: Comparison of individual utility between front-loading (overdraft-enabled) and normal consumption groups. The overall impact of front-loading consumption on individual utility is not significant; however, from an age-based perspective, it leads to a slight increase in utility for younger households (left chart, green bar).

![](https://yqq94nlzjex.feishu.cn/space/api/box/stream/download/asynccode/?code=OWJiNTE3YjViZjE3NjVmZGYxNDBhZGY1OWVlMWU2NTZfRkZFSjF4dUNXMXRjcU1XWTNUS2wyVjcxU3ZZQlI0aGZfVG9rZW46UzFuU2JtNU8zb2NOR1l4eEJ6TWNtcjBSbmdmXzE3NDk4MTg5Nzg6MTc0OTgyMjU3OF9WNA)

​**Figure 4**​: Comparison of GDP trends between the two simulated economies. Overall, front-loading consumption promotes higher long-term GDP growth (blue line).

* In the simulated economy under the front-loading consumption assumption, household consumption, individual utility, and aggregate output differ from those in the normal economy. While front-loading significantly increases consumption among younger households, the corresponding improvement in their utility is less pronounced. Under moderate front-loading behavior, the overall economy exhibits relatively higher GDP growth.


