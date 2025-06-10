# Q5: How do pension systems vary across countries?

## 1. Introduction

### 1.1 Background

Pension systems vary significantly across countries in their structural design, contribution schemes, retirement ages, and replacement rates. For example, Nordic countries feature high taxes and generous benefits, the United States combines individual accounts with Social Security, and China faces structural challenges under rapid population aging. **Understanding these differences helps evaluate the strengths and weaknesses of system designs and guide domestic reforms.**

### 1.2 Comparative Significance

Pension schemes not only secure elderly livelihoods but also shape national savings rates, labor-force participation, and government fiscal burdens. Cross-country variations reflect differing trade-offs between efficiency and equity in welfare-state models.

### 1.3 Research Questions

Using an economic-simulation platform, this study examines the long-term impacts of different national pension systems on economies, specifically:

* How do structural differences (e.g., pay-as-you-go vs. funded schemes) affect government finances and household welfare?
* What is the effect of retirement-age policies on labor supply and aggregate output?
* How do replacement rates and contribution levels influence intergenerational equity and saving behavior?

### 1.4 Research Significance

* **​Feasibility of Institutional Transfer:​** Simulating foreign pension policies in a domestic context provides benchmarks and insights for national reform.
* **​Cross-National Comparative Analysis:​** Evaluating different systems on efficiency, fiscal sustainability, and intergenerational fairness offers lessons on design trade-offs and adaptability.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                            | Role Description                                                                                                            |
| ------------------------ | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Individual             | OLG Model      | Simulate life‐cycle behaviors, capturing sensitivity in retirement, saving, and consumption decisions.                     |
| Government             | Pension Authority                       | Design and implement pension policies, including contribution rates, replacement rates, and subsidy mechanisms.             |
| Firm                 | Perfectly Competitive Market             | Model how wages and employment dynamically adjust with labor‐force participation, affecting the pension contribution base. |
| Bank | No-Arbitrage Platform | Simulate pension‐fund investment returns and fiscal debt costs to assess system sustainability.                            |

### Individual → Overlapping Generations (OLG) Model 

* Use the Overlapping Generations framework to model life‐cycle differences in retirement, saving, and consumption decisions across countries.

### Government → Pension Authority

* Implement and manage various pension-system parameters, including contribution rates, replacement rates, and fiscal subsidies.

### Firm → Perfectly Competitive Market

* Reflect how changes in labor‐force participation under different pension regimes dynamically adjust wages and the contribution base through market mechanisms.

### Bank → No-Arbitrage Platform

* Simulate investment‐return trajectories under funded pension schemes or capture interest‐rate fluctuations tied to government‐bond–based subsidy mechanisms.

---

## 3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type          | Role Description                                                                                                      |
| ------------------------ | ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Individual             | Behavior Cloning Agent | Reproduce how residents of different countries react behaviorally to pension systems, capturing heterogeneity.        |
| Government             | Rule-Based Agent       | Implement rule-based templates of each country’s pension policies, facilitating comparative experiments.             |
| Firm                 | Rule-Based Agent       | Adjust wages via supply–demand rules, supporting market responses under varying demographic structures.              |
| Bank | Rule-Based Agent       | Feedback pension-fund asset changes or fiscal pressure through macro-level rules governing interest and return rates. |

### Individual → Behavior Cloning Agent

* Simulate typical household consumption, saving, and retirement habits across countries, such as “Nordic residents save little and rely on public pensions” and “a high share of Americans delay retirement.”

### Government → Rule-Based Agent

* Implement each country’s pension policy as a set of rules (e.g., “China template,” “US template,” “Sweden template”) to facilitate cross‐system comparisons.

### Firm → Rule-Based Agent

* Model wages determined by labor supply and demand, with rules capturing market wage and employment adjustment processes.

### Bank → Rule-Based Agent

* Use rules to model investment returns or government‐bond yields based on demographic structure and fiscal expenditure changes.

---

## 4. Illustrative Experiments

### Experiment 1: Cross‐National Pension Policy Comparison

* **Experiment Description:**
  Predefine pension‐policy templates for several countries (e.g., China, the United States, Sweden). Users select any template to run the simulation and observe impacts on economic output, public finances, individual welfare, and intergenerational equity.
* **Involved Social Roles:**
  * *Individual: ​*Saving, retirement, and benefit‐claim strategies over the life cycle
  * *Government: ​*Setting contribution rates, replacement rates, and subsidy rules
  * *Firm: ​*Labor‐force participation and wage responses under different regimes
* **AI Agents:**
  * *Individual:* Behavior Cloning Agent
  * *Government:* Rule‐Based Agent (switching by country template)
  * *Firm:* Rule‐Based Agent
* **Experimental Variables:**
  * Policy template selection (e.g., USA / China / Sweden)
  * Replacement rate (e.g., 40%, 60%, 80%)
  * Statutory retirement age (e.g., 60, 65, 67)

