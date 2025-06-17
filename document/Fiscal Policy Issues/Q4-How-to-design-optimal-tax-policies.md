# Q4: How to design optimal tax policies?

## 1. Introduction

### 1.1 Optimal Tax Policy

* Taxation is the government’s core tool for **resource allocation, income redistribution, and macroeconomic stabilization.** Different tax structures (e.g., labor‐income taxes, capital‐gains taxes, consumption taxes) have varied impacts on economic efficiency, income distribution, and fiscal sustainability. Designing an “optimal tax policy” means maximizing social welfare—balancing growth incentives and equity—while ensuring fiscal balance.
* Traditional optimal‐tax theory minimizes distortionary effects without harming efficiency. However, real economies feature heterogeneous agents, income inequality, and political constraints. Reinforcement Learning (RL) offers a powerful “policy search” approach to dynamically explore the optimal strategy space.

### 1.2 Research Questions

Using an economic‐simulation platform, this study investigates how a government can employ RL algorithms to dynamically optimize the tax mix, specifically:

* Trade‐offs among labor‐income taxes, consumption taxes, and capital taxes.
* Effects of rate adjustments on GDP, the Gini coefficient, and fiscal surplus.
* Feedback mechanisms from tax policy to household saving, labor‐force participation, and consumption.

### 1.3 Research Significance

* **Building an Intelligent Policy‐Evaluation Tool:** Deploy RL to create a “simulate–feedback–optimize” loop, equipping policymakers with advanced tools for experimental policy design and institutional assessment.
* **Achieving Dynamic Growth–Equity Balance:** Use multi‐objective optimization to finely tune tax systems for efficiency and fairness, enhancing the responsiveness and adaptability of fiscal frameworks.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                            | Role Description                                                                                                                                                |
| ------------------------ | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Ramsey Model                             | Simulate rational agents’ decisions on labor supply, consumption, and saving in response to changes in tax structures.                             |
| Government             | Fiscal Authority                     | Formulate and implement tax policies, adjusting tax rates to achieve growth, equity, and fiscal‐sustainability objectives.                                     |
| Firm                | Perfect Competition             | Model wages and prices determined by supply and demand, capturing how taxes transmit through labor and goods markets as the policy feedback channel.            |
| Bank | No-Arbitrage Platform | Build savings‐investment return mechanisms, simulating interest‐rate changes and their effects on household asset allocation and capital‐market equilibrium. |

### Individual → Ramsey Model

* Households, as rational economic agents, make decisions on labor supply, consumption, and savings based on utility‐maximization principles. Changes in tax policy alter their marginal choices, providing essential micro‐level feedback that underpins government policy optimization.

### Government → Fiscal Authority

* The government, as the architect and executor of tax policy, adjusts tax‐rate structures in line with economic conditions and macro objectives. Its core function is to achieve a dynamic balance among growth, equity, and fiscal sustainability.

### Firm → Perfect Competition 

* Wages and prices are determined by market mechanisms, reflecting the transmission channels of tax policy through labor and goods markets.

### Bank → No-Arbitrage Platform

* Tax policies influence saving and investment behaviors; the financial system, via interest‐rate mechanisms, feeds back changes that restore economic equilibrium.

---

## 3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type    | Role Description                                                                                                                     |
| ------------------------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Rule-Based Agent | Use fixed behavioral rules to create a stable experimental environment, facilitating evaluation of policy marginal effects.          |
| Government             | RL Agent         | Employ reinforcement learning to explore the tax‐policy space, dynamically optimizing GDP, income distribution, and fiscal balance. |
| Firm                 | Rule-Based Agent | Model wages and employment reacting to tax changes via supply–demand mechanism rules.                                               |
| Bank | Rule-Based Agent | Adjust interest rates and capital returns based on rules governing savings behavior and tax burdens.                                 |

### Individual → Rule-Based Agent

* Household behavior is modeled with fixed decision rules (e.g., “if wages rise, consumption increases; if tax burden rises, labor supply decreases”) within a life-cycle framework. This design provides a stable, interpretable feedback environment during the government’s tax-optimization process, making it easier to isolate policy marginal effects.

### Government → RL Agent

* The government agent uses reinforcement learning to explore optimal tax-policy combinations (labor tax, capital tax, consumption tax, etc.) over multiple simulation rounds, updating its strategy based on environmental feedback. Its objective is to find dynamically optimal solutions under multiple constraints (e.g., GDP, Gini coefficient, fiscal balance), demonstrating adaptive policy learning.

### Firm → Rule-Based Agent

* Market mechanisms adjust wages, prices, and other variables according to supply–demand and tax-burden rules, making rule-based modeling appropriate.

### Bank → Rule-Based Agent

* Interest-rate and capital-return rules dynamically adjust in response to government policies and household behavior.

---

## 4. Illustrative Experiments

### Experiment 1: RL-Based Optimal Tax-Structure Policy Trial

* **Experiment Description:** In the simulated economic environment, the government can use a reinforcement learning (RL) agent to automatically learn the optimal mix and rates of labor-income, consumption, and capital taxes. In this experiment, we compare the government’s use of reinforcement learning methods (DDPG), economic rule-based methods (Seaz Tax), and the real tax rates set by the U.S. federal government (2022), and discuss the impact of different tax rate settings and tax structures on the macroeconomy.
* **Involved Social Roles:**
  * *Individual:* Ramsey Model
  * *Government:* Treasury Department
  * *Market: ​*Perfectly Competitive Market
  * *Financial Institutions:* Arbitrage-Free Financial Intermediaries
* **AI**​**​ Agents:**
  * *Individual:* Rule-Based Agent
  * *Government:* RL Agent/Rule-Based Agent/Data-Based Agent
  * *Market: ​*Rule-Based Agent
  * *Financial Institutions: ​*Rule-Based Agent
* **Experimental Variables:**
  * Different government department agents and their corresponding tax structures.
  * Macro indicators: GDP, wealth Gini coefficient, average household wealth
* **Visualized Experimental Results:**
![Fiscal Q4 P1](../img/Fiscal%20Q4%20P1.png)

​**Figure 1**​: Comparison of household wealth under different tax policies at T=192years. The tax system trained by the RL-Agent (red bar) results in higher average household wealth, with the average wealth of the wealthier households (blue bar) significantly higher than the other two tax systems. The simulated economy using the Seaz rule (left chart) has the second highest average household wealth, while the simulated economy using the real U.S. tax system (right chart) shows the lowest average household wealth.

![Fiscal Q4 P2](../img/Fiscal%20Q4%20P2.png)

​**Figure 2**:At T=192 years, the phenomenon reflected in household wealth is identical, where the tax system trained by the RL-Agent maximizes consumption across different wealth tiers of households.

![Fiscal Q4 P3](../img/Fiscal%20Q4%20P3.png)

​**Figure 3**​: Comparison of long-term GDP growth levels under different tax policies. The RL-Agent economy has the fastest GDP growth (green line), followed by the economy with the Seaz rule (blue line). The simulated economy using the real U.S. tax system experiences the lowest household consumption and GDP growth (yellow line).

![Fiscal Q4 P4](../img/Fiscal%20Q4%20P4.png)

​**Figure 4**​: As time progresses, all tax strategies significantly reduce the wealth gap. However, when the government uses the RL-Agent (green line), the long-term wealth disparity remains relatively higher.

  
