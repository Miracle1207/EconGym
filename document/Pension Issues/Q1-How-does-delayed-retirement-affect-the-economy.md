# Q1: How does delayed retirement affect the economy?

## 1. Introduction

### 1.1 Delay Retirement Policy

The Delay Retirement Policy refers to raising the statutory retirement age so that workers remain active in the **labor market ​**for a longer period, thereby affecting the economic behavior of individuals, firms, governments, and financial markets.

### 1.2 Background of the Study

Globally, population aging has become increasingly severe, placing substantial fiscal pressure on ​**pension systems**​. Many countries are considering or have already implemented delayed retirement policies to relieve pension payment burdens, boost labor supply, and promote economic growth.

### 1.3 Research Questions

This study uses an economic simulation platform to investigate the **“economic impacts of delayed retirement policies,”** specifically examining:

* **GDP**​**​ Effects:** Does delaying retirement contribute to economic growth?
* **Wealth Distribution:** How does the distribution of wealth between older and younger cohorts change?
* **Pension System Sustainability:** How does delayed retirement affect government finances and pension disbursements?
* **Capital Market Stability:** How are household savings and investment behaviors influenced?

### 1.4 Research Significance

* **Policy Guidance:** Considering that delayed retirement may yield diverse economic effects—such as impacts on labor markets, wage levels, capital markets, and social equity—this research provides strong guidance for policymaking.
* **Balancing Equity and Efficiency:** The study helps clarify how delayed retirement redistributes wealth and opportunities across generations, thereby supporting policy designs that reconcile social fairness with economic efficiency.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                              |
| ------------------------ | --------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Individual            | OLG Model   | Study labor supply, consumption, and saving behavior across different age cohorts             |
| Government             | Pension Authority            | Formulate and adjust retirement-age policy, affecting pension expenditures and fiscal balance |
| Government             | Fiscal Authority                   | Evaluate the impact of delayed retirement on tax revenue and social security funds            |
| Firm                 | Perfect Competition          | Analyze how firms adjust wages, hiring, and production strategies                             |
| Bank | No-Arbitrage Platform | Investigate how capital markets are influenced by delayed retirement                          |

### **Individual → Overlapping Generations (OLG) Model**

* The OLG model can simulate labor supply, consumption, and saving decisions across age cohorts, making it ideal for studying the impact of delayed retirement.

### **Government → Pension Authority & Fiscal Authority**

* The Pension Policy Department directly manages retirement policies, while the Fiscal Authority analyzes fiscal sustainability; together they comprehensively assess policy outcomes.

### **Firm → Perfect Competition**

* Real-world markets often approximate perfect competition, facilitating observations of how firms adjust wages and hiring strategies.

### **Bank → No-Arbitrage Platform**

* To study household saving and capital market fluctuations, and to evaluate the effects of delayed retirement on financial stability.

---

## ​3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type    | Role Description                                                                                         |
| ------------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------- |
| Individual             | Rule-Based Agent | Define rules to simulate consumption, saving, and retirement decisions across different age cohorts.     |
| Government             | Rule-Based Agent | Model the real‐world policy rules used by governments to design appropriate delayed retirement schemes. |
| Bank                 | Rule-Based Agent | Specify firms’ labor demand, wage-setting, and production decision rules.                               |
| Firm | Rule-Based Agent | Set the investment behaviors and interest‐rate adjustment mechanisms of financial institutions.         |

### **Individual → Rule-Based Agent**

* Predefined rules simulate labor supply, consumption, and saving decisions by age group—e.g., how working‐age individuals balance wage income and savings, and how retirees adjust consumption patterns. Rule-Based Agents suit modeling stable decision rules (e.g., age‐based consumption–saving trade-offs). In contrast, RL agents demand large datasets to train and may struggle to capture long‐term rational decision patterns.

### **Government → Rule-Based Agent**

* Rules derived from historical data predict how policy adjustments affect finances at different retirement ages. Compared to RL Agents, Rule-Based Agents offer stronger policy consistency when evaluating fixed retirement policies.

### **Firm → Rule-Based Agent**

* Economic theory specifies how firms adjust production and wages based on labor supply and demand. Rule-Based Agents model firm behavior in competitive markets, whereas RL agents may introduce non-rational actions that destabilize the simulation.

### **Bank → Rule-Based Agent**

* The model describes interest rate and return adjustments—e.g., how changes in rates affect savings returns and household portfolio choices. Rule-Based Agents precisely set market adjustment mechanisms; data-driven methods may be limited by **historical data ​**and perform poorly under novel economic conditions.

---

## **​4.​**​**Illustrative Experiments**

### Experiment 1: Impact of Different Retirement Ages on Economic Growth

* **Experiment Description:**
  The simulation platform implements retirement policies and compares economic trajectories under retirement ages of 60, 65, and 70.
* **Involved Social Roles: ​**
  * *Individual: ​*OLG Model
  * *Government: ​*Pension Authority & Fiscal Authority
* **AI**​**​ Agents:**
  * *Individual:* Rule-Based
  * *Government: ​*Rule-Based
* **Experimental Variables:**
  * Retirement age (60, 65, 70)
  * Total GDP and its growth trend
* **Visualized Experimental Results：**

![Pension Q2 P1](../img/Pension%20Q2%20P1.png)

**Figure 1:** The yellow, green, and blue lines represent GDP trajectories in a simulated economy of 1,000 households under statutory retirement ages of 70, 65, and 60, respectively. It is observed that economies with earlier retirement ages exhibit higher total GDP, although the difference is less pronounced when household count is 100.

* Delaying retirement does not raise aggregate output in the long run. One reason may be that extended working years reduce households’ time and willingness to consume, interrupting their life-cycle consumption and saving plans.

