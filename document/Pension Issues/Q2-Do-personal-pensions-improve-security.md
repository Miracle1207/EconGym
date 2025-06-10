# Q2: Do personal pensions improve security?

## ​1.​ Introduction

### 1.1 Definition of the Issue

The **Individual Pension Policy** refers to government-sponsored personal retirement saving schemes designed to supplement the public pension system. These schemes typically include tax incentives, investment incentives, and mandatory contributions.

### 1.2 Social Practice of Individual Pensions

* **United States:** A mature individual pension framework has existed since the 1970s with the establishment of the 401(k) plan, which encourages citizens to save for retirement on a tax-deferred basis.
* **China:** The individual pension system was formally established in 2022 and rolled out nationwide in 2024. As the “third pillar” of the retirement framework, it plays a key role in reinforcing the overall pension structure.

### 1.3 Research Questions

Using an economic simulation platform, this study examines the ​**effectiveness of individual pension policies**​, focusing on:

* **Aggregate Economic ​**​​**Output**​: How do individual pension policies influence total social GDP?
* **Household Income:** What is the impact on household earnings?
* **Capital Stock and Returns:** How are national capital accumulation and investment returns affected?

### 1.4 Research Significance

* **Policy Reference for an Aging Society:**  Under demographic aging, public pension funds face growing fiscal strain. By promoting individual pension plans, governments can bolster personal retirement reserves and ease the burden on public finances.
* **Guidance for Personal Retirement Investment:**  Individuals must decide whether to participate in personal pension schemes and how much to contribute. This study helps households make more informed decisions when engaging with these plans.

---

## ​2. ​Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                                                |
| ------------------------ | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Individual             | OLG Model                             | Simulate individual and household consumption, saving, and investment decisions                                 |
| Government             | Pension Authority            | Formulate and adjust individual pension policies, affecting saving and consumption behavior                     |
| Firm                 | Perfect Competiton          | Assess how firms react to changes in individual pension policies, such as wage setting and investment decisions |
| Bank | No-Arbitrage Platform | Model how capital markets absorb and allocate individual pension savings                                        |

### **Individual → Overlapping Generations (OLG) Model**

* Captures how pension policies affect savings, investment, and retirement planning across age cohorts. The OLG framework yields precise insights into intergenerational behavior.

### **Government →  Pension Authority**

* Designs and adjusts individual pension regulations, directly influencing personal saving, consumption, and investment. Focuses specifically on pension rules, whereas the Ministry of Finance oversees the broader government budget.

### **Firm → Perfect Competiton**

* Firms’ production, investment, and wage decisions respond to changes in personal pension incentives. Perfectly Competitive Market ensures that increased households' savings are fully reflected in capital-market prices.

### **Bank → No-Arbitrage Platform**

* Channel pension contributions into various financial investments and determine capital allocation. Arbitrage-Free Financial Institutions simulate the impact of different investment strategies, evaluating how pension reforms shock capital markets.

---

## 3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type                             | Role Description                                                                              |
| ------------------------ | ------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Individual             | Behavior Cloning Agent / Rule-Based Agent | Learn or define real individual behavior patterns to simulate saving and consumption decisions |
| Government             | Data-Based Agent / Rule-Based Agent       | Formulate individual pension policies based on historical data or adjust policy parameters    |
| Firm                 | Rule-Based Agent                          | Simulate firms’ investment and wage‐setting decisions                                       |
| Bank | Rule-Based Agent                          | Assess how pension savings influence capital markets                                          |

### **Individual → Behavior Cloning (BC) Agent / Rule-Based Agent**

* The impact of individual pension policies on household saving and consumption behaviors needs to be simulated based on actual household decision patterns.A Behavior Cloning Agent, trained on real-world household decision data, can more accurately mimic savings–consumption choices across income and age cohorts.
* A Rule-Based Agent can explicitly encode life-cycle saving and consumption rules—e.g., setting age-, income-, and retirement-expectation–specific marginal propensities to consume and save. When micro-level data or longitudinal behavioral tracking are lacking, rule-based methods grounded in established literature (such as the life-cycle hypothesis and behavioral economics models) ensure both interpretability and controllability.
* By contrast, reinforcement learning–based agents may converge to implausible “optimal” strategies.

### **Government → Data-Based Agent / Rule-Based Agent**

* The formulation of government pension policies depends on empirical data—such as tax revenues, investment return rates, and the effects of retirement-age changes.
* A Data-Based Agent is well suited for calibrating policy parameters using historical datasets and forecasting long-term fiscal sustainability.
* A Rule-Based Agent is appropriate for implementing a fixed individual pension framework, which is particularly practical in environments where the personal pension system remains under development.

### **Firm → Rule-Based Agent**

* Models firms’ investment and wage-setting rules (e.g., adjusting capital outlays based on marginal returns). Rule-Based Agents are cost-effective and align with economic theory compared to reinforcement learning.

### **Bank → Rule-Based Agent**

* Sets target returns and risk-management strategies for pension fund investments under market yield and macroeconomic constraints. Rules ensure predictable simulations; data-driven or RL methods may be limited by historical data or computational complexity.

---

## **​4.​**​**Illustrative Experiment**

### Experiment 1: How do tax incentives for personal pension savings impact social GDP?

* **Experiment Description:**
  Simulate the impact of varying tax-exemption thresholds on the GDP of the virtual economy.
* **Involved Social Roles:**
  
  * *Individual: ​*OLG Model
  * *Government: ​*Pension Authority
* **AI**​​**​ Agents**​:
  
  * *Individual: ​*Rule-Based Agent
  * *Government:* Data-Based Agent
* **Experimental Variables:**
  
  * Tax-exemption levels (e.g., +10%, +15%, +20%)
  * Long-term GDP performance under each scenario
* **Visualized Experimental Results：**
![Pension Q3 P1](../img/Pension%20Q3%20P1.png)

  **Figure 1:** GDP trajectories for household populations of 1,000 and 100 under differing exemption levels. Higher exemptions correlate with lower GDP.
  
  * Higher tax incentives for individual pension savings are associated with lower aggregate GDP, possibly because such incentives prompt households to sacrifice some consumption in favor of increased pension contributions, thereby reducing overall demand for goods and leading to a decline in GDP.

---

### Experiment 2: How do tax incentives for pensions affect household income?

* **Experiment Description:**
  Simulate effects of varying tax-exemption levels on long-term individual incomes.
* **Involved Social Roles:**
  
  * *Individual: ​*OLG Model
  * *Government: ​*Pension Authority
* **AI**​​**​ Agents**​:
  
  * *Individual: ​*Rule-Based Agent
  * *Government:* Data-Based Agent
* **Experimental Variables:**
  
  * Tax-exemption levels (+10%, +15%, +20%)
  * Long-term income comparisons across age and income deciles
* **Visualized Experimental Results：**
![Pension Q3 P2](../img/Pension%20Q3%20P2.png)

  **Figure 2:** Bar charts showing income distributions by age group (left) and income bracket (right) under different exemption levels. Lower exemptions lead to larger income declines.
  
  * Generous tax incentives for personal pension contributions are associated with higher long-term household incomes: by increasing households’ willingness to participate in pension schemes, these exemptions channel savings into banks and financial markets, which in turn support greater income growth over time.

