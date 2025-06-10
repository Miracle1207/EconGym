# Q4: How does work-life balance impact well-being?

## 1. Introduction

### 1.1  **Introduction to Work–Life Balance**

Work–life balance denotes a sustainable, coordinated state in which individuals allocate time and energy between career pursuits and personal life. In modern economies—particularly among​**​ Generation Z**​—concern for work–life balance has surged. Under high-intensity work regimes, employees are increasingly aware of the long-term costs of overextending their physical and mental health. Survey evidence shows that younger workers increasingly prioritize flexible scheduling, remote work options, and personal autonomy, seeking to balance meaningful work with overall life satisfaction.

Real-world examples include:

* China’s “996” work schedule has sparked intense debate, leading many young people to adopt the “lying flat” lifestyle or migrate away from first-tier cities.
* Several European countries are piloting four-day workweeks to enhance well-being and sustain productivity over the long run.
* Companies are increasingly embedding employee well-being metrics into formal HR policies.

### **1.2  Research Questions**

In economics, utility is the key variable for measuring individual well‐being and satisfaction. Although excessive labor may boost short‐term income, it can reduce total utility in the long run due to health deterioration, mental fatigue, or strained family relationships. Based on an economic simulation platform, this study investigates “How does work–life balance affect aggregate output and individual utility over the life cycle?” Specifically, we address:

* How does a more balanced work pattern impact aggregate social output?
* Can a “work–life balance” strategy ​**yield lifetime utility gains**​?

### **1.3  Research Significance**

* **Reforming Modern Work Regimes:**  Does today’s work system need reform to satisfy a new generation’s demand for quality of life? How can we optimally structure work hours and modalities to maximize both social welfare and individual utility?
* **Labor and Behavioral Economics in Well-Being:**  This experiment quantifies well-being using a work–life balance framework, assessing whether higher measured well-being leads to greater realized individual utility and, in turn, fosters broader social development.

---

## **​2. Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                                                                                                                     |
| ------------------------ | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | OLG Model                             | To simulate variations in work-leisure preferences across life-cycle stages (youth, middle age, older age) and capture the trajectory of long-term utility evolution.                |
| Firm                 | Perfect Competition         | Firms adjust employment arrangements (e.g., flexible working hours) to attract labor in line with worker preferences, reflecting market supply–demand equilibrium mechanisms.       |
| Bank | No-Arbitrage Platform | To offer life-cycle financial products (retirement savings, insurance, etc.) that allow individuals to smooth consumption and hedge risks under changing work–life balance regimes. |

### Individual → **Overlapping Generations (OLG) Model**

* The Overlapping Generations framework simulates ​**age‑specific labor and consumption choices**​. Preferences for work–life balance vary across the life cycle—young workers may ​**strive for career growth**​, whereas middle‑aged and older cohorts prioritize ​**health and family time**​.

### Government → Not Applicable

* In the work–life balance experiments, the government must coordinate across multiple departments, for example: the Ministry of Labor enforces maximum working‐hour limits; the pension authority calibrates relevant pension regulations; and the tax authority adapts fiscal rules to the evolving social environment.

### Firm → Perfect Competition

* Firms compete for talent through ​**wages and flexible work policies**​. Workers choose environments that best match their balance preferences, forcing companies to adapt HR strategies.

### Bank →No-Arbitrage Platform

* Provide ​**life‑cycle financial products**​—retirement accounts, health insurance, liquidity support. As work–life patterns shift, so do saving needs and demand for these services.

---

## **​3. Selected Agent Algorithms**

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type     | Role Description                                                                                                                                                                                                                 |
| ------------------------ | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | RL Agent          | Simulate households’ work–life balance decisions using reinforcement learning.                                                                                                                                                 |
| Firm                 | Rule‑Based Agent | Firms adapt hiring strategies and workplace arrangements (e.g., offering flexible hours) according to labor-market supply–demand dynamics and worker preferences, exhibiting predictable behavior.                              |
| Bank | Rule‑Based Agent | Financial institutions deliver standardized life-cycle services—such as savings advice or insurance products—based on individuals’ life-cycle stage and income volatility, making them well-suited for rule-based simulation. |

### **Individual → RL Agent**

* Reinforcement learning excels at modeling optimal household​**​ decision‐making**​. Here, each household maximizes utility to choose its work–life balance, more accurately reflecting the pursuit of “work–life balance.” In contrast, rule‐based agents offer no flexibility for parameter adjustment, and purely data‐driven agents can only infer strategies from historical observations.

### **Firm → Rule-Based Agent**

* In a Perfectly Competitive Market, firms tailor employment terms (e.g., flexible working hours) to labor‐market preferences. These adaptive behaviors can be effectively captured by rule‐based agents.

### **Bank → Rule-Based Agent**

* Financial institutions offer a range of products (e.g., liquidity support when working hours fall) based on labor‐market conditions. Rule‐based agents adjust financial strategies according to predefined rules to fulfill simulation requirements.

---

## **​4. Illustrative Experiment**

### Experiment 1:Impact of Work–Life Balance on Society

* **Experiment Description:**
  Simulate individuals’ strategies for allocating work and leisure across different life‐cycle stages, and measure the long‐term impact on aggregate social production.
* **Involved Social Roles:**
  * *Individual: ​*OLG model
  * *Firm:* Perfectly Competitive Market
* **AI Agents:**
  * *Individual: ​*BC Agent/RL Agent
  * *Firm: ​*Rule‐Based Agent
* **Experimental Variables:**
  * Whether households adopt a work–life balance strategy (Behavior Cloning vs. RL Agent)
  * Level of socio‐economic growth
  * Level of social welfare
* **Visualization of Results:**

![Individual Q6 P1](../img/Individual%20Q6%20P1.png)

**Figure 1**: When households adopt a “work–life balance” strategy (green line), aggregate GDP is lower than under the standard work regime (blue line), but after year 60 the gap narrows and the two GDP paths converge.

![Individual Q6 P2](../img/Individual%20Q6%20P2.png)

**Figure 2**: When households adopt a “work–life balance” strategy (green line), social welfare increases and remains elevated over time.

* By maximizing individual utility and choosing a work–life balance lifestyle, households experience slower economic growth in the short term compared to the baseline scenario, but social welfare rises markedly. In the long run, the GDP gap between the two scenarios narrows, while the welfare gains from the work–life balance approach persist.

