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

| Social Role | Selected Type       | Role Description                                                                                                       | Observation                                                                                                                                          | Action                                                       | Reward                                               |
| ----------- | ------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| Individual  | OLG Model           | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals. | $$o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$$<br/>Private: assets, education, age<br/>Global: distributional statistics                                  | — (same as above)<br/>*OLG*: old agents $$\lambda_t^i = 0$$ | — (same as above)<br/>OLG includes pension if retired |
| Firm       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                 | /                                                                                                                                                    | /                                                            | Zero (long-run)                                      |
| Bank       | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives.   | /                                                                                                                                                    | No rate control                                              | No profit                                            |


---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
The Overlapping Generations framework simulates ​**age‑specific labor and consumption choices**​. Preferences for work–life balance vary across the life cycle—young workers may ​**strive for career growth**​, whereas middle‑aged and older cohorts prioritize ​**health and family time**​.

**Government → Not Applicable**  
In the work–life balance experiments, the government must coordinate across multiple departments, for example: the Ministry of Labor enforces maximum working‐hour limits; the pension authority calibrates relevant pension regulations; and the tax authority adapts fiscal rules to the evolving social environment.

**Firm → Perfect Competition**  
Firms compete for talent through ​**wages and flexible work policies**​. Workers choose environments that best match their balance preferences, forcing companies to adapt HR strategies.

**Bank →Non-Profit Platform **  
Provide ​**life‑cycle financial products**​—retirement accounts, health insurance, liquidity support. As work–life patterns shift, so do saving needs and demand for these services.

---

## **​3. Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | RL Agent          | Simulate households’ work–life balance decisions using reinforcement learning.                                                                                                                                                 |
| Firm                 | Rule‑Based Agent | Firms adapt hiring strategies and workplace arrangements (e.g., offering flexible hours) according to labor-market supply–demand dynamics and worker preferences, exhibiting predictable behavior.                              |
| Bank | Rule‑Based Agent | Financial institutions deliver standardized life-cycle services—such as savings advice or insurance products—based on individuals’ life-cycle stage and income volatility, making them well-suited for rule-based simulation. |

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
* **Visualized Experimental Results:**

![Individual Q4 P1](../img/Individual%20Q4%20P1.png)

**Figure 1**: When households adopt a “work–life balance” strategy (green line), aggregate GDP is lower than under the standard work regime (blue line), but after year 60 the gap narrows and the two GDP paths converge.

![Individual Q4 P2](../img/Individual%20Q4%20P2.png)

**Figure 2**: When households adopt a “work–life balance” strategy (green line), social welfare increases and remains elevated over time.

* By maximizing individual utility and choosing a work–life balance lifestyle, households experience slower economic growth in the short term compared to the baseline scenario, but social welfare rises markedly. In the long run, the GDP gap between the two scenarios narrows, while the welfare gains from the work–life balance approach persist.

