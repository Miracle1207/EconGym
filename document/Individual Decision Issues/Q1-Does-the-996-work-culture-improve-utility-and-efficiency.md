# Q1: Does the “996” work culture improve utility and efficiency?

## **1.Introduction**


### **1.1  Introduction to the "996" Phenomenon and Societal Concerns**

The term **"996"** is a humorous expression in Chinese internet culture, referring to an overloaded work system: working from ​**9 a.m. to 9 p.m.**​, six days a week, which results in ​**72 working hours per week**​, far exceeding international labor standards (according to the ​**International Labour Organization (ILO)**​, working more than **48 hours per week** is considered "long working hours"). Excessive working hours can lead to various issues, such as:

* **Health**​​**​ Issues for Workers**: Long-term high-intensity work can cause chronic fatigue, anxiety, depression, and other health problems.
* **Deterioration of Work-Life Balance**: The time available for personal social activities and family life is greatly reduced, potentially decreasing overall life satisfaction.
* **Decline in Productivity and Creativity**: Research indicates that working beyond reasonable hours does not necessarily improve productivity, and excessive fatigue can lead to **decision-making errors** and decreased efficiency.

### **1.2 Research Questions**

This study focuses on the following key questions:

* **The Impact of Excessive Working Hours on Society**: How do excessive working hours affect macroeconomic variables such as societal output, income inequality, and social consumption?
* **How "996" Affects Long-Term Personal Utility**: Does the "996" system lead to a decline in personal long-term utility and changes in lifestyle?

### **1.3 Research Significance**

* **Evaluation of the Family and Social Utility of Overtime Work:**  This research investigates whether **excessive working hours** can genuinely enhance ​**long-term personal utility**​, or if it only brings ​**short-term economic benefits**​. Additionally, the study considers whether individuals' long working hours can truly contribute to the increase of ​**societal output**​.
* **Policy Guidance for Labor Laws:**  If excessive working hours have severe negative effects on **personal utility** and ​**social development**​, the government should implement **stricter** and more ​**comprehensive labor policies**​. This study, by examining the societal impact of different working hours, aims to provide **data references** for the improvement of ​**labor laws**​.

---

## **2.Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                                                                                                    |
| ------------------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | OLG Model                             | Age is a key factor influencing work duration, consumption behavior, and health status.                                                                             |
| Firm                 | Perfectly Competition          | Using the perfectly competitive market assumption, this examines the impact of the "996" model on business labor costs, output efficiency, and innovation capacity. |
| Bank | No-Arbitrage Platform | Studies whether increased working hours affect the capital markets, such as personal savings rates, loan demand, and other financial variables.                     |

---

### Rationale for Selected Roles

**Individual  → Overlapping Generations (OLG) Model**  
An individual's **age** is a key determinant of work duration, consumption behavior, and health status. Under the ​**Overlapping Generations (OLG) model**​, individuals at different age stages (young labor force, middle-aged labor force, retirees) have distinct preferences and constraints, allowing for a more precise analysis of the long-term effects of excessive work hours on personal welfare. For instance, younger individuals may be inclined to work overtime to accumulate wealth, while middle-aged individuals may focus more on the balance between work and family. Retirees, on the other hand, are influenced by previous savings and government pension policies.

**Government→ No Specific Role**  
This study focuses on the behavioral responses of households and firms to increased working hours, without introducing government-driven policy adjustments. The aim is to isolate the effects of extended work schedules (such as the "996" model) on the economy and individual welfare. Government interventions like taxation, regulation, or welfare policies are intentionally excluded to ensure that the outcomes reflect endogenous behavioral and market responses alone.

**Firm  → Perfect Competition**  
Firms in the market primarily seek ​**profit maximization**​, and thus they may adopt the **"996" model** to boost short-term productivity. However, excessively long working hours may lead to a decline in ​**labor productivity**​, hinder **innovation** capabilities, and potentially cause an imbalance in the labor market's supply and demand. Companies need to strike a balance between work hours and production efficiency, while also considering **worker welfare** and ​**talent attraction**​.

**Bank → No-Arbitrage Platform**  
The primary role of **financial institutions** is to provide ​**loans**​, ​**investment opportunities**​, and ​**asset management**​. Long working hours may influence **workers' consumption** and ​**savings decisions**​. For example, excessive working hours may suppress consumption, leading to an increase in ​**household savings rates**​, which in turn can impact the ​**capital markets**​.

---

## **3.Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Behavior Cloning Agent | Simulate individual behavior characteristics as working hours increase, based on historical data.                                  |
| Government             | RL Agent               | The government formulates optimal policies and flexibly adjusts labor regulations or interventions.                                |
| Firm                 | Rule-Based Agent       | In a perfectly competitive market, firms do not have control over product pricing, and their strategies follow fixed market rules. |
| Bank | Rule-Based Agent       | Rule-Based Agent ensures that the behavior of financial institutions remains relatively stable.                                    |


## **4.Illustrative Experiment**

### **Experiment 1: The Impact of Excessive Working Hours on Society**

* **Experiment Description**:
  Analyze how the "996" labor model impacts macro-level society.
* **Involved Social Roles:**
  * *Individual: ​*OLG Model
  * *Firm:* Perfectly Competitive Market
  * *Bank:* No-Arbitrage Platform
* **AI Agents**:
  * *Individual: ​*BC Agent
  * *Firm: ​*Rule-Based Agent
  * *Bank: ​*Rule-Based Agent
* **Experimental Variables**:
  * Weekly Working Hours (comparison of 40 hours per week and 60 hours per week).
  * Societal GDP Level and Growth Trend.
  * Societal Income Inequality (e.g., changes in the Gini coefficient).

```Python
#The maximum weekly working hours are extended to 60 or 72 hours per week.
#working_hours_max representing the maximum annual working hours.
Working_hours_max= 3120
Working_hours_max= 4160
```

* **​ Visualized Experimental Results：**

![Individual Q1 P1](../img/Individual%20Q1%20P1.png)


**Figure 1**: The "996" work model (green line) leads to a long-term increase in societal output.

![Individual Q1 P2](../img/Individual%20Q1%20P2.png)

![Individual Q1 P3](../img/Individual%20Q1%20P3.png)

**Figure 2, Figure 3**: The "996" work model (green line) slightly reduces the wealth Gini coefficient, indicating a slight decrease in the wealth gap in society, but it has little to no effect on income inequality.

* The "996" work model can effectively increase societal GDP output starting from the first period and continue this growth in the long term.
* The impact of "996" on societal income inequality is relatively small, with the long working hours model slightly reducing the wealth gap.

