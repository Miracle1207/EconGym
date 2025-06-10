# Q2: How does age affect consumption patterns?

## 1.Introduction

### 1.1 **Consumption‑Propensity Differences Across Age Cohorts**

Consumption Propensity Differences refer to systematic variations in spending behavior, patterns, and preferences across different age cohorts. Specifically, this encompasses differences in:

* **Marginal Propensity to Consume:** ​​​ (MPC) the proportion of disposable income devoted to consumption
* **Determinants of Consumption Decisions:** such as income, wealth, expected lifespan, and health status
* **Spending Habits and Priority of Expenditures**

Dividing adults into four groups:Youth (18–30), Early Middle Age (31–45), Late Middle Age (46–65), and Senior (> 65).We can reasonably infer the following patterns based on economic theory and observed social phenomena:

* **Youth (18–30):**  Young adults prioritize immediate gratification and experiential spending. They often depend on short-term income streams and credit-financed consumption, resulting in a high marginal propensity to consume.Driven by quality‐of‐life demands, they allocate a larger share of disposable income to consumption.
* **Early-middle-aged (31–45):**  With growing family responsibilities, this cohort’s MPC declines. They begin saving for the future, particularly for education, housing, and retirement. Consumption is more grounded in stable income and long‐term household planning.
* **Late-middle-aged (46–65):**  Individuals in this stage are typically at their career peak and focus increasingly on retirement and health security. Their MPC decreases further as they prioritize savings for retirement and future needs, leading to more stable spending and reduced non‐essential expenditures.
* **Elderly (65 and older):**  Seniors typically live on fixed pensions or accumulated savings, concentrating consumption on daily necessities and healthcare. With lower income, their MPC is low, and spending centers on essential goods and medical support.

### 1.2**​ Research Questions**

This study leverages an economic simulation platform to examine differences in consumption propensities across age cohorts. Specifically, it covers:

* **Consumption and Saving Behaviors:** Investigate how marginal propensity to consume and saving rates vary by age group, and briefly analyze the potential societal implications of these differences.
* **Income–Consumption Relationship:** Explore how changes in income influence consumption behavior in different age cohorts.

### 1.3 **Research Significance**

Population‐structure shifts are a critical determinant of both macroeconomic outcomes and household‐level decisions. With global population aging intensifying, understanding consumption‐behavior differences across age cohorts is of paramount importance:

* **Consumption Behavior in an Aging Society:**  As the world’s population ages rapidly, this experiment uses consumption‐propensity differences to shed light on how demographic changes influence aggregate demand and economic growth.
* **Corporate Marketing Strategies:**  Technological transformations are profoundly reshaping consumption habits across age groups. Insights from this study can help firms tailor product and marketing strategies to better align with the preferences of each cohort.

## 2.Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                | Role Description                                                                                                                              |
| ------------------------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | OLG Model / Ramsey Model     | The OLG framework captures life‑cycle effects on consumption; the Ramsey model distinguishes age cohorts under an infinite‑horizon setting. |
| Firm                 | Perfect Competition | Adjusts product supply and prices in response to age‑specific demand shifts.                                                                 |
| Bank | Commercial Banks             | Provide savings and loan services, influencing liquidity constraints and intertemporal consumption preferences across age groups.             |

### **Individual →  Overlapping Generations (OLG) Model /  Ramsey Model**

* From the OLG perspective, differences in consumption propensities are fundamentally an intergenerational issue.The OLG framework explicitly **distinguishes consumption decisions, saving behaviors, and budget constraints across age cohorts.**
* In contrast, while the infinite‐horizon Ramsey model assumes perpetual individual lifespans, it can still differentiate age‐related variations in consumption propensity.

### **Government →  No Specific Role**

* This study focuses on the endogenous differences in consumption and saving behaviors across age cohorts, driven by life-cycle characteristics such as income, health status, and household composition. It **deliberately excludes any form of fiscal intervention (e.g., taxation, subsidies, or social transfers) in order to isolate and observe the pure behavioral dynamics of different age groups.** The absence of government policies ensures that observed consumption-propensity differences stem solely from individual decision-making under varying demographic and economic conditions.

### **Firm →Perfect Competition**

* To study consumption‐propensity differences, we need to **examine how markets respond to varying demand across age cohorts**. A perfectly competitive market most directly reflects the impact of demand shifts on prices and supply, making it ideal for analyzing age‐specific consumption effects. In contrast, monopolistic or oligopolistic structures can distort price signals and mask true consumer behavior, rendering them unsuitable for this research.

### **Bank → Commercial Banks**

* Commercial banks, by offering deposit and lending services, directly shape consumption–saving decisions across age groups. Younger cohorts may rely more on borrowing (e.g., student loans, mortgages), whereas seniors focus on savings yields and pension management. By comparison, arbitrage‐free financial intermediaries are more concerned with overall capital‐market efficiency and interact less directly with individual consumers, making them less suitable for this study.

---

## 3.Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type     | Role Description                                                                                                                          |
| ------------------------ | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | RL Agent          | Each household maximizes utility by using reinforcement learning to optimize age‑specific consumption‑saving decisions and preferences. |
| Government             | Rule‑Based Agent | Executes taxation and transfer policies according to explicit fiscal rules.                                                               |
| Firm                 | Rule‑Based Agent | Adjusts prices and supply under predefined rules to match demand shifts and keep the market in equilibrium.                               |
| Bank | Rule‑Based Agent | Delivers standardized financial services—uniform risk assessment and product pricing—for all age cohorts.                               |

### **​Individual →​RL Agent**

* Differences in **consumption propensities** across age cohorts involve complex **utility‐maximization ​**problems. Households must continually adjust their consumption–saving strategies to accommodate changing life‐cycle stages. Reinforcement learning agents can dynamically explore and learn optimal consumption–saving policies, adapting to income shocks, risk preferences, and evolving needs at each age stage. Especially under uncertainty (e.g., health risks, income shocks), RL agents develop forward‐looking consumption strategies that more realistically simulate how families adjust spending based on experience.

### **Government → Rule‐Based Agent**

* Fiscal policy follows well‐defined rules and regulations—such as tax codes, welfare eligibility criteria, and budget‐balance requirements. Rule‐based agents accurately simulate these institutional decision processes, including progressive tax schedules, social‐security benefit formulas, and transfer‐payment mechanisms.

### **Firm → Rule‐Based Agent**

* In a perfectly competitive market, equilibrium emerges from supply–demand interactions. Rule‐based agents can isolate and clearly trace how changes in consumption patterns impact the market without being confounded by more complex market dynamics.

### **Bank → Rule‐Based Agent**

* When studying ​**consumption‐propensity differences**​, financial institutions primarily provide stable service infrastructure rather than optimize strategies. Rule‐based agents model standardized banking procedures—loan approval criteria, interest‐rate setting formulas, and account‐management rules—each grounded in explicit risk‐management frameworks and regulatory mandates. Using rule‐based agents ensures consistency and predictability of financial services, keeping the focus on consumer behavior rather than financial‐institution strategy optimization. In contrast, reinforcement learning could introduce unnecessary complexity in financial decisions, while purely **data‐driven methods** may lack the flexibility to accommodate varying financial needs across age cohorts.

---

## 4.Illustrative Experiment

### **Experiment 1: Baseline Consumption Propensity Analysis**

* **Experiment Description:**
  Build benchmark models of consumption propensity for each age group based on statistical survey data, and record their consumption outcomes when run on the simulated economic platform.
* **Involved Social Roles:**
  * *Firm: ​*Perfect Competition
  * *Individual: ​​*Ramsey ​Model
* **AI**​**​ Agents:**
  * *Firm: ​*Rule‐Based Agent
  * *Individual: ​*RL Agent
  * *Bank: ​*Rule‐Based Agent
* **Experimental Variables:**
  * Four age groups: Youth (18–30), Early Middle Age (31–45), Late Middle Age (46–65), Senior (> 65)
  * Each group’s average and marginal propensity to consume

```Python
#c denotes the initial consumption ratio​ ​(i.e., proportion of income consumed).
#Parameters are initialized based on LLM-informed recommendations and U.S. statistical data tracking.

For each individual in the household population:
    If 18 ≤ age ≤ 30:
        Set c ≈ 0.65 ± small random noise
        # Young adults: high consumption, early-stage overconsumption
    Else if 31 ≤ age ≤ 45:
        Set c ≈ 0.55 ± small random noise
        # Early middle age: career building, moderate consumption
    Else if 46 ≤ age ≤ 65:
        Set c ≈ 0.45 ± small random noise
        # Late middle age: stable income, family burden, more balanced
    Else if age > 65:
        Set c ≈ 0.35 ± small random noise
        # Elderly: consumption needs decline, medical costs rise
```

* **​ Visualized Experimental Results：**

![Individual Q3 P1](../img/Individual%20Q3%20P1.png)

**Figure 1:** Compared to the baseline scenario (left panel), the simulated U.S. consumption patterns under age‐group–specific MPC settings (middle and right panels) show higher consumption by the middle class (green bars) than in the baseline.

* As age increases, the marginal propensity to consume (MPC) of adult cohorts exhibits a general downward trend.
* The age‐stratified simulation environment better stimulates consumption among the middle‐class cohort. Incorporating age layering alongside the platform’s existing income stratification enables a more accurate characterization of household groups and yields a more complete model.

