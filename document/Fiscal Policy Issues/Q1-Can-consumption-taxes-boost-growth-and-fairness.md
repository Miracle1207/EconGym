# Q1: Can consumption taxes boost growth and fairness?

## 1. Introduction

### 1.1 Social Context of Increasing Consumption Tax

Increasing consumption tax refers to the imposition of taxes on the purchase of goods and services. Amid global economic slowdown and rising income inequality, governments around the world seek new revenue sources to fund public services and **social welfare**​**​ ​**programs. As a potential policy tool, the effects of increasing consumption tax require in-depth research.

Taking the United States as an example, consumption tax typically consists of a comprehensive sales tax, composed of fixed State Sales Tax and varying Local Sales Taxes. The top three U.S. states by combined sales tax rates (state + local taxes) are:

* Louisiana: 9.56%
* Tennessee: 9.55%
* Arkansas: 9.45%

### 1.2 Research Questions

Based on an economic simulation platform, this study investigates whether increasing consumption tax can stimulate economic growth and enhance social equity. Specific questions include:

* Impact of consumption tax on overall socio-economic activities (GDP, social welfare levels).
* Impact of consumption tax on income inequality (Gini coefficient).
* Impact of increased consumption tax on different income groups (wealth, consumption, household utility).

### 1.3 Research Significance

* **Evaluating dual impacts on economy and distribution:**
  As an indirect tax, consumption tax has advantages such as a broad tax base and high collection efficiency but may disproportionately burden low-income groups. Assessing its effects on household wealth, consumption, and utility through simulation platforms helps fully understand its ​**redistributive effects**​, guiding more rational tax policy designs.
* **Providing policy ​**​**insights**​**​ for balancing fiscal revenue and social equity:**
  Given increasing fiscal expenditure demands and worsening income inequality, increasing consumption tax becomes a policy option. This research reveals trade-offs between economic growth and social equity under different tax rates, offering quantitative support for achieving fiscal sustainability and social justice.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                                                                                                                                              |
| ------------------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Ramsey Model/OLG Model                | Age and other household attributes have minimal impact on the government's consumption tax policy. Used to analyze long-term macroeconomic effects, particularly changes in saving and consumption behaviors. |
| Government             | Fiscal Authority                   | Formulate and adjust consumption tax policies and evaluate their impacts on public finance.                                                                                                                   |
| Firm            | Perfect Competition          | Observe how changes in consumer demand influence firms' production and pricing strategies.                                                                                                                    |
| Bank  | No-Arbitrage Platform | Study how capital markets respond to consumption tax policies, particularly changes in saving rates and investment behaviors.                                                                                 |

### Rationale for Selected Roles

**Individual → Ramsey Model/OLG Model**  
**Ramsey Model and OLG Model are both suitable for this study.** The Ramsey Model analyzes aggregate macroeconomic responses from representative households’ optimal intertemporal decisions, ideal for studying long-term equilibrium trends. The OLG Model captures heterogeneity across age groups in income, consumption, and tax burdens, enabling analysis of the intergenerational fairness effects of consumption taxes.

**Government → Fiscal Authority**  
The Tax Policy Department directly formulates and implements consumption tax policies, fully simulating tax collection, income redistribution, and fiscal expenditure responses. Compared with pension and central bank departments, the Treasury more accurately reflects the impact of consumption tax on tax structures, government budgets, and social equity.
Pension and monetary policy are not core variables in this study and thus are not used.

**Firm → Perfect Competition**  
Selecting perfectly competitive markets helps eliminate distortions, making the impact of consumption tax policies on supply-demand dynamics, pricing, and distribution clearer.
Monopolistic markets have non-market-determined prices and complex corporate strategies, potentially obscuring the economic effects of consumption taxes and reducing experimental clarity.

**Bank→ No-Arbitrage Platform**  
No-Arbitrage Platform are more suitable for analyzing long-term wealth accumulation and asset allocation responses, without active participation in credit expansion. They clearly reflect savings and investment behavior under policy changes.
Commercial banks involve complex behaviors like lending, interest spreads, and financial risks, less suited for focused macroeconomic policy analysis.

---

## 3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type          | Role Description                                                                                  |
| ------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------- |
| Individual            | Behavior Cloning Agent | Learn behavioral patterns from empirical data via behavior cloning.                               |
| Government             | Data-Based Agent       | Predict changes in public finance after implementing a consumption tax using historical tax data. |
| Firm                 | Rule-Based Agent       | Encode supply–demand rules to simulate consumer behavior under a consumption tax.                |
| Bank | Rule-Based Agent       | Define financial-market operations based on macroeconomic variables.                              |

### Individual → Behavior Cloning Agent

* Behavior Cloning Agents learn consumption and saving behaviors from real household data, ensuring clarity of decision logic and enhancing model authenticity.

### Government → Data-Based Agent

* The government uses historical data to predict consumption tax effects, aiding dynamic adjustments of fiscal policies and improving model accuracy and reliability.

### Firm → Rule-Based Agent

* Market behaviors can be explicitly modeled through price-supply-demand rules, clearly showing the direct transmission paths of consumption tax effects.

### Bank → Rule-Based Agent

* Financial institutions primarily handle asset allocation and interest rate transmission. Rule-based settings clearly express their stable behavior, suitable for macro-policy transmission analysis.

---

## 4. Illustrative Experiment

### Experiment 1: Impact of Increased Consumption Tax on Macroeconomy and Social Welfare

* **Experiment Description:**
  Comparing macroeconomic indicators and welfare levels across different consumption tax rates.
* ​**Involved Social Roles**​:
  * *​Individual：​*OLG Model
  * *​Government：​*Fiscal Authority
* **AI**​**​ Agents：**
  * *​Individual：​*Behavior Cloning Agent
  * *​Government：​*Data-Based Agent
* **Experimental Variables:**
  * Different consumption tax rates (0%, 7%, 9%)
  * Simulated GDP growth trends
  * Simulated social welfare levels
  * Simulated household income inequality (Gini coefficient)
* **Visualized Experimental Results:**

![Fiscal Q3 P1](../img/Fiscal%20Q3%20P1.png)

**Figure 1: ​**Blue, green, and yellow lines represent GDP under 0%, 7%, and 9% consumption tax rates, respectively. Higher taxes (yellow line) slightly increase GDP but show minimal difference compared to no tax.

![Fiscal Q3 P2](../img/Fiscal%20Q3%20P2.png)

**Figure 2: ​**Higher consumption taxes (yellow, green lines) have almost no long-term effect on total social welfare.

![Fiscal Q3 P3](../img/Fiscal%20Q3%20P3.png)

**Figure 3:** Increased consumption tax reduces the income Gini coefficient, indicating it effectively lowers income inequality.

* Increasing the consumption tax can slightly raise GDP in the simulated economy while effectively narrowing income disparities. Moreover, total social welfare remains unchanged under a higher consumption tax regime. Thus, from the perspective of promoting social equity, raising the consumption tax allows revenue to be redistributed—in some form—to lower-income groups more effectively, making it a reasonable tax policy choice.

---

### Experiment 2: Impact of Increased Consumption Tax on Household Wealth and Individual Utility

* **Experiment Description:**
  Comparing household wealth and individual utility across different consumption tax rates.
* ​**Involved Social Roles**​:
  * *​Individual：​*OLG Model
  * *​Government：​*Fiscal Authority
* **AI Agents:**
  * *​Individual：​*Behavior Cloning Agent
  * *​Government：​*Data-Based Agent
* **Experimental Variables:**
  * Different consumption tax rates (0%, 7%, 9%)
  * Household structure stratified by income levels
  * Household wealth levels
  * Person utility
* **Visualized Experimental Results:**

![Fiscal Q3 P4](../img/Fiscal%20Q3%20P4.png)

**Figure 4: ​**Different consumption taxes and household income levels. Blue, green, and yellow lines represent 0%, 7%, and 9% tax rates. Higher consumption taxes result in higher income levels across different age groups (left figure) and economic conditions (right figure), especially benefiting young individuals (25-34 years).

![Fiscal Q3 P5](../img/Fiscal%20Q3%20P5.png)

**Figure 5: ​**Consumption tax effects on household utility are not significant overall, but higher taxes notably reduce utility for individuals aged 25-34 and 45-54 years.

* Higher consumption taxes increase household income, notably benefiting young individuals, but simultaneously reduce utility for certain age groups. This indicates income gains do not fully offset welfare losses from higher consumption costs.

