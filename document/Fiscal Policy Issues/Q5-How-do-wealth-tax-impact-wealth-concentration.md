# Q5: How does wealth tax impact wealth concentration?

## 1. Introduction

### 1.1 Overview of Property Tax

Property tax is a levy on assets held by individuals or households (e.g., real estate, stocks, land), typically assessed as a percentage of net asset value. Over recent decades, many countries have seen wealth concentrate within a small elite, exacerbating inequality, dampening aggregate consumption, and weakening economic dynamism. Because it targets accumulated wealth directly, a well-designed property tax is often viewed as a tool to break intergenerational wealth perpetuation and curb extreme concentration.

### 1.2 Controversial Background on Property Tax

Economic theory suggests a moderate property tax can slow the “Matthew effect” (“the rich get richer”). However, implementation raises concerns: it may trigger capital flight and reduce saving, while imposing burdensome costs on the middle class. Thus, designing and evaluating property-tax regimes requires dynamic simulation and institutional experimentation.

### 1.3 Research Questions

Using an economic-simulation platform, this study examines the long-term impacts of property tax on household behavior and wealth distribution, specifically:

* Can a property tax effectively reduce wealth concentration?
* Do households systematically change their saving, labor, and investment behaviors under different property-tax structures?

### 1.4 Research Significance

* **Assessing Taxation’s Role in Social Equity:** Analyze whether property tax helps disrupt “dynastic wealth” and promotes more balanced resource allocation.
* **Building a Behavioral-Feedback Tax-Simulation Mechanism:** Model how households adjust behavior in response to various property-tax regimes, providing a microfoundation for tax-policy design.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                           | Role Description                                                                                                                                                |
| ------------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Ramsey Model                            | Households optimize consumption and savings over their life-cycle. Wealth tax directly influences their savings behavior and intergenerational wealth planning. |
| Government             | Fiscal Authority                     | The government sets wealth tax rates and designs redistribution policies aimed at mitigating wealth inequality while maintaining fiscal sustainability.         |
| Firm                 | Perfect Competition            | Wealth taxes affect capital accumulation and labor allocation indirectly through changes in prices and wages in competitive markets.                            |
| Bank | No-Arbitrage Platform | Financial intermediaries adjust investment strategies and asset portfolios in response to long-term shifts in returns caused by wealth taxation.                |

### Households → Ramsey Model

* Households face life-cycle income and wealth-accumulation trajectories and make labor, consumption, and saving choices based on utility-maximization principles. They are the direct responders and transmission channel for tax policy.

### Government → Fiscal Authority 

* The government is responsible for designing the property-tax system, levying asset taxes, and adjusting tax-rate structures to achieve redistribution goals and fiscal balance.

### Firm → Perfect Competition

* Wages and goods prices are determined by supply and demand. Tax policies indirectly influence firm and household behavior through these market mechanisms.

### Bank → No-Arbitrage Platform

* Asset prices and capital returns are affected by tax-burden changes over the long run. The financial system adjusts savings and investment allocations to restore intertemporal equilibrium.

---

## 3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type     | Role Description                                                                                                                                |
| ------------------------ | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             |  Behavior Cloning          | Household behavior is modeled by learning from empirical data, enabling the simulation to replicate realistic decision-making under varying tax regimes.              |
| Government             | Rule‐Based Agent | Define multiple property‐tax brackets as experimental inputs and observe their effects on macroeconomic and distributional indicators.         |
| Firm                 | Rule‐Based Agent | Adjust wages and prices according to labor‐market rules, transmitting the marginal effects of tax burdens on economic activity.                |
| Bank | Rule‐Based Agent | Adjust interest rates and returns based on changes in savings and capital accumulation, reflecting taxation’s impact on financial equilibrium. |

### **Individual → BC Agent**

* When facing different property tax rates, households must dynamically balance labor, savings, and investment decisions. A Behavior Cloning (BC) Agent, trained on real-world data, can **replicate more realistic decision-making patterns**, thereby enhancing the fidelity of the simulation.

### **Government → Rule‐Based Agent**

* The government sets multiple property‐tax rules (e.g., 0%, 1%, 2%, 3%) and holds the regime constant during experiments. It observes social feedback to evaluate each tax schedule’s redistributive and fiscal effects.

### **Firm → Rule‐Based Agent**

* Wages, employment, and goods prices adjust via supply–demand rules, ensuring simulated market feedback to tax changes is both realistic and controllable.

### **Bank → Rule‐Based Agent**

* Interest‐rate and capital‐return mechanisms map post‐tax changes in saving and investment through simple rules, aiding simulation of asset‐price trends and market stability.

---

## 4.Illustrative Experiments

### Experiment 1: Household Economic Behavior under Different Property Tax Rates

* **Experiment Description: ​**Simulate several property tax regimes (0%, 3%, 5%) and compare their impacts on individual economic indicators including utility, consumption level, wealth accumulation, savings rate, and labor supply.
* **Involved Social Roles:**
  * Individual: Ramsey Model
  * Government: Fiscal Authority
* **AI**​**​ Agents:**
  * Individual: BC Agent
  * Government: Rule-Based Agent
* **Experimental Variables:**
  * Property tax rate (0%, 3%, 5%)
  * Household utility
  * Savings rate
  * Labor supply
  * Consumption expenditure
* **Visualized Experimental Results:**

![Wealth Tax Impact](../img/Fiscal%20Q5P6.jpeg)

**Figure 1:** Simulated economies under different property tax rates. Left: 0%; Middle: 5%; Right: 3%.

![Wealth Tax Impact](../img/Fiscal%20Q5P1.jpeg)

**Figure 2:** Property tax reduces short-term wealth accumulation across all income levels. Households accumulate the most wealth in the absence of property tax (left).

![Wealth Tax Impact](../img/Fiscal%20Q5P3.jpeg)

**Figure 3:** Short-term utility is only slightly affected by property tax. Average household utility in the 0% tax economy (left) is marginally higher than in the 5% tax case (middle).

![Wealth Tax Impact](../img/Fiscal%20Q5P4.jpeg)

**Figure 4:** In the long run, higher property tax rates significantly lower household utility across all income groups.

![Wealth Tax Impact](../img/Fiscal%20Q5P2.jpeg)

**Figure 5:** Property tax suppresses short-run consumption.

* Higher property tax rates significantly slow down personal wealth accumulation and suppress household consumption. In the long run, reduced consumption and slower wealth growth lead to lower individual utility levels across the economy.

### **Experiment 2: Macro-Social Impacts of Property Tax Policy**

* **Experiment Description:**  Evaluate the long-term impact of different property tax rates (0%, 3%, 5%) on macroeconomic performance (e.g., GDP) and social indicators (e.g., wealth inequality) using a multi-agent economic simulation.
* **Involved Social Roles:**
  * Individual: Ramsey Model
  * Government: Fiscal Authority
* **AI**​**​ Agents:**
  * Individual: Behavior Cloning (BC) Agent
  * Government: Rule-Based Agent
* **Experimental Variables:**
  * Property tax rate (0%, 3%, 5%)
  * Wealth Gini coefficient
  * Simulated economy’s GDP
* **Visualized Experimental Results:**

![Wealth Tax Impact](../img/Fiscal%20Q5P8.jpeg)

**Figure 6:** Property tax temporarily reduces wealth inequality (years 60–120), but in the long run (post year 150), higher tax rates lead to increasing inequality.

![Wealth Tax Impact](../img/Fiscal%20Q5P7.jpeg)

**Figure 7:** GDP grows fastest under the 0% property tax scenario (blue). The economy with the highest tax rate (green) exhibits the lowest long-run GDP level.

![Wealth Tax Impact](../img/Fiscal%20Q5P5.jpeg)

**Figure 8:** Social welfare is maximized in the absence of property tax (blue).

* Although property tax can reduce inequality in the short term, it significantly suppresses GDP growth and eventually results in even greater wealth disparity over time.


