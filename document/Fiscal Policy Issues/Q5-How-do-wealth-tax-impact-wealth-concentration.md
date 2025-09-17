# Q5: How does wealth tax impact wealth concentration?

## 1. Introduction

### 1.1 Overview of Property Tax

Property tax is a levy on assets held by individuals or households (e.g., real estate, stocks, land), typically assessed as a percentage of net asset value. Over recent decades, many countries have seen wealth concentrate within a small elite, exacerbating inequality, dampening aggregate consumption, and weakening economic dynamism. Because it targets accumulated wealth directly, a well-designed property tax is often viewed as a tool to break intergenerational wealth perpetuation and curb extreme concentration.

### 1.2 Controversial Background on Property Tax

Economic theory suggests a moderate property tax can slow the “Matthew effect” (“the rich get richer”). However, implementation raises concerns: it may trigger capital flight and reduce saving, while imposing burdensome costs on the middle class. Thus, designing and evaluating property-tax regimes requires dynamic simulation and institutional experimentation.

### 1.3 Research Questions

Using an economic-simulation platform, this study investigates the long-term impacts of property tax on household behavior and macroeconomic outcomes, specifically:

* **Household Consumption:** How does a property tax affect household consumption patterns across different wealth groups?
* **Household Income:** What are the impacts of property tax on labor income and overall household earnings?
* **Individual Utility:** Does the introduction of property tax improve or reduce individual welfare (utility) under different tax rates?
* **Social Equity:** To what extent does property tax mitigate wealth inequality and promote social fairness?
* **Economic Growth:** How does property taxation influence aggregate savings, investment, and long-term GDP growth?

### 1.4 Research Significance

* **Assessing Taxation’s Role in Social Equity:** Analyze whether property tax helps disrupt “dynastic wealth” and promotes more balanced resource allocation.
* **Building a Behavioral-Feedback Tax-Simulation Mechanism:** Model how households adjust behavior in response to various property-tax regimes, providing a microfoundation for tax-policy design.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                                                                               | Observation                                                                                               | Action                                                                                 | Reward                                   |
| ----------- | ------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------- |
| Individual  | Ramsey Model        | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.        | $$o_t^i = (a_t^i, e_t^i)$$<br>Private: assets, education<br>Global: distributional statistics             | $$a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$$<br>Asset allocation, labor, investment | $$r_t^i = U(c_t^i, h_t^i)$$ (CRRA utility) |
| Government  | Fiscal Authority    | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution.             | $$o_t^g = \{ B_{t-1}, W_{t-1}, P_{t-1}, \pi_{t-1}, Y_{t-1}, \mathcal{I}_t \}$$<br>Public debt, wage, price level, inflation, GDP, income dist. | $$a_t^{\text{fiscal}} = \{ \boldsymbol{\tau}, G_t \}$$<br>Tax rates, spending          | GDP growth, equality, welfare            |
| Firm       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.           | /                                                                                                         | /                                                                                    | Zero (long-run)                          |
| Bank       | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives. | /                                                                                                         | No rate control                                                                      | No profit                                |


---

### Rationale for Selected Roles

**Households → Ramsey Model**  
Households face life-cycle income and wealth-accumulation trajectories and make labor, consumption, and saving choices based on utility-maximization principles. They are the direct responders and transmission channel for tax policy.

**Government → Fiscal Authority**   
The government is responsible for designing the property-tax system, levying asset taxes, and adjusting tax-rate structures to achieve redistribution goals and fiscal balance.

**Firm → Perfect Competition**  
Wages and goods prices are determined by supply and demand. Tax policies indirectly influence firm and household behavior through these market mechanisms.

**Bank → Non-Profit Platform**  
Asset prices and capital returns are affected by tax-burden changes over the long run. The financial system adjusts savings and investment allocations to restore intertemporal equilibrium.

---

## 3. Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             |  Behavior Cloning          | Household behavior is modeled by learning from empirical data, enabling the simulation to replicate realistic decision-making under varying tax regimes.              |
| Government             | Rule‐Based Agent | Define multiple property‐tax brackets as experimental inputs and observe their effects on macroeconomic and distributional indicators.         |
| Firm                 | Rule‐Based Agent | Adjust wages and prices according to labor‐market rules, transmitting the marginal effects of tax burdens on economic activity.                |
| Bank | Rule‐Based Agent | Adjust interest rates and returns based on changes in savings and capital accumulation, reflecting taxation’s impact on financial equilibrium. |

## **4. Running the Experiment**

### **4.1 Quick Start**

To run the simulation with a specific problem scene, use the following command:

```Bash
python main.py --problem_scene ""
```

This command loads the configuration file `cfg/`, which defines the setup for the "" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### **4.2 Problem Scene Configuration**

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### **Example ​**​**YAML**​**​ Configuration: ​**

---

## 5.Illustrative Experiments

### Experiment 1: Household Economic Behavior under Different Property Tax Rates

* **Experiment Description:**

 **​ ​**Simulate several property tax regimes (0%, 3%, 5%) and compare their impacts on individual economic indicators including utility, consumption level, wealth accumulation, savings rate, and labor supply.

* **Experimental Variables:**
  
  * Property tax rate (0%, 3%, 5%)
  * Household utility
  * Savings rate
  * Labor supply
  * Consumption expenditure
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each bar group in the visualization to help readers better understand the results.
  
  * **0%\_wealth\_tax\_ramsey\_100\_bc\_tax\_saez (Left group):** Households are modeled using the Ramsey Model with Behavior Cloning (BC) Agents, while the government adopts a ​**Saez rule-based tax policy**​. No wealth tax is imposed, serving as the baseline scenario.
    * Blue bar: Rich households
    * Green bar: Middle-class households
    * Yellow bar: Poor households
    * Red bar: Overall average
  * **5%\_wealth\_tax\_ramsey\_100\_bc\_tax\_saez (Middle group):** Households are modeled using the Ramsey Model with BC Agents, while the government applies a ​**Saez rule-based tax system**​. A **5% wealth tax** is imposed on household assets.
    * Blue bar: Rich households
    * Green bar: Middle-class households
    * Yellow bar: Poor households
    * Red bar: Overall average
  * **3%\_wealth\_tax\_ramsey\_100\_bc\_tax\_saez (Right group):** Households follow the Ramsey Model with BC Agents, and the government adopts the ​**Saez rule-based tax policy**​. A **3% wealth tax** is applied to household assets.
    * Blue bar: Rich households
    * Green bar: Middle-class households
    * Yellow bar: Poor households
    * Red bar: Overall average
* **Visualized Experimental Results:**

![Wealth Tax Impact](../img/Fiscal%20Q5P6.jpeg)

**Figure 1:** Simulated economies under different property tax rates. Left: 0%; Middle: 5%; Right: 3%.

![Wealth Tax Impact](../img/Fiscal%20Q5P1.jpeg)

**Figure 2:** Property tax reduces long-run wealth accumulation across all income levels. Households accumulate the most wealth in the absence of property tax (left).

![Wealth Tax Impact](../img/Fiscal%20Q5P3.jpeg)

**Figure 3:** Short-term utility is only slightly affected by property tax. Average household utility in the 0% tax economy (left) is marginally higher than in the 5% tax case (middle).

![Wealth Tax Impact](../img/Fiscal%20Q5P4.jpeg)

**Figure 4:** In the long run, higher property tax rates significantly lower household utility across all income groups.

![Wealth Tax Impact](../img/Fiscal%20Q5P2.jpeg)

**Figure 5:** Property tax suppresses short-run consumption.

* Higher property tax rates significantly slow down personal wealth accumulation and suppress household consumption. In the long run, reduced consumption and slower wealth growth lead to lower individual utility levels across the economy.

### **Experiment 2: Macro-Social Impacts of Property Tax Policy**

* **Experiment Description:**

  Evaluate the long-term impact of different property tax rates (0%, 3%, 5%) on macroeconomic performance (e.g., GDP) and social indicators (e.g., wealth inequality) using a multi-agent economic simulation.
* **Experimental Variables:**
  
  * Property tax rate (0%, 3%, 5%)
  * Wealth Gini coefficient
  * Simulated economy’s GDP
* **Baselines:**
  
  Below, we provide explanations of the experimental settings in this visualization to help readers better understand the results.
  
  * **0%\_wealth\_tax\_ramsey\_100\_bc\_tax\_saez (blue line):** Households are modeled using the Ramsey Model with Behavior Cloning (BC) Agents, while the government adopts a ​**Saez rule-based tax policy**​. No wealth tax is imposed, serving as the baseline scenario.
  * **5%\_wealth\_tax\_ramsey\_100\_bc\_tax\_saez (green line):** Households are modeled using the Ramsey Model with BC Agents, while the government applies a ​**Saez rule-based tax system**​. A **5% wealth tax** is imposed on household assets.
  * **3%\_wealth\_tax\_ramsey\_100\_bc\_tax\_saez (yellow line):** Households follow the Ramsey Model with BC Agents, and the government adopts the ​**Saez rule-based tax policy**​. A **3% wealth tax** is applied to household assets.
* **Visualized Experimental Results:**

![Wealth Tax Impact](../img/Fiscal%20Q5P8.jpeg)

**Figure 6:** Property tax temporarily reduces wealth inequality (years 60–120), but in the long run (post year 150), higher tax rates lead to increasing inequality.

![Wealth Tax Impact](../img/Fiscal%20Q5P7.jpeg)


**Figure 7:** GDP grows fastest under the 0% property tax scenario (blue). The economy with the highest tax rate (green) exhibits the lowest long-run GDP level.

![Wealth Tax Impact](../img/Fiscal%20Q5P5.jpeg)

**Figure 8:** Social welfare is maximized in the absence of property tax (blue).

* Although property tax can reduce inequality in the short term, it significantly suppresses GDP growth and eventually results in even greater wealth disparity over time.



