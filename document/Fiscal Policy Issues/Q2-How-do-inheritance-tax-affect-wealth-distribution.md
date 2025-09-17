# Q2: How does inheritance tax affect wealth distribution?

## 1. Introduction

### 1.1 Introduction to Estate Tax

Estate Tax is a tax levied on assets inherited upon an individual's death, aiming to regulate intergenerational wealth transfer. As **global wealth inequality** grows, particularly the issue of intergenerational wealth accumulation within families, estate tax has become a crucial tool for reducing social inequality and promoting wealth redistribution. It not only helps diminish the concentration of "hereditary wealth" but also generates fiscal resources for public services and social welfare.

Implementation forms and rates of estate tax vary widely among developed countries. For example, Japan's maximum estate tax rate is 55%, Korea's is 50%, and France's is 60%. The U.S. has an estate tax at the federal level, with an exemption threshold as high as \$11.6 million, though states differ significantly. Connecticut applies a flat rate of 12%, whereas Iowa plans to abolish its estate tax entirely by 2025.

In contrast, China has **yet to implement** an estate tax. However, with increasing population aging and wealth concentration, discussions about introducing an estate tax to achieve fairer wealth redistribution have become more prevalent.

### 1.2 Research Questions

Based on an economic simulation platform, this study investigates how increasing estate tax impacts social wealth accumulation and distribution from both macro and micro perspectives. Specific questions include:

* **Social ​and GDP**: Does estate tax enhance societal output?
* **Individual working hours:** Does a higher estate tax encourage people to work more flexibly?
* **Household wealth and consumption:** Does a higher estate tax encourage quicker consumption of wealth during individuals' lifetimes?

### 1.3 Research Significance

* **Policy Guidance:**  By evaluating estate tax impacts on social output, household behaviors, and wealth distribution through simulation platforms, this study provides quantitative policy foundations for countries like China, which currently lack estate tax mechanisms, assisting in scientifically designing taxation systems and exemption thresholds.
* **Understanding Intergenerational Incentives and Behavioral Mechanisms:**  This study explores its long-term effects on household savings, labor supply, and consumption decisions, shedding light on the dynamic relationship between household behavior and wealth transfers, thereby offering empirical evidence for behavioral economics and public finance research.

---

## ​2.​ Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                                                                                 |
| ------------------------ | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | OLG Model                             | Analyze how households at different wealth levels respond to inheritance-tax policy, including changes in saving, consumption, and labor supply. |
| Government             | Fiscal Authority                  | Design and adjust inheritance-tax policy and assess its impact on public finances.                                                               |
| Firm                 | Perfect Competition      | Observe how shifts in consumer demand affect firms’ production and pricing strategies.                                                          |
| Bank | No-Arbitrage Platform | Study capital-market reactions to inheritance-tax policy, particularly changes in saving rates and investment behavior.                          |

---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
The OLG framework tracks lifecycle patterns of income, saving, and estate transfers, making it an ideal tool to study how inheritance tax influences intergenerational wealth transmission, labor incentives, and consumption decisions.

**Government → Fiscal Authority**  
As the authority that establishes and collects inheritance tax, the Tax Policy Department shapes policy features—such as exemption thresholds and redistribution rules—and directly affects fiscal revenue and aggregate demand. Modeling this actor enables simulation of policy adjustments’ transmission through the public budget.

**Firm → Perfect Competition**  
A perfectly competitive setting ensures efficient price formation, allowing clear observation of how an inheritance tax shifts supply and demand.

**Bank → No-Arbitrage Platform**  
No-Arbitrage Platform reliably reflect household asset allocation and returns across different life stages without introducing leverage dynamics or risk-preference distortions, making them well-suited for analyzing the medium- and long-term effects of inheritance tax on saving behavior and wealth accumulation paths.

---

## ​3.​ Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Rule-Based Agent / Behavior Cloning Agent | Use a rule-based agent to model household decision processes.Employ behavior cloning to learn patterns from empirical data.           |
| Government             | Data-Based Agent                          | Reproduce changes in public finances after inheritance-tax implementation within the simulation environment using historical tax data. |
| Firm                 | Rule-Based Agent                          | Encode market supply–demand rules to simulate consumer behavior under inheritance tax.                                                |
| Bank | Rule-Based Agent                          | Configure financial-market operations based on macroeconomic variables.                                                                |

---

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

## **​5.​**​**Illustrative Experiment**

```Python
# Estate tax logic (triggered upon individual death)
For each simulation step:
    For each individual in the household:
        If the individual dies:
            1. Compute total_wealth at death
            2. If total_wealth > exemption threshold:
                - Apply marginal tax_rate based on wealth tier
                - Tax amount = (total_wealth - threshold) × tax_rate
            3. Distribute post-tax wealth to heirs
            4. Log effects:
                - Government tax revenue increase
                - Heirs' wealth updates
                - Changes in social wealth distribution (for Gini tracking)
```

### Experiment 1: Macroeconomic Impact of Estate Tax

* **Experiment Description: ​**
  
  Compare macroeconomic indicators under different estate tax rates.
* **Experimental Variables:**
  
  * Estate tax rates (0%, 10%, 15%)
  * Social output (GDP)
* **Baselines:**

  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **estate\_15%\_bc\_saez\_100\_OLG (Blue line):** Households are modeled as Behavior Cloning (BC) Agents under the OLG framework, with the estate tax rate set to 15%.
  * **estate\_0%\_bc\_saez\_100\_OLG (Green line):** Households are modeled as Behavior Cloning (BC) Agents under the OLG framework, with the estate tax rate set to 0%.
  * **estate\_10%\_bc\_saez\_100\_OLG (Yellow line):** Households are modeled as Behavior Cloning (BC) Agents under the OLG framework, with the estate tax rate set to 10%.
* **Visualized Experimental Results：**

![Fiscal Q4 P1](../img/Fiscal%20Q4%20P1.png)
  
  **Figure 1:**  Imposing an inheritance tax clearly raises aggregate output, but increasing the rate from 10% to 15% yields little additional benefit.
* Compared to a zero–tax scenario, introducing an inheritance tax improves the efficiency of wealth circulation and thus boosts GDP. The inheritance-tax policy also carries a signaling effect, but further rate increases have diminishing returns for GDP growth.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Experiment 2: Household Impact of Estate Tax

* **Experiment Description: ​**
  
  Compare household income, consumption, and labor behaviors under different estate tax rates.
* **Experimental Variables:**
  
  * Estate tax rates (0%, 10%, 15%)
  * Simulated households stratified by age and income
  * Household wealth
  * Household working hours
* **Baselines:**
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.The bar charts show household wealth distributions under different estate tax policies.
  
  * **Agent Settings:**
    
    * **estate\_0%\_bc\_saez\_100\_OLG (Green bars):** Estate tax rate set to 0%.
    * **estate\_10%\_bc\_saez\_100\_OLG (Yellow bars):** Estate tax rate set to 10%.
    * **estate\_15%\_bc\_saez\_100\_OLG (Blue bars):** Estate tax rate set to 15%.
  * **Panel Interpretation:**
    
    * **Left panel:** Different bar colors represent household wealth by **age cohorts** (e.g., <24, 25–34, 35–44, 45–54, 55–64, 65–74, 75–84, 85+, total).
    * **Right panel:** Different bar colors represent household wealth by **income/wealth classes** (rich, middle, poor, and mean).
* **Visualized Experimental Results：**
  
![Fiscal Q4 P2](../img/Fiscal%20Q4%20P2.png)
  
  **Figure 2: ​**Higher estate taxes consistently reduce household wealth across different age groups and income levels.
  
![Fiscal Q4 P3](../img/Fiscal%20Q4%20P3.png)
  
  **​ Figure 3:** Higher estate taxes enhance labor incentives for low-income households but discourage labor among middle-income groups; minimal age-based effects observed.
* Estate taxes decrease household wealth.
* Estate taxes exhibit clear labor incentives differences based on income: enhancing labor incentives among low-income groups while discouraging middle-income groups.

## 4. Illustrative Experiment

```Python
# Estate tax logic (triggered upon individual death)
For each simulation step:
    For each individual in the household:
        If the individual dies:
            1. Compute total_wealth at death
            2. If total_wealth > exemption threshold:
                - Apply marginal tax_rate based on wealth tier
                - Tax amount = (total_wealth - threshold) × tax_rate
            3. Distribute post-tax wealth to heirs
            4. Log effects:
                - Government tax revenue increase
                - Heirs' wealth updates
                - Changes in social wealth distribution (for Gini tracking)
```

### Experiment 1: Macroeconomic Impact of Estate Tax

* **Experiment Description: ​**
  Compare macroeconomic indicators under different estate tax rates.
* **Involved Social Roles:**
  
  * *Individual:* OLG Model
  * *Government: ​*Fiscal Authority
* **AI Agents:**
  
  * *Individual: ​*Behavior Cloning Agent
  * *Government: ​*Data-Based Agent
* **Experimental Variables:**
  
  * Estate tax rates (0%, 10%, 15%)
  * Social output (GDP)
* **Visualized Experimental Results：**

  
  **Figure 1:** The blue, yellow, and green lines represent social GDP under inheritance-tax rates of 15%, 10%, and 0%, respectively. Imposing an inheritance tax clearly raises aggregate output, but increasing the rate from 10% to 15% yields little additional benefit.
* Compared to a zero–tax scenario, introducing an inheritance tax improves the efficiency of wealth circulation and thus boosts GDP. The inheritance-tax policy also carries a signaling effect, but further rate increases have diminishing returns for GDP growth.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Experiment 2: Household Impact of Estate Tax

* **Experiment Description: ​**
  Compare household income, consumption, and labor behaviors under different estate tax rates.
* **Involved Social Roles:**
  
  * *Individual: ​*OLG Model
  * *Government:* Fiscal Authority
* **AI Agents:**
  
  * *Individual:* Behavior Cloning Agent
  * *Government: ​*Data-Based Agent
* **Experimental Variables:**
  
  * Estate tax rates (0%, 10%, 15%)
  * Simulated households stratified by age and income
  * Household wealth
  * Individual working hours
* **Visualized Experimental Results：**

  
  **Figure 2: ​**Higher estate taxes (blue bars) consistently reduce household wealth across different age groups (left) and income levels (right).
  

  
  **​ Figure 3:** Higher estate taxes enhance labor incentives for low-income households but discourage labor among middle-income groups; minimal age-based effects observed.
* Estate taxes decrease household wealth.
* Estate taxes exhibit clear labor incentives differences based on income: enhancing labor incentives among low-income groups while discouraging middle-income groups.

