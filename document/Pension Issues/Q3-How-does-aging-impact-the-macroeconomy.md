# Q3: How does aging impact the macroeconomy?

## 1. Introduction

### 1.1 Definition and Drivers of Population Aging

Population aging refers to the rising share of elderly individuals (typically those aged 65 and above) within a society’s total population. Conventionally, when the elderly population exceeds 7%, a society is classified as “aging”; when it surpasses 14%, it enters the stage of “deep aging.”

The primary drivers of population aging are:

* **Declining fertility:** Economic development, higher female education, and rising childrearing costs lead to lower birth rates and a shrinking young cohort.
* **Increased life expectancy:** Improvements in healthcare and living standards raise average lifespan, expanding the elderly population.
* **Cohort effects:** Large birth cohorts from past high-fertility periods gradually age into the elderly bracket, boosting the proportion of seniors.

### 1.2 Research Questions

Using an economic-simulation platform, this study examines the effects of population aging on national economies, specifically:

* **Capital accumulation, savings rates, and output:** How does aging influence capital formation, aggregate saving rates, and GDP?
* **Behavioral changes by age cohort:** How will consumption, saving, and labor-supply behaviors differ across younger, middle-aged, and elderly groups?
* **Intergenerational wealth transfer:** Does aging alter the mechanisms of wealth transmission between generations?

### 1.3 Research Significance

* **Informing labor-market reform and delayed-retirement policies:**  Simulating labor-supply and demand dynamics under aging supports the design of scientifically grounded retirement ages and participation incentives.
* **Guiding long-term fiscal policy optimization:**  Understanding how demographic shifts affect public finances aids in crafting sustainable pension and social-protection systems.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                                                                          |
| ------------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | OLG Model                             | Simulate demographic aging and its impact on labor supply and consumption.                                                                |
| Government             | Pension Authority                    | Formulate pension policies, adjusting retirement ages and benefit expenditures.                                                           |
| Firm                | Perfectly Competition          | Firms hire labor and produce goods in a perfectly competitive environment, responding to labor-supply changes driven by population aging. |
| Bank | No-Arbitrage Platform | Serve as intermediaries offering saving and lending services, measuring how declines in the savings rate affect financial markets.        |

### Individual → Overlapping Generations (OLG) Model

* To capture heterogeneity in consumption, saving, and labor-supply across age cohorts, we employ the Overlapping Generations (OLG) framework. The OLG model naturally represents intergenerational wealth transfers, demographic shifts, and life-cycle behaviors.

### Government → Pension Authority

* Facing rising pension expenditures and a shrinking tax base, the government’s Pension Department is directly responsible for formulating and adjusting policies related to population aging.

### Firm → Perfectly Competition

* Wages in a perfectly competitive market are set by supply and demand, accurately reflecting price-mechanism adjustments in labor supply driven by demographic aging.

### Bank → No-Arbitrage Platform

* We simulate long-run capital-market structures—investment and return mechanisms—under demographic change. Since this experiment focuses on structural shifts rather than short-term credit behavior, we use a no-arbitrage intermediary rather than a commercial-bank role.

---

## 3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type          | Role Description                                                                                                                             |
| ------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Behavior Cloning Agent | Replicate sensitivity differences of various income groups to interest-rate changes, reflecting realistic saving and consumption behaviors.  |
| Government             | Rule-Based Agent       | Set benchmark rates and policy rules to control the permissible range of interest-margin fluctuations.                                       |
| Firm                 | Rule-Based Agent       | Simulate firms’ direct responses to changes in financing costs, consistent with the perfect-competition assumption.                         |
| Bank | Rule-Based Agent       | Commercial banks set margins according to preset strategies, facilitating the assessment of systemic effects under different margin regimes. |

### Individual → Behavior Cloning Agent

* Household behaviors (e.g., saving rates, retirement decisions) are often driven by historical patterns. Behavior Cloning reproduces the non-rational traits of older cohorts—such as savings inertia and rigid labor supply—observed in real data.

### Government → Rule-Based Agent

* Government fiscal decisions follow rule-based triggers (e.g., “if pension shortfall exceeds a threshold, raise the tax rate”), which aptly captures policy responses tied to explicit economic indicators.

### Firm → Rule-Based Agent

* Wage and employment dynamics are governed by supply–demand rules within a perfectly competitive framework, making rule-based modeling ideal for simulating labor-market adjustments.

### Bank → Rule-Based Agent

* Investment-return adjustments are set according to long-term interest-rate trends and demographic shifts, lending themselves naturally to rule-based agent implementation.

---

## 4. Illustrative Experiments

### Experiment 1: Impact of Population Aging on Household Wealth and Labor Supply

* **Experiment Description:**  Configure different demographic structures (e.g., increase the elderly share from 10% to 20%) and observe changes in household saving, consumption, labor supply, and asset accumulation.
* **Involved Social Roles: ​**
  * *Individual: ​*OLG Model
  * *Government:* Pension Authority
  * *Firm:* Perfectly Competitive Market
* **AI Agents:**
  * *Individual:* Behavior Cloning Agent
  * *Government:* Rule-Based Agent
  * *Firm:* Rule-Based Agent
* **Experimental Variables:**
  * Elderly population share (or demographic parameters)
  * Individual assets and labor‐force participation rate

### Experiment 2: Impact of Population Aging on Social Inequality and Economic Growth

* **Experiment Description:**  Simulate aggregate GDP and the Gini coefficient under varying degrees of aging to assess potential generational wealth divergence.
* **Involved Social Roles: ​**
  * *Individual: ​*OLG Model
  * *Government:* Pension Authority
  * *Firm:* Perfectly Competitive Market
* **AI Agents:**
  * *Individual:* Behavior Cloning Agent
  * *Government:* Rule-Based Agent
  * *Firm:* Rule-Based Agent
* **Experimental Variables:**
  * Degree of population aging
  * GDP growth rate and income Gini coefficient

