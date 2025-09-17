# Q3: Does universal basic income enhance equity?

## 1.​ Introduction

### 1.1 Definition

Universal Basic Income (UBI) is a form of social security that provides every citizen with an unconditional, regular cash transfer. This study examines the extent to which UBI fosters social equity. The idea draws on the National Bureau of Economic Research (NBER) article *“Universal Basic Income in the United States and Advanced Countries.”* Key features are:

* **Adequate generosity** – the grant is large enough to cover basic living expenses, even when recipients have no other income;
* **Non-means-tested or only gradually phased-out** – payments do not immediately cease as personal income rises;
* **Near-universal coverage** – benefits are delivered to almost the entire population rather than to narrowly defined groups (e.g., single mothers).

In short, UBI can be viewed as a perpetual transfer that provides every resident with sufficient resources for a minimal standard of living.

### 1.2 Limitations and Policy Inspiration

UBI faces three major hurdles: (i) ​**fiscal cost**​, (ii) ​**potential work-incentive distortions**​, and (iii) ​**integration with existing welfare programs**​. Nevertheless, rapid advances in automation and AI are reshaping labour markets and eroding traditional jobs. In this context, UBI has been proposed as a mechanism to alleviate poverty, curb inequality, and forestall social unrest. The principles of universality and dignity embedded in UBI still offer valuable guidance for modern welfare reform.

### 1.3 Research Questions

Using an economic-simulation platform, we explore how UBI affects:

* **Income inequality** : Does UBI narrow the gap between rich and poor?
* **Household labour supply** : How will total working hours change?
* **Household wealth accumulation** : What is the impact on asset holdings?

### 1.4 Research Significance

* **Welfare-system reform:**
  The universal and unconditional nature of UBI suggests remedies for incomplete coverage and high eligibility thresholds in current schemes, informing more inclusive policy design.
* **Social protection in the ​AI era:**
  As AI and large models increasingly substitute routine work, a UBI-style safety net can provide reliable income for displaced or transitioning workers. Assessing UBI in this technological setting helps build future-proof social insurance.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                                             |
| ------------------------ | --------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
|      Individual        | OLG Model                             | Analyze how different age cohorts respond to UBI—tracking changes in saving, consumption, and labor supply. |
| Government             | Fiscal Authority                   | Design and adjust UBI policy, and evaluate its impact on public finances.                                    |
| Firm         | Perfect Competition         | Observe shifts in labor demand and supply under UBI, as well as changes in wages and employment.             |
| Bank | No-Arbitrage Platform | Study capital-market reactions to UBI, particularly alterations in saving rates and investment behavior.     |

---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
The OLG framework captures age-specific income, consumption, and saving behaviors, making it well-suited to simulate UBI’s dynamic effects on intergenerational resource allocation. It is preferable to an infinitely heterogeneous agent model for this purpose.

**Government → Fiscal Authority**  
The Fiscal Authority is directly responsible for funding and administering UBI, making it the core institution for assessing fiscal impacts. It aligns more closely with UBI financing duties than a central bank.

**Firm → Perfect Competition**  
A perfectly competitive market eliminates distortions from market power, allowing a clear assessment of UBI’s influence on labor supply.

**Bank → No-Arbitrage Platform**  
UBI may alter saving rates and investment patterns; a no-arbitrage framework is ideal for analyzing these financial dynamics.

---

## 3.​ Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | Rule-Based Agent / Behavior Cloning Agent | Use a rule-based agent to model household decision processes; employ behavior cloning to learn patterns from empirical data. |
| Government             | Data-Based Agent / RL Agent               | Forecast changes in public finances following UBI implementation using historical fiscal data.                               |
| Firm                 | Rule-Based Agent                          | Encode supply–demand rules to simulate labor-market responses under UBI.                                                    |
| Bank | Rule-Based Agent                          | Define financial-market operations based on macroeconomic variables.                                                         |


## 4.​ Illustrative Experiment

### Experiment 1: Impact of UBI on Social Equity

* **Experiment Description: ​**  Compare the effects of two UBI levels on income distribution.
* **Involved Social Roles:**
  * *Firm:* Perfectly Competitive Market
  * *Individual:* OLG Model
  * *Government: ​*Fiscal Authority
* **AI​ Agents:**
  * *Individual: ​*Behavior Cloning Agent
  * *Government: ​*RL Agent
* **Experimental Variables:**
  * UBI amount (UBI = 0 or UBI = 50% of the base wage)
  * Income inequality (measured by the Gini coefficient of income)

```Python
# UBI setting for fairness experiment
# Two UBI levels: 0 and 50% of base wage

For each time period t:
    If UBI is enabled:
        For each household:
            UBI = 0.5 × base_wage
    Else:
        UBI = 0

    # Calculate total income for each household
    total_income = wage_income + investment_income + pension_income + UBI

# Government adjusts expenditure accordingly
    government_expenditure += total UBI distributed
```

* **Visualized Experimental Results：**

![Fiscal Q2 P1](../img/Fiscal%20Q2%20P1.png)

**Figure 1: ​**In the simulation with UBI (green line), the income Gini coefficient is lower than in the economy without UBI, indicating that the UBI policy reduces wealth inequality.

* The UBI policy effectively reduces the gap between rich and poor.

---

### Experiment 2: UBI’s Effect on Household Labor Supply

* **Experiment Description: ​**  Assess how varying UBI levels influence average working hours across income deciles.
* **Involved Social Roles:**
  * *Firm:* Perfectly Competitive Market
  * *Individual:* OLG Model
  * *Government:* Fiscal Authority
* **AI**​**​ Agents:**
  * *Individual: ​*Behavior Cloning Agent
  * *Government: ​*RL Agent
* **Experimental Variables:**
  * UBI level (UBI = 0 or UBI = 50% of the base wage)
  * Average working hours of households by income tier
* **Visualized Experimental Results：**

![Fiscal Q2 P2](../img/Fiscal%20Q2%20P2.png)

**Figure 2: ​**Implementing the UBI policy (right-hand bar chart) reduces labor hours across all income brackets.

* Implementing the UBI policy significantly reduces working hours across all income groups. Note that in our simulation of 100 households, the top 10% income cohort is small, so some high-income households opt out of labor entirely.

