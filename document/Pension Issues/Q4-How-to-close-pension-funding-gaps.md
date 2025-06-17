# Q4: How to close pension funding gaps?

## 1. Introduction

### 1.1 Definition of the Pension Gap

The pension gap refers to the shortfall that arises when projected future pension expenditures under the existing system exceed the sum of government revenues and social‐insurance contributions. This issue intensifies amid accelerating population aging and sustained low fertility, directly threatening the sustainability of the social‐security framework.

### 1.2 Causes of the Pension Gap

The main drivers of the pension gap include:

* **Demographic shifts:** A shrinking workforce combined with a growing retired population raises the dependency ratio.
* **Structural pressures:** Under a pay‐as‐you‐go scheme, current contributions cannot support burgeoning future payouts.
* **Fiscal and investment constraints:** Limited growth in tax revenues and low returns on pension‐fund investments hinder gap closure.

### 1.3 Research Questions

Using an economic‐simulation platform, this study examines the effectiveness of various policy tools in addressing the pension gap, specifically:

* **Government strategies:** How do different policy options (delayed retirement, higher contribution rates, optimized pension‐fund investment) affect fiscal sustainability?
* **Policy‐mix design and evaluation:** How can the government combine policies optimally to reduce the pension gap, and what are the subsequent impacts on economic growth, labor markets, and income distribution?

### 1.4 Research Significance

* **Institutional‐mechanism design:** Assessing how measures such as delayed retirement, contribution‐rate adjustments, and replacement‐rate optimization influence the magnitude of the pension gap.

---

## 2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                                                                                                                 |
| ------------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual           | OLG Model   | Use the OLG framework to capture lifecycle differences in contribution, saving, and post-retirement benefit receipt behavior.                                                    |
| Government             | Pension Authority                    | Manage pension revenue and expenditure balance, including collection of social‐insurance contributions, pension disbursements, fiscal subsidies, and government‐bond issuance. |
| Firm                 | Perfect Competiton          | Model how wages and employment rates—driven by labor supply and demand—indirectly affect total pension contributions.                                                          |
| Bank | No-Arbitrage Platform | Simulate pension‐fund investment behavior and how long-term interest‐rate movements impact fund returns and fiscal support capacity.                                           |

### Individual → Overlapping Generations (OLG) Model

* This study focuses on intergenerational wealth transfers, contribution behaviors, and retirement decisions. The Overlapping Generations (OLG) framework captures how cohorts at different ages respond to the pension system, accurately reflecting household heterogeneity under aging.

### Government → Pension Authority  

* Acting as both system designer and fiscal backer, the Pension Authority must address the pension gap through policy levers such as taxation, bond issuance, and structural reforms.

### Firm → Perfect Competiton

* Wages and employment are determined by supply and demand and directly affect households’ contribution capacity and retirement savings. Embedding this market mechanism in the experiment is essential.

### Bank → No-Arbitrage Platform

* Pension‐fund investment returns influence system sustainability. An arbitrage‐free model ensures proper long‐term rate dynamics and fiscal‐return mechanisms.

---

## 3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type               | Role Description                                                                                                                        |
| ------------------------ | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Behavior Cloning Agent      | Reproduce real-world lifecycle-based decisions on contributions, savings, and retirement using behavior cloning.                        |
| Government             | Rule-Based Agent / RL Agent | Define rule-based triggers (e.g., bond issuance, tax hikes) in response to pension shortfalls, or employ RL to optimize pension policy. |
| Firm                 | Rule-Based Agent            | Map labor-market responses via supply–demand rules, reflecting wage and employment dynamics.                                           |
| Bank | Rule-Based Agent            | Set investment-return adjustments based on long-term interest rates and demographic shifts using explicit rules.                        |

### Individual → Behavior Cloning Agent

* Household decisions exhibit bounded rationality and path dependence (e.g., long-tenured workers favor stable contributions and timely retirement). A Behavior Cloning Agent trained on historical data better captures these realistic strategies.

### Government → Rule-Based Agent / RL Agent

* Pension reforms typically follow fiscal thresholds (e.g., tax hikes when shortfall exceeds a limit). A Rule-Based Agent encodes clear reaction rules to efficiently simulate policy logic.
* For multi-objective trade-offs (balancing pension sustainability, fiscal deficits, and household welfare),the RL Agent enables the government to learn optimal actions—such as adjusting retirement age, tax rates, or subsidy levels—through trial-and-error.

### Firm → Rule-Based Agent

* Wages and employment are driven by supply–demand dynamics. Rule-Based Agents quickly reflect price adjustments arising from shifts in labor-market structure.

### Bank → Rule-Based Agent

* Interest-rate and return mechanisms in financial markets follow economic laws (e.g., diminishing marginal returns to capital). A Rule-Based Agent facilitates building investment-return models linked to demographic shifts and government-bond issuance.

---

## 4. Illustrative Experiments

### 4.1 Experiment: Optimal Pension Policy Solution

* **Experiment Description:**
  Train reinforcement learning models using the minimization of the pension gap as the reward function, and compare the outcomes across different RL algorithms as well as between RL-based policies and baseline scenarios.
* **Involved Social Roles:**
  * *Government:* Pension Department
  * *individual*​*s:* OLG Model
* **AI**​**​ Agents:**
  * *Government:* RL Agent/Rule-Based Agent
  * *Individual*​*s:* RL Agent/Behavior Cloning Agent/Rule-Based Agent
* **Experimental Variables:**
  * Pension replacement rate
  * GDP impact
* **Visualized Experimental Results：**

![Pension Q4 P1](../img/PensionQ4P1.png)

**Figure 1:** Pension outcomes under different training strategies, considering four combinations of household and government policies: BC\_DDPG, BC\_PPO, BC\_Rule-Based, and PPO\_PPO (with the first referring to the household strategy and the second to the government strategy).From the age-based breakdown (left panel), the BC\_PPO combination yields the highest total pension surplus. RL-based government strategies significantly reduce pension gaps among young and middle-aged groups, with the PPO\_PPO strategy achieving the smallest pension deficit for young individuals.From the wealth-based breakdown (right panel), the BC\_PPO strategy again results in the highest overall pension surplus. The PPO\_PPO combination substantially lowers the pension gap for wealthy households (blue bars).

![Pension Q4 P2](../img/PensionQ4P2.png)

**Figure 2:** GDP trajectories under different training strategies. The BC\_DDPG combination achieves both stronger long-term GDP growth and a longer simulation duration (blue line), while the PPO\_PPO strategy results in the lowest GDP level.

* Although RL strategies are effective in reducing the pension gap, this optimization may come at the cost of **economic growth—particularly ​**when households also adopt RL-based decision-making. Overall, the combination where households follow Behavior Cloning and the government adopts an RL Agent strikes the best balance between sustained economic development and minimizing the pension gap.

