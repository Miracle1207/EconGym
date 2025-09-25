# Q4: How does work-life balance impact well-being?

## 1. Introduction

### 1.1  **Introduction to Work–Life Balance**

Work–life balance denotes a sustainable, coordinated state in which individuals allocate time and energy between career pursuits and personal life. In modern economies—particularly among​**​ Generation Z**​—concern for work–life balance has surged. Under high-intensity work regimes, employees are increasingly aware of the long-term costs of overextending their physical and mental health. Survey evidence shows that younger workers increasingly prioritize flexible scheduling, remote work options, and personal autonomy, seeking to balance meaningful work with overall life satisfaction.

Real-world examples include:

* China’s “996” work schedule has sparked intense debate, leading many young people to adopt the “lying flat” lifestyle or migrate away from first-tier cities.
* Several European countries are piloting four-day workweeks to enhance well-being and sustain productivity over the long run.
* Companies are increasingly embedding employee well-being metrics into formal HR policies.


### **1.2 Research Questions**

This study uses an economic simulation platform to investigate the ​**economic impacts of work–life balance**​, specifically examining:

* **GDP**​**​ Effects:** How do different work–life balance regimes influence aggregate economic output?
* **Labor Supply:** How does work–life balance affect labor participation and productivity across cohorts?
* **Household Consumption:** How do shifts in work intensity and leisure time impact household consumption patterns?

### **1.3  Research Significance**

* **Reforming Modern Work Regimes:**  Does today’s work system need reform to satisfy a new generation’s demand for quality of life? How can we optimally structure work hours and modalities to maximize both social welfare and individual utility?
* **Labor and Behavioral Economics in Well-Being:**  This experiment quantifies well-being using a work–life balance framework, assessing whether higher measured well-being leads to greater realized individual utility and, in turn, fosters broader social development.

---

## **​2. Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role | Selected Type       | Role Description                                                                                                       | Observation                                                                                                                                          | Action                                                       | Reward                                               |
| ----------- | ------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| **Individual**  | Ramsey Model        | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets.         | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
| **Firm**       | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses.                 | /                                                                                                                                                    | /                                                            | Zero (long-run)                                      |
| **Bank**       | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives.   | /                                                                                                                                                    | No rate control                                              | No profit                                            |


---

### Rationale for Selected Roles

**Individual → Ramsey Model**  
In the Ramsey model, individuals’ choices in balancing life and work are more universal and rational, and the experiment aims to replicate consumption and labor decisions under the most rational circumstances.

**Government → Any Type**  
In the work–life balance experiments, the government must coordinate across multiple departments, for example: the Ministry of Labor enforces maximum working‐hour limits; the pension authority calibrates relevant pension regulations; and the tax authority adapts fiscal rules to the evolving social environment.

**Firm → Perfect Competition**  
Firms compete for talent through ​wages and flexible work policies. Workers choose environments that best match their balance preferences, forcing companies to adapt HR strategies.

**Bank →Non-Profit Platform**   
Provide ​life‑cycle financial products—retirement accounts, health insurance, liquidity support. As work–life patterns shift, so do saving needs and demand for these services.

---

## **​3. Selected Agent Algorithms**

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual             | RL Agent          | Simulate households’ work–life balance decisions using reinforcement learning.                                                                                                                                                 |
| Firm                 | Rule‑Based Agent | Firms adapt hiring strategies and workplace arrangements (e.g., offering flexible hours) according to labor-market supply–demand dynamics and worker preferences, exhibiting predictable behavior.                              |
| Bank | Rule‑Based Agent | Financial institutions deliver standardized life-cycle services—such as savings advice or insurance products—based on individuals’ life-cycle stage and income volatility, making them well-suited for rule-based simulation. |

---


## 4. Running the Experiment

### 4.1 Quick Start

To run the simulation with a specific problem scene, use the following command:

```bash
python main.py --problem_scene "work_life_well_being"
```

This command loads the configuration file `cfg/work_life_well_being.yaml`, which defines the setup for the "work_life_well_being" problem scene. Each problem scene is associated with a YAML file located in the `cfg/` directory. You can modify these YAML files or create your own to define custom tasks.

### 4.2 Problem Scene Configuration

Each simulation scene has its own parameter file that describes how it differs from the base configuration (`cfg/base_config.yaml`). Given that EconGym contains a vast number of parameters, the scene-specific YAML files only highlight the differences compared to the base configuration. For a complete description of each parameter, please refer to the comments in `cfg/base_config.yaml`.

### Example YAML Configuration: `work_life_well_being.yaml`

```yaml
Environment:
  env_core:
    problem_scene: "work_life_well_being"
    episode_length: 300
  Entities:
    - entity_name: 'government'
      entity_args:
        params:
          type: "pension" # central_bank gov

    - entity_name: 'households'
      entity_args:
        params:
          type: 'ramsey' #or OLG Model
          h_max: 2512

    - entity_name: 'market'
      entity_args:
        params:
          type: "perfect"   # ['perfect', 'monopoly', 'monopolistic_competition', 'oligopoly']


    - entity_name: 'bank'
      entity_args:
        params:
          type: 'non_profit'


Trainer:
  house_alg: "ppo"  #The BC Agent can also be chosen in this experiment.
  gov_alg: "rule_based" #The PPO Agent can also be chosen in this experiment.
  firm_alg: "rule_based"
  bank_alg: "rule_based"
  seed: 1
  epoch_length: 300
  cuda: False
#  n_epochs: 300
```
---

## **​5.​**​**Illustrative Experiment**

### Experiment :  Impact of Work–Life Balance on Society

* **Experiment Description:**
  
  Simulate individuals’ strategies for allocating work and leisure across different life‐cycle stages, and measure the long‐term impact on aggregate social production.
* **Experimental Variables:**
  
  * Whether households adopt a work–life balance strategy (Behavior Cloning vs. RL Agent)
  * Level of socio‐economic growth
  * Level of social welfare
* **Baselines:**
  
  Below, we provide explanations of the experimental settings corresponding to each line in the visualization to help readers better understand the results.
  
  * **​base\_bc\_ppo\_100\_OLG (Blue line):​**Households are modeled as ​**Behavior Cloning Agents**​, while the government is a **PPO-based RL Agent** optimizing fiscal policy dynamically.The economy operates under the **OLG Model** with ​**100 households**​, serving as the baseline setting.
  * **​balance\_ppo\_ppo\_100\_OLG (Green line):​**Both households and the government are modeled as **​PPO-based RL Agents.​**The economy operates under the **OLG Model** with **100 households.**
* **Visualization of Results:**

![Individual Q4 P1](../img/Individual%20Q4%20P1.png)

​**Figure 1**​: When households adopt a “work–life balance” strategy, aggregate GDP is lower than under the standard work regime, but after year 60 the gap narrows and the two GDP paths converge.

![Individual Q4 P2](../img/Individual%20Q4%20P2.png)

​**Figure 2**​: When households adopt a “work–life balance” strategy, social welfare increases and remains elevated over time.

* By maximizing individual utility and choosing a work–life balance lifestyle, households experience slower economic growth in the short term compared to the baseline scenario, but social welfare rises markedly. In the long run, the GDP gap between the two scenarios narrows, while the welfare gains from the work–life balance approach persist.

