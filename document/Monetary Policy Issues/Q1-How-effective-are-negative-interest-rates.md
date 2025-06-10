# Q1: How effective are negative interest rates?

## 1. Introduction

### 1.1 Negative Interest Rate Policy

A Negative Interest Rate Policy (NIRP) occurs when a central bank sets certain policy rates below zero to incentivize bank lending and corporate investment, thereby stimulating economic growth. This measure is typically employed during periods of deflationary pressure, insufficient demand, or prolonged economic stagnation.

For example, in January 2016 the Bank of Japan introduced a negative rate on part of its excess reserves, lowering the rate to –0.1%. This policy aimed to counteract long-term stagnation and shrinking domestic demand arising from an aging population by promoting credit expansion and investment.

### 1.2 Research Questions

Using an economic-simulation platform, this study examines the macroeconomic effects of a negative interest rate policy, focusing on:

* **GDP Growth:** Can NIRP effectively boost aggregate output?
* **Wealth and Income Distribution:** Under sustained negative rates, does the distribution of wealth and income change significantly (e.g., does inequality widen or shrink)?

### 1.3 Research Significance

* **Assessing Monetary Policy Effectiveness at Low Rates:**  With many economies at or near the zero lower bound, NIRP has become a final policy tool. Evaluating its impact on output and investment helps delineate its limits and risks.
* **Insights**​**​ into Long-Term Growth Dynamics:**  Countries like Japan have paired NIRP with demographic aging and growth stagnation. Understanding this experience aids in balancing growth and distribution during structural transitions.

---

## ​2. Selected Economic Roles

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                | Role Description                                                                                   |
| ------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Individual             | Ramsey Model                 | Optimize consumption and saving decisions in response to changing interest rates.                  |
| Government             | Central Bank                 | Establish negative interest-rate policy, adjust the policy rate, and monitor inflation targets.    |
| Firm                 | Perfect Competition | Reflect how firms adjust production scale, labor demand, and investment based on financing costs.  |
| Bank | Commercial Banks             | Combine bank lending behavior with policy implementation to comprehensively assess policy impacts. |

### Individual → Ramsey Model

* The Ramsey model assumes **infinitely lived agents** with rational expectations who optimize consumption and saving in response to interest-rate changes. This framework is well suited to analyze how a negative-rate policy influences households’ propensities to consume and save, thereby affecting macroeconomic dynamics.
* Since this experiment focuses on the aggregate impact of negative rates rather than intergenerational decision-making, we employ the Ramsey model instead of an OLG setup.

### Government → Central Bank

* Negative interest-rate policy is a standard monetary-policy tool set and executed by the central bank, encompassing rate setting, reserve requirements, and asset-purchase operations. Compared with the treasury, the central bank is the appropriate authority for simulating systemic effects of rate changes on the economy.

### Firm → Perfect Competition

* In a perfectly competitive market, prices are determined by supply and demand. This setting helps clearly identify the transmission channels through which negative rates affect real output, labor supply, and capital investment, avoiding distortions from market power.

### Bank → Commercial Banks

* Commercial banks **link lending behavior with policy implementation**. Modeling them explicitly allows us to simulate how negative rates prompt banks to expand credit, thereby spurring corporate investment and promoting overall economic growth.

---

## 3. Selected Agent Algorithms

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role            | AI Agent Type          | Role Description                                                                                                                            |
| ------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual             | Behavior Cloning Agent | The BC Agent learns saving and consumption patterns from real household data to simulate behavior under changing interest rates.            |
| Government             | Rule-Based Agent       | Central-bank policy adjustments follow fixed rules.                                                                                         |
| Firm                 | Rule-Based Agent       | The market responds according to supply–demand rules (e.g., lower rates boost investment); the simple rule-based approach ensures control. |
| Bank | Rule-Based Agent       | Commercial banks respond to base-rate changes by expanding loans or cutting deposit rates.                                                  |

### Individual → BC Agent

* The BC Agent can learn household consumption and saving behaviors under different interest-rate environments, capturing non-linear adjustment paths that rule-based agents struggle to model and thus uniting interpretability with practicability.

### Government → Rule-Based Agent

* In this experiment’s crisis context, the central bank enforces a fixed negative-rate policy (e.g., –1%), and the rule-based agent simulates this preset monetary-policy framework; reinforcement-learning approaches may introduce unintended policy deviations.

### Firm → Rule-Based Agent

* The market mechanism hinges on supply, demand, and price adjustments; a rule-based agent can quickly reproduce price responses to interest-rate changes, facilitating clear evaluation of transmission channels and avoiding unnecessary strategic noise.

### Bank → Rule-Based Agent

* Financial institutions’ reactions to rate changes manifest in asset-yield adjustments and lending behavior; a rule-based agent efficiently encodes these stable logics, making it ideal for assessing negative-rate impacts on saving returns and capital allocation.

---

## 4. Illustrative Experiment

### Experiment 1: Evaluation of the Economic Effects of a Negative-Rate Policy

* **Experiment Description: ​**  Compare a baseline economy with one that falls into a crisis in year 10 but implements a negative-interest-rate policy to assess how such a policy aids recovery.
* **Involved Social Roles:**
  * *Government: ​*Central Bank
  * *Firm: ​*Perfectly Competitive Market
  * *Bank: ​*Commercial Banks
* **AI Agents:**
  * *Individual: ​*BC Agent
  * *Government: ​*Rule-Based Agent
  * *Firm:* Rule-Based Agent
  * *Bank: ​*Rule-Based Agent
* **Experimental Variables:**
  * Crisis onset in year 10 vs. a normally operating economy
  * Negative-rate policy (r = –1%)
  * Aggregate GDP
  * Income-inequality indicator (e.g., Gini coefficient)

```Python
# Initial setup: simulate an economic crisis
At time T = 10, the economy enters a crisis scenario 
Simulate an economic downturn by reducing productivity and lowering commodity prices.

# Policy intervention: central bank sets negative interest rate
Central Bank sets nominal interest rate to r = -1%
Commercial banks respond by lowering deposit and lending rates

# Policy transmission mechanism
For each commercial bank:
    - Lower deposit rates for households
    - Increase lending to firms
For each firm:
    - Access cheaper credit
    - Expand investment and hire more workers
    - Anticipate future output growth

# Macroeconomic feedback
Update system-wide variables:
    - Total investment 
    - GDP
    - Employment
```

* **​ Visualized Experimental Results：**

![Monetary Q1 P1](../img/Monetary%20Q1%20P1.png)

**Figure 1:** At Year 10, the orange line represents an economy that has entered a crisis (slowed GDP growth). After the government implements a negative‐rate policy, the crisis economy narrows its GDP gap with the normally growing economy (red line) over the medium and long term, demonstrating the policy’s supportive role in recovery.

![Monetary Q1 P2](../img/Monetary%20Q1%20P2.png)

**Figure 2:** Before the crisis, income inequality in both simulated economies is roughly the same. However, after implementing negative rates, the crisis economy (orange) experiences a steadily rising Gini coefficient, increasingly diverging from the normal economy (red).

* A negative‐rate policy can help restore economic vitality and stabilize GDP growth during a crisis. However, it also widens income inequality, indicating that complementary, more equitable fiscal measures are necessary to prevent the policy from exacerbating the wealth gap.

