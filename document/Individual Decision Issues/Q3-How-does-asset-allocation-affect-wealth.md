# Q3: How does asset allocation affect wealth?

## **​1. Introduction**

### **1.1 Definition of the Issue**

This study examines how individuals allocate their**​ ​**​**disposable income** between **savings** (low risk, low return) and **risky investments** (e.g., stocks, mutual funds, cryptocurrencies) characterized by high volatility. Specifically, it investigates how to determine the savings–investment mix based on market conditions, personal risk preferences, and life-cycle stage, and explores how different allocation behaviors influence long-term wealth-accumulation trajectories and immediate utility levels.

### **1.2 Research Questions**

This study leverages an economic simulation platform to examine “How do asset‐allocation behaviors (saving vs. investing) affect individual wealth and utility?” It focuses on the following questions:

* How do different asset‐allocation strategies influence the trajectory of individual wealth accumulation?
* What constitutes the optimal asset‐allocation strategy?

### **1.3 Research Significance**

* **Intertemporal Wealth Allocation and Financial Guidance:**  In modern economies, personal finance behaviors critically affect individual economic well‐being. While savings offer stability, their returns are limited; risk‐taking investments can boost wealth but carry uncertainty and downside risk. As financial markets grow increasingly complex, different socio‐economic groups exhibit varying risk tolerances and financial literacies. A simulation platform is therefore essential for testing strategies and uncovering behavioral mechanisms.
* **Knowledge Base for Financial Education:**  Through this simulation experiment, the government can gain deeper insights into citizens’ asset allocation behaviors and their long‐term societal impacts. These findings will help guide financial literacy programs and inform policies that encourage saving or investment.

---

## **​2. Selected Economic Roles**

As an example, we selected the following roles from the social role classification of the economic simulation platform. These roles align with the core understanding of the issue and are convenient to implement from an experimental perspective:

| Social Role            | Selected Type                         | Role Description                                                                                 |
| ------------------------ | --------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Individual             | Ramsey Model                          | Simulates individual saving–investment decisions and long‑term wealth trajectories.            |
| Bank | Commercial Banks                      | Offer a savings channel and set the benchmark interest rate.                                     |
| Bank | No-Arbitrage Platform  | Model the returns and volatility of risky investments.                                           |
| Firm                 | Perfect Competition          | Provides the macro backdrop of economic growth and capital returns, shaping investment behavior. |

### **Individual →Ramsey Model**

* The Ramsey Model analyzes **individuals’ continuous saving and investment decisions over their life spans**, emphasizing intertemporal optimization and utility maximization. It thus captures personal financial behavior and its impact on long-term wealth accumulation.

### **Government → Not Applicable**

* In this experiment, we focus on the interaction between households and financial institutions (providing liquidity/lending or modeling high-risk investments) to assess how different financial behaviors affect individual utility, wealth trajectories, and, by extension, broader economic outcomes. **We do not assign a dedicated government department.**

### **Firm → Perfect Competition**

* A perfectly competitive market provides a baseline backdrop for both economic growth and investment returns. Firm profits influence market yields, and this setting realistically simulates overall growth rates and capital returns, underpinning the valuation of high-risk investments.

### **Bank → Commercial Banks / No-Arbitrage Platform**

* **Commercial Banks:** Act as the core conduit for savings, offering stable returns and liquidity guarantees. Changes in bank deposit rates directly influence asset-allocation decisions.
* **No-Arbitrage Platform:** Represent high-risk investment channels in the market, modeling the uncertain returns of volatile assets. They determine the potential returns and volatility faced by investors in the risk-taking segment.

---

## **​3.Selected Agent Algorithms**

*(This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.)*

| Social Role                           | AI Agent Type     | Role Description                                                                                           |
| --------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| Individual                            | RL Agent          | Optimizes the saving–investment mix, balancing risk and return to maximize wealth and utility.            |
| Commercial Banks                      | Rule‑Based Agent | Supplies stable savings returns and models the interest‑rate mechanism.                                   |
| No-Arbitrage Platform | Rule‑Based Agent | Generates uncertain investment returns, creating a realistic market‑risk environment.                     |
| Market                                | Rule‑Based Agent | Delivers background capital‑return rates and macroeconomic conditions that influence investment behavior. |

### **Individual → RL Agent**

* Reinforcement learning is well suited to optimize decisions in complex, uncertain environments. The agent dynamically adjusts its savings–investment ratio based on historical return experience to maximize intertemporal utility. By contrast, rule‐based agents lack flexibility, and purely data‐driven methods are constrained by static historical training data.

### **Financial Institutions(Commercial Banks) → Rule‐Based Agent**

* Commercial banks operate under well‐defined institutional rules (e.g., fixed or slowly adjusting deposit rates), making rule‐based modeling appropriate for simulating their behavior.

### **Financial Institutions(No-Arbitrage Platform)  → Rule‐Based Agent**

* Risk‐asset returns can be specified to follow a given probability distribution, generating controlled return paths. These institutions do not actively optimize investments but serve as exogenous market conditions.

### **Firm → Rule‐Based Agent**

* Firms’ revenue behaviors derive from production functions and market mechanisms, which can be effectively simulated via rule‐based models of fundamental economic dynamics.

---

## **​4. Illustrative Experiments**

### Experiment 1: Impact of Asset Allocation on Individual Wealth

* **Experiment Description:**
  Compare households that allocate funds to **risky assets** (stocks, crypto) versus **risk‑free assets** (deposits, government bonds) and track long‑run wealth paths.
* **Involved Social Roles: ​**
  * *Individual: ​*Ramsey Model
  * *Financial Institutions: ​*Commercial banks & No-Arbitrage Platform
* **AI Agents**
  * *Households (Option 1):* RL Agent
  * *Households (Option 2):* Rule-Based Agent with fixed savings–investment ratio
  * *Financial Institutions: ​*Rule-Based Agent
* **Experimental Variables:**
  * Risky vs. risk‑free allocation
  * Return‑distribution parameters
  * Final net wealth, consumption level, utility trajectory

```python
# Simulating household wealth under different asset allocation strategies

For each time period:
    For each household:
        # Step 1: Decide how to split money
        - Choose % to invest in risky assets (e.g., stocks)
        - Put the rest in safe assets (e.g., savings)
        # Step 2: Calculate total return
        - Risky part: high return, high risk
        - Safe part: low, stable return
        # Step 3: Update wealth
        - New wealth = previous wealth + investment return - consumption
        # Step 4: Record consumption and satisfaction (utility)

# After many rounds:
    - Compare final wealth, total utility, and inequality across households
```

* **​ Visualized Experimental Results：**

![Individual Q5 P1](../img/Individual%20Q5%20P1.png)

**Figure 1. ​**Long-run individual wealth accumulation under risky-asset allocations (blue bars) versus risk-free allocations (green bars). On average, households that avoid risky investments accumulate greater wealth over time, consistent with empirical findings that many investors underperform when exposed to high-volatility assets.

![Individual Q5 P2](../img/Individual%20Q5%20P2.png)

**Figure 2:** Households that engage in risky investments exhibit lower average working hours.

![Individual Q5 P3](../img/Individual%20Q5%20P3.png)

**Figure 3:** Households with risky investments achieve higher long-term cumulative rewards.

* Households allocating wealth to risky assets accumulate lower long-term personal wealth compared to those that avoid risky investments, consistent with empirical evidence that most investors underperform over time.
* Interestingly, households taking on risky investments work fewer hours on average, which in part contributes to their higher long-term personal utility.

---

### Experiment 2 : Macroeconomic Impact of Asset‑Allocation Behavior

* **Experiment Description:**
  Divide households into two groups—those engaging in risky investments (e.g., equities, cryptocurrencies) and those holding only risk-free assets (e.g., bank deposits, government bonds)—and model societies composed of these different household types. Compare the long-term GDP growth trajectories of each society to simulate how heterogeneous asset-allocation behaviors affect aggregate production over time.
* **Involved Social Roles:**
  * *Individual: ​*Ramsey Model
  * *Financial Institutions: ​*Commercial Banks & No-Arbitrage Platform
* **AI Agents:**
  * *Households (Option A):* RL Agent
  * *Households (Option B): ​*Rule-Based Agent
  * *Financial Institutions: ​*Rule-Based Agent
* **Experimental Variables:**
  * Risky Investment vs. Risk-Free Investment
  * Long-Term GDP Trajectory
* **​ Visualized Experimental Results：**

![Individual Q5 P4](../img/Individual%20Q5%20P4.png)

**Figure 4:** The blue line represents a society composed of households engaging in risky investments, while the green line represents a society of households holding only risk-free assets. Over the long run, the society with risky investments exhibits relatively slower GDP growth.

* Over a 300-year observation period, both societies display nearly identical GDP levels and growth trends for the first 100 years. From year 100 to 150, the risky-investment society’s GDP notably exceeds that of the risk-free society. However, for t > 150, the risk-free society’s GDP clearly surpasses the risky-investment society’s, and the gap widens over time.
* Nevertheless, the risky-investment society’s GDP continues to exhibit a sustained positive growth trend over the long term.

