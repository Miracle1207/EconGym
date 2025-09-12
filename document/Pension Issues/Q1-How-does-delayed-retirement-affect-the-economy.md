# Q1: How does delayed retirement affect the economy?

## 1. Introduction

### 1.1 Delayed Retirement Policy

The **Delayed Retirement Policy** refers to raising the statutory retirement age so that workers remain active in the labor market for a longer period. This directly affects the economic behavior of individuals, firms, governments, and financial markets.

### 1.2 Background of the Study

Globally, population aging has become increasingly severe, placing substantial fiscal pressure on **pension systems**. Many countries are considering or have already implemented delayed retirement policies to relieve pension payment burdens, boost labor supply, and promote economic growth.

### 1.3 Research Questions

This study uses an economic simulation platform to investigate the **economic impacts of delayed retirement policies**, specifically examining:

* **Labor Supply:** How does delayed retirement affect total labor supply?  
* **GDP Effects:** Does delaying retirement contribute to economic growth?  
* **Pension System Sustainability:** How does delayed retirement influence government finances and pension disbursements?  
* **Aging Pressure:** What is the impact of delayed retirement on the overall aging burden in the economy?  
* **Welfare Effects:** How does delayed retirement affect individual benefits and overall social welfare?  

### 1.4 Research Significance

* **Policy Guidance:** Considering that delayed retirement may yield diverse economic effects—such as impacts on labor markets, wage levels, capital markets, and social equity—this research provides strong guidance for policymaking.  
* **Balancing Equity and Efficiency:** The study clarifies how delayed retirement redistributes wealth and opportunities across generations, supporting policy designs that reconcile social fairness with economic efficiency.  

---

## 2. Selected Economic Roles

As an example, the following table shows the economic roles most relevant to the delayed retirement question, along with their role descriptions, observations, actions, and reward functions. For further details, please refer to [our paper](https://arxiv.org/pdf/2506.12110).

| Social Role | Selected Type       | Role Description                                             | Observation                                                  | Action                                                       | Reward                                   |
| ----------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| Individual  | OLG Model           | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals. | $o_t^i = (a_t^i, e_t^i, \text{age}_t^i)$<br/>Private: assets, education, age<br/>Global: distributional statistics | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br/>Asset allocation, labor, investment, old agents $\lambda_t^i = 0$ | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility) |
| Government  | Pension Authority   | Pension Authority manages intergenerational transfers by setting retirement age, contribution rates, and pension payouts. | $o_t^g = \{ B_{t-1}, W_{t-1}, P_{t-1}, \pi_{t-1}, Y_{t-1}, \mathcal{I}_t \}$<br/>Public debt, wage, price level, inflation, GDP, income dist. | $a_t^{\text{pension}} = \{ \text{age}^r, \tau_p, k \}$<br/>Retirement age, contribution rate, growth rate | Pension fund sustainability              |
| Firm        | Perfect Competition | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses. | /                                                            | /                                                            | Zero (long-run)                          |
| Bank        | Non-Profit Platform | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives. | /                                                            | No rate control                                              | No profit                                |

---

### Rationale for Selected Roles

**Individual → Overlapping Generations (OLG) Model**  
In aging-related questions, it is essential to choose the OLG model, since it explicitly models the **age feature**. The Ramsey model instead treats individuals as infinitely-lived households, which is more suitable for questions that are not age-sensitive.  

**Government → Pension Authority**  
For aging and retirement questions, the **Pension Authority** is necessary because it directly controls retirement age and pension transfers. Other government agents (e.g., Fiscal Authority, Central Bank) may also be included in extended experiments.  

**Firm → Perfect Competition**  
When market competition details are not the focus, firms can be modeled as perfectly competitive. This assumes market clearing and eliminates strategic complexity, providing a simplified baseline.  

**Bank → Non-Profit Platform**  
When banking interest-rate dynamics are not the focus, the Non-Profit Platform serves as a neutral financial intermediary, applying uniform rates without profit-seeking behavior.  

---

## 3. Selected Agent Algorithms

This section provides a recommended agent configuration. Users are encouraged to adjust agent types based on the specific needs of their experiments.

| Economic Role | Agent Algorithm        | Description                                                  |
| ------------- | ---------------------- | ------------------------------------------------------------ |
| Individual    | Behavior Cloning Agent | To study policy effects, individuals are trained to mimic realistic human responses. We adopt **behavior cloning** using real-world data (e.g., SCF 2022) to train individual policies. |
| Government    | Rule-Based Agent       | Since delayed retirement is defined by statutory retirement age, this can be directly configured in EconGym as a rule-based policy. |
| Firm          | Rule-Based Agent       | Perfect competition implies market clearing and first-order optimality conditions, consistent with rule-based methods. |
| Bank          | Rule-Based Agent       | Non-Profit Platforms have no interest-rate control authority, and thus can be modeled as rule-based intermediaries. |



---

## 4. 参数设置





## **​4.​**​**Illustrative Experiments**

### Experiment: Impact of Different Retirement Ages on Economic Growth

* **Experiment Description:**
  
  我们测试了不同的退休年龄（RA=60,63,65,67,70）对应的经济效应。
  
* **Visualized Experimental Results：**

![Pension Q2 P1](../img/Pension%20Q2%20P1.png)

**Figure 1:** The yellow, green, and blue lines represent GDP trajectories in a simulated economy of 1,000 households under statutory retirement ages of 70, 65, and 60, respectively. It is observed that economies with earlier retirement ages exhibit higher total GDP, although the difference is less pronounced when household count is 100.

* Delaying retirement does not raise aggregate output in the long run. One reason may be that extended working years reduce households’ time and willingness to consume, interrupting their life-cycle consumption and saving plans.

