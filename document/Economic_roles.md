# Core Economic Roles and Agent Modeling in EconGym

EconGym provides a unified framework to simulate heterogeneous economic roles as agents in a Markov game. This document introduces the **core economic roles** and details the **agent modeling setup** (observations, actions, and rewards) for each role.

---

## Core Economic Roles

Each role type captures distinct modeling features and aligns with specific scenarios, enabling flexible composition of diverse simulations.

| **Roles**      | **Role Type**            | **Description**                                              | **Typical Scenarios**                   |
| -------------- | ------------------------ | ------------------------------------------------------------ | --------------------------------------- |
| **Individual** | Ramsey model             | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets. | Wealth distribution, long-term dynamics |
|                | OLG model                | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals. | Retirement policy, demographic shifts   |
| **Government** | Fiscal Authority         | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution. | Fiscal policy, redistribution           |
|                | Central Bank             | Central Bank adjusts nominal interest rates and reserve requirements, transmitting monetary policy to households and firms. | Monetary policy, inflation control      |
|                | Pension Authority        | Pension Authority manages intergenerational transfers by setting retirement age, contribution rates, and pension payouts. | Aging society, pension system reform    |
| **Bank**       | Non-Profit Platforms     | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives. | Simplified setting                      |
|                | Commercial Bank          | Commercial Banks strategically set deposit and lending rates to maximize profits, subject to central bank constraints. | Policy impact on financial markets      |
| **Firm**       | Perfect Competition      | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses. | Equilibrium analysis                    |
|                | Monopoly                 | Monopoly Firms set prices and wages to maximize profits under aggregate demand constraints. | Market power, regulation                |
|                | Oligopoly                | Oligopoly Firms engage in strategic competition, anticipating household responses and rival actions. | Cournot, collusion, AI pricing          |
|                | Monopolistic Competition | Monopolistic Competitors offer differentiated products with CES demand and endogenous entry, supporting studies of consumer preference and market variety. | Branding, pricing strategy              |

---

## Agent Modeling

Building on the economic roles, EconGym models each heterogeneous role as a distinct agent in a **Markov game**.  

- Each agent has a **role-specific observation space**, **action space**, and **reward function**.  
- Given its private observation, each agent selects actions and receives rewards.  
- Environment transitions are defined by economic mechanisms (see Appendix C in the paper).  

This setup allows heterogeneous agents to interact in diverse scenarios.  

---

### MDP Elements for Economic Agents

The following table summarizes the observation $o_t$, action $a_t$, and reward $r_t$ for each agent type. Notation and economic meaning are annotated for clarity.

| **Category**   | **Variant**                  | **Observation $o_t$**                                        | **Action $a_t$**                                             | **Reward $r_t$**                                             |
| -------------- | ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Individual** | **Ramsey model**             | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: wealth distribution, education distribution, wage rate, price_level, lending rate, deposit_rate | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility)                     |
|                | **OLG model**                | $o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$<br/>Private: assets, education, age<br/>Global: same as above | — (same as above)<br/>*OLG*: old agents $\lambda_t^i = 0$    | — (same as above)<br/>*OLG includes pension if retired*      |
| **Government** | **Fiscal Authority**         |\$\$o\_t^g = (\\mathcal{A}\_{t},\\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, B\_{t-1})\$\$  <br> Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, debt. | $a_t^{\text{fiscal}} = ( \boldsymbol{\tau}, G_t )$<br>Tax rates, spending | GDP growth, equality, welfare                                |
|                | **Central Bank**             |\$\$o\_t^g = (\\mathcal{A}\_{t}, \\mathcal{E}\_{t-1}, W\_{t-1}, P\_{t-1}, r^{l}\_{t-1}, r^{d}\_{t-1}, \\pi\_{t-1}, g\_{t-1})\$\$ <br>Wealth distribution, education distribution, wage rate, price level, lending rate, deposit_rate, inflation rate, growth rate. | $a_t^{\text{cb}} = ( \phi_t, \iota_t )$<br>Reserve ratio, benchmark rate | Inflation/GDP stabilization                                  |
|                | **Pension Authority**        | \$\$o\_t^g = ( F\_{t-1}, N\_{t}, N^{old}\_{t}, \\text{age}^r\_{t-1}, \\tau^p\_{t-1}, B\_{t-1}, Y\_{t-1}) \$\$ <br>Pension fund, current population, old individuals number, last retirement age, last contribution rate, debt, GDP | $a_t^{\text{pension}} = ( \text{age}^r_t, \tau^p_t, k )$<br>Retirement age, contribution rate, growth rate | Pension fund sustainability                                  |
| **Bank**       | **Non-Profit Platform**      | /                                                            | No rate control                                              | No profit                                                    |
|                | **Commercial Bank**          | $o_t^{\text{bank}} = ( \iota_t, \phi_t, r^l_{t-1}, r^d_{t-1}, loan, F_{t-1} )$<br>Benchmark rate, reserve ratio, last lending rate, last deposit_rate, loans, pension fund. | $a_t^{\text{bank}} = ( r^d_t, r^l_t )$<br>Deposit, lending decisions | $r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$<br>Interest margin |
| **Firm**       | **Perfect Competition**      | /                                                            | /                                                            | Zero (long-run)                                              |
|                | **Monopoly**                 | $o_t^{\text{mono}} = ( K_t, Z_t, r_{t-1}^l )$<br>Production capital, productivity, lending rate | $a_t^{\text{mono}} = ( p_t, W_t )$<br>Price and wage decisions | $r_t^{\text{mono}} = p_t Y_t - W_t L_t - R_t K_t$<br>Profits = Revenue – costs |
|                | **Oligopoly**                | $o_t^{\text{olig}} = ( K_t^j,  Z_t^j, r_{t-1}^l)$<br>Production capital, productivity, lending rate | $a_t^{\text{olig}} = ( p_t^j, W_t^j )$<br>Price and wage decisions for firm $j$ | $r_t^{\text{olig}} = p_t^j y_t^j - W_t^j L_t^j - R_t K_t^j$<br>Profits = Revenue – costs for firm $j$ |
|                | **Monopolistic Competition** | $o_t^{\text{mono-comp}} = ( K_t^j,  Z_t^j, r_{t-1}^l )$<br> Production capital, productivity, lending rate. Here, $j$ denotes the firm index. | $a_t^{\text{mono-comp}} = ( p_t^j, W_t^j )$<br>Price and wage decisions for firm $j$ | $r_t^{\text{mono-comp}} = p_t^j y_t^j - W_t^j L_t^j - R_t K_t^j$<br>Profits = Revenue – costs for firm $j$ |


---

## Agent Algorithms

| **Agent Algorithm**             | **Description**                                              | **Example Use Case**                                         |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Reinforcement Learning (RL)** | Learns through trial-and-error to optimize long-term cumulative rewards. Well-suited for solving dynamic decision-making problems. | Dynamic environments requiring optimal decision-making.      |
| **Large Language Model (LLM)**  | Generates decisions based on internal knowledge and language understanding. Exhibits human-like behavior patterns. | Simulating realistic decision-making with human-like behavior, or targeting unstructured text data. |
| **Behavior Cloning (BC)**       | Imitates real-world behavior by training on empirical data. Enables realistic micro-level behavior. | Individual households following BC policies from the [2022 Survey of Consumer Finances data](https://www.federalreserve.gov/econres/scfindex.htm). |
| **Economic Method**             | Uses classical rule-based policies from economics literature (e.g., Taylor rule, Saez Tax). Provides direct comparisons between economic theory and AI-based methods. | Central banks (Taylor rule) or fiscal agents (Saez Tax).     |
| **Rule-based Method**           | Encodes domain knowledge or user-defined heuristics (e.g., IMF’s fiscal adjustment rule). Provides interpretable, human-crafted policies. | E.g., “save more when young for retirement”.                 |
| **Real-Data**                   | Replays actual policy trajectories based on historical data (e.g., U.S. federal tax rates). Enables benchmarking against real-world policy outcomes. | U.S. federal tax rates, retirement age schedules.            |

Each algorithm has its own strengths. EconGym supports benchmarking them under the same economic role, or combining different algorithms across roles in a shared scenario. In the following experiments, we showcase how these algorithms generate diverse policy outcomes across tasks.
