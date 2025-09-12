# Core Economic Roles and Agent Modeling in EconGym

EconGym provides a unified framework to simulate heterogeneous economic roles as agents in a Markov game. This document introduces the **core economic roles** and details the **agent modeling setup** (observations, actions, and rewards) for each role.

---

## Core Economic Roles

Each role type captures distinct modeling features and aligns with specific scenarios, enabling flexible composition of diverse simulations.

| **Roles**       | **Role Type**         | **Description** | **Typical Scenarios** |
|------------------|-----------------------|-----------------|-----------------------|
| **Individual**   | Ramsey model          | Ramsey agents are infinitely-lived households facing idiosyncratic income shocks and incomplete markets. | Wealth distribution, long-term dynamics |
|                  | OLG model             | OLG agents are age-specific and capture lifecycle dynamics between working-age (Young) and retired (Old) individuals. | Retirement policy, demographic shifts |
| **Government**   | Fiscal Authority      | Fiscal Authority sets tax policy and spending, shaping production, consumption, and redistribution. | Fiscal policy, redistribution |
|                  | Central Bank          | Central Bank adjusts nominal interest rates and reserve requirements, transmitting monetary policy to households and firms. | Monetary policy, inflation control |
|                  | Pension Authority     | Pension Authority manages intergenerational transfers by setting retirement age, contribution rates, and pension payouts. | Aging society, pension system reform |
| **Bank**         | Non-Profit Platforms  | Non-Profit Platforms apply a uniform interest rate to deposits and loans, eliminating arbitrage and profit motives. | Simplified setting |
|                  | Commercial Bank       | Commercial Banks strategically set deposit and lending rates to maximize profits, subject to central bank constraints. | Policy impact on financial markets |
| **Firm**         | Perfect Competition   | Perfectly Competitive Firms are price takers with no strategic behavior, ideal for baseline analyses. | Equilibrium analysis |
|                  | Monopoly              | Monopoly Firms set prices and wages to maximize profits under aggregate demand constraints. | Market power, regulation |
|                  | Oligopoly             | Oligopoly Firms engage in strategic competition, anticipating household responses and rival actions. | Cournot, collusion, AI pricing |
|                  | Monopolistic Competition | Monopolistic Competitors offer differentiated products with CES demand and endogenous entry, supporting studies of consumer preference and market variety. | Branding, pricing strategy |

---

## Agent Modeling

Building on the economic roles, EconGym models each heterogeneous role as a distinct agent in a **Markov game**.  

- Each agent has a **role-specific observation space**, **action space**, and **reward function**.  
- Given its private observation, each agent selects actions and receives rewards.  
- Environment transitions are defined by economic mechanisms (see Appendix C in the paper).  

This setup allows heterogeneous agents to interact in diverse scenarios.  

---

## MDP Elements for Economic Agents

The following table summarizes the observation $o_t$, action $a_t$, and reward $r_t$ for each agent type. Notation and economic meaning are annotated for clarity.

| **Category** | **Variant** | **Observation $o_t$** | **Action $a_t$** | **Reward $r_t$** |
|--------------|-------------|-------------------------|--------------------|--------------------|
| **Individual** | **Ramsey model** | $o_t^i = (a_t^i, e_t^i)$<br>Private: assets, education<br>Global: distributional statistics | $a_t^i = (\alpha_t^i, \lambda_t^i, \theta_t^i)$<br>Asset allocation, labor, investment | $r_t^i = U(c_t^i, h_t^i)$ (CRRA utility) |
|              | **OLG model** | $o_t^i = (a_t^i, e_t^i,\text{age}_t^i)$<br/>Private: assets, education, age<br/>Global: distributional statistics | — (same as above)<br/>*OLG*: old agents $\lambda_t^i = 0$ | — (same as above)<br/>*OLG includes pension if retired* |
| **Government** | **Fiscal Authority** | $o_t^g = \{ B_{t-1}, W_{t-1}, P_{t-1}, \pi_{t-1}, Y_{t-1}, \mathcal{I}_t \}$<br>Public debt, wage, price level, inflation, GDP, income dist. | $a_t^{\text{fiscal}} = \{ \boldsymbol{\tau}, G_t \}$<br>Tax rates, spending | GDP growth, equality, welfare |
|              | **Central Bank** | — (same as above) | $a_t^{\text{cb}} = \{ \phi_t, \iota_t \}$<br>Reserve ratio, benchmark rate | Inflation/GDP stabilization |
|              | **Pension Authority** | — (same as above) | $a_t^{\text{pension}} = \{ \text{age}^r, \tau_p, k \}$<br>Retirement age, contribution rate, growth rate | Pension fund sustainability |
| **Bank**     | **Non-Profit Platform** | / | No rate control | No profit |
|              | **Commercial Bank** | $o_t^{\text{bank}} = \{ \iota_t, \phi_t, A_{t-1}, K_{t-1}, B_{t-1} \}$<br>Benchmark rate, reserve ratio, deposits, loans, debts | $a_t^{\text{bank}} = \{ r^d_t, r^l_t \}$<br>Deposit, lending decisions | $r = r^l_t (K_{t+1} + B_{t+1}) - r^d_t A_{t+1}$<br>Interest margin |
| **Firm**     | **Perfect Competition** | / | / | Zero (long-run) |
|| **Monopoly** | $o_t^{\text{mono}} = \{ K_t, L_{t}, Z_t, p_{t-1}, W_{t-1} \}$<br>Capital, labor, productivity, last price/wage | $a_t^{\text{mono}} = \{ p_t, W_t \}$<br>Price and wage decisions | $r_t^{\text{mono}} = p_t Y_t - W_t L_t - R_t K_t$<br>Profits = Revenue – costs |
|| **Oligopoly** | $o_t^{\text{olig}} = \{ K_t^j, L_t^j, Z_t^j, p_{t-1}^j, W_{t-1}^j \}$<br>Firm-specific capital, labor, productivity, last price/wage. Here, $j$ denotes the firm index. | $a_t^{\text{olig}} = \{ p_t^j, W_t^j \}$<br>Price and wage decisions for firm $j$ | $r_t^{\text{olig}} = p_t^j y_t^j - W_t^j L_t^j - R_t K_t^j$<br>Profits = Revenue – costs for firm $j$ |
|| **Monopolistic Competition** | $o_t^{\text{mono-comp}} = \{ K_t^j, L_t^j, Z_t^j, p_{t-1}^j, W_{t-1}^j \}$<br>Firm-specific capital, labor, productivity, last price/wage. Here, $j$ denotes the firm index. | $a_t^{\text{mono-comp}} = \{ p_t^j, W_t^j \}$<br>Price and wage decisions for firm $j$ | $r_t^{\text{mono-comp}} = p_t^j y_t^j - W_t^j L_t^j - R_t K_t^j$<br>Profits = Revenue – costs for firm $j$ |


---
