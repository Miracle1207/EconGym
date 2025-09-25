# 🌐 EconGym: A Scalable AI Testbed with Diverse Economic Tasks


![EconGym Overview](document/img/EconGym%20V2.jpg "EconGym Structure Overview")

**EconGym** is a modular and scalable simulation testbed designed for AI-driven economic analysis, policy optimization, and algorithm benchmarking. Built upon rigorous economic modeling, it supports **25+ benchmark tasks** involving heterogeneous agents—households, firms, banks, and governments—with structured observation, action, and reward interfaces.

Users can simulate multi-agent economic dynamics by flexibly combining economic roles and agent algorithms (e.g., rule-based, reinforcement learning, large language models), enabling reproducible research across disciplines.

---

## 📘 User Manual

We provide a detailed **User Manual** that covers **25 key economic research questions**, each accompanied by:

- An introduction to the economic issue  
- Recommended choices of economic roles and agent algorithms  
- Baseline experimental results and visualizations  
- A YAML configuration file for running the environment directly

🔗 See the [`document/`](document/) folder for full documentation and research use cases.

Below is an overview of the 25 economic benchmark problems and their corresponding YAML configuration files:

---

### 🏦 Pension Issues

| No.  | Research Question                                           | YAML File                       |
| ---- | ----------------------------------------------------------- | ------------------------------- |
| Q1   | [`How does delayed retirement affect the economy?`](document/Pension%20Issues/Q1-How-does-delayed-retirement-affect-the-economy.md) | `delayed_retirement.yaml`       |
| Q2   | [`Do personal pensions improve security?`](document/Pension%20Issues/Q2-Do-personal-pensions-improve-security.md) | `personal_pension.yaml`         |
| Q3   | [`How does aging impact the macroeconomy?`](document/Pension%20Issues/Q3-How-does-aging-impact-the-macroeconomy.md) | `population_aging.yaml`         |
| Q4   | [`How to close pension funding gaps?`](document/Pension%20Issues/Q4-How-to-close-pension-funding-gaps.md) | `pension_gap.yaml`              |
| Q5   | [`How do pension systems vary across countries?`](document/Pension%20Issues/Q5-How-do-pension-systems-vary-across-countries.md) | `pension_across_countries.yaml` |

### 💰 Fiscal Policy Issues

| No.  | Research Question                                            | YAML File                     |
| ---- | ------------------------------------------------------------ | ----------------------------- |
| Q1   | [`Can consumption taxes boost growth and fairness?`](document/Fiscal%20Policy%20Issues/Q1-Can-consumption-taxes-boost-growth-and-fairness.md) | `consumption_tax.yaml`        |
| Q2   | [`How does inheritance tax affect wealth distribution?`](document/Fiscal%20Policy%20Issues/Q2-How-do-inheritance-tax-affect-wealth-distribution.md) | `estate_tax.yaml`             |
| Q3   | [`Does universal basic income enhance equity?`](document/Fiscal%20Policy%20Issues/Q3-Does-universal-basic-income-enhance-equity.md) | `universal_basic_income.yaml` |
| Q4   | [`How to design optimal tax policies?`](document/Fiscal%20Policy%20Issues/Q4-How-to-design-optimal-tax-policies.md) | `optimal_tax.yaml`            |
| Q5   | [`How does wealth tax impact wealth concentration?`](document/Fiscal%20Policy%20Issues/Q5-How-do-wealth-tax-impact-wealth-concentration.md) | `wealth_tax.yaml`             |

### 🏛️ Monetary Policy Issues

| No.  | Research Question                                            | YAML File                |
| ---- | ------------------------------------------------------------ |--------------------------|
| Q1   | [`How effective are negative interest rates?`](document/Monetary%20Policy%20Issues/Q1-How-effective-are-negative-interest-rates.md) | `negative_interest.yaml` |
| Q2   | [`How to control inflation via monetary policy?`](document/Monetary%20Policy%20Issues/Q2-How-to-control-inflation-via-monetary-policy.md) | `inflation_control.yaml` |
| Q3   | [`What are the long-term effects of quantitative easing?`](document/Monetary%20Policy%20Issues/Q3-What-are-the-long-term-effects-of-quantitative-easing.md) | `QE.yaml`                |
| Q4   | [`How to set optimal bank rate spreads?`](document/Monetary%20Policy%20Issues/Q4-How-to-set-optimal-bank-rate-spreads.md) | `optimal_monetary.yaml`  |
| Q5   | [`How to coordinate monetary and fiscal policies?`](document/Monetary%20Policy%20Issues/Q5-How-to-coordinate-monetary-and-fiscal-policies.md) | `dbl_government.yaml`    |

### ⚖️ Market Competition Issues

| No.  | Research Question                                            | YAML File                       |
| ---- | ------------------------------------------------------------ |---------------------------------|
| Q1   | [`How does technology drive long-term growth?`](document/Market%20Competition%20Issues/Q1-How-does-technology-drive-long-term-growth.md) | `technology.yaml`               |
| Q2   | [`How do monopolies affect resources and welfare?`](document/Market%20Competition%20Issues/Q2-How-do-monopolies-affect-resources-and-welfare.md) | `monopoly.yaml`                 |
| Q3   | [`What is algorithmic collusion in oligopolies?`](document/Market%20Competition%20Issues/Q3-What-is-algorithmic-collusion-in-oligopolies.md) | `oligopoly.yaml`                |
| Q4   | [`How does product diversity affect welfare?`](document/Market%20Competition%20Issues/Q4-How-does-product-diversity-affect-welfare.md) | `monopolistic_competition.yaml` |

### 👤 Individual Decision-Making Issues

| No.  | Research Question                                            | YAML File                         |
| ---- | ------------------------------------------------------------ |-----------------------------------|
| Q1   | [`Does the “996” work culture improve utility and efficiency?`](document/Individual%20Decision%20Issues/Q1-Does-the-996-work-culture-improve-utility-and-efficiency.md) | `work_hard.yaml`                  |
| Q2   | [`How does age affect consumption patterns?`](document/Individual%20Decision%20Issues/Q2-How-does-age-affect-consumption-patterns.md) | `age_consumption.yaml`            |
| Q3   | [`How does asset allocation affect wealth?`](document/Individual%20Decision%20Issues/Q3-How-does-asset-allocation-affect-wealth.md) | `asset_allocation.yaml`           |
| Q4   | [`How does work-life balance impact well-being?`](document/Individual%20Decision%20Issues/Q4-How-does-work-life-balance-impact-well-being.md) | `work_life_well_being.yaml`       |
| Q5   | [`How does over-leveraged consumption impact the economy?`](document/Individual%20Decision%20Issues/Q5-How-does-over-leveraged-consumption-impact-the-economy.md) | `over_leveraged_consumption.yaml` |
| Q6   | [`How do market structures shape consumer behavior?`](document/Individual%20Decision%20Issues/Q6-How-do-market-structures-shape-consumer-behavior.md) | `market_type.yaml`                |

---

## 🔧 Installation

We recommend using `conda` for environment management.

1. **Create and activate a new Python environment**

   ```bash
   conda create -n EconGym python=3.10
   conda activate EconGym
   ```

2. **Install PyTorch**
    Refer to [https://pytorch.org](https://pytorch.org/) for installation instructions specific to your system.

3. **Clone the repository and install dependencies**

   ```bash
   cd EconGym
   pip install -r requirements.txt
   ```

4. **Install MPI support**

   ```bash
   conda install mpi4py
   ```


---

## 🚀 How to Run Simulations

You can either use **predefined economic tasks** (fast start), or **customize your own experiments** (advanced use).

### ✅ Option 1: Run Predefined Scenarios (Fast Start)

We provide ready-to-use YAML configurations under `cfg/`:

```bash
python main.py --problem_scene "optimal_tax"
```

This command loads `cfg/optimal_tax.yaml`, which specifies the simulation setup. Each `--problem_scene` matches a YAML file in `cfg/`. You can customize these YAMLs or create new ones for your own tasks.

💡 For the full list of predefined tasks and their YAML files, refer to the **User Manual** section above.

**Available options for `--problem_scene` include:**

```text
delayed_retirement
universal_basic_income
wealth_tax
market_type
personal_pension
```

To run multi-government coordination, set `--problem_scene` to one of the supported scenarios, such as:
```bash
python main.py --problem_scene "tre_government"
```
Available options include: "tre_government" and "dbl_government".


------

### 🧪 Option 2: Define Your Own Economic Scenario (Advanced Users)

Advanced users can fully customize the problem setup by modifying parameters in a YAML file. You may also create a new file named `"task_name.yaml"` under the `cfg/` directory and run:

```bash
python main.py --problem_scene "task_name"
```

The most important configurations in the YAML file are the **economic roles** and **agent algorithms**. Other parameters can be tailored to your needs and are explained in more detail in our paper.


#### 1. Configure Economic Roles in YAML

Each role is defined with a `type` field. For example:

```yaml
- entity_name: 'government'
  entity_args:
    params:
      type: "pension"

- entity_name: "households"
  entity_args:
    params:
      type: 'OLG'

- entity_name: 'market'
  entity_args:
    params:
      type: "perfect"

- entity_name: 'bank'
  entity_args:
    params:
      type: 'non_profit'
```

**Supported types:**

| Role       | Type Options                                                 |
| ---------- | ------------------------------------------------------------ |
| Household  | `ramsey`, `OLG`                                              |
| Government | `tax`, `pension`, `central_bank`                             |
| Market     | `perfect`, `monopoly`, `oligopoly`, `monopolistic_competition` |
| Bank       | `non_profit`, `commercial`                                   |

#### 2. Select Agent Algorithms

EconGym supports:

- `rule_based` – Hardcoded expert rules (e.g., IMF policy)
- `data_based` – Real policy data (e.g., U.S. retirement age = 67)
- `bc` – Behavior cloning
- `ddpg`, `ppo` – Reinforcement learning
- `llm` – Large language models

Example in YAML:

```yaml
Trainer:
  house_alg: "bc"
  gov_alg: "llm"
  firm_alg: "rule_based"
  bank_alg: "ddpg"
```

You can define your own algorithm in `agents/`.

------



## 📊 Analysis and Visualization

After each simulation, EconGym will store time-series interaction data in `viz/data/`.

### 🔢 Metrics and Evaluation Logic

All evaluation metrics are defined in `runner.py`. In particular:

- `self.init_economic_dict()`: Specifies which economic indicators are initialized and tracked.
- `self._evaluate_agent()`: Uses the `eval_econ` list to determine which variables are saved for visualization and analysis.

Users can freely customize this by modifying `runner.py` to track additional variables.

**Default tracked metrics include:**

```python
eval_econ = ["gov_reward", "house_reward", "social_welfare", "per_gdp", "income_gini",
 "wealth_gini", "years", "GDP", "gov_spending", "house_total_tax",
 "house_income_tax", "house_wealth_tax", "house_wealth", "house_income",
 "house_consumption", "house_pension", "house_work_hours", "total_labor",
 "WageRate", "house_age"]
```

### 📈 Visualization Interface

EconGym includes an interactive dashboard powered by Flask, allowing users to visualize economic metrics over time.

#### 🧭 Step 1: Launch the interface

Run the following command:

```bash
python viz_index.py
```

This will start a local web server.

#### 🌐 Step 2: Open the visualization dashboard

Visit the following address in your browser:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

You will see an interface like this:

<img src="document/img/readme/viz.png" style="width:50%;" />

#### 📊 Step 3: Select and generate charts

* Choose the metric you want to visualize.
* Click **"Generate Chart"** to render the chart.

You’ll see dynamic visualizations similar to the examples below:

---

### 🎥 Dynamic Visualizations

1. **Wealth distribution over time**
   Households under the **Ramsey model**
   ![Wealth](document/img/readme/wealth.gif)

2. **Consumption distribution over time**
   Households under the **Ramsey model**
   ![Consumption](document/img/readme/consumption.gif)

3. **Age-specific consumption over time**
   Individuals under the **OLG model**, with age-heterogeneous behavior
   ![OLG Consumption](document/img/readme/OLG_consumption.gif)

4. **Pension flow over time**
   Under the **OLG model**:

   * Positive values indicate receiving pensions
   * Negative values indicate contributing to pension insurance
     ![Pension](document/img/readme/wealth.gif)

---

### 📈 Time-Series Plots

5. **GDP over time**
   ![GDP](document/img/readme/GDP.png)

6. **Social welfare trajectory**
   ![Social Welfare](document/img/readme/social_welfare.png)

7. **Wealth Gini index over time**
   ![Wealth Gini](document/img/readme/wealth_gini.png)


------

## 📁 Repository Structure

```text
EconGym/
├── README.md               # Project introduction and instructions
├── requirements.txt        # Python dependencies

# 🔧 Configuration and Scenario Definitions
├── cfg/                    # YAML configs for 25+ economic benchmark tasks
│   ├── *.yaml              # Each defines a full scenario setup
│   ├── calibrate*.py       # Scripts for calibrating reward functions and parameters

# 🧠 Agents and Algorithms
├── agents/                 # Implementation of agent types
│   ├── rule_based.py       # Rule-based logic agents
│   ├── data_based_agent.py # Agents trained on real-world data
│   ├── bc_agent.py         # Behavioral cloning agents
│   ├── ddpg_agent.py       # Deep Deterministic Policy Gradient
│   ├── ppo_agent.py        # Proximal Policy Optimization
│   ├── llm_agent.py        # Large language model agent interface
│   ├── models.py           # Shared neural network architectures
│   ├── utils.py            # Agent-level utilities
│   ├── log_path.py         # Logging paths
│   ├── data/, real_data/   # Datasets and resources

# 🧩 Economic Entities and Logic
├── entities/               # Core economic actors
│   ├── household.py        # Household behavior
│   ├── bank.py             # Private bank logic
│   ├── central_bank_gov.py # Central bank policies
│   ├── tax_gov.py          # Tax authority logic
│   ├── pension_gov.py      # Pension management
│   ├── government.py       # Fiscal government logic
│   └── market.py           # Market  = Firm Agent

# 🌍 Simulation Environment
├── env/                    # Environment engine and evaluation logic
│   ├── env_core.py         # Markov game environment core
│   └── evaluation.py       # Performance evaluation logic

# 🏃 Execution Scripts
├── main.py                 # Entry point for YAML-based runs
├── runner.py               # Run controller and training loop
├── arguments.py            # Command-line argument management

# 📊 Visualization and Post-analysis
├── indicator.py            # Generate metrics and export to Excel
├── viz/                    # Data visualization (Flask + charts)
│   ├── chart.py            # Chart generation logic
│   ├── templates/          # HTML templates
│   ├── data/, models/      # Saved results
├── viz_index.py            # Flask app for interactive result exploration

# 🌐 Static Assets for Web Interface
├── static/                 # Web frontend resources
│   ├── css/, js/, img/     # UI elements for Flask dashboard

# ⚙️ Utilities
├── utils/                  # General-purpose tools
│   ├── config.py           # Load and manage configurations
│   ├── episode.py          # Episode buffer and management
│   ├── experience_replay.py# Replay buffer for training
│   ├── normalizer.py       # Data normalization
│   └── seeds.py            # Random seed control
```


## Acknowledgments

We sincerely thank [Prof. Bo Li](https://liboecon.com/) from Peking University for his valuable discussions and feedback throughout the development of this project.  
As an outstanding economist, Prof. Bo Li provided critical guidance on the theoretical foundations of the economic models in EconGym, significantly enhancing the rigor and realism of the platform.
