# 🌐 EconGym: A Scalable AI Testbed with Diverse Economic Tasks


![EconGym Overview](document/img/EconGym%20V2.jpg "EconGym Structure Overview")

**EconGym** is a modular and scalable simulation testbed designed for AI-driven economic analysis, policy optimization, and algorithm benchmarking. Built upon rigorous economic modeling, it supports **25+ benchmark tasks** involving heterogeneous agents—households, firms, banks, and governments—with structured observation, action, and reward interfaces.

Users can simulate multi-agent economic dynamics by flexibly combining economic roles and agent algorithms (e.g., rule-based, reinforcement learning, large language models), enabling reproducible research across disciplines.



**Paper**: [EconGym: A Scalable AI Testbed with Diverse Economic Tasks](https://arxiv.org/abs/2506.12110)

**Web Page**: [EconGym Official Page](https://miracle1207.github.io/econgym_page/)



---

## 📣 News

[09/26/2025] 🎉 The **EconGym** code is officially released! Dive into the platform and start simulating multi-agent economic dynamics!

[09/19/2025] 🎉 [**EconGym**](https://neurips.cc/virtual/2025/poster/121445) accepted for NeurIPS 2025!



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
python main.py --problem_scene "optimal_monetary"
```

This command loads `cfg/optimal_monetary.yaml`, which specifies the simulation setup. Each `--problem_scene` matches a YAML file in `cfg/`. You can customize these YAMLs or create new ones for your own tasks.

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
│   ├── *.yaml              # Each defines a full scenario setup (e.g., tax models, pension models)
│   ├── calibrate*.py       # Scripts for calibrating reward functions and parameters

# 🧠 Agents and Algorithms
├── agents/                 # Implementation of agent types
│   ├── __init__.py         # Initialization file for the agents module
│   ├── behavior_cloning/   # Behavioral cloning agents (learning from demonstrations)
│   │   ├── __init__.py     # Initialization file for behavior cloning agents
│   │   ├── bc_agent.py     # Main file for behavior cloning agents
│   │   └── trained_models  # Directory for storing trained models
│   ├── data/               # Datasets and resources for agents
│   │   └── advanced_scfp2022_1110.csv  # Example dataset for training
│   ├── data_based_agent.py # Data-based agent trained on real-world data
│   ├── llm/                # Large Language Model (LLM) agent interface
│   │   ├── llm_agent.py    # LLM agent main file
│   │   └── prompts.py      # Prompt templates for LLMs
│   ├── log_path.py         # File for managing log paths and logging configuration
│   ├── models/             # Directory for storing generated trajectory data
│   ├── rl/                 # Reinforcement Learning (RL) agents
│   │   ├── __init__.py     # Initialization for RL agents
│   │   ├── ddpg_agent.py   # DDPG agent
│   │   ├── models.py       # Shared neural network architectures for RL agents
│   │   ├── ppo_agent.py    # PPO agent
│   │   └── sac_agent.py    # SAC agent
│   ├── rule_based/         # Rule-based agents
│   │   ├── __init__.py     # Initialization file for rule-based agents
│   │   ├── bank.py         # Bank logic (non-profit/commercial)
│   │   ├── government.py   # Government policy
│   │   ├── households.py   # Household behavior and decision-making
│   │   ├── market.py       # Market interaction logic (monopoly, perfect competition ...)
│   │   └── rules_core.py   # Core logic for rule-based systems
│   ├── saez.py             # Fiscal policy based on Saez Tax model
│   └── utils.py            # General utility functions for agents

# 🧩 Economic Entities and Logic
├── entities/               # Core economic actors in the system
│   ├── __init__.py         # Initialization file for entities module
│   ├── bank.py             # Private bank logic (including non-profit and commercial types)
│   ├── base.py             # Base class for all entities (inheritance)
│   ├── government.py       # Fiscal government logic (tax collection, etc.)
│   ├── households.py       # Household behavior (savings, consumption, etc.)
│   └── market.py           # Firm interactions in market (wage and price setting)

# 🌍 Simulation Environment
├── env/                    # Environment engine and evaluation logic
│   ├── __init__.py         # Initialization for environment module
│   ├── env_core.py         # Core logic for Markov game environment
│   └── set_observation.py  # Observation setup for all agents (state space)

# 🏃 Execution Scripts
├── main.py                 # Entry point for running the simulation (YAML-based configuration)
├── runner.py               # Controller for running experiments and training loops
├── arguments.py            # Command-line argument management for running scripts

# 📊 Visualization and Post-analysis
├── indicator.py            # Generate performance metrics and export them (e.g., to Excel)
├── viz/                    # Data visualization (Flask + charts)
│   ├── chart.py            # Logic for generating plots and charts
│   ├── templates/          # HTML templates for visualization (Flask dashboard)
│   ├── data/               # Directory for storing visualized data
│   ├── models/             # Saved models used for visualization
│   └── viz_index.py        # Flask app for interactive visualization of simulation results

# 🌐 Static Assets for Web Interface
├── static/                 # Web frontend resources (CSS, JS, Images)
│   ├── css/                # Stylesheets for web UI
│   ├── js/                 # JavaScript files for web UI interactivity
│   └── img/                # Image files for UI elements (e.g., icons)

# ⚙️ Utilities
├── utils/                  # General-purpose utility functions and modules
│   ├── config.py           # Functionality to load and manage configuration files
│   ├── episode.py          # Episode buffer and management logic
│   ├── experience_replay.py# Replay buffer for training reinforcement learning agents
│   ├── normalizer.py       # Data normalization (scaling and transformations)
│   └── seeds.py            # Management of random seeds for reproducibility

```

## 🤖 Natural-Language Interaction Layer (`nl_agent/`)

`nl_agent/` is a thin layer on top of EconGym that lets you drive simulations in plain
language. Given a natural-language economic question, an LLM decides whether EconGym can
model it, produces a structured scenario configuration, validates it, runs the real
simulation, and explains the resulting numbers back in natural language.

The layer **does not modify the EconGym core**. It only calls existing entry points
(`utils.config.load_config`, `main.py`'s `select_agent` / `setup_government_agents`,
`env.env_core.EconomicSociety`, and `runner.Runner`) from the outside.

### Architecture

A question flows through four stages:

1. **Planner** (LLM) — turns the question into one or more `ScenarioConfig` objects.
   Comparison questions (e.g. *"what happens if we raise X?"*) automatically yield a
   *baseline* plus one or more *treatment* configs. If the question is ambiguous, the
   Planner asks a clarifying question instead of guessing; if EconGym cannot model it,
   the Planner explains which capability boundary it hits.
2. **Validation** (pure Python) — checks value ranges, required role combinations, and
   whether each parameter is actually effective under the chosen algorithm. On failure,
   the concrete errors are fed back to the Planner for up to two repair attempts.
3. **Config & run** — each `ScenarioConfig` is merged with `cfg/base_config.yaml` into a
   full config (`build_full_config`; multi-government safe, and a human-readable scene
   diff is also written to `cfg/generated/`), then executed via `EconomicSociety` +
   `Runner`.
4. **Narrator** (LLM) — writes the explanation using **only** the real values returned by
   the simulation (deltas and percentage changes are precomputed in Python), and always
   lists caveats such as early termination or single-seed variance.

### 1. Installation

In addition to the core EconGym dependencies (see [Installation](#-installation) above):

```bash
conda activate EconGym
pip install openai anthropic pydantic pyyaml
```

The `openai` SDK also backs the OpenRouter provider (an OpenAI-compatible API reached by
changing the `base_url`); `anthropic` is only needed for the native Anthropic API.

### 2. Configuration

The default provider is **OpenRouter** with the model `anthropic/claude-sonnet-4.5`.
Create a `.env` file in the repository root (kept local, not committed):

```
OPENROUTER_API_KEY=your_key_here
```

To use the native Anthropic API instead, set both variables:

```
ANTHROPIC_API_KEY=your_key_here
LLM_PROVIDER=anthropic
```

`LLM_PROVIDER` accepts `openrouter` (default) or `anthropic`. Both implement the same
`LLMClient` protocol, so the Planner and Narrator code is unchanged either way.

### 3. Usage

#### Web UI (recommended)

```bash
conda activate EconGym
python nl_agent/webapp.py     # then open http://127.0.0.1:5050
```

The web UI is a two-step workflow:

1. **Plan.** After you submit a question, the Planner returns the proposed economic-agent
   setup, parameters, and per-agent algorithms within a few seconds, along with an
   estimated runtime (rule-based / behavior-cloning = seconds; reinforcement learning =
   minutes; per-step LLM decisions = several minutes). Nothing is simulated yet.
2. **Run.** Click **"确认配置，开始仿真" (Confirm & Run)** to launch the simulation. It runs
   in a background thread while the page polls `GET /api/run_status` to display a live
   **progress bar** — training epochs (`X / n_epochs`) for RL, decision-call count for
   LLM agents, and *config i / N* across comparison runs — so long jobs never look frozen.

If the proposed setup is not what you want, click **"修改配置" (Edit)** to change any field
(entity types, parameters, and algorithms as dropdowns or number inputs; governments can be
added or removed). **"应用修改" (Apply)** re-validates the edited config (schema + semantic
rules) before accepting it.

Backend endpoints: `POST /api/plan`, `POST /api/validate`, `POST /api/run` (returns a
`job_id`), and `GET /api/run_status`.

#### Command line

```bash
python nl_agent/cli.py                # interactive REPL
python nl_agent/cli.py --plan-only    # show the generated config without running
```

The REPL keeps a conversation open: ambiguous questions trigger a follow-up prompt, and the
output includes the explanation, caveats, the exact scenario config used, and the raw
numbers (so you can verify the explanation against the source data).

#### Python API

```python
from nl_agent.pipeline import answer_question

result = answer_question(
    "How would raising the retirement age to 70 affect the pension system and GDP?"
)
print(result.status)
print(result.narrator_output.narrative)
for caveat in result.narrator_output.caveats:
    print("-", caveat)
```

### 4. Package layout

```
nl_agent/
├── schema.py               # Config schema: ScenarioConfig / HouseholdConfig / GovernmentConfig / ...
├── capability_manifest.py  # Hand-maintained manifest of what EconGym can model (fed to the Planner)
├── planner.py              # Natural language  -> ScenarioConfig (PlannerOutput)
├── validation.py           # Semantic rules (e.g. pension requires OLG; parameter-effectiveness checks)
├── config_layer.py         # ScenarioConfig -> full EconGym config (build_full_config)
├── run_scenario.py         # Build EconomicSociety, assemble agents, run Runner, return metrics
├── narrator.py             # Real metrics -> natural-language explanation (NarratorOutput)
├── pipeline.py             # Orchestration entry points: plan_question / run_and_narrate / answer_question
├── llm_client.py           # Provider-agnostic structured-output client (OpenRouter / Anthropic)
├── webapp.py               # Flask web UI
├── cli.py                  # Interactive CLI
└── scenarios.py            # Hand-written example scenarios (no LLM)
```

### 5. Notes and limitations

- **Providers.** OpenRouter (default; structured output via `json_object` + validated
  retries) and the native Anthropic API (structured output via `tool_use`) are supported
  and switched with `LLM_PROVIDER`. OpenAI/DeepSeek extension points are in place — adding
  one is a single new `LLMClient` implementation.
- **Non-determinism.** Planner output is sampled, so the same question may yield slightly
  different parameters (e.g. household count) across calls, changing the absolute
  magnitudes. Directional conclusions (higher/lower) are generally stable.
- **Execution modes.** Non-RL / non-LLM configs use `runner.test()` (seconds). RL configs
  (`ppo`/`ddpg`/`sac`) train via `runner.run()` and are then evaluated once (minutes or
  more at realistic scale). `llm` government/bank/market roles use EconGym's built-in
  `agents/llm/llm_agent.py`, making one real API call per step; validation caps
  `episode_length × eval_episodes ≤ 150` to keep cost bounded.
- **Default algorithms.** Households default to `bc` (behavior cloning on real U.S. SCF
  data, no training cost); government/firm/bank default to `rule_based`. The Planner
  upgrades to more expensive algorithms per question type — `ppo`/`ddpg`/`sac` for
  optimal/adaptive-behavior questions, `llm` for "reason like an expert decision-maker"
  questions. See `capability_manifest.py` §5.
- **Multi-government coordination.** `ScenarioConfig.governments` is a list of 1–3 distinct
  government types, enabling fiscal + central-bank (cf. the built-in `dbl_government` task)
  or fiscal + central-bank + pension (`tre_government`). Each government may set its own
  `alg`. The first government is the `main_gov` (households' taxes/pensions and macro
  metrics resolve against it). Multi-government merging is handled by `build_full_config`
  rather than `utils.config.load_config`, whose `deep_merge_entities` collapses entities by
  name and would drop duplicate `government` entries.
- **Parameter effectiveness.** Many YAML parameters only take effect under specific
  algorithms — e.g. tax parameters (`tau`/`xi`) are overridden by hardcoded values under
  `gov_alg=rule_based`, and `CRRA`/`IFE` do not affect decisions under non-RL household
  algorithms. The full mapping lives in `capability_manifest.py`.
- **Degenerate Gini values.** When an aggressive policy (e.g. a heavy wealth tax chosen by
  an `llm`/`ppo` government) drives the economy toward collapse, EconGym's `gini_coef()`
  may return values slightly outside `[0, 1]` (or negative). This is a numerical edge case
  in EconGym itself, not a bug in this layer; such values must not be read as "more equal."
- **`agents/llm/llm_agent.py`.** Originally keyed on `OPENAI_API_KEY`; it now routes through
  OpenRouter (`OPENROUTER_API_KEY`) when `llm_name` contains a `/` (e.g.
  `anthropic/claude-sonnet-4.5`), without affecting existing OpenAI/DeepSeek usage. It
  still supports government/bank/market roles only, not households.

---

## Citation and Acknowledgments

We would like to express our sincere gratitude to [Prof. Bo Li](https://liboecon.com/) from Peking University for his invaluable discussions and feedback throughout the development of this project. As a distinguished economist, Prof. Li provided critical insights into the theoretical foundations of the economic models in EconGym, significantly enhancing the platform’s rigor and realism.

If you use `EconGym` in your research or projects, we kindly ask you to cite the following:

```bibtex
@inproceedings{mi2025econgym,
  title={EconGym: A Scalable AI Testbed with Diverse Economic Tasks},
  author={Qirui Mi and Qipeng Yang and Zijun Fan and Wentian Fan and Heyang Ma and Chengdong Ma and Siyu Xia and Bo An and Jun Wang and Haifeng Zhang},
  journal={Advances in Neural Information Processing Systems},
  year={2025},
}
```

