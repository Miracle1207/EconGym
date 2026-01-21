
---

# Behavior Cloning Agents (BC Agents)

This directory contains the implementation of Behavior Cloning (BC) agents used within the **EconGym** framework. It includes different decision-making architectures tailored for computational economic simulations.

## 1. Core Agent Architectures

We provide two primary agent implementations based on the timing of decision-making and strategic interaction:

| Agent File | Decision Logic | Game Theory Model |
| --- | --- | --- |
| `bc_agent.py` | **Simultaneous** | Agents act concurrently in a standard BC setup. |
| `bc_agent_stackelberg.py` | **Sequential** | Follows a **Stackelberg Game** model (Leader-Follower dynamics). |

---

## 2. Configuration & Scope (`bc_agent_stackelberg.py`)

The Stackelberg variant is currently configured with the following specifications:

* **Input/Output Dimensions**:
* `obs_dim`: 10 or 11
* `action_dim`: 2 or 3


* **Model Architecture**: `MFActor` (Mean-Field Actor).
* **Expert Action Sampling**: Implements `sample_expert_action` using **similarity retrieval based on 6 distinct metrics**.
* **Primary Dataset**: `agents\data\advanced_scfp2022_1110_extension.csv`

> [!NOTE]
> **Customization**: For task-specific modifications or hyperparameter tuning, please focus on:
> `bc_agent_stackelberg.py`, `experience_replay.py`, and `runner.py`.

---

## 3. Testing with Pre-trained Models

To use a pre-trained model when `bc_test=True` is enabled, please note the following constraints:

* **Supported Models**: Currently, only the **Ramsey model** (within `households`) is supported.
* **Model Checkpoint**: The system expects the weights to be stored at:
`agents\behavior_cloning\trained_models\ramsey_bc_binet.pt`

---

## 4. How to Run

To deploy the Stackelberg-based behavior cloning agent for your specific task, follow these steps:

1. Navigate to the `cfg/` directory.
2. Locate the configuration file corresponding to your task.
3. Modify the `house_alg` parameter to use the Stackelberg agent:

```yaml
# Inside your task configuration file (e.g., cfg/your_task.yaml)
house_alg: "bc_stackelberg"

```

4. Ensure other environment parameters (like `bc_test`) are aligned with your objective (training vs. inference).

