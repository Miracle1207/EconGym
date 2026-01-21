
---

# SMFRL: Stackelberg Mean-Field Reinforcement Learning

This subfolder implements the **SMFRL** algorithm, a reinforcement learning framework designed for hierarchical decision-making in large-scale systems.

## 1. About the Algorithm

The SMFRL algorithm is based on the research presented in the following paper. It addresses the complexity of macroeconomic policy learning by modeling the interaction between a central authority (Leader) and a massive population of agents (Followers) as a **Dynamic Stackelberg Mean-Field Game (DSMFG)**.

### Reference

If you use this algorithm in your research, please cite:

```bibtex
@article{mi2025learning,
  title={Learning Macroeconomic Policies through Dynamic Stackelberg Mean-Field Games},
  author={Mi, Qirui and Zhao, Zhiyu and Ma, Chengdong and Xia, Siyu and Song, Yan and Yang, Mengyue and Wang, Jun and Zhang, Haifeng},
  journal={ECAI},
  year={2025}
}

```

* **GitHub Repository**: [Miracle1207/DSMFG](https://github.com/Miracle1207/DSMFG)
* **ArXiv Preprint**: [arXiv:2403.12093](https://arxiv.org/abs/2403.12093)

---

## 2. Core Logic

The algorithm decomposes the hierarchical game into two interdependent learning processes:

* **Follower Logic (`bi_mfrl`)**: Uses Mean-Field Reinforcement Learning to handle the high-dimensional state space of numerous agents.
* **Leader Logic (`bi_ddpg`)**: Uses a DDPG-based approach to optimize macroeconomic policy (e.g., tax or interest rates) given the followers' collective response.

---

## 3. How to Run

To enable the SMFRL algorithm, you must modify the `Trainer` section in your configuration file located in `cfg/`.

### Configuration in EconGym

Within the **EconGym** environment, the Leader and Follower algorithms are tightly coupled to maintain the Stackelberg equilibrium. You **must** enable both simultaneously:

```yaml
# cfg/your_task_config.yaml
Trainer:
  house_alg: "bi_mfrl"   # Follower algorithm
  gov_alg: "bi_ddpg"     # Leader algorithm

```

> [!IMPORTANT]
> **Implementation Note**:
> * **In EconGym**: `bi_mfrl` and `bi_ddpg` are interdependent and must be used together.
> * **In Original DSMFG**: These two components are decoupled and can theoretically be used or replaced independently.
> 
> 

---

## 4. Key Differences from Standard MARL

Unlike standard multi-agent algorithms where agents act simultaneously, SMFRL explicitly models the **temporal priority** of the government (Leader) and the **statistical distribution** (Mean-Field) of the households (Followers). This makes it particularly suitable for tasks involving:

* Fiscal and Monetary policy optimization.
* Large-scale population simulations.

---
