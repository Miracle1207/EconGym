"""
配置层：ScenarioConfig -> EconGym 能运行的形式。

设计原则："包一层"，不碰 EconGym 内核：
- 复用 main.py 里已经写好的 select_agent / setup_government_agents，不重新实现一遍agent装配逻辑。
- ScenarioConfig 产出一份"场景差异"结构（和 cfg/*.yaml 里那些内置任务长得一模一样），
  与 cfg/base_config.yaml 合并后交给 EconomicSociety。任何没被我们覆盖的字段
  （action_dim/action_space/real_action_max...）都保证来自 EconGym 官方默认值，而不是我们凭空编的。

为什么不直接用 utils.config.load_config 做合并：它的 deep_merge_entities 用 entity_name 做字典 key，
同名 entity 会被折叠成一个——单政府时没问题，但多政府场景（Entities 里有多个 entity_name='government'
条目，参考内置的 dbl_government.yaml / tre_government.yaml）会丢政府。所以合并逻辑在这里自己做：
语义与 load_config 完全一致，只是按"条目"而非按"名字"合并 Entities，天然支持多政府。
"""
from __future__ import annotations

from pathlib import Path

import yaml
from omegaconf import OmegaConf

from .schema import ScenarioConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = REPO_ROOT / "cfg" / "generated"


def to_scene_dict(config: ScenarioConfig) -> dict:
    """把 ScenarioConfig 转成"场景差异"字典，结构与 cfg/*.yaml 完全一致。"""

    # ---- households ----
    household_params: dict = {
        "type": config.households.type,
        "households_n": config.households.households_n,
    }
    for field_name in ("CRRA", "IFE", "stock_alpha"):
        value = getattr(config.households, field_name)
        if value is not None:
            household_params[field_name] = value

    household_entity: dict = {
        "entity_name": "households",
        "entity_args": {"params": household_params},
    }
    if "OLG" in config.households.type:
        olg_block = {}
        if config.households.birth_rate is not None:
            olg_block["birth_rate"] = config.households.birth_rate
        if config.households.initial_working_age is not None:
            olg_block["initial_working_age"] = config.households.initial_working_age
        if olg_block:
            household_entity["entity_args"]["OLG"] = olg_block

    # ---- governments（1~N 个，参考内置 dbl_government.yaml / tre_government.yaml 的多政府结构） ----
    government_entities: list[dict] = []
    for gov in config.governments:
        gov_params: dict = {"type": gov.type}
        if gov.gov_task is not None:
            gov_params["gov_task"] = gov.gov_task
        if gov.tax_type is not None:
            gov_params["tax_type"] = gov.tax_type
        if gov.retire_age is not None:
            gov_params["retire_age"] = gov.retire_age
        if gov.contribution_rate is not None:
            gov_params["contribution_rate"] = gov.contribution_rate
        government_entities.append(
            {"entity_name": "government", "entity_args": {"params": gov_params}}
        )

    # ---- market / bank ----
    market_params: dict = {"type": config.market.type}
    market_entity: dict = {
        "entity_name": "market",
        "entity_args": {"params": market_params},
    }
    if config.market.firm_n != 1:
        market_entity["entity_args"][config.market.type] = {"firm_n": config.market.firm_n}
    bank_entity = {
        "entity_name": "bank",
        "entity_args": {"params": {"type": config.bank.type}},
    }

    # ---- env_core (环境级常量参数，始终生效，见 nl_agent/validation.py 的说明) ----
    env_core: dict = {
        "problem_scene": config.problem_scene,
        "episode_length": config.episode_length,
    }
    for field_name in ("consumption_tax_rate", "estate_tax_rate", "estate_tax_exemption"):
        for gov in config.governments:
            value = getattr(gov, field_name)
            if value is not None:
                env_core[field_name] = value
                break  # 同名冲突由校验层拦截，这里取第一个非空值

    # ---- Trainer ----
    trainer: dict = {
        "house_alg": config.algorithms.house_alg,
        "gov_alg": config.algorithms.gov_alg,
        "firm_alg": config.algorithms.firm_alg,
        "bank_alg": config.algorithms.bank_alg,
        "seed": config.seed,
        "eval_episodes": config.eval_episodes,
        "epoch_length": config.episode_length,
        "cuda": False,
        "test": not config.uses_rl(),  # 非 RL 组合：单次评估；RL 组合：先训练，再单独评估一次
    }
    # 每个政府显式写一个 {type}_gov_alg（main.py 的 setup_government_agents 优先读这个 key，
    # 找不到才回退 gov_alg）——这正是 dbl_government.yaml / tre_government.yaml 的用法。
    for gov_type, alg in config.gov_algs().items():
        trainer[f"{gov_type}_gov_alg"] = alg
    if config.uses_rl():
        trainer["n_epochs"] = config.n_epochs
    if config.uses_llm_agent():
        # 不设置的话会掉回 base_config.yaml 的默认值 "gpt-4o-mini"（不含"/"），
        # agents/llm/llm_agent.py 就走不到 OpenRouter 分支、认不出 OPENROUTER_API_KEY。
        trainer["llm_name"] = "anthropic/claude-sonnet-4.5"

    return {
        "Environment": {
            "env_core": env_core,
            "Entities": government_entities + [household_entity, market_entity, bank_entity],
        },
        "Trainer": trainer,
    }


def build_full_config(config: ScenarioConfig):
    """base_config.yaml + 场景差异 -> 完整可运行的 OmegaConf 配置。

    合并语义与 utils.config.load_config 一致，唯一区别：Entities 按"条目"逐个与 base 里的
    同名模板合并（而不是按 entity_name 折叠成字典），因此同时挂多个 government 不会丢。
    """
    base_cfg = OmegaConf.load(REPO_ROOT / "cfg" / "base_config.yaml")
    scene_dict = to_scene_dict(config)
    scene_cfg = OmegaConf.create(scene_dict)

    merged = OmegaConf.merge(base_cfg, scene_cfg)

    base_by_name = {
        e["entity_name"]: e
        for e in OmegaConf.to_container(base_cfg.Environment.Entities, resolve=True)
    }
    merged_entities = []
    for scene_entity in scene_dict["Environment"]["Entities"]:
        base_entity = base_by_name[scene_entity["entity_name"]]
        merged_entities.append(
            OmegaConf.to_container(
                OmegaConf.merge(OmegaConf.create(base_entity), OmegaConf.create(scene_entity)),
                resolve=True,
            )
        )
    merged.Environment.Entities = OmegaConf.create(merged_entities)
    merged.Environment.env_core.problem_scene = config.problem_scene
    return merged


def write_scenario_yaml(config: ScenarioConfig) -> Path:
    """把"场景差异"YAML 写入 cfg/generated/（供人查看/留档；实际运行用 build_full_config）。"""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    scene_dict = to_scene_dict(config)
    out_path = GENERATED_DIR / f"{config.problem_scene}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(scene_dict, f, sort_keys=False, allow_unicode=True)
    return out_path
