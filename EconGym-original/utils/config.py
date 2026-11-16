from pathlib import Path
from omegaconf import OmegaConf


def deep_merge_entities(base_entities, override_entities):
    """
    Merge two lists of entities by matching `entity_name`. Override values where provided.
    """
    merged = []
    override_map = {e['entity_name']: e for e in override_entities}
    
    for base_entity in base_entities:
        name = base_entity['entity_name']
        if name in override_map:
            merged_entity = OmegaConf.merge(base_entity, override_map[name])
        else:
            merged_entity = base_entity
        merged.append(merged_entity)
    
    # Add new entities that exist in override but not in base
    base_names = {e['entity_name'] for e in base_entities}
    for name, override_entity in override_map.items():
        if name not in base_names:
            merged.append(override_entity)
    
    return merged

def load_config(problem_scene):
    """
    Load base configuration and merge with problem_scene-specific configuration using OmegaConf.
    """
    base_config_path = Path("cfg/base_config.yaml")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base configuration file not found at {base_config_path}")
    base_cfg = OmegaConf.load(base_config_path)
    
    scene_config_path = Path(f"cfg/{problem_scene}.yaml")
    if not scene_config_path.exists():
        raise FileNotFoundError(f"The configuration file for '{problem_scene}' was not found. Please make sure that cfg/{problem_scene}.yaml exists.")

    scene_cfg = OmegaConf.load(scene_config_path)
    
    # For special scenes, use scene config directly
    if problem_scene in {"tre_government", "dbl_government", "sgl_government"}:
        return scene_cfg
    
    # Otherwise, merge base config with scene-specific config
    cfg = OmegaConf.merge(base_cfg, scene_cfg)
    
    # Manually deep-merge Environment.Entities by entity_name
    if (
            'Environment' in base_cfg
            and 'Entities' in base_cfg.Environment
            and 'Environment' in scene_cfg
            and 'Entities' in scene_cfg.Environment
    ):
        base_entities = OmegaConf.to_container(base_cfg.Environment.Entities, resolve=True)
        override_entities = OmegaConf.to_container(scene_cfg.Environment.Entities, resolve=True)
        merged_entities = deep_merge_entities(base_entities, override_entities)
        cfg.Environment.Entities = OmegaConf.create(merged_entities)
    
    # Ensure problem_scene is set
    cfg.Environment.env_core.problem_scene = problem_scene
    return cfg

