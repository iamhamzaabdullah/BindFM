"""
BindFM Config Loader
--------------------
Loads YAML training configs and returns (BindFMConfig, TrainingConfig) pairs.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Tuple

from model.bindfm import BindFMConfig
from training.train import TrainingConfig


def load_config(yaml_path: str, key: str) -> Tuple[BindFMConfig, TrainingConfig]:
    """
    Load a named training config from YAML.

    Args:
        yaml_path: Path to training_configs.yaml
        key:       Config name, e.g. "small_stage0", "full_stage2"

    Returns:
        (BindFMConfig, TrainingConfig) tuple
    """
    with open(yaml_path) as f:
        all_configs = yaml.safe_load(f)

    if key not in all_configs:
        available = list(all_configs.keys())
        raise KeyError(f"Config key '{key}' not found. Available: {available}")

    cfg = all_configs[key]

    # Build model config
    model_size = cfg.get("model_size", "small")
    if model_size == "small":
        model_cfg = BindFMConfig.small()
    elif model_size == "medium":
        model_cfg = BindFMConfig.medium()
    elif model_size == "full":
        model_cfg = BindFMConfig.full()
    else:
        model_cfg = BindFMConfig()

    # Override any model fields from YAML
    model_fields = {
        "n_encoder_layers", "d_encoder_hidden", "d_encoder_edge",
        "d_encoder_out", "n_rbf", "cutoff_angst", "n_trunk_layers",
        "d_single", "d_pair", "n_assay_types",
        "d_affinity_hidden", "d_struct_hidden", "d_gen_hidden", "max_gen_atoms",
    }
    for field in model_fields:
        if field in cfg:
            setattr(model_cfg, field, cfg[field])

    # Build training config
    train_fields = {
        "stage", "lr", "min_lr", "weight_decay", "grad_clip",
        "warmup_steps", "total_steps", "batch_size", "accumulate_grad",
        "encoder_lr_scale", "save_every", "eval_every", "checkpoint_dir",
        "use_amp", "decoy_ratio", "freeze_trunk", "freeze_encoder",
    }
    train_cfg = TrainingConfig()
    for field in train_fields:
        if field in cfg:
            setattr(train_cfg, field, cfg[field])

    return model_cfg, train_cfg
