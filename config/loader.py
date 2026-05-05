from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().with_name("model_hparams.yaml")


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path).expanduser() if path else CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    return data


def get_model_profile(profile_name: str) -> dict[str, Any]:
    cfg = load_config()
    profiles = cfg.get("model_profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Unknown model profile: {profile_name}")
    return deepcopy(profiles[profile_name])


def get_model_kwargs(profile_name: str) -> dict[str, Any]:
    profile = get_model_profile(profile_name)
    kwargs = profile.get("model_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError(f"model_kwargs of profile '{profile_name}' must be a mapping")
    return deepcopy(kwargs)


def get_preprocess_kwargs(profile_name: str) -> dict[str, Any]:
    profile = get_model_profile(profile_name)
    kwargs = profile.get("preprocess", {})
    if not isinstance(kwargs, dict):
        raise ValueError(f"preprocess of profile '{profile_name}' must be a mapping")
    return deepcopy(kwargs)


def get_training_variants(key: str = "clip_6s_variants") -> list[dict[str, Any]]:
    cfg = load_config()
    training = cfg.get("training", {})
    variants = training.get(key, [])
    if not isinstance(variants, list):
        raise ValueError(f"training.{key} must be a list")
    return deepcopy(variants)


def get_training_default_variant(key: str = "clip_6s_default_variant") -> str:
    cfg = load_config()
    training = cfg.get("training", {})
    name = training.get(key, "alpha")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"training.{key} must be a non-empty string")
    return name.strip()


def get_training_model_kwargs(
    variant_name: str | None = None,
    variants_key: str = "clip_6s_variants",
    default_key: str = "clip_6s_default_variant",
) -> dict[str, Any]:
    variants = get_training_variants(variants_key)
    target_name = variant_name or get_training_default_variant(default_key)
    for item in variants:
        if item.get("name") == target_name:
            kwargs = item.get("model_kwargs", {})
            if not isinstance(kwargs, dict):
                raise ValueError(
                    f"model_kwargs of variant '{target_name}' must be a mapping"
                )
            return deepcopy(kwargs)
    raise KeyError(f"Unknown training variant: {target_name}")
