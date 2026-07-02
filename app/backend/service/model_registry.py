from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from fastapi import HTTPException
from torch import nn

from app.backend.service.model_load import inspect_model_weights, load_model
from config.loader import get_training_default_variant, get_training_variants

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = APP_ROOT / "model_prob"


@dataclass(frozen=True)
class ModelEntry:
    name: str
    display_name: str
    weights_path: Path
    relative_path: str
    variant: str
    variant_source: str
    num_classes: int
    probability_mode: str


_registry_lock = Lock()
_model_cache: dict[str, nn.Module] = {}
_active_model_name: str | None = None


def _configured_variants() -> set[str]:
    return {
        str(item.get("name", "")).strip()
        for item in get_training_variants()
        if str(item.get("name", "")).strip()
    }


def _infer_variant(weights_path: Path) -> tuple[str, str]:
    env_variant = os.getenv("AI_MODEL_VARIANT")
    if env_variant and env_variant.strip():
        return env_variant.strip(), "env"

    stem = weights_path.stem.lower()
    for variant in sorted(_configured_variants()):
        if variant.lower() in stem:
            return variant, "filename"

    return get_training_default_variant(), "config_default"


def _iter_model_files() -> list[Path]:
    if not MODEL_ROOT.exists():
        return []

    return sorted(
        [path.resolve() for path in MODEL_ROOT.glob("*.safetensors") if path.is_file()],
        key=lambda path: path.name.lower(),
    )


def _build_registry() -> dict[str, ModelEntry]:
    registry: dict[str, ModelEntry] = {}
    for path in _iter_model_files():
        variant, variant_source = _infer_variant(path)
        metadata = inspect_model_weights(str(path))
        name = path.name
        registry[name] = ModelEntry(
            name=name,
            display_name=path.stem.replace("_", " ").title(),
            weights_path=path,
            relative_path=str(path.relative_to(REPO_ROOT)),
            variant=variant,
            variant_source=variant_source,
            num_classes=int(metadata["num_classes"]),
            probability_mode=str(metadata["probability_mode"]),
        )
    return registry


def _serialize_model(model: ModelEntry, active_name: str | None) -> dict[str, object]:
    return {
        "name": model.name,
        "display_name": model.display_name,
        "relative_path": model.relative_path,
        "variant": model.variant,
        "variant_source": model.variant_source,
        "num_classes": model.num_classes,
        "probability_mode": model.probability_mode,
        "active": model.name == active_name,
    }


def _default_model_name(registry: dict[str, ModelEntry]) -> str:
    if not registry:
        raise HTTPException(
            status_code=500,
            detail=f"No .safetensors models were found in {MODEL_ROOT}",
        )

    env_path = os.getenv("AI_MODEL_WEIGHTS_PATH")
    if env_path:
        resolved_path = Path(env_path).expanduser().resolve()
        for model in registry.values():
            if model.weights_path.resolve() == resolved_path:
                return model.name

    env_model_name = os.getenv("AI_ACTIVE_MODEL")
    if env_model_name and env_model_name in registry:
        return env_model_name

    default_variant = get_training_default_variant()
    for model in registry.values():
        if model.variant == default_variant:
            return model.name

    return next(iter(registry))


def list_models() -> list[dict[str, object]]:
    registry = _build_registry()
    active_name = get_active_model_entry().name if registry else None
    return [_serialize_model(model, active_name) for model in registry.values()]


def get_active_model_entry() -> ModelEntry:
    global _active_model_name
    registry = _build_registry()
    if not registry:
        raise HTTPException(
            status_code=500,
            detail=f"No .safetensors models were found in {MODEL_ROOT}",
        )

    with _registry_lock:
        if _active_model_name not in registry:
            _active_model_name = _default_model_name(registry)
        return registry[_active_model_name]


def select_model(model_name: str) -> dict[str, object]:
    global _active_model_name
    registry = _build_registry()
    if model_name not in registry:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_name}' was not found in {MODEL_ROOT.relative_to(REPO_ROOT)}"
            ),
        )

    selected = registry[model_name]
    try:
        with _registry_lock:
            if model_name not in _model_cache:
                _model_cache[model_name] = load_model(
                    weights_path=str(selected.weights_path),
                    num_classes=selected.num_classes,
                    variant_name=selected.variant,
                )
            _active_model_name = model_name
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load model '{model_name}': {exc}",
        ) from exc

    return {
        "message": "Model selected",
        "selected_model": _serialize_model(selected, selected.name),
    }


def get_active_model() -> nn.Module:
    _, model = get_active_model_state()
    return model


def get_active_model_state() -> tuple[ModelEntry, nn.Module]:
    global _active_model_name
    registry = _build_registry()
    if not registry:
        raise HTTPException(
            status_code=500,
            detail=f"No .safetensors models were found in {MODEL_ROOT}",
        )

    with _registry_lock:
        if _active_model_name not in registry:
            _active_model_name = _default_model_name(registry)

        active_model = registry[_active_model_name]
        cached_model = _model_cache.get(active_model.name)
        if cached_model is None:
            cached_model = load_model(
                weights_path=str(active_model.weights_path),
                num_classes=active_model.num_classes,
                variant_name=active_model.variant,
            )
            _model_cache[active_model.name] = cached_model
    return active_model, cached_model


def is_model_loaded(model_name: str | None = None) -> bool:
    target_name = model_name or get_active_model_entry().name
    return target_name in _model_cache
