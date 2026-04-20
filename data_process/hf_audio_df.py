from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled with clear runtime error
    yaml = None

try:
    from datasets import Audio, load_dataset
except ModuleNotFoundError:  # pragma: no cover - handled with clear runtime error
    Audio = None
    load_dataset = None


def _require_dependencies() -> None:
    if yaml is None:
        raise ModuleNotFoundError(
            "Missing dependency 'PyYAML'. Install with: pip install pyyaml"
        )
    if load_dataset is None:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        )


def _read_dataset_catalog(yaml_path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datasets = raw.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("Invalid YAML format: 'datasets' must be a list.")

    catalog: dict[str, dict[str, Any]] = {}
    for item in datasets:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        catalog[name] = item
    return catalog


def _resolve_configs(
    catalog: Mapping[str, dict[str, Any]],
    names: Sequence[str],
    expected_media_type: str,
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    missing = [name for name in names if name not in catalog]
    if missing:
        raise ValueError(f"Dataset name(s) not found in YAML: {missing}")

    for name in names:
        cfg = dict(catalog[name])
        media_type = str(cfg.get("media_type", "")).strip().lower()
        if media_type and media_type != expected_media_type:
            raise ValueError(
                f"Dataset '{name}' has media_type='{media_type}', expected '{expected_media_type}'."
            )
        cfg["name"] = name
        resolved.append(cfg)
    return resolved


def _detect_audio_columns(
    ds: Any,
    cfg: Mapping[str, Any],
    explicit_audio_column: str | None,
) -> list[str]:
    columns: list[str] = []

    if explicit_audio_column:
        columns.append(explicit_audio_column)

    cfg_cols = cfg.get("data_columns")
    if isinstance(cfg_cols, list):
        columns.extend([str(c) for c in cfg_cols if c])

    features = getattr(ds, "features", {}) or {}
    if Audio is not None:
        for col_name, feature in features.items():
            if isinstance(feature, Audio):
                columns.append(col_name)

    common_candidates = [
        "audio",
        "audio_path",
        "speech",
        "wav",
        "mp3",
        "flac",
        "output_audio",
    ]
    existing_cols = set(getattr(ds, "column_names", []) or [])
    for col_name in common_candidates:
        if col_name in existing_cols:
            columns.append(col_name)

    deduped: list[str] = []
    seen = set()
    for col in columns:
        if col not in seen:
            deduped.append(col)
            seen.add(col)
    return deduped


def _safe_audio_extension(audio_obj: Any) -> str:
    if isinstance(audio_obj, dict):
        src_path = audio_obj.get("path")
        if isinstance(src_path, str):
            suffix = Path(src_path).suffix.lower()
            if suffix:
                return suffix
    return ".wav"


def _write_audio_object(audio_obj: Any, output_file: Path) -> bool:
    # HF Audio feature is usually a dict with one or more fields:
    # {"array", "sampling_rate", "path", "bytes"}.
    if isinstance(audio_obj, dict):
        arr = audio_obj.get("array")
        sr = int(audio_obj.get("sampling_rate") or 16000)
        src_path = audio_obj.get("path")
        raw_bytes = audio_obj.get("bytes")

        if arr is not None:
            wave = np.asarray(arr)
            if wave.ndim == 2 and wave.shape[0] <= 8:
                wave = wave.T
            sf.write(output_file, wave, sr)
            return True

        if isinstance(src_path, str) and src_path and Path(src_path).exists():
            shutil.copy2(src_path, output_file)
            return True

        if isinstance(raw_bytes, (bytes, bytearray)):
            if output_file.suffix.lower() == ".wav":
                try:
                    with io.BytesIO(raw_bytes) as bio:
                        wave, sr = sf.read(bio, dtype="float32")
                    sf.write(output_file, wave, sr)
                    return True
                except Exception:
                    pass
            output_file.write_bytes(raw_bytes)
            return True

    if isinstance(audio_obj, str) and Path(audio_obj).exists():
        shutil.copy2(audio_obj, output_file)
        return True

    return False


def _extract_first_audio(row: Mapping[str, Any], candidates: Sequence[str]) -> tuple[Any, str] | tuple[None, None]:
    for col in candidates:
        if col not in row:
            continue
        value = row[col]
        if value is None:
            continue
        return value, col
    return None, None


def build_binary_audio_df_from_yaml(
    yaml_path: str | Path,
    real_dataset_names: Sequence[str],
    synthetic_dataset_names: Sequence[str],
    *,
    hf_config_by_name: Mapping[str, str] | None = None,
    hf_split_by_name: Mapping[str, str] | None = None,
    max_samples_per_dataset: int | None = None,
    max_samples_by_name: Mapping[str, int] | None = None,
    explicit_audio_column_by_name: Mapping[str, str] | None = None,
    output_audio_dir: str | Path = "crop_data/hf_audio",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a binary (real vs synthetic) audio DataFrame from a YAML dataset catalog.

    Returns:
      full_df, train_df, test_df

    Output DataFrame columns (minimum):
      - path (compatible with SonicDataset)
      - audio_path (alias of path)
      - label (real=1, synthetic=0)
    """
    _require_dependencies()

    if not real_dataset_names:
        raise ValueError("real_dataset_names must contain at least one dataset name.")
    if not synthetic_dataset_names:
        raise ValueError("synthetic_dataset_names must contain at least one dataset name.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    catalog = _read_dataset_catalog(yaml_path)
    real_cfgs = _resolve_configs(catalog, real_dataset_names, expected_media_type="real")
    syn_cfgs = _resolve_configs(catalog, synthetic_dataset_names, expected_media_type="synthetic")

    hf_config_by_name = hf_config_by_name or {}
    hf_split_by_name = hf_split_by_name or {}
    max_samples_by_name = max_samples_by_name or {}
    explicit_audio_column_by_name = explicit_audio_column_by_name or {}

    output_root = Path(output_audio_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    per_dataset_counters: dict[str, int] = {}

    def collect_from_one(cfg: Mapping[str, Any], label: int) -> None:
        ds_name = str(cfg["name"])
        repo_id = str(cfg.get("path") or "").strip()
        if not repo_id:
            raise ValueError(f"Dataset '{ds_name}' is missing 'path' in YAML.")

        config_name = hf_config_by_name.get(ds_name)
        split_name = hf_split_by_name.get(ds_name, "train")
        ds = load_dataset(repo_id, name=config_name, split=split_name)

        limit = max_samples_by_name.get(ds_name, max_samples_per_dataset)
        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                return
            ds = ds.select(range(min(limit, len(ds))))

        audio_columns = _detect_audio_columns(
            ds=ds,
            cfg=cfg,
            explicit_audio_column=explicit_audio_column_by_name.get(ds_name),
        )
        if not audio_columns:
            raise ValueError(
                f"No audio column found for dataset '{ds_name}'. "
                "Set explicit_audio_column_by_name for this dataset."
            )

        media_tag = "Real" if label == 1 else "Synthetic"
        out_dir = output_root / ds_name / media_tag.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        per_dataset_counters.setdefault(ds_name, 0)

        for item in ds:
            audio_obj, used_col = _extract_first_audio(item, audio_columns)
            if audio_obj is None:
                continue

            per_dataset_counters[ds_name] += 1
            clip_id = per_dataset_counters[ds_name]
            ext = _safe_audio_extension(audio_obj)
            output_file = out_dir / f"{media_tag.lower()}_{clip_id:06d}{ext}"

            if not _write_audio_object(audio_obj, output_file):
                continue

            rows.append(
                {
                    "filename": output_file.name,
                    "path": str(output_file),
                    "audio_path": str(output_file),
                    "fake_type": media_tag,
                    "label": int(label),
                    "source_dataset": ds_name,
                    "hf_repo": repo_id,
                    "hf_split": split_name,
                    "audio_column": used_col,
                }
            )

    for cfg in real_cfgs:
        collect_from_one(cfg, label=1)
    for cfg in syn_cfgs:
        collect_from_one(cfg, label=0)

    if not rows:
        raise RuntimeError("No audio samples were extracted from selected datasets.")

    full_df = pd.DataFrame(rows)
    if full_df["label"].nunique() < 2:
        raise RuntimeError(
            "Need both classes for binary training. Check real/synthetic dataset selections."
        )

    stratify = full_df["label"] if full_df["label"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    full_df = full_df.reset_index(drop=True)

    return full_df, train_df, test_df
