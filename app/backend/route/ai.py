from __future__ import annotations

import os
import tempfile
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, File, HTTPException, UploadFile
from torch import nn

from app.backend.service.model_load import load_model, pred
from app.backend.service.data_process import preprocess
from config.loader import get_training_default_variant

router = APIRouter(prefix="/ai", tags=["AI Music Detection"])

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "model_alpha.safetensors"
MODEL_WEIGHTS_PATH = Path(
    os.getenv("AI_MODEL_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH))
).expanduser()

_model: nn.Module | None = None
_model_lock = Lock()


def _get_active_variant() -> tuple[str, str]:
    env_variant = os.getenv("AI_MODEL_VARIANT")
    if env_variant and env_variant.strip():
        return env_variant.strip(), "env"
    return get_training_default_variant("clip_6s_default_variant"), "config_default"


def _get_model() -> nn.Module:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_WEIGHTS_PATH.exists():
                    raise HTTPException(
                        status_code=500,
                        detail=f"Model weights not found: {MODEL_WEIGHTS_PATH}",
                    )
                _model = load_model(str(MODEL_WEIGHTS_PATH))
    return _model


@router.get("/health")
def health():
    model_variant, model_variant_source = _get_active_variant()
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "weights_path": str(MODEL_WEIGHTS_PATH),
        "model_variant": model_variant,
        "model_variant_source": model_variant_source,
    }


@router.post("/detect")
async def detect_ai_music(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio file name")

    suffix = Path(file.filename).suffix or ".wav"
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(await file.read())

        waveform = preprocess(str(tmp_path))
        infer = pred(_get_model(), waveform)

        probs = infer["probs"]
        pred_idx = int(infer["pred_idx"])
        ai_score = float(probs[0]) if len(probs) > 0 else 0.0
        human_score = float(probs[1]) if len(probs) > 1 else 0.0

        return {
            "filename": file.filename,
            "pred_idx": pred_idx,
            "label": "ai" if pred_idx == 0 else "human",
            "ai_music_detected": pred_idx == 0,
            "scores": {
                "human": human_score,
                "ai": ai_score,
            },
            "logits": infer["logits"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        await file.close()
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
