from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.backend.service.inference import detect_full_crop, detect_random_crop
from app.backend.service.model_registry import (
    get_active_model_entry,
    get_active_model_state,
    is_model_loaded,
    list_models,
    select_model,
)

router = APIRouter(prefix="/ai", tags=["AI Music Detection"])


class SelectModelRequest(BaseModel):
    model_name: str


def _active_model_payload() -> dict[str, object]:
    active_model = get_active_model_entry()
    return {
        "name": active_model.name,
        "display_name": active_model.display_name,
        "relative_path": active_model.relative_path,
        "variant": active_model.variant,
        "variant_source": active_model.variant_source,
        "num_classes": active_model.num_classes,
        "probability_mode": active_model.probability_mode,
    }


async def _save_upload_to_temp(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())
    return tmp_path


async def _run_detection(file: UploadFile, mode: str) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded video file name")

    tmp_path: Path | None = None

    try:
        tmp_path = await _save_upload_to_temp(file)
        active_model, model = get_active_model_state()

        if mode == "full_crop":
            return detect_full_crop(
                file_path=str(tmp_path),
                filename=file.filename,
                model=model,
                active_model=active_model,
            )

        return detect_random_crop(
            file_path=str(tmp_path),
            filename=file.filename,
            model=model,
            active_model=active_model,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        await file.close()
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


@router.get("/health")
def health():
    active_model = get_active_model_entry()
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(active_model.name),
        "active_model": _active_model_payload(),
        "available_models_count": len(list_models()),
        "available_detection_modes": ["random_6s_crop", "full_crop"],
    }


@router.get("/models")
def get_models():
    return {
        "active_model": _active_model_payload(),
        "models": list_models(),
    }


@router.post("/models/select")
def set_active_model(payload: SelectModelRequest):
    return select_model(payload.model_name)


@router.post("/detect")
async def detect_ai_music(file: UploadFile = File(...)):
    return await _run_detection(file, mode="random_6s_crop")


@router.post("/detect/random-crop")
async def detect_random_crop_endpoint(file: UploadFile = File(...)):
    return await _run_detection(file, mode="random_6s_crop")


@router.post("/detect/full-crop")
async def detect_full_crop_endpoint(file: UploadFile = File(...)):
    return await _run_detection(file, mode="full_crop")
