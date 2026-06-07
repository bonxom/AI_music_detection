from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from app.backend.service.data_process import build_contiguous_chunks, preprocess
from app.backend.service.model_load import pred, predict_batch
from app.backend.service.model_registry import ModelEntry


def _base_response(
    *,
    filename: str,
    detection_mode: str,
    active_model: ModelEntry,
    ai_probability: float,
    non_ai_probability: float,
    pred_idx: int,
    logits: list[float],
) -> dict[str, object]:
    return {
        "filename": filename,
        "processing_status": "completed",
        "detection_mode": detection_mode,
        "selected_model": {
            "name": active_model.name,
            "display_name": active_model.display_name,
            "variant": active_model.variant,
            "num_classes": active_model.num_classes,
            "probability_mode": active_model.probability_mode,
        },
        "pred_idx": pred_idx,
        "label": "ai" if pred_idx == 0 else "human",
        "ai_generated_probability": round(ai_probability * 100, 2),
        "ai_generated_probability_score": ai_probability,
        "human_probability_score": non_ai_probability,
        "ai_music_detected": pred_idx == 0,
        "scores": {
            "human": non_ai_probability,
            "ai": ai_probability,
        },
        "logits": logits,
    }


def detect_random_crop(
    file_path: str,
    filename: str,
    model: nn.Module,
    active_model: ModelEntry,
) -> dict[str, object]:
    waveform = preprocess(file_path)
    infer = pred(model, waveform)

    return _base_response(
        filename=filename,
        detection_mode="random_6s_crop",
        active_model=active_model,
        ai_probability=float(infer["ai_prob"]),
        non_ai_probability=float(infer["non_ai_prob"]),
        pred_idx=int(infer["pred_idx"]),
        logits=[float(value) for value in infer["logits"]],
    )


def detect_full_crop(
    file_path: str,
    filename: str,
    model: nn.Module,
    active_model: ModelEntry,
    sample_rate: int = 16000,
    chunk_seconds: int = 6,
    batch_size: int = 32,
) -> dict[str, object]:
    chunk_samples = int(sample_rate * chunk_seconds)
    chunks, starts, total_samples = build_contiguous_chunks(
        file_path,
        sample_rate=sample_rate,
        chunk_samples=chunk_samples,
    )

    device = next(model.parameters()).device
    ai_probs: list[float] = []
    non_ai_probs: list[float] = []
    logits_rows: list[list[float]] = []

    with torch.no_grad():
        for batch_start in range(0, chunks.size(0), batch_size):
            batch = chunks[batch_start : batch_start + batch_size].to(device)
            infer = predict_batch(model, batch)
            ai_probs.extend(float(value) for value in infer["ai_probs"])
            non_ai_probs.extend(float(value) for value in infer["non_ai_probs"])
            logits_rows.extend(
                [[float(logit) for logit in row] for row in infer["logits"]]
            )

    best_idx = int(torch.tensor(non_ai_probs).argmin().item())
    best_start_sample = int(starts[best_idx])
    best_end_sample = min(best_start_sample + chunk_samples, total_samples)
    best_ai_probability = float(ai_probs[best_idx])
    best_non_ai_probability = float(non_ai_probs[best_idx])
    pred_idx = 0 if best_ai_probability >= 0.5 else 1

    response = _base_response(
        filename=filename,
        detection_mode="full_crop",
        active_model=active_model,
        ai_probability=best_ai_probability,
        non_ai_probability=best_non_ai_probability,
        pred_idx=pred_idx,
        logits=logits_rows[best_idx],
    )
    response["full_crop_summary"] = {
        "num_chunks": int(len(non_ai_probs)),
        "count_chunks_non_ai_lt_50": int(
            sum(probability < 0.5 for probability in non_ai_probs)
        ),
        "best_chunk_index": best_idx,
        "best_non_ai_prob": best_non_ai_probability,
        "best_ai_prob": best_ai_probability,
        "best_start_sec": round(best_start_sample / sample_rate, 4),
        "best_end_sec": round(best_end_sample / sample_rate, 4),
        "all_non_ai_probs": non_ai_probs,
    }
    response["source_path"] = Path(file_path).name
    return response
