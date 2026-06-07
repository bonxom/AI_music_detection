from __future__ import annotations

import os
from functools import lru_cache

import torch
from safetensors.torch import load_file
from torch import nn

from config.loader import get_training_model_kwargs
from model_audio_input.model import SpecTTTra


@lru_cache(maxsize=32)
def inspect_model_weights(weights_path: str) -> dict[str, int | str]:
    state_dict = load_file(weights_path)

    classifier_weight = state_dict.get("classifier.weight")
    classifier_bias = state_dict.get("classifier.bias")

    if classifier_weight is not None:
        num_classes = int(classifier_weight.shape[0])
    elif classifier_bias is not None:
        num_classes = int(classifier_bias.numel())
    else:
        raise KeyError(
            f"Unable to determine classifier output size for weights: {weights_path}"
        )

    probability_mode = "sigmoid_non_ai" if num_classes == 1 else "softmax_ai_human"
    return {
        "num_classes": num_classes,
        "probability_mode": probability_mode,
    }


def load_model(
    weights_path: str,
    num_classes: int | None = None,
    variant_name: str | None = None,
) -> nn.Module:
    """Load SpecTTTra with safetensors weights using the inferred classifier size."""
    variant_name = variant_name or os.getenv("AI_MODEL_VARIANT")
    if num_classes is None:
        metadata = inspect_model_weights(weights_path)
        num_classes = int(metadata["num_classes"])

    model_kwargs = get_training_model_kwargs(variant_name=variant_name)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["expected_samples"] = 96000

    model = SpecTTTra(**model_kwargs)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_batch(model: SpecTTTra, waveform_batch: torch.Tensor) -> dict[str, object]:
    with torch.no_grad():
        logits = model(waveform_batch)

    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)

    output_dim = int(logits.shape[-1])

    if output_dim == 1:
        non_ai_probs = torch.sigmoid(logits.view(-1))
        ai_probs = 1.0 - non_ai_probs
        pred_indices = torch.where(
            ai_probs >= 0.5,
            torch.zeros_like(ai_probs, dtype=torch.long),
            torch.ones_like(ai_probs, dtype=torch.long),
        )
        probability_mode = "sigmoid_non_ai"
    else:
        probs = torch.softmax(logits, dim=-1)
        ai_probs = probs[:, 0]
        non_ai_probs = probs[:, 1] if probs.shape[-1] > 1 else 1.0 - ai_probs
        pred_indices = torch.argmax(probs, dim=-1)
        probability_mode = "softmax_ai_human"

    return {
        "pred_indices": pred_indices.tolist(),
        "ai_probs": ai_probs.tolist(),
        "non_ai_probs": non_ai_probs.tolist(),
        "logits": logits.tolist(),
        "output_dim": output_dim,
        "probability_mode": probability_mode,
    }


def pred(model: SpecTTTra, waveform: torch.Tensor) -> dict[str, object]:
    batch_result = predict_batch(model, waveform)
    pred_indices = batch_result["pred_indices"]
    ai_probs = batch_result["ai_probs"]
    non_ai_probs = batch_result["non_ai_probs"]
    logits = batch_result["logits"]

    return {
        "pred_idx": int(pred_indices[0]),
        "ai_prob": float(ai_probs[0]),
        "non_ai_prob": float(non_ai_probs[0]),
        "logits": logits[0],
        "output_dim": int(batch_result["output_dim"]),
        "probability_mode": str(batch_result["probability_mode"]),
    }
