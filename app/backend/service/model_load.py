import os

from torch import nn
from model_audio_input.model import SpecTTTra
from safetensors.torch import load_file
import torch
from config.loader import get_training_model_kwargs

def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load SpecTTTra with safetensors weights for GASBench-style inference."""
    variant_name = os.getenv("AI_MODEL_VARIANT")
    model_kwargs = get_training_model_kwargs(variant_name=variant_name)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["expected_samples"] = 96000
    model = SpecTTTra(**model_kwargs)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def pred(model: SpecTTTra, waveform: torch.Tensor):
    with torch.no_grad():
        logits = model(waveform)  # [1, 2]
        probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    return {
        "pred_idx": pred_idx,
        "probs": probs.squeeze(0).tolist(),
        "logits": logits.squeeze(0).tolist(),
    }
