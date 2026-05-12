import torch
from torch import nn
import torchaudio
from safetensors.torch import load_file
import math
from config.loader import get_training_model_kwargs
from layer.tokenizer import STTokenizer

class SpecTTTra(nn.Module):
    def __init__(
        self,
        input_spec_dim, input_temp_dim,
        embed_dim,
        f_clip, t_clip,
        num_heads, num_layers,
        dim_feedforward=2048,
        num_classes=2,
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        f_min=20.0,
        f_max=8000.0,
        expected_samples=96000, # 6s
        pos_drop_rate=0.0,
    ):
        super(SpecTTTra, self).__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.embed_dim = embed_dim
        self.f_clip = f_clip
        self.t_clip = t_clip
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.expected_samples = expected_samples

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=input_spec_dim,
            f_min=f_min,
            f_max=f_max,
            center=True,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        self.tokenizer = STTokenizer(
            input_spec_dim=input_spec_dim,
            input_temp_dim=input_temp_dim,
            t_clip=t_clip,
            f_clip=f_clip,
            embed_dim=embed_dim
        )

        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.transformer_layers,
            num_layers=num_layers
        )

        self.pooling = nn.AdaptiveAvgPool1d(1) # ép từ sequence len L xuống 1
        
        self.classifier = nn.Linear(
            in_features=embed_dim,
            out_features=num_classes
        )

    def forward(self, x):
        # Accept waveform with shape [B, T] or [B, 1, T].
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected input shape [B, {self.expected_samples}] or [B, 1, {self.expected_samples}]")

        x = x.to(torch.float32)

        # Check valid của x.size đã được crop chưa
        if x.size(1) != self.expected_samples:
            raise ValueError(f"Expected [B, {self.expected_samples}], got {tuple(x.shape)}")


        x = self.mel_transform(x)
        x = self.to_db(x)
        
        # If runtime frame count differs, align to configured temporal dim.
        if x.size(-1) < self.input_temp_dim:
            x = nn.functional.pad(x, (0, self.input_temp_dim - x.size(-1)))
        elif x.size(-1) > self.input_temp_dim:
            x = x[:, :, : self.input_temp_dim]
        
        tokens = self.tokenizer(x) # (B, F/f + T/t, Embed_size) = (B, N, E)
        tokens = self.pos_drop(tokens) # (B, N, E)
        output = self.transformer(tokens) # (B, N, E)

        # Vì hiện tại ở dạng (B, N, E) tức là sequence len là N 
        # do đó cần chuyển về (B, E, N) cho phù hợp với đầu vào AvgPool
        output = output.transpose(1, 2) # (B, E, N)
        pooled = self.pooling(output) # (B, E, 1)
        pooled = pooled.squeeze(-1) # (B, E)

        logits = self.classifier(pooled) # (B, num_classes)

        return logits


def load_model(
    weights_path: str,
    num_classes: int = 2,
    variant_name: str | None = None,
) -> nn.Module:
    """Load SpecTTTra with safetensors weights for GASBench-style inference."""
    model_kwargs = get_training_model_kwargs(variant_name=variant_name)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["expected_samples"] = 96000
    model = SpecTTTra(**model_kwargs)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model
