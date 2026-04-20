import torch
from torch import nn
from layer.tokenizer import STTokenizer

class SpecTTTra(nn.Module):
    def __init__(
        self,
        input_spec_dim, input_temp_dim,
        embed_dim,
        f_clip, t_clip,
        num_heads, num_layers,
        pos_drop_rate=0.0
    ):
        super(SpecTTTra, self).__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.embed_dim = embed_dim
        self.f_clip = f_clip
        self.t_clip = t_clip
        self.num_heads = num_heads
        self.num_layers = num_layers

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
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.transformer_layers,
            num_layers=num_layers
        )

        self.pooling = nn.AdaptiveAvgPool1d(1) # ép từ sequence len L xuống 1
        
        self.classifier = nn.Linear(
            in_features=embed_dim,
            out_features=1
        )

    def forward(self, x):
        # Tránh MelSpectrogram ép shape thành (B, 1, F, T) -> Luôn ở (B, F, T)
        if x.dim() == 4:
            x = x.squeeze(1)
        
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
