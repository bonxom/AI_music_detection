import torch
from torch import nn
import torchaudio
from safetensors.torch import load_file
import math

class STTokenizer(nn.Module):
    def __init__(
            self,
            input_spec_dim, input_temp_dim,
            t_clip, f_clip,
            embed_dim,
        ):
        super().__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.t_clip = t_clip
        self.f_clip = f_clip
        self.embed_dim = embed_dim

        # n_token = [(input - kernel_size) / stride + 1]
        # stride = kernel_size = clip_size
        self.num_spec_token = math.floor(
            (input_spec_dim - f_clip ) / f_clip + 1
        )
        self.num_temp_token = math.floor(
            (input_temp_dim - t_clip ) / t_clip + 1
        )
        self.num_tokens = self.num_spec_token + self.num_temp_token

        self.temporal_tokenizer = Tokenizer1D(
            input_dim=input_spec_dim,
            embed_dim=embed_dim,
            clip_size=t_clip
        )

        self.spectro_tokenizer = Tokenizer1D(
            input_dim=input_temp_dim,
            embed_dim=embed_dim,
            clip_size=f_clip
        )

    def forward(self, x):
        temp_input = x # (B, F, T)
        temp_tokens = self.temporal_tokenizer(temp_input) # (B, T/t_clip, embed_dim)

        spec_input = x.permute(0, 2, 1) # (B, T, F)
        spec_tokens = self.spectro_tokenizer(spec_input) # (B, F/f_clip, embed_dim)

        spectro_temporal_tokens = torch.cat(
            (temp_tokens, spec_tokens),
            dim=1
        ) # (B, T/t + F/f, embed_dim)

        return spectro_temporal_tokens


    
class Tokenizer1D(nn.Module):
    def __init__(
        self,
        input_dim, 
        embed_dim,
        clip_size, # thời gian hoặc tần số
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=clip_size, # trượt dọc theo trục (thời gian hoặc tần số)
            stride=clip_size, # kernel và stride cho thấy mỗi lần lấy 1 t_clip, ko chồng lấn -> ~len/t_clip
        )
        self.act = nn.GELU()
        self.pos_encoder = SinusoidPosEncoding(token_dim=embed_dim) 

    def forward(self, x): # x shape = (input_channel, chiều dài) 
                         #shape = (F, T) or (T, F). Ví dụ với (F, T): Temp_tokenizer
        x = self.conv1d(x) # (embed_dim, T / t_clip)
        x = self.act(x)
        x = x.transpose(1, 2) # ( T/t_clip, embed_dim) -> to pos_encode
        x = self.pos_encoder(x) # k cần norm ở cuối vì dùng sinusoid, nếu dùng learn pe thì cần thêm norm để chuẩn hóa
        return x # (T/t_clip, embed_dim)
        #(B, F, T)
# -> cắt theo time thành các đoạn dài t_clip
# -> mỗi đoạn thành 1 vector dim = embed_dim
# -> ra (B, T/t_clip, embed_dim)
        

class SinusoidPosEncoding(nn.Module):
    def __init__(self, token_dim, max_len=5000):
        super(SinusoidPosEncoding, self).__init__()
        pe = torch.zeros(max_len, token_dim) # shape(max_len, token_dim)
        position = torch.arange(0, max_len, dtype=torch.float) #  (maxlen, )
        position = position.unsqueeze(1) # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, token_dim, 2, dtype=torch.float) # 2 -> step = 2
            * (-torch.log(torch.tensor(10000.0)) / token_dim)
        ) # (token_dim // 2)
        # even pos
        pe[:, 0::2] = torch.sin(position * div_term) # (max_len, token_dim // 2)
        # odd pos
        pe[:, 1::2] = torch.cos(position * div_term) # (max_len, token_dim // 2)
        
        pe = pe.unsqueeze(0) # (1, max_len, token_dim)

        #register_buffer dùng để đăng ký một tensor vào nn.Module 
        # như một phần state của model nhưng không phải parameter để học. 
        # Trong docs, PyTorch nói buffer là “non-learnable aspects of computation”; 
        # buffer mặc định là persistent, nên sẽ được lưu trong state_dict, 
        # và cũng được truy cập như một thuộc tính của module.

        self.register_buffer("pe", pe)


    def forward(self, x):
        # shape: (batch_size, seq_len, token_dim)
        x = x + self.pe[:, : x.size(1), :] # gắn thêm pos encoding vào x
        return x

class SpecTTTra(nn.Module):
    def __init__(
        self,
        input_spec_dim, input_temp_dim,
        embed_dim,
        f_clip, t_clip,
        num_heads, num_layers,
        num_classes=2,
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        f_min=20.0,
        f_max=8000.0,
        expected_samples=80000, # 5s
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
        # Accept waveform with shape [B, 80000] or [B, 1, 80000].
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError("Expected input shape [B, 80000] or [B, 1, 80000]")

        x = x.to(torch.float32)

        # Enforce fixed 5s length for stable token dimensions.
        if x.size(1) < self.expected_samples:
            pad = self.expected_samples - x.size(1)
            x = nn.functional.pad(x, (0, pad))
        elif x.size(1) > self.expected_samples:
            x = x[:, : self.expected_samples]

        x = self.mel_transform(x)
        x = self.to_db(x)
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-8)

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


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load SpecTTTra with safetensors weights for GASBench-style inference."""
    model = SpecTTTra(
        input_spec_dim=128,
        input_temp_dim=157,
        embed_dim=256,
        f_clip=8,
        t_clip=4,
        num_heads=8,
        num_layers=4,
        num_classes=num_classes,
    )
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model
