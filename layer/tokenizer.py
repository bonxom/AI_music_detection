import torch
from torch import nn
import math
from pos_encoding import SinusoidPosEncoding

class STTokenizer(nn.Module):
    def __init__(
            self,
            input_spec_dim, input_temp_dim,
            t_clip, f_clip,
            embed_dim,
        ):
        super(self, STTokenizer).__init__()
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
            input_dim=input_temp_dim,
            embed_dim=embed_dim,
            clip_size=t_clip
        )

        self.spectro_tokenizer = Tokenizer1D(
            input_dim=input_spec_dim,
            embed_dim=embed_dim,
            clip_size=f_clip
        )

    def foward(self, x):
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
        super(self, Tokenizer1D).__init__()
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
        

