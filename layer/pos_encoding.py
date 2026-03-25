import torch
from torch.nn import nn

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