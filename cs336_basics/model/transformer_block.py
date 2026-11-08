import torch
import torch.nn as nn
from .multihead_self_attention import MultiHeadSelfAttention
from .positionwise_feedforward import FFN_positionwise
from .rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                theta: int | None = None, 
                max_seq_len: int | None = None
            ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta) # rope is implemented in here

        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN_positionwise(d_model, d_ff)

    def forward(self, x):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        attentions = x + self.attn(self.ln1(x), token_positions)
        return attentions + self.ffn(self.ln2(attentions))