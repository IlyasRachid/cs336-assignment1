import torch
import torch.nn as nn
from .scaled_dot_product_attention import ScaledDotProductAttention
from .linear import Linear
from .rope import RotaryPositionalEmbedding

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "make sure the num_heads divides d_model"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        self.dot_product_attention = ScaledDotProductAttention()

    def forward(
            self,
            in_features: torch.Tensor,
            token_positions: torch.Tensor | None = None
            ):
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)

        q = Q.reshape(*Q.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        k = K.reshape(*K.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        v = V.reshape(*V.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)

        seq_len = Q.shape[-2]
        positions = None
        if token_positions is not None:
            positions = token_positions.unsqueeze(-2)
                
        if self.rope is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        
        
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device)
        )
        attentions = self.dot_product_attention(q, k, v, mask)
        attention = attentions.transpose(-2, -3).contiguous()
        attention = attention.reshape(*Q.shape[:-1], self.num_heads * self.d_k)

        return self.output_proj(attention)
