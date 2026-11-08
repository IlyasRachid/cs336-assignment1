import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.d_k = d_k

        positions = torch.arange(max_seq_len, device=device)
        k = torch.arange(d_k // 2, device=device)

        freqs = theta ** (-2*k / d_k)

        angles = positions[:, None] * freqs[None, :]

        self.register_buffer(
            "cos", torch.cos(angles), persistent=False
        )
        self.register_buffer(
            "sin", torch.sin(angles), persistent=False
        )
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions (..., seq_len)
        # cos(token_positions) (..., seq_len, d_k // 2)
        x_pairs = x.reshape(*x.shape[:-1], self.d_k//2, 2) # (..., seq_len, d_k // 2, 2)
        cos_idx = self.cos[token_positions] # (seq_len, d_k/2)
        sin_idx = self.sin[token_positions] # (seq_len, d_k/2)

        a = x_pairs[..., 0]
        b = x_pairs[..., 1]

        a_rot = a * cos_idx - b * sin_idx
        b_rot = a * sin_idx + b * cos_idx

        rotated = torch.stack((a_rot, b_rot), dim=-1)
        return rotated.reshape(x.shape).to(x.dtype)