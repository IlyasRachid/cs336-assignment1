import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        assert d_model != 0, "d_model should be bigger than 0"
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device)
        )


    def forward(self, x: torch.Tensor):
        in_type = x.dtype
        x = x.to(torch.float32) # prevents overflowing
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        x = x.to(in_type)
        return x * self.weight
