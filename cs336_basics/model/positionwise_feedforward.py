import torch 
import torch.nn as nn
import math
from .linear import Linear

class FFN_positionwise(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, dtype=None, device=None):
        super().__init__()

        if d_ff is None:
            d_ff = math.ceil((d_model * 8/3) / 64) * 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype) # d_model -> d_ff
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype) # d_ff -> d_model
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) # d_model -> d_ff


    def forward(self, x: torch.Tensor):
        v = self.w1(x)
        silu = (v) * torch.sigmoid(v)
        glu = silu * self.w3(x)
        return self.w2(glu)