import torch
import torch.nn as nn
from typing import Iterable

@torch.no_grad()
def gradient_clip(params: Iterable[nn.Parameter], M: float, eps: float =1e-6):
    params = [p for p in params if p.grad is not None]
    if not params:
        return
    total_sq = sum(p.grad.norm() ** 2 for p in params)
    total_norm = total_sq.sqrt()
    if total_norm > M:
        scale = M / (total_norm + eps)
        for p in params:
            p.grad.mul_(scale)