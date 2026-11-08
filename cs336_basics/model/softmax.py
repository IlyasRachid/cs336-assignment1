import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1):
        m = torch.max(x, dim=dim, keepdims=True).values
        exp = torch.exp(x - m)
        return exp / torch.sum(exp, dim=dim, keepdim=True)