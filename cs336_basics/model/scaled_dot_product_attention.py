import torch
import torch.nn as nn
from .softmax import Softmax

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        d_k = queries.shape[-1]

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = self.softmax(scores, dim=-1)

        return weights @ values