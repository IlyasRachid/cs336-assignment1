import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )    

        with torch.no_grad():
            nn.init.trunc_normal_(
                self.weight,
                mean=0.0,
                std=std,
                a=-3*std,
                b=3*std,
            )

    def forward(self, x):
        return x @ self.weight.T