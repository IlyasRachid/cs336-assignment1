import torch
import torch.nn as nn
from typing import Optional, Callable
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.1):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "lambda": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2) = group["lr"], group["betas"]
            eps, weight_decay = group["eps"], group["lambda"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if not state:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v, t = state["m"], state["v"], state["t"]
                grad = p.grad

                m.mul_(b1).add_(grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                alpha_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.mul_(1 - lr * weight_decay)  
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                state["t"] = t + 1
        return loss