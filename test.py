import torch
import torch.nn as nn

from collections import defaultdict

from cs336_basics.model.transformer_lm import Transformer

model = Transformer(
    vocab_size = 68,
    context_length = 32,
    num_layers = 2,
    d_model = 6,
    num_heads = 3,
    d_ff = 16,
    rope_theta = 1000
)

for name, p in model.named_parameters():
    print(name)


def group_of(name):
    parts = name.split(".") # layers.1.ffn.w3.weight, ln_final.weight...
    return ".".join(parts[:-1]) or name

@torch.no_grad()
def param_stats(model):
    g2, w2 = defaultdict(float), defaultdict(float)
    for name, p in model.named_parameters():
        k = group_of(name)
        w2[k] += p.pow(2).sum().item()
        if p.grad is not None:
            g2[k] += p.grad.pow(2).sum().item()
    return (
        {f"gradnorm/{k}": v**0.5 for k, v in g2.items()},
        {f"weightnorm/{k}": v**0.5 for k, v in w2.items()}
    )