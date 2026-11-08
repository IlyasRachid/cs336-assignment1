import torch 
import torch.nn as nn

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    c = logits.amax(dim=-1, keepdim=True)
    shifted = logits - c
    lse = torch.logsumexp(shifted, dim=-1)
    target_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (lse - target_logits).mean()

