import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str = "cuda"):
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    inputs  = np.stack([x[s : s + context_length] for s in starts])
    targets = np.stack([x[s + 1 : s + context_length + 1] for s in starts])
    return (
        torch.from_numpy(inputs.astype(np.int64)).to(device),
        torch.from_numpy(targets.astype(np.int64)).to(device),
    )