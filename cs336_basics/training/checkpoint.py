import torch
import os
import typing

def save_checkpoint(
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
        ):
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
        model: torch.nn.Module, 
        optimizer:  torch.optim.Optimizer,
        ):
    
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["iteration"]