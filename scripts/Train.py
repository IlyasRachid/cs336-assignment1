import dataclasses
import torch
import numpy as np
from pathlib import Path
import time
import json

from cs336_basics.paths import ROOT, TOKENS, RUNS
from cs336_basics.training.adamW import AdamW
from cs336_basics.training.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.training.get_batch import get_batch
from cs336_basics.training.cross_entropy import cross_entropy
from cs336_basics.training.learning_rate_scheduling import cosine_schedule
from cs336_basics.training.gradient_clipping import gradient_clip
from cs336_basics.training.monitoring import WandbMonitor
from cs336_basics.config import parse_configs, build_model

### Helper functions ###
def group_of(name):
    parts = name.split(".")
    return ".".join(parts[:-1]) or name

@torch.no_grad()
def _grouped_l2_norms(model, *, gradients, metric_prefix):
    squared_norms = {}
    for name, p in model.named_parameters():
        tensor = p.grad if gradients else p
        if tensor is None:
            continue

        key = group_of(name)
        contribution = tensor.detach().float().square().sum()
        if key in squared_norms:
            squared_norms[key] = squared_norms[key] + contribution
        else:
            squared_norms[key] = contribution

    return {
        f"{metric_prefix}/{key}": float(squared_norm.sqrt().item())
        for key, squared_norm in squared_norms.items()
    }


def gradient_norms(model):
    return _grouped_l2_norms(model, gradients=True, metric_prefix="gradnorm")


def weight_norms(model):
    return _grouped_l2_norms(model, gradients=False, metric_prefix="weightnorm")


def project_path(configured_path, default_path):
    path = Path(configured_path) if configured_path else default_path
    return path if path.is_absolute() else ROOT / path

@torch.no_grad()
def evaluate(model, valid_data, batch_size, context_length, device,  n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(valid_data, batch_size, context_length, device)
        losses.append(cross_entropy(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


### Training loop ###
def main():
    model_cfg, train_cfg = parse_configs()

    run_dir = RUNS / train_cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = project_path(train_cfg.ckpt_path, run_dir / "ckpt.pt")
    log_path = project_path(train_cfg.log_path, run_dir / "train_log.jsonl")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = project_path(train_cfg.train_path, TOKENS / "tinystories_train.bin")
    valid_path = project_path(train_cfg.valid_path, TOKENS / "tinystories_valid.bin")

    with open(run_dir / "config.json", "w") as f:
        json.dump({"model": dataclasses.asdict(model_cfg),
                "train": dataclasses.asdict(train_cfg),
                "resolved": {"train_path": str(train_path), "valid_path": str(valid_path),
                                "ckpt_path": str(ckpt_path), "log_path": str(log_path)}},f, indent=2)

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(valid_path, dtype=np.uint16, mode="r")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(model_cfg).to(device)

    optimizer = AdamW(
        model.parameters(), 
        lr=train_cfg.lr_max, 
        betas=(train_cfg.beta1, train_cfg.beta2), 
        weight_decay=train_cfg.weight_decay, 
        eps=train_cfg.eps
        )

    start_it = 0
    if ckpt_path.exists():
        start_it = load_checkpoint(ckpt_path, model, optimizer)

    monitor_config = {
        "model": dataclasses.asdict(model_cfg),
        "train": dataclasses.asdict(train_cfg),
        "resolved": {
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "ckpt_path": str(ckpt_path),
            "log_path": str(log_path),
        },
    }
    with WandbMonitor(
        run_dir=run_dir,
        run_name=train_cfg.run_name,
        project=train_cfg.wandb_project,
        config=monitor_config,
        enabled=train_cfg.use_wandb,
    ) as monitor:
        with open(log_path, "a") as log_file:
            t0 = time.time()
            model.train()
            for it in range(start_it, train_cfg.cosine_iters):
                lr = cosine_schedule(
                    it,
                    train_cfg.lr_max,
                    train_cfg.lr_min,
                    train_cfg.warmup_iters,
                    train_cfg.cosine_iters,
                )
                for g in optimizer.param_groups:
                    g["lr"] = lr

                x, y = get_batch(train_data, train_cfg.batch_size, model_cfg.context_length, device)

                loss = cross_entropy(model(x), y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                log_train_stats = it % 50 == 0
                if log_train_stats:
                    # Capture raw gradients before clipping changes their norms.
                    grads_train = gradient_norms(model)

                gradient_clip(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

                if log_train_stats:
                    # Capture weights after the optimizer update for this iteration.
                    train_stats = dict(grads_train)
                    train_stats.update(weight_norms(model))
                    rec_train = {"it": it, "split": "train", "loss": loss.item(), "lr": lr, "time": time.time() - t0}
                    rec_train.update(train_stats)
                    monitor.log_train(
                        loss=rec_train["loss"],
                        learning_rate=lr,
                        iteration=it,
                        elapsed=rec_train["time"],
                        metrics=train_stats,
                    )
                    log_file.write(json.dumps(rec_train) + "\n")
                    log_file.flush()

                if it % train_cfg.eval_interval == 0 or it == train_cfg.cosine_iters - 1:
                    eval_loss = evaluate(
                        model,
                        valid_data,
                        train_cfg.batch_size,
                        model_cfg.context_length,
                        device,
                        n_batches=20,
                    )
                    rec_val = {"it": it, "split": "valid", "loss": eval_loss, "lr": lr, "time": time.time() - t0}
                    monitor.log_validation(
                        loss=rec_val["loss"],
                        iteration=it,
                        elapsed=rec_val["time"],
                    )
                    log_file.write(json.dumps(rec_val) + "\n")
                    log_file.flush()

                if it % train_cfg.ckpt_interval == 0 and it > 0:
                    save_checkpoint(model, optimizer, it + 1, ckpt_path)

        save_checkpoint(model, optimizer, train_cfg.cosine_iters, ckpt_path)


if __name__ == "__main__":
    main()
        

    
