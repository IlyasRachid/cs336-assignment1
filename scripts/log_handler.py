import json
from pathlib import Path
import matplotlib.pyplot as plt
from cs336_basics.paths import RUNS

def load_log(path):
    recs = [json.loads(line) for line in open(path)]
    split = lambda s: [r for r in recs if r["split"] == s]
    return split("train"), split("valid")


def load_wandb_run(entity: str, project: str, run_id: str):
    """Load one W&B run into the same ``(train, valid)`` format as ``load_log``.

    The loader preserves logged training metrics such as ``gradnorm/...`` and
    ``weightnorm/...``. It maps W&B's ``train/loss`` and ``valid/loss`` fields
    to ``loss`` so the existing plotting code can use either data source.
    """
    try:
        import pandas as pd
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Loading W&B data requires the 'wandb' and 'pandas' packages. "
            "Install them before calling load_wandb_run()."
        ) from exc

    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    train, valid = [], []

    for row in run.scan_history():
        # Keep every logged metric, dropping only pandas' missing-value markers.
        clean_row = {
            key: value
            for key, value in row.items()
            if value is not None and not (isinstance(value, float) and pd.isna(value))
        }
        if "valid/loss" in clean_row:
            clean_row["loss"] = clean_row.pop("valid/loss")
            clean_row["split"] = "valid"
            valid.append(clean_row)
        elif "train/loss" in clean_row:
            clean_row["loss"] = clean_row.pop("train/loss")
            clean_row["split"] = "train"
            train.append(clean_row)

    return train, valid


def plot_run_data(train, valid, x: str = "it", out: str = "curves.png", axis: str = "loss", ylim=None, label="run"):
    """Plot train/validation records returned by ``load_log`` or ``load_wandb_run``."""
    fig, ax = plt.subplots(figsize=(8, 5))
    train_with_axis = [r for r in train if axis in r and x in r]
    if not train_with_axis:
        raise ValueError(f"Metric {axis!r} or x-axis {x!r} is not present in training records.")

    line, = ax.plot([r[x] for r in train_with_axis], [r[axis] for r in train_with_axis],
                    alpha=0.4, label=f"{label} train")
    valid_with_axis = [r for r in valid if axis in r and x in r]
    if valid_with_axis:
        ax.plot([r[x] for r in valid_with_axis], [r[axis] for r in valid_with_axis],
                color=line.get_color(), marker="o", ms=3, label=f"{label} valid")
    ax.set_xlabel("gradient steps" if x == "it" else "wall-clock time (s)")
    ax.set_ylabel(axis)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=150, bbox_inches="tight")


def plot_run_comparison(runs, x: str = "it", out: str = "curves.png", axis: str = "loss", ylim=None):
    """Plot one metric for multiple runs.

    ``runs`` maps labels to ``(train_records, valid_records)`` pairs, such as
    the values returned by ``load_wandb_run``.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (train, valid) in runs.items():
        train_points = [r for r in train if axis in r and x in r]
        if not train_points:
            raise ValueError(f"Metric {axis!r} or x-axis {x!r} is missing from {label!r} training data.")

        line, = ax.plot([r[x] for r in train_points], [r[axis] for r in train_points],
                        alpha=0.45, label=f"{label} train")
        valid_points = [r for r in valid if axis in r and x in r]
        if valid_points:
            ax.plot([r[x] for r in valid_points], [r[axis] for r in valid_points],
                    color=line.get_color(), marker="o", ms=3, label=f"{label} valid")

    ax.set_xlabel("gradient steps" if x == "it" else "wall-clock time (s)")
    ax.set_ylabel(axis)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=150, bbox_inches="tight")


def plot_runs(runs: dict[str, str], x: str = "it", out: str = "curves.png", axis: str = "loss", ylim=None):
    """Plot JSONL runs; use ``plot_run_data`` for records loaded from W&B."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, path in runs.items():
        train, valid = load_log(path)
        train_with_axis = [r for r in train if axis in r]
        if not train_with_axis:
            raise ValueError(f"Metric {axis!r} is not present in training records for {label!r}.")

        line, = ax.plot([r[x] for r in train_with_axis], [r[axis] for r in train_with_axis],
                        alpha=0.4, label=f"{label} train")
        valid_with_axis = [r for r in valid if axis in r]
        if valid_with_axis:
            ax.plot([r[x] for r in valid_with_axis], [r[axis] for r in valid_with_axis],
                    color=line.get_color(), marker="o", ms=3, label=f"{label} valid")
    ax.set_xlabel("gradient steps" if x == "it" else "wall-clock time (s)")
    ax.set_ylabel(axis)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_name', required=True)
    parser.add_argument('-o', '--out_path', default=None,
                        help="optional output directory (default: the run directory)")
    parser.add_argument('-a', '--axis', default="loss")
    args = parser.parse_args()

    run_dir = RUNS / args.run_name
    log_path = run_dir / "train_log.jsonl"
    runs = {args.run_name: str(log_path)}
    out_dir = Path(args.out_path) if args.out_path else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    filename_metric = args.axis.replace("/", "_")
    plot_runs(runs, x="it", axis=args.axis, out=str(out_dir / f"{filename_metric}_vs_steps.png"))
    plot_runs(runs, x="time", axis=args.axis, out=str(out_dir / f"{filename_metric}_vs_time.png"))
