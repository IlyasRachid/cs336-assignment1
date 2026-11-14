import json
from pathlib import Path
import matplotlib.pyplot as plt
from cs336_basics.paths import RUNS

def load_log(path):
    recs = [json.loads(line) for line in open(path)]
    split = lambda s: [r for r in recs if r["split"] == s]
    return split("train"), split("valid")

def plot_runs(runs: dict[str, str], x: str = "it", out: str = "curves.png", axis: str = "loss", ylim=None):
    """runs: {label: path_to_jsonl}; x: 'it' (gradient steps) or 'time' (wall-clock seconds)."""
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
