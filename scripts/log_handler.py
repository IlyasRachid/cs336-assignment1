import json
from pathlib import Path
import matplotlib.pyplot as plt
from cs336_basics.paths import RUNS

def load_log(path):
    recs = [json.loads(line) for line in open(path)]
    split = lambda s: [r for r in recs if r["split"] == s]
    return split("train"), split("valid")

def plot_runs(runs: dict[str, str], x: str = "it", out: str = "curves.png", ylim=None):
    """runs: {label: path_to_jsonl}; x: 'it' (gradient steps) or 'time' (wall-clock seconds)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, path in runs.items():
        train, valid = load_log(path)
        line, = ax.plot([r[x] for r in train], [r["loss"] for r in train],
                        alpha=0.4, label=f"{label} train")
        ax.plot([r[x] for r in valid], [r["loss"] for r in valid],
                color=line.get_color(), marker="o", ms=3, label=f"{label} valid")
    ax.set_xlabel("gradient steps" if x == "it" else "wall-clock time (s)")
    ax.set_ylabel("loss")
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
    args = parser.parse_args()

    run_dir = RUNS / args.run_name
    log_path = run_dir / "train_log.jsonl"
    runs = {args.run_name: str(log_path)}
    out_dir = Path(args.out_path) if args.out_path else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_runs(runs, x="it", out=str(out_dir / "loss_vs_steps.png"))
    plot_runs(runs, x="time", out=str(out_dir / "loss_vs_time.png"))
