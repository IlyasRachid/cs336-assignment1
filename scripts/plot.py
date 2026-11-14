"""Plot one metric across local training runs or W&B runs.

Examples:
    python -m scripts.plot --runs run1 run2
    python -m scripts.plot --runs run1 run2 --wandb --axis valid/loss
"""
import argparse
import json
from pathlib import Path

from cs336_basics.paths import RUNS
from scripts.log_handler import load_log, load_wandb_run, plot_run_comparison


def _load_wandb_run_by_name(run_name: str):
    run_dir = RUNS / run_name
    metadata_path = run_dir / "wandb_run.json"
    config_path = run_dir / "config.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing W&B metadata file: {metadata_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config file: {config_path}")

    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    entity = metadata.get("entity")
    project = metadata.get("project") or config.get("train", {}).get("wandb_project")
    run_id = metadata.get("id")
    if not entity or not project or not run_id:
        raise ValueError(
            f"Could not resolve W&B entity/project/id for {run_name!r}; "
            f"check {metadata_path} and {config_path}."
        )
    return load_wandb_run(entity, project, run_id)


def main():
    parser = argparse.ArgumentParser(description="Plot a metric across training runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="run directory names")
    parser.add_argument("--wandb", action="store_true", help="load runs from W&B using local run metadata")
    parser.add_argument("--axis", default="loss", help="metric key to plot (default: loss)")
    parser.add_argument("--x-axis", choices=("it", "time"), default="it", help="plot against steps or time")
    parser.add_argument("--out", default=None, help="output image path")
    args = parser.parse_args()

    runs = {}
    for run_name in args.runs:
        if args.wandb:
            runs[run_name] = _load_wandb_run_by_name(run_name)
        else:
            log_path = RUNS / run_name / "train_log.jsonl"
            if not log_path.is_file():
                raise FileNotFoundError(f"Training log not found: {log_path}")
            runs[run_name] = load_log(log_path)

    output = Path(args.out) if args.out else Path("plots") / f"{args.axis.replace('/', '_')}_vs_{args.x_axis}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_run_comparison(runs, x=args.x_axis, axis=args.axis, out=str(output))
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
