import json
from pathlib import Path
from typing import Mapping, Optional


class WandbMonitor:
    """Optional W&B loss tracking with run IDs persisted alongside checkpoints."""
    def __init__(self, run_dir: Path, run_name: str, project: str, config: dict, enabled: bool):
        self._run = None
        if not enabled:
            return

        import wandb
        
        metadata_path = run_dir / "wandb_run.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("project") != project:
                raise ValueError(
                    f"W&B project for this run is {metadata.get('project')!r}, "
                    f"but the requested project is {project!r}. "
                    "Use a new run_name to start a separate W&B run."
                )
            run_id = metadata.get("id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"Invalid W&B run ID in {metadata_path}")
        else:
            run_id = wandb.util.generate_id()
            metadata = {"id": run_id, "project": project, "entity": None}
            self._write_metadata(metadata_path, metadata)

        run = wandb.init(
            project=project,
            entity=metadata.get("entity"),
            name=run_name,
            id=run_id,
            resume="allow",
            config=config,
        )

        try:
            if metadata.get("entity") is None and run.entity:
                metadata["entity"] = run.entity
                self._write_metadata(metadata_path, metadata)
            run.define_metric("*", step_metric="it")
        except BaseException:
            run.finish(exit_code=1)
            raise

        self._run = run

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish(exit_code=1 if exc_type is not None else 0)
        return False

    @staticmethod
    def _write_metadata(path: Path, metadata: dict) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary_path.replace(path)

    def log_train(
        self,
        *,
        loss: float,
        learning_rate: float,
        iteration: int,
        elapsed: float,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        if self._run is not None:
            payload = {
                "train/loss": loss,
                "lr": learning_rate,
                "it": iteration,
                "time": elapsed,
            }
            if metrics:
                payload.update(metrics)
            self._run.log(payload)

    def log_validation(self, *, loss: float, iteration: int, elapsed: float) -> None:
        if self._run is not None:
            self._run.log({
                "valid/loss": loss,
                "it": iteration,
                "time": elapsed,
            })

    def finish(self, exit_code: int) -> None:
        if self._run is not None:
            run = self._run
            self._run = None
            run.finish(exit_code=exit_code)
