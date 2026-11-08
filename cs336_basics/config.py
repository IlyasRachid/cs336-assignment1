import argparse
from dataclasses import dataclass
import json

from cs336_basics.model.transformer_lm import Transformer
from cs336_basics.paths import ROOT

TRAINING_CONFIG = ROOT / "artifacts" / "run_config.json"

@dataclass
class TrainConfig:
    batch_size: int = 128
    lr_max: float = 1e-3
    lr_min: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_iters: int = 200
    cosine_iters: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    ckpt_interval: int = 1000
    train_path: str | None = None 
    valid_path: str | None = None
    ckpt_path: str | None = None
    log_path: str | None = None 
    run_name: str = "baseline"
    use_wandb: bool = False
    wandb_project: str = "cs336-a1"
    seed: int = 0

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 1e5


def parse_configs(argv: list[str] | None = None) -> tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(
        description="Load a named model and training config from artifacts/run_config.json."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config name from artifacts/run_config.json (for example: sanity_cpu_tiny).",
    )
    args = parser.parse_args(argv)
    try:
        return build_config(args.config)
    except KeyError as exc:
        parser.error(str(exc))

def build_model(cfg: ModelConfig) -> Transformer:
    return Transformer(
        cfg.vocab_size, 
        cfg.context_length, 
        cfg.num_layers, 
        cfg.d_model, 
        cfg.num_heads, 
        cfg.d_ff, 
        cfg.rope_theta
        )

def build_config(config_name: str) -> tuple[ModelConfig, TrainConfig]:
    with TRAINING_CONFIG.open(encoding="utf-8") as f:
        configs = json.load(f)

    try:
        config = configs[config_name]
    except KeyError as exc:
        available = ", ".join(sorted(configs))
        raise KeyError(
            f"Unknown config {config_name!r} in {TRAINING_CONFIG}. Available configs: {available}"
        ) from exc

    model_cfg = ModelConfig(**config["model_config"])
    train_cfg = TrainConfig(**config["train_config"])
    return model_cfg, train_cfg
