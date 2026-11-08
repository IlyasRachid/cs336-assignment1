# Experiment configurations

`artifacts/run_config.json` is the source of truth for model and training settings. Select an entry by name; training and the GPU benchmark load the same `ModelConfig` and `TrainConfig` from that entry.

- `sanity_cpu_tiny`: small 10-step TinyStories sanity run with W&B logging.
- `tinyStories_exp_1`: larger TinyStories experiment.

Run commands from the repository root.

## Benchmark a GPU

The benchmark creates synthetic token batches on the GPU, so it does not need the TinyStories token files. It uses the selected config's model dimensions and batch size.

```bash
uv run python -m cs336_basics.utils --config tinyStories_exp_1 --warmup 5 --steps 20
```

To also estimate MFU, add `--peak_fp32_tflops` with that GPU's theoretical FP32 peak in TFLOPs/s.

## Train

Training uses the configured token paths. The sample sanity run is:

```bash
uv run python -m scripts.Train --config sanity_cpu_tiny
```

For the larger experiment, select `tinyStories_exp_1`. To continue a run, use the same config and preserve its checkpoint and W&B run metadata. For a fresh run, add a new entry in `artifacts/run_config.json` with a new `run_name`.

## Plot and generate

```bash
uv run python scripts/log_handler.py --run_name sanity_cpu_tiny
uv run python scripts/generate.py --run_name sanity_cpu_tiny
```
