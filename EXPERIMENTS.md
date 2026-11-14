# Experiment configurations

`artifacts/run_config.json` is the source of truth for model and training settings. Select an entry by name; training and the GPU benchmark load the same `ModelConfig` and `TrainConfig` from that entry.

- `sanity_cpu_tiny`: small 10-step TinyStories sanity run with W&B logging.
- `tinyStories_exp_1`: larger TinyStories experiment.

Run commands from the repository root.

## Tokenizers and token files

The implementations live in `cs336_basics`; the scripts are command-line entry points.
BPE training requires `g++` with C++20 support. The trainer builds automatically into
`artifacts/bin/bpe` on first use and rebuilds when its source changes. Before timing
BPE training or running the assignment's speed test, build it once:

```bash
uv run python -c 'from cs336_basics.tokenization.tokenizer import get_bpe_binary; get_bpe_binary()'
```

Train a TinyStories tokenizer and prepare the training and validation files:

```bash
uv run python -m scripts.tokenizer --input data/TinyStoriesV2-GPT4-train.txt --output-dir artifacts/tokenizers/tinystories_10k --vocab-size 10000
uv run python -m scripts.prepare_tokens
uv run python -m scripts.prepare_tokens --input data/TinyStoriesV2-GPT4-valid.txt --output artifacts/tokens/tinystories_valid.bin
```

Tokenizer training writes `vocab.pkl` and `merges.pkl`. Use `--tokenizer-dir` when
encoding or generating with a tokenizer other than the default TinyStories tokenizer.
Both preprocessing commands accept `--num-workers`; use `1` for a small local check.

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

## Plot runs

Compare one metric across multiple runs. By default, the plotter reads each
run's local `runs/<run_name>/train_log.jsonl` file:

```bash
uv run python -m scripts.plot --runs sanity_cpu_tiny tinyStories_exp_1
```

The default metric is `loss`, which includes both training and validation
curves. Choose a different logged metric with `--axis`; for example, use a
parameter's exact key to compare its weight or gradient norm:

```bash
uv run python -m scripts.plot --runs run_lr_1e-3 run_lr_3e-3 --axis gradnorm/layers.0.attn.q_proj
uv run python -m scripts.plot --runs run_lr_1e-3 run_lr_3e-3 --axis weightnorm/layers.0.attn.q_proj --x-axis time
```

Use `--x-axis it` (the default) to plot against training steps, or
`--x-axis time` for elapsed seconds. The image is saved under `plots/` by
default. Set `--out path/to/plot.png` to choose a different output path.

To fetch history from W&B instead of local JSONL files, add `--wandb`:

```bash
uv run python -m scripts.plot --runs run_lr_1e-3 run_lr_3e-3 --wandb
```

W&B mode uses the run ID, entity, and project saved in each run's
`runs/<run_name>/wandb_run.json` and `config.json`. The run must have been
logged with W&B from this project so those files exist. Install `wandb` and
`pandas` in the environment before using `--wandb`. W&B validation history
currently contains loss only; gradient and weight norm curves are available
for training records.

## Generate

```bash
uv run python scripts/generate.py --run_name sanity_cpu_tiny
```
