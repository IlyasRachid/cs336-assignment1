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

## Plot and generate

```bash
uv run python scripts/log_handler.py --run_name sanity_cpu_tiny
uv run python scripts/generate.py --run_name sanity_cpu_tiny
```
