import argparse
import torch

from cs336_basics.config import ModelConfig, TrainConfig, build_config, build_model
from cs336_basics.training.gradient_clipping import gradient_clip

GiB_BYTES = 1024**3
FP32_BYTES = 4


def count_params(model_config: ModelConfig) -> int:
    """Count trainable parameters for this untied, bias-free Transformer."""
    vocab_size = model_config.vocab_size
    d_model = model_config.d_model
    num_layers = model_config.num_layers
    d_ff = model_config.d_ff

    # Token embedding + LM head, attention and FFN projections, RMSNorm weights.
    return (
        2 * vocab_size * d_model
        + num_layers * (2 * d_model + 4 * d_model**2 + 3 * d_model * d_ff)
        + d_model
    )


def model_weight_memory(
    model_config: ModelConfig, bytes_per_parameter: int = FP32_BYTES
) -> int:
    """Estimate bytes for model weights alone (FP32 by default)."""
    return count_params(model_config) * bytes_per_parameter


def training_state_memory(
    model_config: ModelConfig,
    parameter_bytes: int = FP32_BYTES,
    gradient_bytes: int = FP32_BYTES,
    optimizer_state_bytes: int = 2 * FP32_BYTES,
) -> int:
    """Estimate weights + gradients + Adam's two moment tensors, in bytes."""
    bytes_per_parameter = parameter_bytes + gradient_bytes + optimizer_state_bytes
    return count_params(model_config) * bytes_per_parameter


def activation_memory(
    model_config: ModelConfig,
    batch_size: int,
    bytes_per_activation: int = FP32_BYTES,
) -> int:
    """Rough retained-activation estimate; this is not a CUDA peak measurement."""
    vocab_size = model_config.vocab_size
    seq_len = model_config.context_length
    d_model = model_config.d_model
    num_layers = model_config.num_layers
    d_ff = model_config.d_ff
    num_heads = model_config.num_heads

    activation_elements = (
        num_layers
        * (
            8 * batch_size * seq_len * d_model
            + 4 * batch_size * seq_len * d_ff
            + 2 * batch_size * num_heads * seq_len**2
        )
        + batch_size * seq_len * d_model
        + 2 * batch_size * seq_len * vocab_size
        + batch_size * seq_len
        + 1
    )
    return activation_elements * bytes_per_activation


def total_memory(model_config: ModelConfig, batch_size: int) -> int:
    """Approximate training memory in bytes, excluding CUDA/runtime overhead."""
    return training_state_memory(model_config) + activation_memory(model_config, batch_size)


def forward_flops(model_config: ModelConfig, batch_size: int) -> int:
    """Approximate matrix-multiplication FLOPs for one forward batch.

    A multiply-add counts as two FLOPs. Elementwise ops, norms, and softmax are
    omitted, so treat this as an estimate rather than an exact operation count.
    """
    num_layers = model_config.num_layers
    seq_len = model_config.context_length
    d_model = model_config.d_model
    d_ff = model_config.d_ff
    vocab_size = model_config.vocab_size

    return batch_size * (
        num_layers
        * (
            8 * seq_len * d_model**2
            + 4 * seq_len**2 * d_model
            + 6 * seq_len * d_model * d_ff
        )
        + 2 * seq_len * d_model * vocab_size
    )


def training_flops(model_config: ModelConfig, batch_size: int) -> int:
    """Approximate FLOPs for one training step (forward + backward)."""
    return 3 * forward_flops(model_config, batch_size)


def total_training_flops(
    model_config: ModelConfig, batch_size: int, iterations: int
) -> int:
    """Approximate FLOPs for a run with the given number of optimizer steps."""
    return training_flops(model_config, batch_size) * iterations


def model_flop_utilization(achieved_tflops_per_sec: float, peak_fp32_tflops: float) -> float:
    """Return estimated MFU as a fraction of the GPU's theoretical FP32 peak."""
    if peak_fp32_tflops <= 0:
        raise ValueError("peak_fp32_tflops must be positive")
    return achieved_tflops_per_sec / peak_fp32_tflops


def gpu_train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    warmup: int,
    measured_steps: int,
):
    """Benchmark optimizer steps using synthetic token batches generated on the GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_train requires a CUDA-enabled PyTorch device")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if measured_steps <= 0:
        raise ValueError("measured_steps must be positive")

    from cs336_basics.training.adamW import AdamW
    from cs336_basics.training.cross_entropy import cross_entropy
    import time

    device = torch.device("cuda", torch.cuda.current_device())
    model = build_model(model_cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.lr_max,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
        eps=train_cfg.eps,
    )

    # Reuse fixed synthetic inputs so timing measures model training, not RNG or I/O.
    batch_shape = (train_cfg.batch_size, model_cfg.context_length)
    x = torch.randint(model_cfg.vocab_size, batch_shape, device=device)
    y = torch.randint(model_cfg.vocab_size, batch_shape, device=device)

    def run_step():
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(x), y)
        loss.backward()
        gradient_clip(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

    model.train()
    for _ in range(warmup):
        run_step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(measured_steps):
        run_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    step_time = elapsed / measured_steps
    tokens_per_sec = (
        train_cfg.batch_size * model_cfg.context_length * measured_steps / elapsed
    )
    flops_per_step = training_flops(model_cfg, train_cfg.batch_size)
    tflops_per_sec = flops_per_step / step_time / 1e12
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    return step_time, tokens_per_sec, tflops_per_sec, peak_allocated, peak_reserved


def main():
    parser = argparse.ArgumentParser(description="Estimate model size and benchmark training throughput.")
    parser.add_argument(
        "--config",
        required=True,
        help="Config name from artifacts/run_config.json.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up optimizer steps.")
    parser.add_argument("--steps", type=int, default=20, help="Measured optimizer steps.")
    parser.add_argument(
        "--peak_fp32_tflops",
        type=float,
        default=None,
        help="GPU theoretical FP32 peak in TFLOPs/s, used to estimate MFU.",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.peak_fp32_tflops is not None and args.peak_fp32_tflops <= 0:
        parser.error("--peak_fp32_tflops must be positive")

    model_config, train_config = build_config(args.config)
    batch_size = train_config.batch_size

    params = count_params(model_config)
    weights = model_weight_memory(model_config)
    train_state = training_state_memory(model_config)
    activations = activation_memory(model_config, batch_size)
    memory = total_memory(model_config, batch_size)
    flops_per_step = training_flops(model_config, batch_size)
    run_flops = total_training_flops(
        model_config, batch_size, train_config.cosine_iters
    )

    print(f"Parameters: {params:,}")
    print(f"FP32 weights: {weights / GiB_BYTES:.3f} GiB")
    print(f"FP32 training state (weights, grads, Adam moments): {train_state / GiB_BYTES:.3f} GiB")
    print(f"Approx retained activations: {activations / GiB_BYTES:.3f} GiB")
    print(f"Approx training memory (excludes CUDA/runtime overhead): {memory / GiB_BYTES:.3f} GiB")
    print(f"Estimated training FLOPs per step: {flops_per_step / 1e12:.6f} TFLOPs")
    print(f"Estimated FLOPs for {train_config.cosine_iters:,} steps: {run_flops / 1e15:.6f} PFLOPs")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        step_time, tokens_per_sec, tflops_per_sec, peak_allocated, peak_reserved = gpu_train(
            model_config, train_config, args.warmup, args.steps
        )
        print(f"Average training-step time: {step_time:.4f} seconds")
        print(f"Throughput: {tokens_per_sec:,.1f} tokens/second")
        print(f"Throughput: {tokens_per_sec * 3600:,.0f} tokens/hour")
        print(f"Measured compute rate: {tflops_per_sec:.4f} TFLOPs/second")
        if args.peak_fp32_tflops is not None:
            mfu = model_flop_utilization(tflops_per_sec, args.peak_fp32_tflops)
            print(f"Estimated MFU: {mfu:.2%}")
        else:
            print("MFU not calculated; pass --peak_fp32_tflops for this GPU's FP32 peak.")
        print(f"Peak CUDA allocated: {peak_allocated / GiB_BYTES:.3f} GiB")
        print(f"Peak CUDA reserved: {peak_reserved / GiB_BYTES:.3f} GiB")
    else:
        print("CUDA is unavailable; skipped the runtime benchmark.")


if __name__ == "__main__":
    main()
