#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ -x "$script_dir/.venv/bin/python" ]]; then
    python_cmd=("$script_dir/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
    python_cmd=(uv run python)
elif command -v python3 >/dev/null 2>&1; then
    python_cmd=(python3)
else
    echo "Could not check PyTorch GPU support: Python and uv were not found." >&2
    exit 2
fi

"${python_cmd[@]}" - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"Could not check PyTorch GPU support: importing torch failed ({exc}).")
    raise SystemExit(2)

print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("CUDA GPU support for PyTorch: unavailable")
    if torch.version.cuda is None:
        print("This PyTorch build does not include CUDA support.")
    else:
        print(f"This PyTorch build includes CUDA {torch.version.cuda}, but no usable GPU is visible.")
    raise SystemExit(0)

try:
    sample = torch.tensor([1.0], device="cuda")
    result = (sample + 1).item()
    torch.cuda.synchronize()
except Exception as exc:
    print(f"CUDA is reported available, but a test operation failed ({exc}).")
    raise SystemExit(1)

print("CUDA GPU support for PyTorch: available")
print(f"CUDA runtime: {torch.version.cuda}")
for device_id in range(torch.cuda.device_count()):
    print(f"GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
print(f"Test operation: passed (1 + 1 = {result:.0f})")
PY
