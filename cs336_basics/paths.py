from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # .../assignment1-basics-main
DATA = ROOT / "data"
TOKENIZERS = ROOT / "artifacts" / "tokenizers"
TOKENS = ROOT / "artifacts" / "tokens"
RUNS = ROOT / "runs"