import torch
import json


from cs336_basics.config import ModelConfig, build_model
from cs336_basics.paths import RUNS

@torch.no_grad()
def generate(
    run_name: str,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    p: float,
    vocab: dict[int, bytes],
    eos_id: int
    ) -> str:

    run_dir = RUNS / run_name
    saved = json.load(open(run_dir / "config.json"))
    model_cfg = ModelConfig(**saved["model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_cfg).to(device)
    ckpt_path = saved.get("resolved", {}).get("ckpt_path", str(run_dir / "ckpt.pt"))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = list(input_ids)

    for _ in range(max_new_tokens):
        x = torch.tensor(ids[-model_cfg.context_length:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0, -1]    # (vocab_size,)

        probs = torch.softmax(logits / max(temperature, 1e-9), dim=-1)
        sorted_probs, sorted_idx = probs.sort(descending=True)
        keep = (sorted_probs.cumsum(-1) - sorted_probs) < p
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum()
        next_id = sorted_idx[torch.multinomial(sorted_probs, 1)].item()

        ids.append(next_id)
        if next_id == eos_id:
            break

    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def main(argv=None):
    import argparse
    from cs336_basics.tokenization.tokenization import Tokenizer
    from cs336_basics.paths import TOKENIZERS

    parser = argparse.ArgumentParser(description="Generate text from a saved run.")
    parser.add_argument("-r", "--run_name", required=True)
    parser.add_argument("--tokenizer-dir", type=str, default=str(TOKENIZERS / "tinystories_10k"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args(argv)

    from pathlib import Path
    tokenizer_dir = Path(args.tokenizer_dir)
    tok = Tokenizer.from_files(tokenizer_dir / "vocab.pkl", tokenizer_dir / "merges.pkl", ["<|endoftext|>"])
    print(generate(args.run_name, tok.encode(args.prompt), args.max_new_tokens,
                   args.temperature, args.top_p, tok.vocab, tok.bytes_ids[b"<|endoftext|>"]))


if __name__ == "__main__":
    main()
