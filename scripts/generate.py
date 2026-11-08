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


if __name__ == "__main__":

    from cs336_basics.tokenization.tokenization import Tokenizer 
    from cs336_basics.paths import TOKENIZERS
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_name', required=True)

    run_name = parser.parse_args().run_name

    VOCAB_PATH = str(TOKENIZERS / "tinystories_10k/vocab.pkl")
    MERGES_PATH = str(TOKENIZERS / "tinystories_10k/merges.pkl")

    tok = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"])
    eos_id = tok.bytes_ids[b"<|endoftext|>"]

    prompt = "Once upon a time"
    ids = tok.encode(prompt)

    # 1. normal sampling
    print(generate(run_name, ids, 50, temperature=1.0, p=0.9,
                   vocab=tok.vocab, eos_id=eos_id))

    # 2. temperature ~ 0 must be deterministic (greedy)
    a = generate(run_name, ids, 20, 1e-9, 0.9, tok.vocab, eos_id)
    b = generate(run_name, ids, 20, 1e-9, 0.9, tok.vocab, eos_id)
    assert a == b, "temperature ~0 should be deterministic"

    # 3. tiny p keeps only the top token, so it should also be deterministic
    a = generate(run_name, ids, 20, 1.0, 1e-9, tok.vocab, eos_id)
    b = generate(run_name, ids, 20, 1.0, 1e-9, tok.vocab, eos_id)
    assert a == b, "p ~0 should be deterministic"

    # 4. max_new_tokens is respected (no eos expected from an untrained model)
    out_ids = tok.encode(generate(run_name, ids, 5, 1.0, 1.0, tok.vocab, eos_id))
    print("generated ok, new tokens ~", len(out_ids) - len(ids))

    # 5. prompt longer than the context window must not crash
    run_dir = RUNS / run_name
    saved = json.load(open(run_dir / "config.json"))
    context_length = saved["model"]["context_length"]
    long_ids = ids * (context_length // len(ids) + 2)
    generate(run_name, long_ids, 3, 1.0, 0.9, tok.vocab, eos_id)
    print("long prompt ok")
