import dataclasses
import importlib
import json
import pickle

import numpy as np
import torch

from cs336_basics.config import ModelConfig, build_model
from cs336_basics.tokenization.prepare_tokens import encode_file
from cs336_basics.tokenization.tokenization import Tokenizer
from cs336_basics.tokenization.tokenizer import main as train_tokenizer


def test_tokenizer_artifacts_and_encoding(tmp_path):
    text = "hello hello world<|endoftext|>hello world"
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text)
    output = tmp_path / "tokenizer"
    train_tokenizer(["--input", str(corpus), "--output-dir", str(output),
                     "--vocab-size", "270", "--num-workers", "1"])

    vocab_path, merges_path = output / "vocab.pkl", output / "merges.pkl"
    with vocab_path.open("rb") as f:
        vocab = pickle.load(f)
    with merges_path.open("rb") as f:
        merges = pickle.load(f)
    assert isinstance(vocab, dict)
    assert isinstance(merges, list)
    assert len(vocab) <= 270
    assert b"<|endoftext|>" in vocab.values()

    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    tokens = tmp_path / "tokens" / "train.bin"
    count = encode_file(corpus, vocab_path, merges_path, ["<|endoftext|>"], tokens, num_workers=1)
    ids = np.fromfile(tokens, dtype=np.uint16).tolist()
    assert count == len(ids)
    assert ids == tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_imported_generation_uses_configured_checkpoint(tmp_path, monkeypatch):
    generation = importlib.import_module("cs336_basics.inference.generate")
    monkeypatch.setattr(generation, "RUNS", tmp_path)
    cfg = ModelConfig(vocab_size=8, context_length=4, num_layers=1,
                      d_model=8, num_heads=2, d_ff=16)
    run_dir = tmp_path / "example"
    run_dir.mkdir()
    checkpoint = tmp_path / "custom.pt"
    torch.save({"model": build_model(cfg).state_dict()}, checkpoint)
    (run_dir / "config.json").write_text(json.dumps({
        "model": dataclasses.asdict(cfg), "resolved": {"ckpt_path": str(checkpoint)},
    }))
    vocab = {i: bytes([97 + i]) for i in range(8)}
    prompt = [0, 1] * 4  # Exercise context-window truncation as well.
    result = generation.generate("example", prompt, 3, 1.0, 1e-9, vocab, -1)
    assert result.startswith("abababab")
    assert len(result) == len(prompt) + 3
    assert result == generation.generate("example", prompt, 3, 1.0, 1e-9, vocab, -1)


def test_scripts_share_package_implementations():
    for script, package, function in [
        ("scripts.tokenizer", "cs336_basics.tokenization.tokenizer", "tokenize"),
        ("scripts.prepare_tokens", "cs336_basics.tokenization.prepare_tokens", "encode_file"),
        ("scripts.generate", "cs336_basics.inference.generate", "generate"),
    ]:
        assert getattr(importlib.import_module(script), function) is getattr(importlib.import_module(package), function)
