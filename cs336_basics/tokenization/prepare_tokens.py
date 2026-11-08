import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

from cs336_basics.tokenization.tokenization import Tokenizer
from cs336_basics.tokenization.tokenizer import split_chunks

_tok = None

def _init_worker(vocab_path, merges_path, special_tokens):
    global _tok
    _tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

def _encode_range(path, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return np.array(_tok.encode(text), dtype=np.uint16)

def encode_file(path, vocab_path, merges_path, special_tokens, out_path, num_workers=32):
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if num_workers == 1:
        _init_worker(vocab_path, merges_path, special_tokens)
        arr = _encode_range(path, 0, Path(path).stat().st_size)
        arr.tofile(out_path)
        return len(arr)

    bounds = split_chunks(Path(path), special_tokens[0].encode("utf-8"), 8 * num_workers)
    with ProcessPoolExecutor(num_workers, initializer=_init_worker,
                             initargs=(vocab_path, merges_path, special_tokens)) as ex:
        futs = [ex.submit(_encode_range, path, bounds[k], bounds[k + 1])
                for k in range(len(bounds) - 1)]
        total = 0
        with open(out_path, "wb") as out:
            for f in tqdm(futs):
                arr = f.result()
                arr.tofile(out)
                total += len(arr)
    return total



def main(argv=None):
    import argparse
    from cs336_basics.paths import DATA, TOKENIZERS, TOKENS

    parser = argparse.ArgumentParser(description="Encode a corpus as uint16 token IDs.")
    parser.add_argument("--input", type=Path, default=DATA / "TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZERS / "tinystories_10k")
    parser.add_argument("--output", type=Path, default=TOKENS / "tinystories_train.bin")
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    total = encode_file(args.input, args.tokenizer_dir / "vocab.pkl", args.tokenizer_dir / "merges.pkl",
                        ["<|endoftext|>"], args.output, args.num_workers)
    dt = time.perf_counter() - t0
    print(f"{total} tokens in {dt:.2f}s")


if __name__ == "__main__":
    main()
