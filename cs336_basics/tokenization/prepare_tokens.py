import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
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


def main():

    text_path = "/storage/Irachid/assignments/assignment1-basics-main/data/TinyStoriesV2-GPT4-train.txt"
    vocab_path = "/storage/Irachid/assignments/assignment1-basics-main/cs336_basics/vocab_tinyStories.pkl"
    merges_path = "/storage/Irachid/assignments/assignment1-basics-main/cs336_basics/merges_tinyStories.pkl"

    out_path = "/storage/Irachid/assignments/assignment1-basics-main/cs336_basics/token_ids/tinyStories_train_ids.bin"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    special_tokens = ["<|endoftext|>"]

    num_workers = 32

    t0 = time.perf_counter()
    total = encode_file(text_path, vocab_path, merges_path, special_tokens, out_path, num_workers)
    dt = time.perf_counter() - t0

    nbytes = os.path.getsize(text_path)
    print(f"{total} tokens, {nbytes / total:.2f} bytes/token, {nbytes / dt / 1e6:.1f} MB/s")

    ids = np.memmap(out_path, dtype=np.uint16, mode="r")
    assert len(ids) == total
    print("max id:", ids.max())

    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    print(tok.decode(ids[:200].tolist()))
    

if __name__=="__main__":
    main()