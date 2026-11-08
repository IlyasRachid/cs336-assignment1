from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import regex as re
import time
import pickle
import resource
import subprocess
from tqdm import tqdm


# helper functions
def split_chunks(input_path: Path, special_token: bytes, num_chunks: int) -> list[int]:
    with open(input_path, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
    
        mini_chunk_size = 4096
        
        chunk_size = file_size // num_chunks
        
        chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)  
                
                if mini_chunk == b'':
                    chunk_boundaries[bi] = file_size
                    break
            
                idx = mini_chunk.find(special_token)
                if idx != -1:
                    chunk_boundaries[bi] = initial_position + idx
                    break
                initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize_chunk(path: Path, start: int, end: int, pattern, split_pat):
    with open(path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    counts = Counter()
    for piece in split_pat.split(text):
        counts.update(m.group() for m in pattern.finditer(piece))
        
    return {tok.encode("utf-8"): n for tok, n in counts.items()}
    

def train_bpe(pre_tokens, vocab, vocab_size):
    words = [list(w) for w in pre_tokens]
    freqs = list(pre_tokens.values())
    merges = []
    
    pair_counts = defaultdict(int)
    pair_words = defaultdict(set)

    for i, word in enumerate(words):
        for a, b in zip(word, word[1:]):
            pair = (a, b)
            pair_counts[pair] += freqs[i]
            pair_words[pair].add(i)

    iterations = vocab_size - len(vocab)
    for _ in tqdm(range(iterations), desc="merges"):
    #for _ in range(iterations):
        if not pair_counts:
            break
        
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]))
        
        a, b = best
        merges.append((vocab[a], vocab[b]))
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]

        for idx in list(pair_words[best]):
            old = words[idx]
            new, i = [], 0
            while i < len(old):
                if i < len(old) - 1 and old[i] == a and old[i+1] == b:
                    new.append(new_id)
                    i += 2
                else:
                    new.append(old[i])
                    i += 1
                    
            if len(new) == len(old):
                continue

            f = freqs[idx]
            for p in zip(old, old[1:]):
                pair_counts[p] -= f
                if pair_counts[p] == 0:
                    del pair_counts[p]
            for p in zip(new, new[1:]):
                pair_counts[p] += f
                pair_words[p].add(idx)
            words[idx] = new
            
            for p in set(zip(old, old[1:])) - set(zip(new, new[1:])):
                pair_words[p].discard(idx)
                if not pair_words[p]:
                    del pair_words[p]


        pair_words.pop(best, None)
        pair_counts.pop(best, None)
    
    return vocab, merges


import subprocess
import tempfile
from tqdm import tqdm

def train_bpe_cpp(pre_tokens, vocab, vocab_size, special_tokens):
    num_merges = vocab_size - len(vocab) - len(special_tokens)

    with tempfile.TemporaryFile(mode="w+t") as inp, \
         tempfile.TemporaryFile(mode="w+t") as out:

        # Serialize input without building one giant string
        inp.write(f"{len(pre_tokens)} {num_merges}\n")

        for word, freq in tqdm(
            pre_tokens.items(),
            total=len(pre_tokens),
            desc="Preparing BPE input"
        ):
            inp.write(
                f"{len(word)} {' '.join(map(str, word))} {freq}\n"
            )

        inp.flush()
        inp.seek(0)

        # Launch C++ trainer
        with subprocess.Popen(
            ["./bpe"],
            stdin=inp,
            stdout=out,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as proc:

            with tqdm(
                total=num_merges,
                desc="BPE merges",
                unit="merge"
            ) as bar:

                for line in proc.stderr:
                    if line.startswith("PROGRESS "):
                        completed = int(line.split()[1])
                        bar.update(max(0, completed - bar.n))
                    else:
                        tqdm.write(line.rstrip())

            if proc.wait() != 0:
                raise RuntimeError("C++ BPE training failed")

        out.seek(0)

        merges = [
            tuple(bytes.fromhex(x) for x in line.split())
            for line in out
        ]

    for a, b in merges:
        vocab[len(vocab)] = a + b

    return vocab, merges

#--------------------------------------------------------------------------------------------------------------     

def tokenize(input_path: str, vocab_size: int, special_tokens: list[str], num_workers: int = 8):
    path = Path(input_path)
    assert path.is_file(), "file doesn't exist"

    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    
    assert vocab_size >= len(special_tokens) + 256, "max_vocab_size too low"
    
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    split_pat = re.compile("|".join(re.escape(t) for t in special_tokens))
    
    num_chunks = 4 * num_workers
    
    chunks = split_chunks(path, "<|endoftext|>".encode("utf-8"), num_chunks)
    pre_tokens = Counter()

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(pretokenize_chunk, path, chunks[k], chunks[k+1], PAT, split_pat) for k in range(len(chunks) - 1)
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="pretokenize"):
            pre_tokens.update(fut.result())

    #vocab, merges = train_bpe(pre_tokens, vocab, vocab_size)
    vocab, merges = train_bpe_cpp(pre_tokens, vocab, vocab_size)

    for tok in special_tokens:
            vocab[len(vocab)] = tok.encode("utf-8")
    
    return vocab, merges


#--------------------------------------------------------------------------------------------------------------

def main():
    num_workers = 16
    t0 = time.perf_counter()
    vocab, merges = tokenize("/storage/Irachid/assignments/assignment1-basics-main/data/owt_train.txt", 32000, ["<|endoftext|>"], num_workers)
    total = time.perf_counter() - t0

    with open("vocab_owl.pkl", "wb") as f: pickle.dump(vocab, f)
    with open("merges_owl.pkl", "wb") as f: pickle.dump(merges, f)

    longest = max(vocab.values(), key=len)
    print(f"total: {total:.1f}s")
    print(f"peak RSS main: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
    print(f"peak RSS largest worker: {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1e6:.2f} GB")
    print(len(longest), longest)
    

if __name__ == "__main__":
    main()
    