import regex as re
import pickle
from collections.abc import Iterable, Iterator
from itertools import islice

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.special_tokens = set(special_tokens or [])
        vocab = dict(vocab)
        for tok in self.special_tokens:
            encoded = tok.encode("utf-8")
            if encoded not in vocab.values():
                vocab[len(vocab)] = encoded

        self.vocab = vocab
        self.bytes_ids = {b: id for id, b in vocab.items()}
        self.merges = merges
        self.cache = {}
        self.ranks = {
            (self.bytes_ids[a], self.bytes_ids[b]): i
            for i, (a, b) in enumerate(merges)
        }
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.split_pattern = (
            re.compile("(" + "|".join(map(re.escape, tokens)) + ")")
            if tokens else None)

    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocabulary = pickle.load(f)
        
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocabulary, merges, special_tokens)

    
    def process_chunk(self, chunk):
        all_ids = []

        if not chunk:
            return all_ids

        if chunk in self.special_tokens:
            return [self.bytes_ids[chunk.encode("utf-8")]]

        for pre_token in self.PAT.finditer(chunk):
            key = pre_token.group()
            hit = self.cache.get(key)
            if hit is not None:
                all_ids.extend(hit)
                continue 

            ids = [self.bytes_ids[bytes([b])] for b in key.encode("utf-8")]
            while len(ids) > 1:
                pair_ids = zip(ids, ids[1:])

                best_pair = min(
                    pair_ids,
                    key=lambda p: self.ranks.get(p, float("inf"))
                )

                if best_pair not in self.ranks:
                    break

                new_id = self.bytes_ids[
                    self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
                ]

                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids

            all_ids.extend(ids)
            if len(self.cache) > 1_000_000:
                self.cache.clear()
            self.cache[key] = ids

        return all_ids


    def encode(self, text: str):
        chunks = self.split_pattern.split(text) if self.split_pattern else [text]
        return [
            token_id 
            for chunk in chunks
            for token_id in self.process_chunk(chunk)
        ]
        
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)


    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")



def compression_ratio(tokenizer, documents):
    text = "".join(documents)
    num_bytes = len(text.encode("utf-8"))
    num_tokens = len(tokenizer.encode(text))

    return num_bytes / num_tokens

def main():
    text_path = "/storage/Irachid/assignments/assignment1-basics-main/data/owt_valid.txt"
    vocab_path = "/storage/Irachid/assignments/assignment1-basics-main/cs336_basics/vocab_owl.pkl"
    merges_path = "/storage/Irachid/assignments/assignment1-basics-main/cs336_basics/merges_owl.pkl"

    num_workers = 16
    
    with open(text_path, "r") as f:
        text = "".join(islice(f, 100))
    
    tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    encoded_ids = tokenizer.encode(text, num_workers)
    decoded = tokenizer.decode(encoded_ids)
    comp_ratio = compression_ratio(tokenizer, text)

    print(f"Compression ratio: {comp_ratio:.2f}")
    print(decoded == text)
    

if __name__=="__main__":
    main()