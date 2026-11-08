import random
import subprocess
import time
import statistics

from tokenizer import train_bpe

def test_randomized(trials=100):
    rng = random.Random(42)

    for trial in range(trials):
        n = rng.randint(1, 30)
        num_merges = rng.randint(1, 30)

        pre_tokens = {}

        for _ in range(n):
            word = tuple(
                rng.randrange(256)
                for _ in range(rng.randint(0, 20))
            )
            pre_tokens[word] = rng.randint(1, 20)

        vocab = {i: bytes([i]) for i in range(256)}

        lines = [f"{len(pre_tokens)} {num_merges}"]

        for word, freq in pre_tokens.items():
            lines.append(
                f"{len(word)} {' '.join(map(str, word))} {freq}"
            )

        result = subprocess.run(
            ["./bpe"],
            input="\n".join(lines) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )

        cpp_merges = [
            tuple(bytes.fromhex(x) for x in line.split())
            for line in result.stdout.splitlines()
        ]

        _, py_merges = train_bpe(
            pre_tokens,
            vocab,
            256 + num_merges,
        )

        assert cpp_merges == py_merges, (
            f"Mismatch at trial {trial}\n"
            f"C++: {cpp_merges}\n"
            f"Python: {py_merges}"
        )


def benchmark(n=10_000, num_merges=1_000, repetitions=5):
    rng = random.Random(42)
    pre_tokens = {}

    while len(pre_tokens) < n:
        word = tuple(
            rng.randrange(256)
            for _ in range(rng.randint(5, 20))
        )
        pre_tokens[word] = rng.randint(1, 20)

    lines = [f"{len(pre_tokens)} {num_merges}"]

    for word, freq in pre_tokens.items():
        lines.append(
            f"{len(word)} {' '.join(map(str, word))} {freq}"
        )

    cpp_input = "\n".join(lines) + "\n"

    python_times = []
    cpp_times = []

    for _ in range(repetitions):
        # C++ benchmark
        result = subprocess.run(
            ["./bpe"],
            input=cpp_input,
            text=True,
            capture_output=True,
            check=True,
        )

        cpp_times.append(float(result.stderr.strip()))

        # Python benchmark
        vocab = {i: bytes([i]) for i in range(256)}

        start = time.perf_counter()
        _, py_merges = train_bpe(
            pre_tokens,
            vocab,
            256 + num_merges,
        )
        python_times.append(time.perf_counter() - start)

        cpp_merges = [
            tuple(bytes.fromhex(x) for x in line.split())
            for line in result.stdout.splitlines()
        ]

        assert cpp_merges == py_merges, "Merge sequences differ!"

    py_median = statistics.median(python_times)
    cpp_median = statistics.median(cpp_times)

    print(f"Words: {n}, Merges: {num_merges}")
    print(f"Python: {py_median:.4f}s")
    print(f"C++:    {cpp_median:.4f}s")
    print(f"Speedup: {py_median / cpp_median:.2f}x")



if __name__=="__main__":
    test_randomized()
    benchmark()