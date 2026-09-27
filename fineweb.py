"""
FineWeb-Edu dataset prep (SMALL, learning-scale version of Karpathy's fineweb.py)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Streams a small slice of the dataset (instead of downloading the full
10B-token sample), tokenizes it with the GPT-2 tokenizer, and writes it
to disk as .npy shards -- same format/logic as the real script, just
scaled down so it runs quickly on a local machine.

Run as:
$ python fineweb_small.py
Will save shards to the local directory "edu_fineweb_small".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm              # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb_small"
remote_name = "sample-10BT"     # same underlying dataset config Karpathy uses
num_docs_to_take = 5000         # <-- learning-scale: only pull this many documents
shard_size = int(1e6)           # 1M tokens per shard (Karpathy uses 100M -- too big for this)

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_dir)

# defined at module scope (not inside __main__) so that worker processes,
# which re-import this module on Windows, also have access to these
ENC = tiktoken.get_encoding("gpt2")
EOT = ENC._special_tokens["<|endoftext|>"]


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [EOT]  # the special <|endoftext|> token delimits all documents
    tokens.extend(ENC.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    print(f"Streaming {num_docs_to_take} documents from {remote_name}...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)
    fw_small = fw.take(num_docs_to_take)

    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw_small, chunksize=16):

            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # current shard is full: write it, start a new one, carry over the remainder
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write whatever's left as the final shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
            shard_index += 1

    print(f"Done. Wrote {shard_index} shard(s) to {DATA_CACHE_DIR}")


if __name__ == "__main__":
    main()