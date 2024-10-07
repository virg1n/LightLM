import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "tokens are too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None

for doc in tqdm(fw, desc="Processing documents"):
    tokens = tokenize(doc)

    # Check if there is enough space in the current shard for the new tokens
    if token_count + len(tokens) < shard_size:
        all_tokens_np[token_count:token_count + len(tokens)] = tokens
        token_count += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        # Write the current shard and start a new one
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}")
        # Split the document into whatever fits in this shard; the remainder goes to the next one
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        # Populate the next shard with the leftovers of the current document
        all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
        token_count = len(tokens) - remainder

if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])