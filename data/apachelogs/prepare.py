import os
import requests
import tiktoken
import numpy as np
import random

# download the apache log dataset
# ideally download https://github.com/naman-modi/datasets/blob/main/logs/apache/combined.log.zip, unzip, and store as input.txt in this folder
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/naman-modi/datasets/main/logs/apache/log_1.log'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)


with open(input_file_path, 'rb') as input_file:
    # Read lines while decoding with the specified encoding
    lines = input_file.readlines()
    lines = [line.decode('utf-8', errors='replace') for line in lines]

# shuffle log lines before spliting
random.shuffle(lines)

n = len(lines)
train_data = ''.join(lines[:int(n * 0.9)])
val_data = ''.join(lines[int(n * 0.9):])

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
