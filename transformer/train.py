import os
import time
import math
import pickle
from contextlib import nullcontext

# import sys

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from tokenizer.basic_bpe import BasicTokenizer

# hyperparameters
block_size = 32
batch_size = 16
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
vocab_size = 512

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# import sys; sys.exit(0)


torch.manual_seed(327)


### unneeded tokenization stuff

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# with open('/Users/samswitz/GitHub/micro-research/transformer/data/input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
# print('big load of data')

# tokenizer = BasicTokenizer()
# tokenizer.train(text, vocab_size)

# # Train and test splits
# data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
# print('successfully trained tokenizer')
# import sys; sys.exit(0)

# n = int(0.9*len(data)) # first 90% will be train, rest val
# train_data = data[:n]
# val_data = data[n:]

###


data_dir = os.path.join('data')
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # data = train_data if split == 'train' else val_data
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=1024, dropout=dropout) # start with model_args from command line

conf = GPTConfig(**model_args)

model = GPT(conf)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')