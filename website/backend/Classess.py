import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math
import tiktoken
import inspect
import os


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

vocab_size = 50304 #50257
batch_size = 2**19
mini_batches = 8
time_stamps = 512
context_len = 1024
emb_neur = 768 #1024
epochs = 19073
num_blocks = 12 #16
num_heads = 12 #16
dropout_neur = 0.2
data_dir = "edu_fineweb10B"
log_dir = "log"
checkpoints_frequency = 5000
log_file = os.path.join(log_dir, f"log.txt")
val_log_file = os.path.join(log_dir, f"val_log.txt")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_lr_steps = 500
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95


enc = tiktoken.get_encoding("gpt2")



ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed(1337)


def get_lr(epoch):
    if epoch < warmup_lr_steps:
        return (max_lr * (epoch+1)/warmup_lr_steps)
    if epoch > epochs:
        return min_lr
    loc = (epoch - warmup_lr_steps)/(epochs - warmup_lr_steps)
    coef = 0.5 * (1.0 + math.cos(math.pi * loc))
    return min_lr + coef * (max_lr - min_lr)

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader():
    def __init__(self, B, T, cur_process, num_processes, data_dir, split):
        self.B = B
        self.T = T
        self.cur_process = cur_process
        self.cur_shard = 0
        self.num_processes = num_processes
        self.data_dir = data_dir

        shards = os.listdir(self.data_dir)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(self.data_dir, s) for s in shards]
        self.shards = shards

        self.tokens = load_tokens(self.shards[self.cur_shard])
        
        self.current_step = cur_process * B * T

        print(f"loaded ~{len(self.tokens)*len(self.shards)} tokens")


    def reset(self):
        self.cur_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.cur_process
        
    def next_batch(self):
        B, T = self.B, self.T
        
        self.current_step += B * T * self.num_processes
        tokens = self.tokens[self.current_step:self.current_step+B*T+1]
        x = (tokens[:-1]).view(B, T)
        y = (tokens[1:]).view(B, T)
        if (self.current_step+B*T* self.num_processes + B*T+1)  > len(self.tokens):
            self.cur_shard = (self.cur_shard+1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.cur_shard])
            self.current_step = self.cur_process * B * T
        return x, y


class SelfAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.qkv = nn.Linear(emb_neur, 3 * emb_neur)
        self.proj = nn.Linear(emb_neur, emb_neur)
        self.proj.COMES_TO_RESIDUAL = 1
        # self.dropout = nn.Dropout(dropout_neur)

    def forward(self, idx):

        B, T, C = idx.shape
        qkv = self.qkv(idx)
        q, k, v = qkv.split(emb_neur, dim=2)
        q = q.view(B, T, num_heads, C//num_heads).transpose(1, 2) # B, nh, T, hs
        k = k.view(B, T, num_heads, C//num_heads).transpose(1, 2) # B, nh, T, hs
        v = v.view(B, T, num_heads, C//num_heads).transpose(1, 2) # B, nh, T, hs

        # attention = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1]))
        # attention = torch.tril(attention[:, :, :T, :T])
        
        # attention = attention.masked_fill(attention == 0, float("-inf"))
        # attention = F.softmax(attention, dim=-1)
        # out = attention @ v # B, nh, T, hs 
        

        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = attention.transpose(2, 1).contiguous().view(B, T, C)
        out = self.proj(out)
        # out = self.dropout(out)

        return out
        


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(emb_neur, 4 * emb_neur),
        #     nn.GELU(),
        #     nn.Linear(4 * emb_neur, emb_neur),
        #     nn.Dropout(dropout_neur),
        # )
        self.upl = nn.Linear(emb_neur, 4 * emb_neur)
        self.gelu = nn.GELU()
        self.dwnl = nn.Linear(4 * emb_neur, emb_neur)
        self.dwnl.COMES_TO_RESIDUAL = 1

    def forward(self, idx):
        idx = self.upl(idx)
        idx = self.gelu(idx)
        idx = self.dwnl(idx)
        return idx
        # return self.net(idx)


class Block(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.attentions = SelfAttention(num_heads)
        self.ffn = FeedForward()
        self.ln1 = nn.LayerNorm(emb_neur)
        self.ln2 = nn.LayerNorm(emb_neur)

    def forward(self, idx):
        idx = idx + self.attentions(self.ln1(idx))
        idx = idx + self.ffn(self.ln2(idx))
        return idx

        
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens_embedding = nn.Embedding(vocab_size, emb_neur)
        self.position_embedding = nn.Embedding(context_len, emb_neur)
        self.blocks = nn.Sequential( *[Block(num_heads) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(emb_neur)
        self.ll_head = nn.Linear(emb_neur, vocab_size)

        self.tokens_embedding.weight = self.ll_head.weight

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        std = (1.0 / math.sqrt(emb_neur))
        if isinstance(module, nn.Linear):
            if hasattr(module, "COMES_TO_RESIDUAL"):
                std *= (1.0)/(math.sqrt(2*num_blocks))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    # I have taken this function [configure_optimizers] from Karpathy's nanoGPT
    # https://github.com/karpathy/nanoGPT
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        embedded_tokens = self.tokens_embedding(idx) # B, T, emb_neur
        embedded_position = self.position_embedding(torch.arange(T, device=device)) # T, emb_neur
        
        idx = embedded_tokens + embedded_position # B, T, emb_neur
        idx = self.blocks(idx)
        idx = self.ln(idx)
        logits = self.ll_head(idx)
        
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optmizer, T_max=lr_steps, eta_min=min_lr)