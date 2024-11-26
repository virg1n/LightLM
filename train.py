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

from model import Transformer

# class MyLLM():
#     def __init__(self, model: Transformer, tokenizer, model_args):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.args = model_args
vocab_size = 50304 #50257
batch_size = 2**19
mini_batches = 8
time_stamps = 512
context_len = 1024

data_dir = "edu_fineweb10B"
log_dir = "log"
checkpoints_frequency = 2000
log_file = os.path.join(log_dir, f"log.txt")
val_log_file = os.path.join(log_dir, f"val_log.txt")

epochs = 19000

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
    print(f"device: {device}")
    

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
mini_epochs = int(batch_size / (mini_batches * time_stamps * ddp_world_size)) #number of mini-batches to get 0.5M batch

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
    


torch.set_float32_matmul_precision('high')

m = Transformer(
    device, vocab_size, mini_batches, context_len, num_layers=12, num_dims=1024, multiple_of=4, num_heads=16, num_kv_heads=4, use_cache=False, eps = 1e-6, ffn_dim_multiplier=None
)
m = m.to(device)
m = torch.compile(m)
#making loss average from all gpus
if ddp:
    m = DDP(m, device_ids=[ddp_local_rank]) 
raw_m = m.module if ddp else m

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

data_loader = DataLoader(mini_batches, time_stamps, cur_process=ddp_rank, num_processes=ddp_world_size, data_dir=data_dir, split="train")
val_loader = DataLoader(mini_batches, time_stamps, cur_process=ddp_rank, num_processes=ddp_world_size, data_dir=data_dir, split="val")

# I have taken this function [configure_optimizers] from Karpathy's nanoGPT
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_lr_steps = 700
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95

optmizer = raw_m.configure_optimizers(weight_decay, max_lr, (beta1, beta2), device)

for epoch in range(epochs):
    t0 = time.time()
    last_epoch = epochs - 1
    # validation loss check + save model weights every 'checkpoints_frequency' steps
    if epoch % 300 == 0 or epoch == last_epoch:
        m.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = m(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum.item()}")
            with open(val_log_file, "a") as f:
                f.write(f"epoch:{epoch} val_loss:{val_loss_accum.item():.5f}\n")
            if epoch > 0 and (epoch % checkpoints_frequency == 0 or last_epoch):
                checkpoint_path = os.path.join(log_dir, f"model_{epoch:05d}.pt")
                checkpoint = {
                    'model': raw_m.state_dict(),
                    'optimizer':optmizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss_accum.item()
                }

                torch.save(checkpoint, checkpoint_path)
        
    m.train()
    accumulated_loss = 0.0
    optmizer.zero_grad()
    # using accumulated loss
    for mini_epoch in range(mini_epochs):
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
    
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = m(x, y)
        loss /= mini_epochs
        accumulated_loss += loss.detach()
    
        if ddp:
            m.require_backward_grad_sync = (mini_epoch == mini_epochs-1)
        loss.backward()
    if ddp:
        dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG)
    
    norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    # scheduler.step()

    # change lr
    lr = get_lr(epoch)
    for param_group in optmizer.param_groups:
        param_group['lr'] = lr
    optmizer.step()
    
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0

    # wrtie to the file losses
    if master_process and epoch%5==0:
        print(f"epoch: {epoch}, loss: {accumulated_loss:.5f}, norm: {norm:.5f}, time: {dt*1000:.2f}ms, tok/s: {data_loader.B*data_loader.T*mini_epochs*ddp_world_size/dt:.2f}")
        with open(log_file, "a") as f:
            f.write(f"epoch:{epoch} loss:{accumulated_loss.item():.5f}\n")
if ddp:
    destroy_process_group()