# TODO

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
from dataclasses import dataclass
from huggingface_hub import PyTorchModelHubMixin

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import ModelConfig, Transformer, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")
SEED = 1337

checkpoints_frequency = 2000

data_dir = "edu_fineweb10B"             # dataset directory
log_dir = "log"
log_file = os.path.join(log_dir, f"log.txt")
val_log_file = os.path.join(log_dir, f"val_log.txt")

torch.manual_seed(SEED)
if device == 'cuda':
    torch.cuda.manual_seed(SEED)

#DDP
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

# lr scheduler
def get_lr(epoch, warmup_lr_steps, max_lr, min_lr, epochs):
    if epoch < warmup_lr_steps:
        return (max_lr * (epoch+1)/warmup_lr_steps)
    if epoch > epochs:
        return min_lr
    loc = (epoch - warmup_lr_steps)/(epochs - warmup_lr_steps)
    coef = 0.5 * (1.0 + math.cos(math.pi * loc))
    return min_lr + coef * (max_lr - min_lr)


config = ModelConfig(
    vocab_size = 50304,

    num_dims = 1024,
    num_heads = 16,
    num_kv_heads = 4,
    num_layers = 16,

    rmsnorm_eps = 1e-6,
    rope_theta = 1e5,

    batch_size = 2**19,
    mini_batches = 2,
    time_stamps = 512,
    context_len = 1024,
    
    use_cache = False,
    use_flash = False,

    moe_num_experts = 6,
    moe_routed_experts = 1,         #top_k
    moe_eps = 1e-6,
    moe_aux_loss_coef = 0.01,
    moe_shared_experts = 0,         #0 for default MoE. >0 for DeepSeekMoE
    use_lossfreebalance = True
)

model = Transformer(config)
model = model.to(device)
model = torch.compile(model)

mini_epochs = int(2**19 / (16 * 512))
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

data_loader = DataLoader(config.mini_batches, config.time_stamps, cur_process=ddp_rank, num_processes=ddp_world_size, data_dir=data_dir, split="train")
val_loader = DataLoader(config.mini_batches, config.time_stamps, cur_process=ddp_rank, num_processes=ddp_world_size, data_dir=data_dir, split="val")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_lr_steps = 700
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95

optmizer = model.configure_optimizers(weight_decay, max_lr, (beta1, beta2), device)

epochs = 60
update_rate = 1e-5
use_lossfreebalance = True
for epoch in range(epochs):
    t0 = time.time()
    last_epoch = epochs - 1

    accumulated_tt_loss = 0.0
    model.train()
    accumulated_loss = 0.0
    optmizer.zero_grad()
    # using accumulated loss
    for mini_epoch in range(mini_epochs):
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
    
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss, tt_loss = model(x, y)
        loss /= mini_epochs
        if not use_lossfreebalance:
            tt_loss /= mini_epochs
            accumulated_tt_loss += tt_loss.detach()
        
        accumulated_loss += loss.detach()
        
    
        if ddp:
            model.require_backward_grad_sync = (mini_epoch == mini_epochs-1)
        loss.backward()


        if use_lossfreebalance:
            for block in range(0, 16):
                expert_counts = torch.bincount(tt_loss[1].flatten(), minlength=model.blocks[block].ffn.moe_routed_experts)  
                avg_count = expert_counts.float().mean()
                for i, count in enumerate(expert_counts):
                    error = avg_count - count.float()
                    model.blocks[block].ffn.expert_biases.data[i] += update_rate * torch.sign(error)

    if ddp:
        dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG)
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # change lr
    lr = get_lr(epoch, warmup_lr_steps, max_lr, min_lr, epochs)
    for param_group in optmizer.param_groups:
        param_group['lr'] = lr
    optmizer.step()
    
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0

    # wrtie to the file losses
    if master_process and epoch%5==0:
        print(f"epoch: {epoch}, loss: {accumulated_loss:.5f}, tt_loss:{accumulated_tt_loss:.5f}, norm: {norm:.5f}, time: {dt*1000:.2f}ms, tok/s: {data_loader.B*data_loader.T*mini_epochs*ddp_world_size/dt:.2f}")
        with open(log_file, "a") as f:
            f.write(f"epoch:{epoch} loss:{accumulated_loss.item():.5f} tt_loss:{accumulated_tt_loss:.5f}\n")
if ddp:
    destroy_process_group()