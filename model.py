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

# Help functions
def repeat_kv(vct, n_times):
    bsz, cl, num_kv_heads, dm = vct.shape
    if n_times == 1:
        return vct
    else:
        return (
            vct[:, :, :, None, :]
            .expand(bsz, cl, num_kv_heads, n_times, dm)
            .reshape(bsz, cl, num_kv_heads * n_times, dm)
        )


# https://github.com/hkproj/pytorch-llama/blob/main/model.py
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_pos(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps : 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.g * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, num_dim, num_heads, num_kv_heads, batch_size, context_len, use_cache=False, device=device):
        super().__init__()

        self.use_cache = use_cache

        self.num_q_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        self.num_heads = num_heads
        self.num_rep = self.num_q_heads // self.num_kv_heads
        self.head_dim = num_dim // self.num_q_heads

        self.wq = nn.Linear(num_dim, num_dim, bias=False)
        self.wk = nn.Linear(num_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(num_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_dim, num_dim, bias=False)

        self.cache_k = torch.zeros(
            (
                batch_size,
                context_len,
                self.num_kv_heads,
                self.head_dim
            ), device=device
        )

        self.cache_v = torch.zeros(
            (
                batch_size,
                context_len,
                self.num_kv_heads,
                self.head_dim
            ), device=device
        )

    def forward(self, x, freqs_complex, start_pos = 0):
        c_bsz, c_cl, c_dm = x.shape # c_cl = 1
    
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(c_bsz, c_cl, self.num_q_heads, self.head_dim) # B, T, qh, hs
        k = k.view(c_bsz, c_cl, self.num_kv_heads, self.head_dim) # B, T, kh, hs
        v = v.view(c_bsz, c_cl, self.num_kv_heads, self.head_dim) # B, T, vh, hs

        queries = apply_rotary_pos(q, freqs_complex, device=x.device)
        keys = apply_rotary_pos(k, freqs_complex, device=x.device)

        if self.use_cache:
            self.cache_k[:c_bsz, start_pos:start_pos+c_cl] = keys
            self.cache_v[:c_bsz, start_pos:start_pos+c_cl] = v
            
            keys = self.cache_k[:c_bsz, :start_pos+c_cl]
            v = self.cache_v[:c_bsz, :start_pos+c_cl]

        keys = repeat_kv(keys, self.num_rep)
        values = repeat_kv(v, self.num_rep)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention = torch.matmul(queries, keys.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        attention = torch.tril(attention[:, :, :c_cl, :c_cl])
        attention = attention.masked_fill(attention == 0, float("-inf"))

        attention = F.softmax(attention, dim=-1).type_as(queries)
        output = torch.matmul(attention, values)

        output = output.transpose(2, 1).contiguous().view(c_bsz, c_cl, c_dm)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, num_dims, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        
        hidden_dim = 4 * num_dims
        hidden_dim = int(2 * num_dims / 3)

        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(num_dims, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, num_dims, bias=False)
        self.w3 = nn.Linear(num_dims, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, num_dims, multiple_of, num_heads, num_kv_heads, batch_size, context_len, use_cache=False, eps = 1e-6, ffn_dim_multiplier=None,device=device):
        super().__init__()

        self.attention = SelfAttention(num_dims=num_dims, num_heads=num_heads, num_kv_heads=num_kv_heads, batch_size=batch_size, context_len=context_len, use_cache=use_cache, device=device)
        self.ffn = FeedForward(num_dims=num_dims, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier)

        self.norm_attention = RMSNorm(num_dims, eps=eps)
        self.norm_ffn = RMSNorm(num_dims, eps=eps)

    def forward(self, x, freqs_complex, start_pos):
        x = x + self.attention(
            self.norm_attention(x), 
            freqs_complex, 
            start_pos
            )
        
        x = x + self.ffn(
            self.norm_ffn(x)
            )
        return x

        
class Transformer(nn.Module):
    def __init__(self, device, vocab_size, batch_size, context_len, num_layers, num_dims, multiple_of, num_heads, num_kv_heads, use_cache=False, eps = 1e-6, ffn_dim_multiplier=None, device=device):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.context_len = context_len

        self.tokens_embedding = nn.Embedding(self.vocab_size, self.num_dims)

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(Block(
            num_dims=num_dims, multiple_of=multiple_of, num_heads=num_heads, num_kv_heads=num_kv_heads, batch_size=batch_size, context_len=context_len, use_cache=use_cache, eps = 1e-6, ffn_dim_multiplier=ffn_dim_multiplier, device=device
            ))

        self.norm = RMSNorm(self.num_dims, eps)
        self.ll_head = nn.Linear(self.num_dims, self.vocab_size, bias=False)

        self.tokens_embedding.weight = self.ll_head.weight

        self.freqs_complex = precompute_theta_pos_frequencies(self.num_dims // self.num_heads, self.context_len * 2, device=device)

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

    def forward(self, x, targets=None, start_pos=0):
        _, seq_len = x.shape
        
        x = self.tokens_embedding(x) #
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        for block in self.blocks:
            x = block(x, freqs_complex=freqs_complex, start_pos=start_pos)

        x = self.norm(x)
        logits = self.ll_head(x)
        
        
        if targets is None:
            loss = None
        else:
            c_bsz, c_cl, c_dm = logits.shape
            logits = logits.view(c_bsz*c_cl, c_dm)
            targets = targets.view(c_bsz*c_cl)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, x, max_tokens, use_cache=False):
        for c_tkn_pos in range(max_tokens):
            if use_cache:
                logits, _ = self.forward(x[:, -1], start_pos=c_tkn_pos)
            else:
                logits, _ = self.forward(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x

