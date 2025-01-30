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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")
SEED = 1337

torch.manual_seed(SEED)
if device == 'cuda':
    torch.cuda.manual_seed(SEED)

# assert batch_size % (mini_batches * time_stamps) == 0, "batch_size is not devided by B and T"
# mini_epochs = int(batch_size / (mini_batches * time_stamps)) #number of mini-batches to get 0.5M batch

@dataclass
class ModelConfig:
    vocab_size: int

    num_dims: int
    num_heads: int                      # querry heads
    num_kv_heads: int
    num_layers: int
    # ffn_hidden_dims: int                # 

    batch_size: int
    mini_batches: int
    time_stamps: int
    context_len: int
    use_cache: bool                     # KV-cache
    use_flash: bool                     # flash attention

    # moe_type: str                     # DeepSeek/default
    moe_num_experts: int                # total number of experts
    moe_routed_experts: int             # top_k (how many experts to choose per token)
    moe_eps: float = 1e-6
    moe_aux_loss_coef: float = 0.01
    moe_shared_experts: int = 0         # number of experts shared experts for DeepSeekMoE
    use_lossfreebalance: bool = False   # AUXILIARY-LOSS-FREE LOAD BALANCING STRATEGY FOR MIXTURE-OF-EXPERTS from DeepSeek https://arxiv.org/pdf/2408.15664

    rmsnorm_eps: float = 1e-6
    rope_theta: float = 1e5

    

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
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
    
# Help function for RoPE
def repeat_kv(vct: torch.Tensor, n_times: int):
    bsz, cl, num_kv_heads, dm = vct.shape
    if n_times == 1:
        return vct
    else:
        return (
            vct[:, :, :, None, :]
            .expand(bsz, cl, num_kv_heads, n_times, dm)
            .reshape(bsz, cl, num_kv_heads * n_times, dm)
        )


# This function was taked from https://github.com/hkproj/pytorch-llama/blob/main/model.py
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
    def __init__(self, config):
        super().__init__()
        self.g = nn.Parameter(torch.ones(config.num_dims))
        self.eps = config.rmsnorm_eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.g * self._norm(x.float()).type_as(x)
    

class GroupedQueryAttention(nn.Module):
    def __init__(self, config, device=device):
        super().__init__()

        self.use_cache = config.use_cache
        self.use_flash = config.use_flash

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads

        self.num_heads = config.num_heads
        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.num_dims // self.num_heads

        self.wq = nn.Linear(config.num_dims, config.num_dims, bias=False)
        self.wk = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.num_dims, config.num_dims, bias=False)

    def forward(self, x, freqs_complex, start_pos = 0):
        c_batch_size, c_context_len, c_dim = x.shape # c_context_len = 1
    
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(c_batch_size, c_context_len, self.num_heads, self.head_dim)      # B, T, qh, hs
        k = k.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim)   # B, T, kh, hs
        v = v.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim)   # B, T, vh, hs

        queries = apply_rotary_pos(q, freqs_complex, device=x.device)
        keys = apply_rotary_pos(k, freqs_complex, device=x.device)

        if self.use_cache:
            # Initialize cache if not exist
            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    (c_batch_size, self.config.context_len, self.num_kv_heads, self.head_dim),
                    device=x.device
                )
                self.cache_v = torch.zeros(
                    (c_batch_size, self.config.context_len, self.num_kv_heads, self.head_dim),
                    device=x.device
                )
            # Update cache
            self.cache_k[:c_batch_size, start_pos:start_pos + c_context_len] = keys
            self.cache_v[:c_batch_size, start_pos:start_pos + c_context_len] = v

            keys = self.cache_k[:c_batch_size, :start_pos + c_context_len]
            v = self.cache_v[:c_batch_size, :start_pos + c_context_len]
            

        if self.use_flash:
            output = F.scaled_dot_product_attention(queries, keys, v, is_causal=True, enable_gqa=True)
            
        else: # Calculate Grouped Query Attention manually
            keys = repeat_kv(keys, self.num_rep)
            values = repeat_kv(v, self.num_rep)
    
            queries = queries.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
    
            attention = torch.matmul(queries, keys.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
    
            attention = torch.tril(attention[:, :, :c_context_len, :c_context_len])
            attention = attention.masked_fill(attention == 0, float("-inf"))
    
            attention = F.softmax(attention, dim=-1).type_as(queries)
            output = torch.matmul(attention, values)

        output = output.transpose(2, 1).contiguous().view(c_batch_size, c_context_len, c_dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Caclulating number of hidden dimensions like in Llama2
        multiple_of=4
        ffn_dim_multiplier=None
        hidden_dim = 4 * config.num_dims
        hidden_dim = int(2 * config.num_dims / 3)

        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(config.num_dims, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.num_dims, bias=False)
        self.w3 = nn.Linear(config.num_dims, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class FFNwMoE(nn.Module): # MoE Layer/DeepSeek MoE Layer
    def __init__(self, config):
        super().__init__()
        # self.moe_type = config.moe_type.lower()
        self.moe_routed_experts = config.moe_routed_experts # top_k
        self.moe_aux_loss_coef = config.moe_aux_loss_coef
        self.moe_eps = config.moe_eps
        self.moe_shared_experts = config.moe_shared_experts
        self.num_experts = config.moe_num_experts

        self.use_lossfreebalance = config.use_lossfreebalance
        
        # Caclulating number of hidden dimensions like in Llama2
        multiple_of=4
        ffn_dim_multiplier=None
        hidden_dim = 4 * config.num_dims
        hidden_dim = int(2 * config.num_dims / 3)

        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim

        self.router = nn.Linear(config.num_dims, self.num_experts, bias=False)
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            self.experts.append(
                nn.ModuleList([
                    nn.Linear(config.num_dims, hidden_dim, bias=False),
                    nn.Linear(hidden_dim, config.num_dims, bias=False),
                    nn.Linear(config.num_dims, hidden_dim, bias=False)
                ]))
        
        # shared experts (for DeepSeekMoE)
        self.shared_experts = nn.ModuleList()
        for _ in range(self.moe_shared_experts):
            self.shared_experts.append(
                nn.ModuleList([
                    nn.Linear(config.num_dims, hidden_dim, bias=False),
                    nn.Linear(hidden_dim, config.num_dims, bias=False),
                    nn.Linear(config.num_dims, hidden_dim, bias=False)
                ]))
            
        # AUXILIARY-LOSS-FREE LOAD BALANCING STRATEGY FOR MIXTURE-OF-EXPERTS from DeepSeek https://arxiv.org/pdf/2408.15664
        if self.use_lossfreebalance:
            self.expert_biases = nn.Parameter(torch.zeros(self.num_experts))
            
    def forward(self, x):
        c_batch_size, c_context_len, c_dim = x.shape
        x_flat = x.view(-1, c_dim)          #c_batch_size * c_context_len, c_dim

        router_out = self.router(x_flat)
        router_probs = F.softmax(router_out, dim=-1) 

        _, topk_indices = router_out.topk(self.moe_routed_experts, dim=-1)
        if not self.use_lossfreebalance:
            topk_probs, _ = router_probs.topk(self.moe_routed_experts, dim=-1)
            expert_mask = F.one_hot(topk_indices[:, 0], self.num_experts).float()
            density = expert_mask.mean(dim=0)
            router_prob_mean = router_probs.mean(dim=0)
            aux_loss = self.moe_aux_loss_coef * torch.sum(density * router_prob_mean) * self.num_experts

        else: # if use_lossfreebalance
            router_out = router_out + self.expert_biases
            router_probs = torch.sigmoid(router_out) # from https://arxiv.org/pdf/2408.15664 paper
            topk_probs = router_probs.gather(-1, topk_indices)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            # In the case of AUXILIARY-LOSS-FREE LOAD BALANCING we pass router_probs, topk_indices as aux_loss for further calculations 
            aux_loss = (router_probs, topk_indices)

        # topk_probs, topk_indices = router_probs.topk(self.moe_routed_experts, dim=-1)
        output = torch.zeros_like(x_flat)

        # expert_mask = torch.zeros_like(router_probs).scatter(-1, topk_indices, 1.0)
        # density = expert_mask.mean(dim=0)
        # router_prob_mean = router_probs.mean(dim=0)

        # cv_expert = density.std() / (density.mean() + self.moe_eps)
        # cv_router = router_prob_mean.std() / (router_prob_mean.mean() + self.moe_eps)
        
        # aux_loss = (cv_expert**2 + cv_router**2) * self.moe_aux_loss_coef

        
        for i in range(self.moe_routed_experts):
            expert_index = topk_indices[:, i]
            expert_probs = topk_probs[:, i]

            for expert_id in range(self.num_experts):
                idx = (expert_id == expert_index).nonzero().squeeze()

                if idx.numel() == 0:
                    continue
                x_for_expert = x_flat[idx]
                w1, w2, w3 = self.experts[expert_id]
                
                expert_output = w2(F.silu(w1(x_for_expert)) * w3(x_for_expert))
                output[idx] += expert_output * expert_probs[idx].unsqueeze(-1)

        # shared exprets(for DeepSeekMoE)
        for shared_expert_id in range(self.moe_shared_experts):
            w1, w2, w3 = self.shared_experts[shared_expert_id]
            expert_output = w2(F.silu(w1(x_flat)) * w3(x_flat))
            output = output + expert_output


        return output.view(c_batch_size, c_context_len, c_dim), aux_loss



class Block(nn.Module):
    def __init__(self, config, device=device):
        super().__init__()

        self.attention = GroupedQueryAttention(config, device=device)
        self.ffn = FFNwMoE(config)

        self.norm_attention = RMSNorm(config)
        self.norm_ffn = RMSNorm(config)

    def forward(self, x, freqs_complex, start_pos):
        x = x + self.attention(
            self.norm_attention(x), 
            freqs_complex, 
            start_pos
            )
        
        ffn_out, aux_loss = self.ffn(
            self.norm_ffn(x)
            )
        x = x + ffn_out
        return x, aux_loss
    

class Transformer(nn.Module, PyTorchModelHubMixin): # extending PyTorchModelHubMixin for save weights as safetensors
    def __init__(self, config):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.num_dims = config.num_dims
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.context_len = config.context_len

        self.use_lossfreebalance = config.use_lossfreebalance

        self.tokens_embedding = nn.Embedding(self.vocab_size, self.num_dims)

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(Block(config))

        self.norm = RMSNorm(config)
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
        
        x = self.tokens_embedding(x)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        total_aux_loss = 0

        for block in self.blocks:
            x, aux_loss = block(x, freqs_complex=freqs_complex, start_pos=start_pos)
            if not self.use_lossfreebalance:
                total_aux_loss += aux_loss

        x = self.norm(x)
        logits = self.ll_head(x)
        
        
        if targets is None:
            loss = None
            tt_loss = None
        else:
            c_batch_size, c_context_len, c_dim = logits.shape
            logits = logits.view(c_batch_size*c_context_len, c_dim)
            targets = targets.view(c_batch_size*c_context_len)
            tt_loss = F.cross_entropy(logits, targets)
            if not self.use_lossfreebalance: loss = tt_loss + total_aux_loss    # in this case, tt_loss its loss w/o aux_loss
            else: 
                                                                                # if we want to use AUXILIARY-LOSS-FREE LOAD BALANCING we pass router_probs, topk_indices as tt_loss
                loss = tt_loss
                tt_loss = aux_loss

        return logits, loss, tt_loss

    @torch.no_grad()
    def generate(self, x, max_tokens, use_cache=False):
        for c_tkn_pos in range(max_tokens):
            if use_cache:
                if c_tkn_pos == 0:
                    logits, _, tt_loss = self.forward(x, start_pos=c_tkn_pos)
                else:
                    logits, _, tt_loss = self.forward(x[:, -1], start_pos=c_tkn_pos)
            else:
                logits, _, tt_loss = self.forward(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x
    

def main():
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

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


if __name__ == "__main__":
    main()