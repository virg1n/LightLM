import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken 
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm
from safetensors import torch as safe_torch
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

enc = tiktoken.get_encoding("gpt2")

@dataclass
class ModelConfig:
    vocab_size: int

    num_dims: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    ffn_hidden_dims: int

    batch_size: int
    mini_batches: int
    time_stamps: int
    context_len: int
    use_cache: bool
    use_flash: bool

    num_experts: int
    moe_topk: int
    moe_eps: float = 1e-6
    moe_aux_loss_coef: float= 0.007

    rmsnorm_eps: float = 1e-6
    rope_theta: float = 1e5

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

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta_vals = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) 
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta_vals).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_pos(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
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

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads

        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.num_dims // self.num_heads

        self.wq = nn.Linear(config.num_dims, config.num_dims, bias=False)
        self.wk = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.num_dims, config.num_dims, bias=False)

        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_complex, start_pos=0):
        c_batch_size, c_context_len, c_dim = x.shape 

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(c_batch_size, c_context_len, self.num_heads, self.head_dim)  # B, T, qh, hs
        k = k.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim)  # B, T, kh, hs
        v = v.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim)  # B, T, vh, hs

        queries = apply_rotary_pos(q, freqs_complex, device=x.device)
        keys = apply_rotary_pos(k, freqs_complex, device=x.device)

        if self.use_cache:
            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    (c_batch_size, self.config.context_len, self.num_kv_heads, self.head_dim),
                    device=x.device
                )
                self.cache_v = torch.zeros(
                    (c_batch_size, self.config.context_len, self.num_kv_heads, self.head_dim),
                    device=x.device
                )
            self.cache_k[:c_batch_size, start_pos:start_pos + c_context_len] = keys
            self.cache_v[:c_batch_size, start_pos:start_pos + c_context_len] = v

            keys = self.cache_k[:c_batch_size, :start_pos + c_context_len]
            v = self.cache_v[:c_batch_size, :start_pos + c_context_len]

        keys = repeat_kv(keys, self.num_rep)
        values = repeat_kv(v, self.num_rep)

        queries = queries.transpose(1, 2) 
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention = torch.matmul(queries, keys.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        attention = torch.tril(attention)
        attention = attention.masked_fill(attention == 0, float("-inf"))

        attention = F.softmax(attention, dim=-1).type_as(queries)
        output = torch.matmul(attention, values)

        output = output.transpose(2, 1).contiguous().view(c_batch_size, c_context_len, c_dim)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        multiple_of = 4
        ffn_dim_multiplier = None
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

class FFNwMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moe_aux_loss_coef = config.moe_aux_loss_coef
        self.moe_eps = config.moe_eps

        multiple_of = 4
        ffn_dim_multiplier = None
        hidden_dim = 4 * config.num_dims
        hidden_dim = int(2 * config.num_dims / 3)

        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim

        self.top_k = config.moe_topk

        self.num_experts = config.num_experts
        self.router = nn.Linear(config.num_dims, self.num_experts, bias=False)
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            self.experts.append(
                nn.ModuleList([
                    nn.Linear(config.num_dims, hidden_dim, bias=False),
                    nn.Linear(hidden_dim, config.num_dims, bias=False),
                    nn.Linear(config.num_dims, hidden_dim, bias=False)
                ]))

    def forward(self, x):
        c_batch_size, c_context_len, c_dim = x.shape
        x_flat = x.view(-1, c_dim)  # c_batch_size * c_context_len, c_dim

        router_out = F.softmax(self.router(x_flat), dim=-1)
        topk_probs, topk_indices = router_out.topk(self.top_k, dim=-1)

        output = torch.zeros_like(x_flat)
        aux_loss = 0.0

        expert_mask = F.one_hot(topk_indices[:, 0], self.num_experts).float()
        density = expert_mask.mean(dim=0)
        router_prob_mean = router_out.mean(dim=0)
        aux_loss = self.moe_aux_loss_coef * torch.sum(density * router_prob_mean) * self.num_experts

        for i in range(self.top_k):
            expert_index = topk_indices[:, i]
            expert_probs = topk_probs[:, i]

            for expert_id in range(self.num_experts):
                idx = (expert_id == expert_index).nonzero(as_tuple=True)[0]

                if len(idx) == 0:
                    continue
                x_for_expert = x_flat[idx]
                w1, w2, w3 = self.experts[expert_id]

                expert_output = w2(F.silu(w1(x_for_expert)) * w3(x_for_expert))
                output[idx] += expert_output * expert_probs[idx].unsqueeze(-1)

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

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config 
        self.vocab_size = config.vocab_size
        self.num_dims = config.num_dims
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.context_len = config.context_len

        self.tokens_embedding = nn.Embedding(self.vocab_size, self.num_dims)

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(Block(config))

        self.norm = RMSNorm(config)
        self.ll_head = nn.Linear(self.num_dims, self.vocab_size, bias=False)

        self.ll_head.weight = self.tokens_embedding.weight

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.num_dims // self.num_heads, 
            self.context_len * 2, 
            device=device,
            theta=config.rope_theta
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
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
            total_aux_loss += aux_loss

        x = self.norm(x)
        logits = self.ll_head(x)
        
        if targets is None:
            loss = None
            tt_loss = None
        else:
            c_batch_size, c_context_len, c_dim = logits.shape
            logits = logits.view(c_batch_size * c_context_len, c_dim)
            targets = targets.view(c_batch_size * c_context_len)
            tt_loss = F.cross_entropy(logits, targets)
            loss = tt_loss + total_aux_loss

        return logits, loss, tt_loss

    @torch.no_grad()
    def generate(self, x, max_tokens, use_cache=False):
        for c_tkn_pos in range(max_tokens):
            if use_cache:
                if c_tkn_pos == 0:
                    logits, _ = self.forward(x, start_pos=c_tkn_pos)
                else:
                    logits, _ = self.forward(x[:, -1], start_pos=c_tkn_pos)
            else:
                logits, _ = self.forward(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x

def load_model_from_safetensors(safetensors_path: str, config: ModelConfig) -> Transformer:
    model = Transformer(config)
    model.to(device)

    safe_dict = safe_torch.load_file(safetensors_path, device=device)

    missing_keys, unexpected_keys = model.load_state_dict(safe_dict, strict=False)
    if missing_keys:
        print(f"Missing keys when loading model: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys when loading model: {unexpected_keys}")

    model.eval()

    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Model compilation failed: {e}. Continuing without compilation.")

    return model

def encode_text(text: str):
    return enc.encode(text)

def compute_choice_logprob(model, prompt_tokens, choice_tokens):
    input_ids = prompt_tokens + choice_tokens
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _, _ = model(input_ids_tensor)

    choice_start = len(prompt_tokens)
    
    relevant_logits = logits[0, choice_start-1:-1, :]  
    target_ids = torch.tensor(choice_tokens, device=device)

    log_probs = F.log_softmax(relevant_logits, dim=-1)
    choice_logprobs = log_probs[range(len(choice_tokens)), target_ids]

    return choice_logprobs.sum().item()

def evaluate_hellaswag(model, num_samples=500):
    dataset = load_dataset("hellaswag", split="validation")
    correct, total = 0, 0

    if num_samples > 0:
        first_example = dataset[0]
        print("HellaSwag first example keys:", first_example.keys())

    for example in tqdm(dataset.select(range(num_samples)), desc="Evaluating HellaSwag"):
        try:
            ctx_a = example.get('ctx_a', '')
            ctx_b = example.get('ctx_b', '')
            ctx_c = example.get('ctx_c', '')
            startphrase = example.get('startphrase', '')
            context = f"{ctx_a} {ctx_b} {ctx_c} {startphrase}".strip()
            choices = example["endings"]
            label = example["label"]
        except KeyError as e:
            print(f"KeyError while accessing HellaSwag example: {e}")
            print("Available keys:", example.keys())
            continue

        prompt_tokens = encode_text(context)
        choice_logprobs = []

        for choice in choices:
            choice_tokens = encode_text(" " + choice)  # Add a space for separation
            ll = compute_choice_logprob(model, prompt_tokens, choice_tokens)
            choice_logprobs.append(ll)

        pred = max(range(len(choices)), key=lambda i: choice_logprobs[i])
        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"HellaSwag accuracy on {num_samples} samples = {accuracy:.4f}")
    return accuracy

def evaluate_mmlu(model, subject="abstract_algebra", split="validation", num_samples=200):
    dataset = load_dataset("lukaemon/mmlu", subject, split=split)
    correct, total = 0, 0

    if num_samples > 0:
        first_example = dataset[0]
        print(f"MMLU-{subject} first example keys:", first_example.keys())

    for example in tqdm(dataset.select(range(num_samples)), desc=f"Evaluating MMLU-{subject}"):
        try:
            question = example["question"]
            correct_label = example["answer"].upper() 
            choices = [example["A"], example["B"], example["C"], example["D"]]
            label_idx = ord(correct_label) - ord("A") 
        except KeyError as e:
            print(f"KeyError while accessing MMLU-{subject} example: {e}")
            print("Available keys:", example.keys())
            continue

        prompt_tokens = encode_text(question)
        choice_logprobs = []

        for choice in choices:
            choice_tokens = encode_text(" " + choice)
            ll = compute_choice_logprob(model, prompt_tokens, choice_tokens)
            choice_logprobs.append(ll)

        pred_idx = max(range(len(choices)), key=lambda i: choice_logprobs[i])
        if pred_idx == label_idx:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"MMLU ({subject}) accuracy on {num_samples} samples = {accuracy:.4f}")
    return accuracy

def evaluate_all_mmlu(model, split="validation", num_samples=50):
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", 
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "econometrics", "electrical_engineering",
        "elementary_mathematics", "formal_logic", "global_facts", "government",
        "health_professions", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_european_history",
        "high_school_geography", "high_school_government_politics",
        "high_school_macroeconomics", "high_school_mathematics",
        "high_school_microeconomics", "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history", "high_school_world_history",
        "human_aging", "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "molecular_biology", "music", "nutrition", "philosophy", "prehistory",
        "professional_accounting", "professional_law", "professional_medicine",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "software_engineering", "us_foreign_policy", "virology"
    ]

    scores = []
    for sbj in subjects:
        print(f"--- Evaluating MMLU Subject: {sbj} ---")
        acc = evaluate_mmlu(model, subject=sbj, split=split, num_samples=num_samples)
        scores.append(acc)
    
    mean_score = sum(scores) / len(scores) if scores else 0
    print(f"Average MMLU Accuracy over {len(subjects)} tasks = {mean_score:.4f}")
    return mean_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom LLM on HellaSwag and MMLU")
    parser.add_argument("--model_path", type=str, default="./model_testing/model.safetensors",
                        help="Path to the model weights in .safetensors format")
    parser.add_argument("--eval_dataset", type=str, choices=["hellaswag", "mmlu", "all_mmlu"], default="hellaswag",
                        help="Dataset to evaluate on: 'hellaswag', 'mmlu', or 'all_mmlu'")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--mmlu_subject", type=str, default="abstract_algebra",
                        help="Specific MMLU subject to evaluate on (if eval_dataset is 'mmlu')")
    args = parser.parse_args()

    config = ModelConfig(
        vocab_size=50304,
        num_dims=1024,
        num_heads=16,
        num_kv_heads=4,
        num_layers=16,
        ffn_hidden_dims=4*1024,

        batch_size=2**19,
        mini_batches=2,
        time_stamps=512,
        context_len=1024,
        use_cache=False,
        use_flash=False,

        num_experts=6,
        moe_topk=1,
        moe_eps=1e-6,
        moe_aux_loss_coef=0.01,

        rmsnorm_eps=1e-6,
        rope_theta=1e5
    )

    print("Loading model...")
    model = load_model_from_safetensors(args.model_path, config)
    print("Model loaded successfully.")

    if args.eval_dataset == "hellaswag":
        evaluate_hellaswag(model, num_samples=args.num_samples)
    elif args.eval_dataset == "mmlu":
        evaluate_mmlu(model, subject=args.mmlu_subject, split="validation", num_samples=args.num_samples)
    elif args.eval_dataset == "all_mmlu":
        evaluate_all_mmlu(model, split="validation", num_samples=args.num_samples)
    else:
        print("Unsupported dataset selected.")

if __name__ == "__main__":
    main()
