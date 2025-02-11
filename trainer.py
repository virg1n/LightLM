import time
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

from datatrove.utils.dataset import DatatroveFolderDataset

@dataclass
class TrainerConfig:
    vocab_size: int                 
    num_epochs: int                 

    use_ddp: bool                   
    use_moe: bool                   # enable mixture-of-experts
    use_lossfreebalance: bool       # use Auxiliary-loss-free load balancing strategy for mixture-of-experts from DeepSeek https://arxiv.org/pdf/2408.15664
    clean_cuda_cache: bool = True   # Helps prevent OOM errors during eval on large models
    use_compile: bool = True        # use torch.compile()
    use_dtype: str = "bfloat16"

    seed: int = 1998                
    max_seq_len: int = 1024         # maximum context length for batch
    batch_size: int = 1             # numbe of batches
    accumulation_steps: int = 1
    
    # Optimizer parameters
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    learning_rate: float = 1e-3
    betas: Tuple[float, float] = (0.90, 0.95)
    update_rate: float = 1e-5  # update_rate of biases for loss-free balancing

    val_ratio: int = 0.005
    steps_for_eval: int = 20                            # number of steps for evaluation
    eval_interval: int = 50

    checkpoints_frequency: int = 500
    path_to_checkpoints: str = "./model_testing"        # path to directory to save checkpoints

    tokenized_dataset_path: str = ""                    # path to directory with tokeized dataset
    eval_log_file: str = "logs/eval.txt"                # path to file to write eval results



class DataLoader():
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.current_epoch = 0
        self.seed = config.seed
        self.token_size = 2 if config.vocab_size < 65535 else 4
        self.rank = rank

        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        if rank == 0:
            print(f"{'Total tokens loaded: '} {self.len_dataset * config.max_seq_len:,}")

        self.train_len_dataset = math.ceil((1-config.val_ratio) * self.len_dataset)
        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        shard_size = self.len_dataset // world_size 
        self.train_start_idx = rank * shard_size
        self.train_end_idx = self.train_start_idx + shard_size
        self.train_current_idx = self.train_start_idx

        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx

    def get_batch(self, current_idx: int, start_idx: int, end_idx: int):
        new_idx = current_idx + self.config.batch_size
        
        x_l, y_l = zip(*[(self.dataset[idx]['input_ids'][:-1], self.dataset[idx]['input_ids'][1:])
                    for idx in range(current_idx, min(new_idx, self.len_dataset))])
        x, y = torch.stack(list(x_l)), torch.stack(list(y_l))
    
        if new_idx >= end_idx:
            new_idx = start_idx
            self.new_epoch()

        return x, y, new_idx

    def next_batch(self, split):
        if split == "train":
            x, y, self.train_current_idx = self.get_batch(self.train_current_idx, self.train_start_idx, self.train_end_idx)
        else: # validation
            x, y, self.val_current_idx = self.get_batch(self.val_current_idx, self.val_start_idx, self.len_dataset)
        return x, y
    
    def reset(self, rank: int = 0, world_size: int = 1):
        self.current_epoch = 0
        self.seed = self.config.seed
        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        shard_size = self.len_dataset // world_size 
        self.train_start_idx = rank * shard_size
        self.train_end_idx = self.train_start_idx + shard_size
        self.train_current_idx = self.train_start_idx

        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx

    def new_epoch(self):
        self.current_epoch += 1
        self.load_dataset(self.seed + self.current_epoch)

    def load_dataset(self, seed: int):
        self.dataset = DatatroveFolderDataset(
            folder_path=self.config.tokenized_dataset_path,
            filename_pattern=os.path.join(self.config.tokenized_dataset_path, "**", "*.ds"),
            seq_len=self.config.max_seq_len,
            token_size=self.token_size,
            recursive=True,
            shuffle=True,
            seed=seed + self.rank
        )

    def num_train_steps(self):
        return math.ceil((self.train_end_idx-self.train_start_idx) / self.config.batch_size)


class Trainer():
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.num_epochs = config.num_epochs

        self.use_moe = config.use_moe
        self.use_lossfreebalance = config.use_lossfreebalance if self.use_moe else False
        self.clean_cuda_cache = config.clean_cuda_cache
        self.dtype = getattr(torch, self.config.use_dtype)

        self.steps_for_eval = config.steps_for_eval
        self.weight_decay = config.weight_decay
        self.update_rate = config.update_rate if self.use_moe else 0

        self.device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu'
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(config.seed)
            n_gpus = torch.cuda.device_count()

        use_compile = self.config.use_compile and self.device == "cuda" and torch.__version__.startswith("2")
        if use_compile:
            self.model = torch.compile(self.model)
            
        # DDP
        if n_gpus > 1 and config.use_ddp:   
            self.ddp = True
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = torch.device(f"cuda:{self.ddp_local_rank}")
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0

            self.model.to(self.device)
            
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
            self.raw_m = model
        else:
            self.ddp = False
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.master_process = True

            if self.device != "cpu":
                self.model.to(self.device)

        if self.master_process:
            print("Device:", self.device)
            print(f"Model's trainable params: {sum([p.data.numel() for p in self.model.parameters() if p.requires_grad]) / 1e6:.2f}M")
            print(f"Tokens per step: {self.config.batch_size * self.config.max_seq_len * self.ddp_world_size * self.config.accumulation_steps}")
            print(f"use {'torch.compile()'}: {use_compile}")
            print(f"Use MoE: {'Yes ' if self.use_moe else 'No'}")
            if self.use_moe:
                print(f"Number of experts: {self.model.blocks[0].ffn.num_experts}")
                print(f"Number of used experts during inference: {self.model.blocks[0].ffn.moe_routed_experts}")
                print(f"Method of aux_loss: {'loss-free-balance' if config.use_lossfreebalance else 'default'}")
                print(f"Number of parameters will be used during inference: {((sum([p.data.numel() for p in self.model.parameters() if p.requires_grad]) - sum(p.numel() for p in self.model.blocks[0].ffn.parameters()) * len(self.model.blocks) * (1-(self.model.blocks[0].ffn.moe_routed_experts + self.model.blocks[0].ffn.moe_shared_experts) / (self.model.blocks[0].ffn.num_experts + self.model.blocks[0].ffn.moe_shared_experts)))) / 1e6:.2f}M")
    
    def step(self, data_loader, accumulation_steps: int,
              num_tokens: int, split: str = "train"):
        """
        Performs single forward/backward pass with gradient accumulation.
            Returns: (total_loss, cross_entropy_loss, number_of_processed_tokens)
        """
        x, y = data_loader.next_batch(split=split)
        x, y = x.to(self.device), y.to(self.device)
        num_tokens += torch.numel(x)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            _, loss, ce_loss = self.model(x, y)

        loss /= accumulation_steps

        loss.backward()
        return loss, ce_loss, num_tokens
    

    def train(self, data_loader):
        num_steps_per_epoch = math.ceil(data_loader.num_train_steps() / self.config.accumulation_steps)

        # Configuration of optimizer and schedulers
        # Using AdamW with cosine decay and warmup - similar to Llama's training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),  
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.weight_decay,
            fused=(self.device=="cuda")
        )
        
        warmup_steps = math.floor(self.config.warmup_ratio * num_steps_per_epoch * self.num_epochs)
        warmup_factor = lambda step: 0.05 + 0.95 * (step / max(warmup_steps, 1))
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_factor
        )

        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=(num_steps_per_epoch * self.num_epochs) - warmup_steps, 
            eta_min=0.1 * self.config.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cos_scheduler],
            milestones=[warmup_steps])

        last_step = num_steps_per_epoch - 1
        self.model.train()

        for epoch in range(self.num_epochs):
            for step in range(num_steps_per_epoch):
                t0 = time.perf_counter()
                accumulated_loss = 0.0
                num_tokens = 0

                ddp_nosync_ctx = self.model.no_sync() if self.ddp else nullcontext()
                with ddp_nosync_ctx:
                    for _ in range(self.config.accumulation_steps - 1):
                        loss, ce_loss, num_tokens = self.step(data_loader, self.config.accumulation_steps, num_tokens, split="train")
                        accumulated_loss += loss

                loss, ce_loss, num_tokens = self.step(data_loader, self.config.accumulation_steps, num_tokens, split="train")
                accumulated_loss += loss.detach()

                # Calculate expert biases using Auxiliary Loss-Free Balance method for MoE (https://arxiv.org/pdf/2408.15664)
                if self.use_moe and self.use_lossfreebalance: 
                    for block in range(len(self.model.blocks)):
                        expert_counts = torch.bincount(ce_loss[1].flatten(), minlength=self.model.blocks[block].ffn.moe_routed_experts)  
                        avg_count = expert_counts.float().mean()
                        for i, count in enumerate(expert_counts):
                            error = avg_count - count.float()
                            self.model.blocks[block].ffn.expert_biases.data[i] += self.update_rate * torch.sign(error)

                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #ToDO

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                t1 = time.perf_counter()

                tokens_per_sec = num_tokens / (t1 - t0)

                # Logging 
                if self.master_process:
                    print(f"Epoch: {epoch} | Step: {step} |  loss: {accumulated_loss:.4f} | norm: {norm:.4f} |  tok/s: {tokens_per_sec}")
                
                # Evaluation 
                if self.master_process and ((step>0 and step % self.config.eval_interval == 0) or step == last_step):
                    self.model.eval() 
                    val_loss = self.eval(data_loader)

                    with open(self.config.eval_log_file, "a") as f:
                        f.write(f"Step: {step * (epoch+1)}, val_loss: {val_loss:.4f}, norm: {norm:.4f}, time: {t1 - t0:.2f}ms, tok/s: {tokens_per_sec:.1f} \n")

                    self.model.train()
                    if self.clean_cuda_cache:
                        torch.cuda.empty_cache()

                # Save Chekpoints
                if self.master_process and ((step % self.config.checkpoints_frequency == 0 and step > 0) or step == last_step):
                    self.save_checkpoints(self.config.path_to_checkpoints, name=str((epoch+1) * step))

    def eval(self, data_loader):
        """
        Evaluates model on validation split using running average of first [steps_for_eval] batches
        """
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(self.steps_for_eval):
                x, y = data_loader.next_batch(split="val")
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    _, loss, ce_loss = self.model(x, y)
                loss /= self.steps_for_eval
                val_loss_accum += loss.detach()
            return val_loss_accum

    def save_checkpoints(self, path: str, name: str):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"model.checkpoint.{name}.pt")
        # self.model.save_pretrained(".checkpoint_path", config=config)
        torch.save(self.model.state_dict(), checkpoint_path)
        print("Checkpoints saved")