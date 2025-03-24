# python train.py

from model import Transformer, ModelConfig
from trainer import Trainer, TrainerConfig, DataLoader

from transformers import AutoTokenizer
import torch

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

tokenizer_id = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token

checkpoint_path = ''
continue_train = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_config = TrainerConfig(
    vocab_size = tokenizer.vocab_size,
    num_epochs = 1,

    use_ddp = False,
    use_moe = False,
    use_lossfreebalance = False,
    clean_cuda_cache = True,
    use_compile = True,
    use_dtype = "bfloat16",

    seed = 1338,
    max_seq_len = 1536, # 1536
    batch_size = 16, # 16,
    accumulation_steps = int(2**19//(1536 * 32)),
    
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    learning_rate = 4e-4,
    betas = (0.90, 0.97),
    update_rate = 5e-6,

    val_ratio = 0.005,
    steps_for_eval = 20,
    eval_interval = 20,

    checkpoints_frequency = 2000,
    path_to_checkpoints = "./model_testing",

    tokenized_dataset_path = "cosmopedia",
    eval_log_file = "log/eval_cosmopedia.txt",
)

config = ModelConfig(
        vocab_size = tokenizer.vocab_size,

        num_dims = 512,
        num_heads = 16,
        num_kv_heads = 4,
        num_layers = 32,
        ffn_hidden_dims = 512 * 4,

        rmsnorm_eps = 1e-6,
        rope_theta = 1e5,
    
        context_len = 1536,
        
        use_cache = False,
        use_flash = True,
        use_moe = False,

        moe_num_experts = 2,
        moe_active_experts = 2,
        moe_eps = 1e-6,
        moe_aux_loss_coef = 0.01,
        moe_shared_experts = 1,
        use_lossfreebalance = False,
    )


model = Transformer(config)
if continue_train:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v 
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

model.to(device)

data_loader = DataLoader(train_config)
trainer = Trainer(train_config, model, tokenizer)
trainer.train(data_loader)