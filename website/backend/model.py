from Classess import GPT
import torch
import torch.nn.functional as F
import tiktoken

import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint_path = './model_fourepochs_76291_125M.pt'
model_from_checkpoint = GPT()

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

state_dict = checkpoint['model']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_state_dict[k[len("_orig_mod."):]] = v 
    else:
        new_state_dict[k] = v

missing_keys, unexpected_keys = model_from_checkpoint.load_state_dict(new_state_dict, strict=False)
if missing_keys:
    print(f"Missing keys: {missing_keys}")
if unexpected_keys:
    print(f"Unexpected keys: {unexpected_keys}")
    
model_from_checkpoint = model_from_checkpoint.to(device)

print("device: ", device)

enc = tiktoken.get_encoding("gpt2")
