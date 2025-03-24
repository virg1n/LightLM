import torch
from transformers import AutoTokenizer

from model import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_id = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token

model = Transformer.from_pretrained("./model_FFN").to(device)

input_ids = tokenizer(["I am a language model,"], return_tensors="pt")['input_ids'].to(device)
idx = model.generate(input_ids, temperature=0.48, top_k=40, max_tokens=30)
print(tokenizer.batch_decode(idx)[0])
