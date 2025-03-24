import re
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_id = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token


def arc(model, save_name, temp=0.25):
    dataset = load_dataset('ai2_arc', 'ARC-Challenge', split='test')

    answer_pattern = re.compile(r'([A-Ea-e])')
    
    def format_prompt(example):
        """Formats an ARC example into a prompt with question and options."""
        prompt = f"Question: {example['question']}\nOptions:\n"
        for label, text in zip(example['choices']['label'], example['choices']['text']):
            prompt += f"{label}) {text}\n"
        prompt += "Answer:"
        return prompt
    
    correct = 0
    total = 0
    
    model.eval() 
    
    for example in tqdm(dataset, desc="Evaluating on ARC"):
        prompt = format_prompt(example)
        # Tokenize and truncate to model's context length
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536
        )['input_ids'].to(device)
        
        with torch.no_grad():
            # Generate answer tokens (max 10 new tokens)
            generated_ids = model.generate(
                input_ids,
                temperature=temp,
                top_k=5,
                max_tokens=10
            )
        
        # Extract only the newly generated tokens (excluding input)
        generated_tokens = generated_ids[:, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # Parse the generated text for the answer
        match = answer_pattern.search(generated_text)
        predicted_answer = match.group(1).upper() if match else None
        correct_answer = example['answerKey']
        
        total += 1
        if predicted_answer == correct_answer:
            correct += 1
    
    accuracy = correct / total
    with open('evaluation.txt', 'a') as f:
        f.write(f"{save_name}: ARC Evaluation Results:\n")
        arc_result = accuracy
        f.write(str(arc_result) + "\n")

    print(f"Accuracy on ARC-Challenge: {accuracy * 100:.2f}%")


def wino(model, save_name, temp=0.25):
    dataset = load_dataset('winogrande', 'winogrande_xl', split='validation')

    answer_pattern = re.compile(r'([A-Ba-b])')
    
    def format_prompt(example):
        """Formats a WinoGrande example into a prompt with options."""
        prompt = (
            f"{example['sentence']}\n"
            f"Options:\n"
            f"A) {example['option1']}\n"
            f"B) {example['option2']}\n"
            "Answer:"
        )
        return prompt
    
    correct = 0
    total = 0
    model.eval()
    
    for example in tqdm(dataset, desc="Evaluating WinoGrande"):
        prompt = format_prompt(example)
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536
        )['input_ids'].to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                temperature=temp,
                top_k=50,
                max_tokens=10
            )
        
        generated_tokens = generated_ids[:, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        match = answer_pattern.search(generated_text)
        predicted_answer = match.group(1).upper() if match else None
        correct_answer = 'A' if example['answer'] == '1' else 'B'
        
        total += 1
        if predicted_answer == correct_answer:
            correct += 1
    
    accuracy = correct / total
    with open('evaluation.txt', 'a') as f:
        f.write(f"{save_name}: Wino Evaluation Results:\n")
        wino_result = accuracy
        f.write(str(wino_result) + "\n\n")

    print(f"Accuracy on WinoGrande: {accuracy * 100:.2f}%")



def eval_model(checkpoint_path, temp=0.25, save_name="model"):
    
    model = Transformer.from_pretrained(checkpoint_path)
        
    model = model.to(device)
    model.eval()
    
    arc(model, f"{save_name}", temp)
    wino(model, f"{save_name}", temp)

eval_model("./model_FFN")