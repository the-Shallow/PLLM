import torch
from src.models.load_model import load_model
from src.pruning.registry import get_pruner

def generate(model, tokenizer, prompt, do_sample=True, temperature=0.8, repetition_penalty=1.15, max_length=100):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_length, do_sample=do_sample, temperature=temperature, repetition_penalty=repetition_penalty)
    return tokenizer.decode(out[0], skip_special_tokens=True)