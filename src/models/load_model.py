from typing import Any, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(cfg):
    name = cfg.get("name")
    device = "mps"

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(name)
    model.to(device)
    model.eval()
    return model, tokenizer