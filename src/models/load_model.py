from typing import Any, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int8":    torch.int8,
    "int16":   torch.int16,
    "int32":   torch.int32,
    "int64":   torch.int64,
    "uint8":   torch.uint8,
    "bool":    torch.bool,
}

def load_model(cfg):
    name = cfg.get("name")
    device = cfg.get("device")
    load_in_8bit = cfg.get("load_in_8bit")
    dtype=cfg.get("dtype", "float16")
    torch_dtype = DTYPE_MAP.get(dtype, torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if(load_in_8bit):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quantization_config, device_map="cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype, device_map="cuda:0")
    
    model.to(device)
    model.eval()
    return model, tokenizer, device