from typing import Any, Dict, Tuple
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_path(profile_cfg, key):
    value = profile_cfg.get("paths", {}).get(key)
    return os.path.expandvars(value) if value else value

def load_model(cfg, profile_cfg):
    name = cfg.get("name")
    device = cfg.get("device", "cuda")

    hf_home = get_path(profile_cfg, "hf_home")
    print(f"HF_HOME for this run: {hf_home if hf_home else 'default'}")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "transformers")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")

    print(os.environ.get("HF_HOME", "HF_HOME not set"), os.environ.get("TRANSFORMERS_CACHE", "TRANSFORMERS_CACHE not set"), os.environ.get("HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_HUB_CACHE not set"))

    
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=os.environ.get("TRANSFORMERS_CACHE", None))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }

    dtype_str = cfg.get("dtype", "fp16")
    torch_dtype = dtype_map.get(dtype_str, torch.float16)

    load_in_8bit = cfg.get("load_in_8bit", False)
    load_in_4bit = cfg.get("load_in_4bit", False)
    
    device_map = cfg.get("device_map", None)

    logger_info = {
        "model" : name,
        "dtype": dtype_str,
        "device": device,
        "device_map": device_map,
        "8bit": load_in_8bit,
        "4bit": load_in_4bit,
        "hf_cache": os.environ.get("HF_HOME", "default")
    }

    print(f"Loading model with config: {logger_info}")

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        cache_dir=os.environ.get("TRANSFORMERS_CACHE", None)
    )

    if device_map is None:
        model.to(device)

    model.eval()
    return model, tokenizer, device