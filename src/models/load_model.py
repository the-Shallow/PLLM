from typing import Any, Dict, Tuple
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import BitsAndBytesConfig

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

def get_path(profile_cfg, key):
    value = profile_cfg.get("paths", {}).get(key)
    return os.path.expandvars(value) if value else value


def resolve_local_model_path(model_name: str) -> str:
    candidate_roots = [
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("TRANSFORMERS_CACHE"),
    ]

    repo_name = f"models--{model_name.replace('/', '--')}"

    for root in candidate_roots:
        if not root:
            continue

        repo_dir = os.path.join(root, repo_name)
        snapshots_dir = os.path.join(repo_dir, "snapshots")

        if os.path.isdir(snapshots_dir):
            snapshots = sorted(os.listdir(snapshots_dir))
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[-1])

    raise FileNotFoundError(
        f"No cached snapshots found for model '{model_name}' in: {candidate_roots}"
    )

def load_model(cfg, profile_cfg):
    name = cfg.get("name")
    device = cfg.get("device", "cuda")
    runtime = profile_cfg.get("runtime").get("offline")
    dtype=cfg.get("dtype", "float16")
    torch_dtype = DTYPE_MAP.get(dtype, torch.float16)
    
    hf_home = get_path(profile_cfg, "hf_home")
    print(f"HF_HOME for this run: {hf_home if hf_home else 'default'}")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "transformers")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    print(os.environ.get("HF_HOME", "HF_HOME not set"), os.environ.get("TRANSFORMERS_CACHE", "TRANSFORMERS_CACHE not set"), os.environ.get("HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_HUB_CACHE not set"))
    local_files_only = False
    if runtime:
        print("Running in offline mode. Will attempt to load model from local cache.")
        local_model_path = resolve_local_model_path(name)
        print(f"Using local model path: {local_model_path}")
        local_files_only = True
    else:
        local_model_path = name
        print(f"Using model name for online loading: {local_model_path}")
        
    # tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=os.environ.get("TRANSFORMERS_CACHE", None))
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }

    dtype_str = cfg.get("dtype", "fp16")
    torch_dtype = dtype_map.get(dtype_str, torch.float16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

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

    model_kwargs = {
    "torch_dtype": torch_dtype,
    "local_files_only": local_files_only,
}

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    # if load_in_8bit:
        # model_kwargs["load_in_8bit"] = True

    # if load_in_4bit:
    #     model_kwargs["load_in_4bit"] = True

    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["device_map"] = device_map or "auto"


    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = device_map or "auto"

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        **model_kwargs
    )

    model.config.use_cache = False
    # if device_map is None:
    #     model.to(device)

    if device_map is None and not load_in_4bit and not load_in_8bit:
        model.to(device)

    model.eval()
    return model, tokenizer, device