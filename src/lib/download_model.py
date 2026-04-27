#!/usr/bin/env python3
"""
HuggingFace Model Downloader for HPC Clusters
----------------------------------------------
Downloads a model into the same cache structure expected by load_model.py.

Cache layout written:
    <hf_home>/
        hub/                        <- HUGGINGFACE_HUB_CACHE
            models--<org>--<n>/
                snapshots/<hash>/   <- what resolve_local_model_path() picks up
        transformers/               <- TRANSFORMERS_CACHE

Usage:
    python download_model.py --model meta-llama/Llama-3.2-3B --hf-home /scratch/$USER/hf_cache
    python download_model.py --model pankajmathur/orca_mini_3b --hf-home /scratch/$USER/hf_cache
    python download_model.py --model meta-llama/Llama-3.2-3B --hf-home /scratch/$USER/hf_cache --token hf_...
    python download_model.py --model meta-llama/Llama-3.2-3B --hf-home /scratch/$USER/hf_cache --verify
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download a HuggingFace model to HPC cache")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.2-3B)"
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help=(
            "Root cache directory — mirrors the hf_home key in your profile YAML. "
            "Hub weights go to <hf_home>/hub, tokenizer files to <hf_home>/transformers. "
            "Falls back to HF_HOME env var, then ~/.cache/huggingface."
        )
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision/branch to download (default: main)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token for gated models (e.g. Llama). "
             "Can also be set via HF_TOKEN env var."
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Download tokenizer only (skip model weights)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "After download, verify that resolve_local_model_path() can find the snapshot "
            "and that the model loads — mimics what load_model.py does in offline mode."
        )
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cache path helpers — mirrors load_model.py's env var conventions exactly
# ---------------------------------------------------------------------------

def resolve_hf_home(hf_home_arg: "str | None") -> Path:
    """
    Resolve the root HF_HOME directory.
    Priority: --hf-home arg > HF_HOME env var > default.
    """
    if hf_home_arg:
        # Expand shell vars and ~, then resolve to absolute path
        expanded = os.path.expandvars(os.path.expanduser(hf_home_arg))
        hf_home  = Path(expanded).resolve()
    elif "HF_HOME" in os.environ:
        hf_home = Path(os.environ["HF_HOME"])
        log.info(f"Using HF_HOME from environment: {hf_home}")
    else:
        hf_home = Path.home() / ".cache" / "huggingface"
        log.warning(f"No --hf-home provided — defaulting to {hf_home}")
        log.warning("On HPC this may fill your home quota. Use --hf-home /scratch/$USER/hf_cache")

    return hf_home


def set_env_vars(hf_home: Path):
    """
    Set the same env vars that load_model.py sets at runtime so that
    huggingface_hub and transformers write to the correct subdirectories.

    load_model.py uses:
        HF_HOME               = hf_home
        HUGGINGFACE_HUB_CACHE = hf_home/hub       <- resolve_local_model_path reads this
        TRANSFORMERS_CACHE    = hf_home/transformers
    """
    hub_cache          = hf_home / "hub"
    transformers_cache = hf_home / "transformers"

    os.environ["HF_HOME"]               = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"]    = str(transformers_cache)

    # Do NOT set offline flags here — we need network access to download
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(key, None)

    hub_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)

    log.info(f"HF_HOME               = {hf_home}")
    log.info(f"HUGGINGFACE_HUB_CACHE = {hub_cache}")
    log.info(f"TRANSFORMERS_CACHE    = {transformers_cache}")

    return hub_cache, transformers_cache


def resolve_local_model_path(model_name: str) -> str:
    """
    Exact copy of the function in load_model.py — used by --verify to confirm
    the downloaded snapshot is discoverable before submitting a GPU job.
    """
    candidate_roots = [
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("TRANSFORMERS_CACHE"),
    ]

    repo_name = f"models--{model_name.replace('/', '--')}"

    for root in candidate_roots:
        if not root:
            continue
        repo_dir      = os.path.join(root, repo_name)
        snapshots_dir = os.path.join(repo_dir, "snapshots")

        if os.path.isdir(snapshots_dir):
            snapshots = sorted(os.listdir(snapshots_dir))
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[-1])

    raise FileNotFoundError(
        f"No cached snapshots found for model '{model_name}' in: {candidate_roots}"
    )


# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------

def check_disk_space(path: Path, min_gb: float = 15.0):
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    log.info(f"Available disk space at {path}: {free_gb:.1f} GB")
    if free_gb < min_gb:
        log.warning(
            f"Only {free_gb:.1f} GB free — large models (7B+) need 15-30 GB. "
            "Download may fail mid-way."
        )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(
    model_id: str,
    hub_cache: Path,
    revision: str,
    token: "str | None",
    skip_model: bool,
):
    try:
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        log.error("Run: pip install transformers huggingface_hub accelerate")
        sys.exit(1)

    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token is None:
        log.warning(
            "No HF token provided. Gated models (e.g. Llama) require one. "
            "Pass --token hf_... or export HF_TOKEN=hf_..."
        )

    # -- Tokenizer (auth check before pulling weights) -----------------------
    log.info("Fetching tokenizer (auth/name check)...")
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=hub_cache,   # writes into hub/ matching HUGGINGFACE_HUB_CACHE
            revision=revision,
            token=hf_token,
        )
        log.info(f"Tokenizer OK — vocab size: {tok.vocab_size}")
    except Exception as e:
        log.error(f"Tokenizer fetch failed: {e}")
        if "401" in str(e) or "gated" in str(e).lower() or "403" in str(e):
            log.error(
                "Access denied. Request model access at huggingface.co "
                "and supply --token hf_..."
            )
        sys.exit(1)

    if skip_model:
        log.info("Skipping model weights (--no-model). Tokenizer is cached.")
        return

    # -- Model weights via snapshot_download ---------------------------------
    # snapshot_download writes to:
    #   hub_cache/models--<org>--<name>/snapshots/<hash>/
    # which is exactly what resolve_local_model_path() in load_model.py expects.
    log.info("Downloading model weights via snapshot_download...")
    log.info("(This may take several minutes for large models)")
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=hub_cache,
            revision=revision,
            token=hf_token,
            ignore_patterns=[       # skip non-PyTorch weight formats to save space
                "*.msgpack",
                "*.h5",
                "flax_model*",
                "tf_model*",
                "rust_model*",
            ],
        )
        log.info(f"Snapshot saved to: {local_path}")
    except Exception as e:
        log.error(f"snapshot_download failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Verify — mimics load_model.py offline boot path
# ---------------------------------------------------------------------------

def verify(model_id: str, token: "str | None"):
    log.info("Verifying snapshot is discoverable by resolve_local_model_path()...")
    try:
        local_path = resolve_local_model_path(model_id)
        log.info(f"resolve_local_model_path() -> {local_path}")
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    log.info("Verifying model loads from local path (offline, mimicking load_model.py)...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Device: {device}")

        # Mirror load_model.py exactly
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=True,
            device_map=device,
        )
        model.eval()
        log.info(f"Model loaded — {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

        inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        log.info(f"Forward pass OK — logits shape: {out.logits.shape}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        log.error(f"Verification failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Print config snippet for profile YAML
# ---------------------------------------------------------------------------

def print_yaml_snippet(hf_home: Path, model_id: str):
    print("\n" + "=" * 65)
    print("Add to your profile YAML (mirrors what load_model.py reads):")
    print("=" * 65)
    print(f"""
paths:
  hf_home: {hf_home}

runtime:
  offline: true   # set false to allow online pulls

model:
  name: {model_id}
  device: cuda
  dtype: fp16
  device_map: null
  load_in_8bit: false
  load_in_4bit: false
""")
    print("=" * 65)
    print("\nOr export manually in your bsub job script:")
    print(f"  export HF_HOME={hf_home}")
    print(f"  export HUGGINGFACE_HUB_CACHE={hf_home}/hub")
    print(f"  export TRANSFORMERS_CACHE={hf_home}/transformers")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    hf_home = resolve_hf_home(args.hf_home)
    hub_cache, _ = set_env_vars(hf_home)

    check_disk_space(hf_home)

    download(
        model_id=args.model,
        hub_cache=hub_cache,
        revision=args.revision,
        token=args.token,
        skip_model=args.no_model,
    )

    if args.verify:
        verify(args.model, args.token)

    print_yaml_snippet(hf_home, args.model)
    log.info("Done.")


if __name__ == "__main__":
    main()