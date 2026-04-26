from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..finetune.load_tune_dataset import load_finetune_dataset
from ..finetune.mask_guard import MaskedFineTuneGuard
from ..pruning.mask_io import load_masks
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

@dataclass
class MaskedConfig:
    model_name: str
    mask_path: str
    output_dir: str

    max_rows = 1000
    max_length = 512
    batch_size = 1
    epochs = 1
    lr = 2e-5

    device = "cuda"
    dtype = torch.bfloat16
    seed = 42


def tokenize_batch(batch, tokenizer, max_length):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    labels = tokens["input_ids"].clone()
    # tokens["labels"] = tokens["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    tokens["labels"] = labels
    return tokens


def collate_fn(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0]
    }

def run_masked_ft(model, tokenizer, fine_tune_cfg, profile_cfg, masks, device):
    profile_cfg = profile_cfg or {}
    dataset_name  = fine_tune_cfg.get("dataset", "alpaca")
    max_rows = int(fine_tune_cfg.get("max_rows", 1000))
    epochs = int(fine_tune_cfg.get("epochs", 1))
    batch_size = int(fine_tune_cfg.get("batch_size", 1))
    lr = float(fine_tune_cfg.get("lr", 2e-5))
    max_length = int(fine_tune_cfg.get("max_length", 512))
    seed = 42
    preserved_pruned = bool(fine_tune_cfg.get("preserve_pruned", True))
    use_lora = bool(fine_tune_cfg.get("use_lora", False))

    guard = None
    if preserved_pruned and masks:
        guard = MaskedFineTuneGuard(model, masks)
        guard.apply_masks()

    if use_lora:
        print(f"Applying LoRA Adapters")
        is_kbit = bool(fine_tune_cfg.get("is_qlora", False))

        if is_kbit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r = int(fine_tune_cfg.get("lora_r", 8)),
            lora_alpha = int(fine_tune_cfg.get("lora_alpha", 16)),
            lora_dropout=float(fine_tune_cfg.get("lora_dropout", 0.05)),
            target_modules=fine_tune_cfg.get(
                "target_modules",["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            bias="none"
        )

        model = get_peft_model(model, lora_cfg)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    
    if device is None:
        device = next(model.parameters()).device

    torch.manual_seed(seed)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # masks = load_masks(cfg.mask_path, device=cfg.device)
    
    print(f"Loading alpaca subset for fine-tuning")
    dataset = load_finetune_dataset(
        dataset_name="wikitext",
        max_rows=max_rows,
        seed=seed
    )

    print(f"Tokenizing fine-tuning dataset")
    tokenized_dataset = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    model.train()
    total_steps = 0
    final_loss = None

    for epoch in range(epochs):
        running_loss = 0.0

        print(f"Starting fine-tuning epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(dataloader):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            # total_norm = torch.norm(torch.stack([
            #         torch.norm(p.grad.detach(), 2)
            #         for p in model.parameters()
            #         if p.grad is not None
            #     ]), 2)

            # print("Grad norm:", total_norm.item())

            if guard is not None and not use_lora:
                guard.zero_pruned_gradients()
            optimizer.step()

            if guard is not None and not use_lora:
                guard.apply_masks()
            optimizer.zero_grad(set_to_none=True)

            loss_value = float(loss.item())
            running_loss += loss_value
            final_loss = loss_value
            total_steps += 1

            if step % 20 == 0:
                if guard is not  None:
                    bad_count = guard.count_pruned_nonzero_weights()
                else:
                    bad_count = None
                avg_loss = running_loss / max(step + 1, 1)

                print(
                    f"epoch={epoch + 1} "
                    f"step={step} "
                    f"loss={loss.item():.4f} "
                    f"avg_loss={avg_loss:.4f} "
                    f"pruned_nonzero={bad_count}"
                )


    model.eval()
    metrics = {
        "fine_tune_enabled": True,
        "fine_tune_dataset": dataset_name,
        "fine_tune_rows": max_rows,
        "fine_tune_epochs": epochs,
        "fine_tune_batch_size": batch_size,
        "fine_tune_lr": lr,
        "fine_tune_steps": total_steps,
        "fine_tune_final_loss": final_loss,
        "fine_tune_preserve_pruned": preserved_pruned,
    }
    print(f"Saved masked fine tuned model to : {metrics}")
    return metrics