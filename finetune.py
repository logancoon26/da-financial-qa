"""
Input:   data/chunks.jsonl   (from clean.py)
Output:  models/lora-adapter/   (the LoRA weights, ~80MB)
         models/merged/          (base model + adapter merged, for deployment)
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH    = "data/chunks.jsonl"
OUTPUT_DIR   = "models/lora-adapter"
MERGED_DIR   = "models/merged"

# LoRA hyperparameters
LORA_RANK    = 16     # adapter rank
LORA_ALPHA   = 32     # scaling: effective scale = alpha/rank = 2.0
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # attention matrices

# Training hyperparameters
EPOCHS       = 1
BATCH_SIZE   = 8      # per-device batch size; keep low due to long sequences
GRAD_ACCUM   = 4      # effective batch size = BATCH_SIZE × GRAD_ACCUM
LR           = 2e-4
MAX_SEQ_LEN  = 512    # max tokens per training example

# How many chunks to use (None = all). Set a small number for quick testing.
MAX_SAMPLES  = 18750


# ── Load and prepare data ─────────────────────────────────────────────────────

def load_dataset(path: str, max_samples: int) -> Dataset:
    """
    Load chunks.jsonl and convert to HuggingFace Dataset.
    We use the "messages" field (instruction format) for SFT.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append({"messages": rec["messages"]})
            if max_samples and len(records) >= max_samples:
                break

    print(f"Loaded {len(records)} training examples")
    return Dataset.from_list(records)


def format_prompt(example: dict, tokenizer) -> dict:
    """
    Convert the messages list to a single string using the model's chat template.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ── Model setup ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    """
    Load the base model in 4-bit NF4 quantization (QLoRA setup).

    BitsAndBytesConfig tells the loader how to quantize
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map={"":0},          # automatically distribute across available GPUs
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token   # Mistral has no pad token by default
    tokenizer.padding_side = "right"            # pad on the right for causal LM

    return model, tokenizer


def attach_lora(model):
    """
    Attach LoRA adapters to the specified attention weight matrices.

    """
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",             
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Print a summary of how many parameters are actually being trained
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset: Dataset):
    """
    Run supervised fine-tuning with SFTTrainer.

    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",      # warm up then cosine decay
        warmup_ratio=0.05,               # 5% of steps for warmup
        fp16=False,                      # use bf16 instead on A100
        bf16=True,
        logging_steps=20,
        save_steps=100,
        save_total_limit=3,              # keep only the 2 most recent checkpoints
        report_to="none",                # set to "wandb" if you want W&B logging
        optim="paged_adamw_8bit",        # 8-bit AdamW: saves GPU memory for optimizer states
    )

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    dataset = dataset.shuffle(seed=42)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer
)

    print(f"\nStarting training — {EPOCHS} epochs, effective batch size {BATCH_SIZE * GRAD_ACCUM}")
    trainer.train()

    # Save only the LoRA adapter weights ( not the full model)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✓ LoRA adapter saved → {OUTPUT_DIR}")


def merge_and_save(model, tokenizer):
    """
    Merge the LoRA adapter into the base model weights and save the full model.

    """
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"✓ Merged model saved → {MERGED_DIR}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import torch, sys
    print(sys.executable)
    print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Fine-tuning on CPU is not practical.")
        print("Run this on Google Colab (A100 runtime) or Lambda Labs.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # Load data
    raw_dataset = load_dataset(DATA_PATH, max_samples=MAX_SAMPLES)

    # Load model and tokenizer
    print(f"Loading {BASE_MODEL} in 4-bit…")
    model, tokenizer = load_model_and_tokenizer()

    # Format prompts using the model's chat template
    dataset = raw_dataset.map(lambda ex: format_prompt(ex, tokenizer))

    # Attach LoRA adapters
    model = attach_lora(model)

    # Train
    train(model, tokenizer, dataset)

    # Merge adapters into base model
    print("\nMerging adapter into base model…")
    merge_and_save(model, tokenizer)


if __name__ == "__main__":
    main()
