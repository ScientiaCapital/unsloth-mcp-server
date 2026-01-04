#!/usr/bin/env python3
"""
Local SFT Training Script for Mac M2
=====================================
Trains Coperniq sales model on Apple Silicon.

Requirements:
    pip install torch transformers peft trl datasets accelerate

Usage:
    python train_local_mac.py
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Check device
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Apple Metal (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("✅ Using CUDA GPU")
else:
    device = "cpu"
    print("⚠️ Using CPU (slower)")

print(f"PyTorch version: {torch.__version__}")

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "sft_training_data.jsonl"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "adapters" / "coperniq_sales_sft_v2"

# Load training data
print(f"\n📂 Loading data from {DATA_FILE}")
with open(DATA_FILE, "r") as f:
    data = [json.loads(line) for line in f]
print(f"   Loaded {len(data)} training examples")

# Model config - using smaller model for Mac
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"\n🤖 Loading model: {MODEL_NAME}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model (no quantization on Mac - MPS doesn't support bitsandbytes well)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    device_map={"": device} if device == "mps" else "auto",
    trust_remote_code=True,
)

print(f"   Model loaded on {device}")
print(f"   Parameters: {model.num_parameters():,}")

# Add LoRA adapters
print("\n🔧 Adding LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Format dataset
def format_chat(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}

print("\n📊 Formatting dataset...")
dataset = Dataset.from_list(data).map(format_chat, remove_columns=["messages", "_metadata"])
print(f"   Dataset ready: {len(dataset)} examples")

# Training config - optimized for Mac M2 16GB
print("\n🚀 Starting training...")
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    max_seq_length=1024,  # Reduced for memory
    per_device_train_batch_size=1,  # Small batch for Mac
    gradient_accumulation_steps=8,  # Accumulate to simulate larger batch
    max_steps=200,
    learning_rate=2e-4,
    logging_steps=10,
    warmup_steps=10,
    save_steps=50,
    fp16=False,  # MPS works better with fp32
    bf16=False,
    optim="adamw_torch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Train!
trainer.train()

# Save model
print(f"\n💾 Saving model to {OUTPUT_DIR}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training complete!")
print(f"   Model saved to: {OUTPUT_DIR}")

# Quick test
print("\n🧪 Testing model...")
model.eval()

SYSTEM = """You are a top-performing SDR for Coperniq, construction software for MEP contractors.
Key differentiators: Asset lifecycle tracking, real-time monitoring, AI features, 30-50% lower cost."""

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "A prospect says they're too small for Coperniq. How do I respond?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=200, temperature=0.7, do_sample=True)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*60)
print("TEST OUTPUT:")
print("="*60)
print(response)
