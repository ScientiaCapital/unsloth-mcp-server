#!/usr/bin/env python3
"""
Coperniq Sales Agent - SFT Fine-Tuning Script
==============================================
Train on sales objection handling, competitor positioning, and MEP-specific scenarios.

REQUIREMENTS:
- Google Colab T4 GPU (free tier works!)
- Or RunPod with any NVIDIA GPU (RTX 3090, A100, etc.)

USAGE:
1. Upload this script and sft_training_data.jsonl to Colab/RunPod
2. Run: python train_coperniq_sft.py

OUTPUT:
- LoRA adapter: ./coperniq-sales-sft/
- Merged model: ./coperniq-sales-sft/merged_16bit/
- GGUF for Ollama: ./coperniq-sales-sft/gguf_q8_0/

Based on Tim's successful finetuning1.ipynb notebook (works!)
"""

import os
import json
import subprocess
import sys
from datetime import datetime

# ============================================
# CONFIGURATION - Same as your working notebook!
# ============================================

# Model: Same tiny model from your Colab notebook
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"  # 0.5B params, fits in 4GB VRAM
MAX_SEQ_LENGTH = 2048  # Plenty for sales conversations
LOAD_IN_4BIT = True  # 4-bit quantization for memory efficiency

# LoRA Config - Same as your notebook (r=16, alpha=16)
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Config - Optimized for ~100 examples
EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4  # Effective batch size = 2 * 4 = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
MAX_STEPS = 60  # Same as your notebook

# Paths
DATA_FILE = "./sft_training_data.jsonl"
OUTPUT_DIR = "./coperniq-sales-sft"

# HuggingFace (optional - set your token)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_REPO = "ScientiaCapital/coperniq-sales-sft"  # Change to your repo
PUSH_TO_HUB = HF_TOKEN is not None

print("="*70)
print("COPERNIQ SALES AGENT - SFT FINE-TUNING")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
print(f"Training: {EPOCHS} epochs, {MAX_STEPS} max steps")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print()

# ============================================
# INSTALL DEPENDENCIES
# ============================================

def install_deps():
    """Install Unsloth and dependencies."""
    print("Installing dependencies...")
    # For Colab
    subprocess.run([sys.executable, "-m", "pip", "install", "unsloth", "-q"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "xformers", "-q"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "trl", "datasets", "-q"], check=False)

# Uncomment if needed (usually pre-installed in Colab)
# install_deps()

# ============================================
# IMPORTS
# ============================================

from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import torch

# ============================================
# CHECK GPU
# ============================================

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_mem:.1f} GB")
else:
    print("WARNING: No GPU detected! Training will be very slow.")
print()

# ============================================
# LOAD MODEL
# ============================================

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect (float16 or bfloat16)
    load_in_4bit=LOAD_IN_4BIT,
)

print("Configuring LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=42,
    use_rslora=False,  # Keep it simple for first training
)

# Count trainable params
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# ============================================
# LOAD DATA
# ============================================

print("\nLoading training data...")

def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

train_data = load_jsonl(DATA_FILE)
print(f"Loaded {len(train_data)} training examples")

# Format for training - apply chat template
def format_example(example):
    """Format ChatML to model's expected format."""
    messages = example.get("messages", [])
    if not messages:
        return {"text": ""}

    # Apply the model's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# Create dataset
train_dataset = Dataset.from_list(train_data).map(format_example)
train_dataset = train_dataset.filter(lambda x: len(x['text']) > 0)

print(f"After formatting: {len(train_dataset)} examples")

# Show sample
print("\n--- Sample Training Example ---")
sample = train_dataset[0]['text'][:500]
print(sample + "..." if len(train_dataset[0]['text']) > 500 else sample)
print("--- End Sample ---\n")

# ============================================
# TRAINING
# ============================================

print("="*70)
print("STARTING TRAINING")
print("="*70)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,  # Pack short sequences for efficiency
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=WARMUP_RATIO,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        save_strategy="steps",
        save_steps=30,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
    ),
)

# Start training
start_time = datetime.now()
print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

trainer_stats = trainer.train()

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()
print(f"\nCompleted in {duration:.1f} seconds ({duration/60:.1f} minutes)")

# ============================================
# TEST THE MODEL
# ============================================

print("\n" + "="*70)
print("TESTING THE MODEL")
print("="*70)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Test prompts
test_prompts = [
    "A prospect says 'Your software looks great but we've tried other systems before and they never stick with our field crews. What makes Coperniq different?'",
    "How do I position Coperniq against ServiceTitan?",
    "The prospect wants to think about it. What's my response?",
]

for prompt in test_prompts:
    print(f"\n📝 PROMPT: {prompt[:80]}...")

    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        use_cache=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()

    print(f"🤖 RESPONSE: {response[:400]}...")

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

# Save LoRA adapter
print("1. Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   ✅ Saved to {OUTPUT_DIR}/")

# Save merged model (for deployment)
print("2. Saving merged 16-bit model...")
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/merged_16bit",
    tokenizer,
    save_method="merged_16bit",
)
print(f"   ✅ Saved to {OUTPUT_DIR}/merged_16bit/")

# Save GGUF for Ollama
print("3. Saving GGUF (q8_0) for Ollama...")
try:
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}/gguf_q8_0",
        tokenizer,
        quantization_method="q8_0",  # High quality quantization
    )
    print(f"   ✅ Saved to {OUTPUT_DIR}/gguf_q8_0/")
except Exception as e:
    print(f"   ⚠️ GGUF export failed: {e}")
    print("   (You can export later with llama.cpp)")

# Optional: Push to HuggingFace Hub
if PUSH_TO_HUB:
    print(f"4. Pushing to HuggingFace Hub ({HF_REPO})...")
    try:
        model.push_to_hub(
            HF_REPO,
            token=HF_TOKEN,
            private=True,  # Team access only
        )
        tokenizer.push_to_hub(
            HF_REPO,
            token=HF_TOKEN,
            private=True,
        )
        print(f"   ✅ Pushed to https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"   ⚠️ Push failed: {e}")
else:
    print("4. Skipping HuggingFace push (HF_TOKEN not set)")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

print(f"""
📊 Training Stats:
   - Examples: {len(train_dataset)}
   - Duration: {duration/60:.1f} minutes
   - Final Loss: {trainer_stats.training_loss:.4f}

📁 Output Files:
   - LoRA adapter: {OUTPUT_DIR}/
   - Merged model: {OUTPUT_DIR}/merged_16bit/
   - GGUF (Ollama): {OUTPUT_DIR}/gguf_q8_0/

🚀 Next Steps:

1. TEST LOCALLY (Ollama):
   cd {OUTPUT_DIR}/gguf_q8_0/
   ollama create coperniq-sales -f Modelfile
   ollama run coperniq-sales "How do I handle price objections?"

2. PUSH TO HUGGINGFACE (share with team):
   export HF_TOKEN=your_token
   python push_to_hub.py --model {OUTPUT_DIR} --repo ScientiaCapital/coperniq-sales-sft

3. CONTINUE WITH GRPO (preference optimization):
   python train_grpo.py --sft-adapter {OUTPUT_DIR}
""")

# Save training report
report = {
    "model": MODEL_NAME,
    "training_examples": len(train_dataset),
    "epochs": EPOCHS,
    "max_steps": MAX_STEPS,
    "duration_seconds": duration,
    "final_loss": trainer_stats.training_loss,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "completed": datetime.now().isoformat(),
}

with open(f"{OUTPUT_DIR}/training_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Training report saved!")
