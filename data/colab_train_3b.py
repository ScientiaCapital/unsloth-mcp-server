#!/usr/bin/env python3
"""
Colab/RunPod Training Script for Qwen2.5-3B
============================================
Trains Coperniq sales model on 679 examples from Gong + battle cards.

Copy these cells to Colab with T4/A100 GPU (Free tier T4 works with 4-bit)

Cell 1: Install Unsloth
------------------------
%%capture
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
"""

# Cell 2: Verify GPU
# ------------------
"""
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""

# Cell 3: Load Model (Qwen2.5-3B-Instruct)
# ----------------------------------------
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",  # 3B params - larger model
    max_seq_length=2048,  # Longer context for sales conversations
    load_in_4bit=True,    # 4-bit quantization for T4 GPU
    dtype=None,           # Auto-detect (bf16 on A100, fp16 on T4)
)

print(f"✅ Loaded {model.config.num_hidden_layers} layer model")

# Cell 4: Add LoRA Adapters
# -------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                  # LoRA rank
    lora_alpha=16,         # LoRA alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Memory efficient
    random_state=42,
)

model.print_trainable_parameters()

# Cell 5: Load Training Data (Upload all_training.jsonl first!)
# ------------------------------------------------------------------
import json

# Upload your all_training.jsonl file first!
# Contains: 8,558 examples (104 synthetic + 575 Gong + 7,879 Close emails)
with open("all_training.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

print(f"📊 Loaded {len(train_data)} training examples")

# Show category/source breakdown
sources = {}
for ex in train_data:
    source = ex.get("_metadata", {}).get("source", "unknown")
    sources[source] = sources.get(source, 0) + 1

print("\nSource breakdown:")
for src, count in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {src}: {count}")

# Cell 6: Format Dataset for Training
# ------------------------------------
from datasets import Dataset

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# Remove metadata column before training
dataset = Dataset.from_list([
    {"messages": ex["messages"]} for ex in train_data
]).map(format_chat, remove_columns=["messages"])

print(f"✅ Dataset formatted: {len(dataset)} examples")

# Cell 7: Configure Training
# --------------------------
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="coperniq_sales_3b_sft",
    max_seq_length=2048,              # Longer context
    per_device_train_batch_size=2,     # 2 fits on T4 with 4-bit
    gradient_accumulation_steps=4,     # Effective batch = 8
    max_steps=2000,                    # ~2 epochs on 8,558 examples
    learning_rate=2e-4,
    logging_steps=50,
    warmup_steps=100,
    save_steps=500,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    seed=42,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Cell 8: Train!
# --------------
print("🚀 Starting training on 8,558 examples (2,000 steps)...")
print("   Sources: 7,879 Close emails + 575 Gong calls + 104 synthetic")
trainer.train()
print("✅ Training complete!")

# Cell 9: Test the Model
# ----------------------
FastLanguageModel.for_inference(model)

SYSTEM = """You are a top-performing SDR for Coperniq, construction software for MEP contractors.
Key differentiators: Asset lifecycle tracking, real-time monitoring, AI features, 30-50% lower cost."""

# Test prompts based on real training data scenarios
test_prompts = [
    "We're looking at ServiceTitan but it seems expensive. What makes Coperniq different?",
    "Write a follow-up email for a solar contractor who went silent after the demo.",
    "How does Coperniq handle project migration from other CRMs like Ascent?"
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt[:60]}...")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).cuda()
    outputs = model.generate(inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Print just the assistant response
    print(response.split("assistant")[-1][:500] if "assistant" in response else response[-500:])

# Cell 10: Save Model
# -------------------
model.save_pretrained("coperniq_sales_3b_sft")
tokenizer.save_pretrained("coperniq_sales_3b_sft")
print("💾 Saved to coperniq_sales_3b_sft/")

# Cell 11: Zip for Download
# -------------------------
import shutil
shutil.make_archive("coperniq_sales_3b_sft", "zip", "coperniq_sales_3b_sft")
print("📦 Created coperniq_sales_3b_sft.zip - download this!")

# Cell 12: Push to HuggingFace (Optional)
# ---------------------------------------
"""
from huggingface_hub import login
login(token="hf_YOUR_TOKEN")  # Or use the token from .env

model.push_to_hub("tmk-ai/coperniq-sales-3b-sft")
tokenizer.push_to_hub("tmk-ai/coperniq-sales-3b-sft")
print("🚀 Pushed to HuggingFace!")
"""
