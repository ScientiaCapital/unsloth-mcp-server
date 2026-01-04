# Coperniq Sales SFT Training - Colab Notebook Cells

**Notebook Name:** `coperniq_sales_sft_training.ipynb`

---

## CELL 1: Install Unsloth (Run first, then restart runtime)

```python
# Cell 1: Install Unsloth
# After running, you'll need to restart the runtime (Runtime > Restart session)

!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

print("✅ Installation complete! Now restart the runtime:")
print("   Runtime > Restart session")
```

---

## CELL 2: Check GPU

```python
# Cell 2: Verify GPU is available
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_mem:.1f} GB")
else:
    print("❌ No GPU detected! Go to Runtime > Change runtime type > T4 GPU")
```

---

## CELL 3: Upload Training Data

```python
# Cell 3: Upload your training data file
# This will show a file picker - select sft_training_data.jsonl

from google.colab import files
uploaded = files.upload()

# Show what was uploaded
for filename in uploaded.keys():
    print(f"✅ Uploaded: {filename} ({len(uploaded[filename])} bytes)")
```

---

## CELL 4: Load Model

```python
# Cell 4: Load Qwen2.5-0.5B with Unsloth optimizations

from unsloth import FastLanguageModel

# Configuration - same as your working notebook!
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 2048

print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # Save memory
)

print(f"✅ Model loaded!")
print(f"   Parameters: {model.num_parameters():,}")
```

---

## CELL 5: Add LoRA Adapters

```python
# Cell 5: Configure LoRA (Low-Rank Adaptation)
# This makes training fast and memory-efficient

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=42,
)

# Count trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA configured!")
print(f"   Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
```

---

## CELL 6: Load Training Data

```python
# Cell 6: Load and format the ChatML training data
import json
from datasets import Dataset

# Load the JSONL file you uploaded
DATA_FILE = "sft_training_data.jsonl"

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

train_data = load_jsonl(DATA_FILE)
print(f"✅ Loaded {len(train_data)} training examples")

# Format for training
def format_example(example):
    messages = example.get("messages", [])
    if not messages:
        return {"text": ""}
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

train_dataset = Dataset.from_list(train_data).map(format_example)
train_dataset = train_dataset.filter(lambda x: len(x['text']) > 0)

print(f"✅ Formatted {len(train_dataset)} examples")

# Show a sample
print("\n--- Sample ---")
print(train_dataset[0]['text'][:500] + "...")
```

---

## CELL 7: Configure Trainer

```python
# Cell 7: Set up the SFT Trainer

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

OUTPUT_DIR = "./coperniq-sales-sft"

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,  # Efficient packing
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # ~3 epochs for 83 examples
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        save_strategy="steps",
        save_steps=30,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    ),
)

print("✅ Trainer configured!")
print(f"   Max steps: 60")
print(f"   Batch size: 2 x 4 = 8 effective")
```

---

## CELL 8: Train! 🚀

```python
# Cell 8: Start training
# This takes ~5-10 minutes on T4 GPU

from datetime import datetime

print("="*50)
print("STARTING TRAINING")
print("="*50)

start_time = datetime.now()
trainer_stats = trainer.train()
duration = (datetime.now() - start_time).total_seconds()

print(f"\n✅ Training complete!")
print(f"   Duration: {duration/60:.1f} minutes")
print(f"   Final loss: {trainer_stats.training_loss:.4f}")
```

---

## CELL 9: Test the Model

```python
# Cell 9: Test your trained model!

FastLanguageModel.for_inference(model)

# Test prompts
test_prompts = [
    "A prospect says 'Your software looks expensive.' How do I respond?",
    "How do I position Coperniq against ServiceTitan?",
    "What are the key pain points MEP contractors face?",
]

for prompt in test_prompts:
    print(f"\n📝 PROMPT: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Get just the assistant's response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()

    print(f"🤖 RESPONSE: {response[:300]}...")
```

---

## CELL 10: Save LoRA Adapter

```python
# Cell 10: Save the trained LoRA adapter

print("Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Saved to {OUTPUT_DIR}/")

# List saved files
!ls -la {OUTPUT_DIR}/
```

---

## CELL 11: Save Merged Model (Full Model)

```python
# Cell 11: Merge LoRA into base model and save
# This creates a standalone model your team can use

print("Merging LoRA into base model...")
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/merged_16bit",
    tokenizer,
    save_method="merged_16bit",
)
print(f"✅ Merged model saved to {OUTPUT_DIR}/merged_16bit/")

!ls -la {OUTPUT_DIR}/merged_16bit/
```

---

## CELL 12: Export to GGUF (for Ollama)

```python
# Cell 12: Export to GGUF format for Ollama

print("Exporting to GGUF (q8_0 quantization)...")
try:
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}/gguf_q8_0",
        tokenizer,
        quantization_method="q8_0",
    )
    print(f"✅ GGUF saved to {OUTPUT_DIR}/gguf_q8_0/")
    !ls -la {OUTPUT_DIR}/gguf_q8_0/
except Exception as e:
    print(f"⚠️ GGUF export failed: {e}")
    print("   (You can export later with llama.cpp)")
```

---

## CELL 13: Download Your Model

```python
# Cell 13: Download the trained model to your computer

from google.colab import files
import shutil

# Zip the model folder
print("Zipping model files...")
shutil.make_archive("coperniq-sales-sft", 'zip', OUTPUT_DIR)
print("✅ Created coperniq-sales-sft.zip")

# Download
files.download("coperniq-sales-sft.zip")
print("\n📥 Downloading... Check your Downloads folder!")
```

---

## CELL 14 (Optional): Push to HuggingFace

```python
# Cell 14: Push to HuggingFace Hub (optional)
# Set your HF token first!

HF_TOKEN = ""  # Set your HuggingFace token here
HF_REPO = "ScientiaCapital/coperniq-sales-sft"

if HF_TOKEN:
    print(f"Pushing to {HF_REPO}...")
    model.push_to_hub(HF_REPO, token=HF_TOKEN, private=True)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN, private=True)
    print(f"✅ Pushed to https://huggingface.co/{HF_REPO}")
else:
    print("Set HF_TOKEN to push to HuggingFace")
```

---

## Summary

| Cell | Purpose | Time |
|------|---------|------|
| 1 | Install Unsloth | 2 min |
| 2 | Check GPU | instant |
| 3 | Upload data | instant |
| 4 | Load model | 30 sec |
| 5 | Add LoRA | instant |
| 6 | Load data | instant |
| 7 | Configure trainer | instant |
| 8 | **TRAIN** | 5-10 min |
| 9 | Test model | 30 sec |
| 10 | Save adapter | 10 sec |
| 11 | Save merged | 30 sec |
| 12 | Export GGUF | 1 min |
| 13 | Download | depends on size |
| 14 | Push to HF | 1 min |

**Total: ~15 minutes**
