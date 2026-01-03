#!/usr/bin/env python3
"""
07-save-model.py - Save your trained model

WHAT THIS DOES:
- Saves the LoRA adapters (the trained parts)
- Creates a zip file for easy download

OUTPUT:
- my_first_finetune/: Directory with adapter files
- my_model.zip: Packaged for download (~50MB)

WHAT'S SAVED:
- adapter_model.safetensors: The trained LoRA weights
- adapter_config.json: LoRA configuration
- tokenizer files: For inference
- NOT the base model (that's downloaded separately)
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import shutil
import os

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "my_first_finetune"


def train_and_save():
    """Train and save the model."""
    print("=" * 50)
    print("Training and saving model")
    print("=" * 50)

    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        load_in_4bit=True,
    )

    # Add LoRA
    print("\n[2/4] Adding LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Create dataset and train
    print("\n[3/4] Training (60 steps)...")
    examples = [
        ("What is 2+2?", "4"), ("What is 3+3?", "6"),
        ("What is 5+5?", "10"), ("What is 10+10?", "20"),
        ("What is 7+3?", "10"), ("What is 8+2?", "10"),
        ("What is 1+1?", "2"), ("What is 4+4?", "8"),
        ("What is 6+6?", "12"), ("What is 9+1?", "10"),
        ("What is 15+15?", "30"), ("What is 20+20?", "40"),
        ("What is 11+11?", "22"), ("What is 12+8?", "20"),
        ("What is 25+25?", "50"), ("What is 30+30?", "60"),
        ("What is 100+100?", "200"), ("What is 50+50?", "100"),
        ("What is 13+7?", "20"), ("What is 16+4?", "20"),
    ]
    dataset = Dataset.from_list([
        {"text": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"}
        for q, a in examples
    ])

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=512,
        args=TrainingArguments(
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            warmup_steps=5, max_steps=60, learning_rate=2e-4, fp16=True,
            logging_steps=10, output_dir="outputs", seed=42,
        ),
    )
    trainer.train()

    # Save
    print("\n[4/4] Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Create zip
    zip_name = "my_model"
    shutil.make_archive(zip_name, 'zip', OUTPUT_DIR)
    zip_size = os.path.getsize(f"{zip_name}.zip") / 1024 / 1024

    print("\n" + "=" * 50)
    print("COMPLETE - Model saved!")
    print(f"  Directory: {OUTPUT_DIR}/")
    print(f"  Zip file: {zip_name}.zip ({zip_size:.1f} MB)")
    print("=" * 50)
    print("\n" + "Files in model directory:")
    for f in os.listdir(OUTPUT_DIR):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  {f}: {size:.1f} KB")
    print("\nNext step: Run 08-inference.py")


if __name__ == "__main__":
    train_and_save()
