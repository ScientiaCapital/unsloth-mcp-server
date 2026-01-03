#!/usr/bin/env python3
"""
train_objection_handler.py - Train a sales objection handling model

PURPOSE:
    Fine-tune DeepSeek-R1 on your best 100 objection handling examples.
    Optimized for 16GB memory (works on M2 Mac or T4 GPU).

PREREQUISITE:
    Run prep_sales_data.py first to create data/objection_training_100.jsonl

LEARNING PATH:
    1. Start with 100 examples (quality over quantity)
    2. Train for 3 epochs
    3. Test if it learned the patterns
    4. Add more data only if needed

USAGE:
    python train_objection_handler.py

OUTPUT:
    models/objection-handler-lora/ - LoRA adapters (~200MB)
"""

import os
import json
import sys

# Ensure output directory exists
os.makedirs("models", exist_ok=True)


def main():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    print("=" * 50)
    print("Objection Handler Training")
    print("=" * 50)
    print("REMEMBER: SFT before GRPO!")
    print("")

    # 1. Load model with 4-bit quantization (fits in 16GB)
    print("[1/6] Loading model with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B",
        max_seq_length=2048,  # Shorter = less memory
        dtype=None,  # Auto-detect (float16 on M2)
        load_in_4bit=True,  # Critical for 16GB
    )
    print(f"Model loaded: {model.num_parameters():,} parameters")

    # 2. Configure LoRA adapters
    print("\n[2/6] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank (higher = more capacity, more memory)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Saves memory!
    )

    # 3. Load filtered training data
    print("\n[3/6] Loading training data...")
    data_file = "data/objection_training_100.jsonl"

    if not os.path.exists(data_file):
        print(f"ERROR: Training data not found: {data_file}")
        print("Run prep_sales_data.py first!")
        sys.exit(1)

    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"Loaded {len(dataset)} training examples")

    # 4. Training config optimized for memory
    print("\n[4/6] Configuring trainer...")
    training_args = TrainingArguments(
        output_dir="models/objection-handler-v1",
        per_device_train_batch_size=1,  # Start small
        gradient_accumulation_steps=4,  # Simulates batch_size=4
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,  # Let it auto-detect
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",  # Memory-efficient optimizer
        warmup_steps=5,
        seed=42,
    )

    # 5. Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",  # Adjust based on your data format
        max_seq_length=2048,
        args=training_args,
    )

    # 6. Train!
    print("\n[5/6] Training...")
    trainer.train()

    # 7. Save LoRA adapters
    print("\n[6/6] Saving LoRA adapters...")
    output_dir = "models/objection-handler-lora"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"LoRA adapters saved to: {output_dir}")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run test_objection_handler.py to test the model")
    print("2. If results are good, export to GGUF for Ollama")
    print("3. Only consider GRPO if you want further improvement")


if __name__ == "__main__":
    main()
