#!/usr/bin/env python3
"""
finetune_model.py - Fine-tune a model with SFT using Unsloth

USAGE:
    python finetune_model.py '{"model_name": "...", "dataset_name": "..."}'

INPUT (JSON):
    - model_name: str (required) - HuggingFace model name
    - dataset_name: str (required) - Dataset name or path
    - output_dir: str (default: "outputs") - Output directory
    - max_seq_length: int (default: 512) - Maximum sequence length
    - lora_rank: int (default: 16) - LoRA rank (r)
    - lora_alpha: int (default: 16) - LoRA alpha
    - learning_rate: float (default: 2e-4) - Learning rate
    - batch_size: int (default: 2) - Per-device batch size
    - gradient_accumulation_steps: int (default: 4) - Gradient accumulation
    - max_steps: int (default: 60) - Maximum training steps
    - dataset_text_field: str (default: "text") - Field containing text
    - load_in_4bit: bool (default: True) - Use 4-bit quantization

OUTPUT:
    JSON with training results

REMEMBER: SFT BEFORE GRPO!
"""

import json
import sys
import os


def main():
    # Parse args
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = json.loads(sys.stdin.read())

    # Required args
    model_name = args.get("model_name")
    dataset_name = args.get("dataset_name")

    if not model_name or not dataset_name:
        print(json.dumps({"error": "model_name and dataset_name are required"}))
        sys.exit(1)

    # Optional args with defaults
    output_dir = args.get("output_dir", "outputs")
    max_seq_length = args.get("max_seq_length", 512)
    lora_rank = args.get("lora_rank", 16)
    lora_alpha = args.get("lora_alpha", 16)
    learning_rate = args.get("learning_rate", 2e-4)
    batch_size = args.get("batch_size", 2)
    gradient_accumulation_steps = args.get("gradient_accumulation_steps", 4)
    max_steps = args.get("max_steps", 60)
    dataset_text_field = args.get("dataset_text_field", "text")
    load_in_4bit = args.get("load_in_4bit", True)

    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading model: {model_name}", file=sys.stderr)

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )

        print("Adding LoRA adapters...", file=sys.stderr)

        # Add LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        print(f"Loading dataset: {dataset_name}", file=sys.stderr)

        # Load dataset
        if os.path.exists(dataset_name):
            # Local file
            if dataset_name.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=dataset_name)
            elif dataset_name.endswith('.json'):
                dataset = load_dataset('json', data_files=dataset_name)
            else:
                dataset = load_dataset(dataset_name)
        else:
            # HuggingFace dataset
            dataset = load_dataset(dataset_name)

        # Get train split
        train_dataset = dataset["train"] if "train" in dataset else dataset

        print(f"Training for {max_steps} steps...", file=sys.stderr)

        # Configure trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=5,
                max_steps=max_steps,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                output_dir=output_dir,
                seed=42,
            ),
        )

        # Train
        trainer.train()

        # Save
        trainer.save_model()

        result = {
            "success": True,
            "output_dir": output_dir,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "max_steps": max_steps,
            "lora_rank": lora_rank,
            "message": "SFT training complete! Model saved to output_dir."
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
