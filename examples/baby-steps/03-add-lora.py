#!/usr/bin/env python3
"""
03-add-lora.py - Add LoRA adapters for training

WHAT THIS DOES:
- Adds LoRA (Low-Rank Adaptation) layers to the model
- These are the ONLY parts we train (not the whole model)
- Result: Train 2.5% of parameters, not 100%

WHY LoRA:
- Much faster training (minutes instead of hours)
- Much less GPU memory needed
- Resulting adapter is tiny (~50MB instead of 2GB+)

LoRA PARAMETERS:
- r=16: Rank of the adaptation matrices (higher = more capacity)
- lora_alpha=16: Scaling factor (usually same as r)
- target_modules: Which layers to adapt (attention + MLP)
"""

from unsloth import FastLanguageModel

# Load model first (from previous step)
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"

def add_lora():
    """Add LoRA adapters to the model."""
    print("=" * 50)
    print("Adding LoRA adapters")
    print("=" * 50)

    # Load base model
    print("\n[1/2] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        load_in_4bit=True,
    )

    # Add LoRA
    print("\n[2/2] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print("COMPLETE - LoRA adapters added")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print("=" * 50)
    print("\nNext step: Run 04-create-dataset.py")

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = add_lora()
