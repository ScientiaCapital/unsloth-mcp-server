#!/usr/bin/env python3
"""
02-load-model.py - Load a model with Unsloth

WHAT THIS DOES:
- Loads the Qwen2.5-0.5B model (smallest good model)
- Uses 4-bit quantization to save GPU memory
- Applies Unsloth optimizations for 2x speed

MODEL CHOICE:
- Qwen2.5-0.5B-Instruct: 500M params, fits in 4GB VRAM
- Great for learning and testing
- Swap to larger models later (3B, 7B, 14B)

MEMORY REQUIREMENTS:
- 4-bit: ~2GB VRAM
- Full precision: ~4GB VRAM
"""

from unsloth import FastLanguageModel

# Configuration - change these to try different models
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"  # Smallest, fastest
MAX_SEQ_LENGTH = 512  # Keep small for speed
LOAD_IN_4BIT = True   # Use less memory

def load_model():
    """Load model with Unsloth optimizations."""
    print("=" * 50)
    print(f"Loading model: {MODEL_NAME}")
    print("=" * 50)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 50)
    print("COMPLETE - Model loaded")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Quantization: {'4-bit' if LOAD_IN_4BIT else 'full precision'}")
    print("=" * 50)
    print("\nNext step: Run 03-add-lora.py")

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()
