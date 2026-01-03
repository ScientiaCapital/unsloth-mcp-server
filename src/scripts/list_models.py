#!/usr/bin/env python3
"""
list_models.py - List supported Unsloth models

USAGE:
    python list_models.py

OUTPUT:
    JSON array of supported model names
"""

import json
import sys

def main():
    try:
        from unsloth import FastLanguageModel

        # Define supported models (same as in TypeScript)
        models = [
            # === RECOMMENDED FOR LEARNING ===
            "unsloth/Qwen2.5-0.5B-Instruct",       # Baby steps - smallest, fastest
            "unsloth/Qwen2.5-1.5B-Instruct",       # Small but capable
            "unsloth/Qwen2.5-3B-Instruct",         # Good balance

            # === YOUR CHOSEN PRODUCTION MODELS ===
            "unsloth/DeepSeek-R1-Distill-Qwen-7B", # Reasoning-focused, great for sales
            "unsloth/Qwen2.5-7B-Instruct",         # General purpose, excellent quality

            # === OTHER 7B MODELS ===
            "unsloth/Llama-3.1-8B-bnb-4bit",
            "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",

            # === LARGER MODELS (need more VRAM) ===
            "unsloth/gemma-2-9b-bnb-4bit",
            "unsloth/gemma-2-27b-bnb-4bit",
            "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",

            # === SMALLER MODELS ===
            "unsloth/Llama-3.2-1B-bnb-4bit",
            "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
            "unsloth/Llama-3.2-3B-bnb-4bit",
            "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            "unsloth/Mistral-Small-Instruct-2409",
            "unsloth/Phi-3.5-mini-instruct",
            "unsloth/Phi-3-medium-4k-instruct",
        ]
        print(json.dumps(models))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
