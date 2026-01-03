#!/usr/bin/env python3
"""
utils.py - Shared utilities for Unsloth MCP scripts

Common functions used across multiple scripts.
"""

import json
import sys
import os


def parse_args():
    """Parse JSON arguments from command line or stdin."""
    if len(sys.argv) > 1:
        return json.loads(sys.argv[1])
    else:
        return json.loads(sys.stdin.read())


def output_json(data):
    """Output JSON result."""
    print(json.dumps(data))


def error_json(message):
    """Output JSON error and exit."""
    print(json.dumps({"error": message, "success": False}))
    sys.exit(1)


def log(message):
    """Log message to stderr (not captured in JSON output)."""
    print(message, file=sys.stderr)


def ensure_dir(path):
    """Ensure directory exists."""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)


def format_chatml(user_message, assistant_message):
    """Format messages in ChatML format for training."""
    return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_message}<|im_end|>"


def load_jsonl(filepath):
    """Load JSONL file into list of dicts."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """Save list of dicts to JSONL file."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def get_device():
    """Get the best available device (cuda > mps > cpu)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Model configs for common setups
DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0,
    "bias": "none",
}

DEFAULT_TRAINING_CONFIG = {
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "max_steps": 60,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 10,
    "seed": 42,
}

# Baby steps model - smallest for learning
BABY_STEPS_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
