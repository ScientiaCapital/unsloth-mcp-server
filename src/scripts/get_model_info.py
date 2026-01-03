#!/usr/bin/env python3
"""
get_model_info.py - Get detailed model information

USAGE:
    python get_model_info.py '{"model_name": "unsloth/Qwen2.5-0.5B-Instruct"}'

INPUT (JSON):
    - model_name: str (required) - Model name or path

OUTPUT:
    JSON with model architecture details
"""

import json
import sys


def main():
    # Parse args
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = json.loads(sys.stdin.read())

    model_name = args.get("model_name")
    if not model_name:
        print(json.dumps({"error": "model_name is required"}))
        sys.exit(1)

    try:
        from transformers import AutoConfig, AutoTokenizer
        import torch

        print(f"Loading config for: {model_name}", file=sys.stderr)

        # Load config (doesn't download full model)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Try to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            vocab_size = len(tokenizer)
        except:
            vocab_size = getattr(config, 'vocab_size', 'unknown')

        # Extract model info
        result = {
            "success": True,
            "model_name": model_name,
            "model_type": getattr(config, 'model_type', 'unknown'),
            "vocab_size": vocab_size,
            "hidden_size": getattr(config, 'hidden_size', 'unknown'),
            "num_hidden_layers": getattr(config, 'num_hidden_layers', 'unknown'),
            "num_attention_heads": getattr(config, 'num_attention_heads', 'unknown'),
            "intermediate_size": getattr(config, 'intermediate_size', 'unknown'),
            "max_position_embeddings": getattr(config, 'max_position_embeddings', 'unknown'),
            "torch_dtype": str(getattr(config, 'torch_dtype', 'unknown')),
        }

        # Add model-specific info
        if hasattr(config, 'num_key_value_heads'):
            result['num_key_value_heads'] = config.num_key_value_heads
        if hasattr(config, 'rope_theta'):
            result['rope_theta'] = config.rope_theta
        if hasattr(config, 'sliding_window'):
            result['sliding_window'] = config.sliding_window

        # Estimate parameter count
        try:
            hidden = config.hidden_size
            layers = config.num_hidden_layers
            vocab = config.vocab_size
            # Rough estimate: embedding + attention + FFN per layer
            params = vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * config.intermediate_size)
            result['estimated_params'] = f"{params / 1e9:.2f}B"
        except:
            pass

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
