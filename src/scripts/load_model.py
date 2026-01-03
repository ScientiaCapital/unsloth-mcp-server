#!/usr/bin/env python3
"""
load_model.py - Load a model with Unsloth optimizations

USAGE:
    python load_model.py '{"model_name": "unsloth/Qwen2.5-0.5B-Instruct", "max_seq_length": 512}'

INPUT (JSON):
    - model_name: str (required) - HuggingFace model name
    - max_seq_length: int (default: 2048) - Maximum sequence length
    - load_in_4bit: bool (default: True) - Use 4-bit quantization
    - use_gradient_checkpointing: bool (default: True) - Enable gradient checkpointing

OUTPUT:
    JSON with model info
"""

import json
import sys


def main():
    # Parse args from stdin or command line
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = json.loads(sys.stdin.read())

    model_name = args.get("model_name")
    if not model_name:
        print(json.dumps({"error": "model_name is required"}))
        sys.exit(1)

    max_seq_length = args.get("max_seq_length", 2048)
    load_in_4bit = args.get("load_in_4bit", True)
    use_gradient_checkpointing = args.get("use_gradient_checkpointing", True)

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth" if use_gradient_checkpointing else False,
        )

        # Get model info
        result = {
            "success": True,
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "num_parameters": model.num_parameters() if hasattr(model, 'num_parameters') else "unknown",
            "vocab_size": len(tokenizer) if tokenizer else "unknown",
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
