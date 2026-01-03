#!/usr/bin/env python3
"""
generate_text.py - Generate text using a fine-tuned model

USAGE:
    python generate_text.py '{"model_path": "outputs/", "prompt": "Hello"}'

INPUT (JSON):
    - model_path: str (required) - Path to model or adapter
    - prompt: str (required) - Input prompt
    - max_new_tokens: int (default: 256) - Max tokens to generate
    - temperature: float (default: 0.7) - Sampling temperature
    - top_p: float (default: 0.9) - Top-p sampling
    - system_prompt: str (optional) - System prompt for chat models

OUTPUT:
    JSON with generated text
"""

import json
import sys


def main():
    # Parse args
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = json.loads(sys.stdin.read())

    model_path = args.get("model_path")
    prompt = args.get("prompt")

    if not model_path or not prompt:
        print(json.dumps({"error": "model_path and prompt are required"}))
        sys.exit(1)

    max_new_tokens = args.get("max_new_tokens", 256)
    temperature = args.get("temperature", 0.7)
    top_p = args.get("top_p", 0.9)
    system_prompt = args.get("system_prompt")

    try:
        from unsloth import FastLanguageModel
        import torch

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        # Enable inference mode
        FastLanguageModel.for_inference(model)

        # Build messages for chat format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        result = {
            "success": True,
            "model_path": model_path,
            "prompt": prompt,
            "response": response,
            "max_new_tokens": max_new_tokens,
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
