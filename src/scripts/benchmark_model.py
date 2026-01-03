#!/usr/bin/env python3
"""
benchmark_model.py - Benchmark model performance

USAGE:
    python benchmark_model.py '{"model_name": "...", "prompt": "Hello"}'

INPUT (JSON):
    - model_name: str (required) - Model name or path
    - prompt: str (required) - Test prompt
    - num_iterations: int (default: 5) - Number of benchmark runs
    - max_new_tokens: int (default: 128) - Tokens per generation

OUTPUT:
    JSON with benchmark results (tokens/sec, memory, etc.)
"""

import json
import sys
import time


def main():
    # Parse args
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = json.loads(sys.stdin.read())

    model_name = args.get("model_name")
    prompt = args.get("prompt")

    if not model_name or not prompt:
        print(json.dumps({"error": "model_name and prompt are required"}))
        sys.exit(1)

    num_iterations = args.get("num_iterations", 5)
    max_new_tokens = args.get("max_new_tokens", 128)

    try:
        from unsloth import FastLanguageModel
        import torch
        import psutil
        import os

        print(f"Loading model: {model_name}", file=sys.stderr)

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        # Prepare for inference
        FastLanguageModel.for_inference(model)

        # Warm-up run
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        _ = model.generate(**inputs, max_new_tokens=10)

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # GPU memory if available
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        print(f"Running {num_iterations} benchmark iterations...", file=sys.stderr)

        # Benchmark runs
        times = []
        tokens_per_second = []

        for i in range(num_iterations):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            start_time = time.time()
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)
            tokens_per_second.append(max_new_tokens / elapsed)

            print(f"  Iteration {i+1}: {elapsed:.3f}s ({max_new_tokens/elapsed:.1f} tok/s)", file=sys.stderr)

        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = {
            "success": True,
            "model_name": model_name,
            "num_iterations": num_iterations,
            "max_new_tokens": max_new_tokens,
            "avg_time_seconds": round(sum(times) / len(times), 3),
            "min_time_seconds": round(min(times), 3),
            "max_time_seconds": round(max(times), 3),
            "avg_tokens_per_second": round(sum(tokens_per_second) / len(tokens_per_second), 2),
            "memory_used_mb": round(final_memory - initial_memory, 2),
            "total_memory_mb": round(final_memory, 2),
        }

        if gpu_memory is not None:
            result["gpu_memory_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
