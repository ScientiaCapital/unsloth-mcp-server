#!/usr/bin/env python3
"""
export_model.py - Export a fine-tuned model to GGUF or HuggingFace

USAGE:
    python export_model.py '{"model_path": "outputs/", "format": "gguf"}'

INPUT (JSON):
    - model_path: str (required) - Path to fine-tuned model
    - format: str (default: "gguf") - Export format: "gguf", "merged_16bit", "merged_4bit"
    - output_path: str (optional) - Output path (default: model_path + format)
    - quantization: str (default: "q8_0") - GGUF quantization method

OUTPUT:
    JSON with export results
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

    model_path = args.get("model_path")
    if not model_path:
        print(json.dumps({"error": "model_path is required"}))
        sys.exit(1)

    export_format = args.get("format", "gguf")
    quantization = args.get("quantization", "q8_0")
    output_path = args.get("output_path")

    if not output_path:
        output_path = f"{model_path}_{export_format}"

    try:
        from unsloth import FastLanguageModel

        print(f"Loading model from: {model_path}", file=sys.stderr)

        # Load the model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        os.makedirs(output_path, exist_ok=True)

        if export_format == "gguf":
            print(f"Exporting to GGUF ({quantization})...", file=sys.stderr)

            # Save to GGUF
            model.save_pretrained_gguf(
                output_path,
                tokenizer,
                quantization_method=quantization,
            )

            result = {
                "success": True,
                "format": "gguf",
                "quantization": quantization,
                "output_path": output_path,
                "message": f"Model exported to GGUF format. Ready for Ollama!"
            }

        elif export_format == "merged_16bit":
            print("Exporting merged 16-bit model...", file=sys.stderr)

            model.save_pretrained_merged(
                output_path,
                tokenizer,
                save_method="merged_16bit",
            )

            result = {
                "success": True,
                "format": "merged_16bit",
                "output_path": output_path,
                "message": "Model exported as merged 16-bit. Push to HuggingFace Hub!"
            }

        elif export_format == "merged_4bit":
            print("Exporting merged 4-bit model...", file=sys.stderr)

            model.save_pretrained_merged(
                output_path,
                tokenizer,
                save_method="merged_4bit",
            )

            result = {
                "success": True,
                "format": "merged_4bit",
                "output_path": output_path,
                "message": "Model exported as merged 4-bit. Smaller but less accurate."
            }

        elif export_format == "lora_only":
            print("Saving LoRA adapters only...", file=sys.stderr)

            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            result = {
                "success": True,
                "format": "lora_only",
                "output_path": output_path,
                "message": "LoRA adapters saved. Use with base model for inference."
            }

        else:
            result = {
                "success": False,
                "error": f"Unknown format: {export_format}. Use: gguf, merged_16bit, merged_4bit, lora_only"
            }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
