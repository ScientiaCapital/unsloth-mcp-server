#!/usr/bin/env python3
"""
08-inference.py - Use your trained model for inference

WHAT THIS DOES:
- Loads your saved LoRA adapter
- Runs inference on new inputs
- Shows how to use the model in production

USAGE:
1. Run after 07-save-model.py creates the adapter
2. Point model_name to your saved adapter folder
3. The base model is downloaded automatically

FOR PRODUCTION:
- Export to GGUF for Ollama (see 09-export-gguf.py)
- Or use this script with your adapter
"""

from unsloth import FastLanguageModel
import torch
import json

# Path to your saved adapter (from 07-save-model.py)
ADAPTER_PATH = "my_first_finetune"

# If adapter doesn't exist, use base model
MODEL_NAME = ADAPTER_PATH if __import__('os').path.exists(ADAPTER_PATH) else "unsloth/Qwen2.5-0.5B-Instruct"


def run_inference(prompt: str, system_prompt: str = None) -> str:
    """Run inference with the model."""
    global model, tokenizer

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    return response


def main():
    """Main inference demo."""
    global model, tokenizer

    print("=" * 50)
    print("Inference Demo")
    print("=" * 50)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Demo 1: Math (what we trained on)
    print("\n" + "-" * 40)
    print("Demo 1: Math questions (trained domain)")
    print("-" * 40)
    for q in ["What is 7+7?", "What is 25+75?", "What is 123+456?"]:
        answer = run_inference(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

    # Demo 2: CRM extraction (new domain)
    print("-" * 40)
    print("Demo 2: CRM extraction (new domain)")
    print("-" * 40)
    crm_prompt = """Just talked to Mike at ABC HVAC, 25 person shop in Phoenix.
They're currently using Procore but unhappy with the pricing.
Looking to switch in Q1. Decision maker is their Operations Director Sarah."""

    system = "Extract structured CRM data from sales notes. Output JSON only."
    answer = run_inference(crm_prompt, system_prompt=system)
    print(f"Input: {crm_prompt[:80]}...")
    print(f"Output:\n{answer}\n")

    # Demo 3: Interactive mode
    print("-" * 40)
    print("Demo 3: Interactive mode")
    print("-" * 40)
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            response = run_inference(user_input)
            print(f"Model: {response}\n")
        except KeyboardInterrupt:
            break

    print("\n" + "=" * 50)
    print("COMPLETE")
    print("Your model is ready for production!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Export to GGUF for Ollama (faster inference)")
    print("2. Swap training data with your real domain data")
    print("3. Train longer with more examples")


if __name__ == "__main__":
    main()
