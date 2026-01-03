#!/usr/bin/env python3
"""
test_objection_handler.py - Test your trained objection handling model

PURPOSE:
    Verify that SFT training worked before moving to GRPO.
    Test on various objection types to check generalization.

USAGE:
    python test_objection_handler.py

EXPECTED RESULTS:
    - Should handle budget objections professionally
    - Should acknowledge concerns and reframe value
    - Should ask discovery questions
    - May not be perfect - that's what GRPO is for!
"""

import json
import sys


def main():
    from unsloth import FastLanguageModel

    print("=" * 50)
    print("Testing Objection Handler Model")
    print("=" * 50)

    # Load base model + LoRA adapters
    print("\nLoading model...")
    model_path = "models/objection-handler-lora"

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"ERROR: Could not load model from {model_path}")
        print(f"Error: {e}")
        print("\nMake sure you ran train_objection_handler.py first!")
        sys.exit(1)

    FastLanguageModel.for_inference(model)  # Speeds up generation

    # Detect device
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Test objections
    test_objections = [
        # Budget objection
        {
            "objection": "budget",
            "message": "Coperniq looks interesting but we're not in budget for new software right now."
        },
        # Timing objection
        {
            "objection": "timing",
            "message": "This sounds good but we're in the middle of a big project. Maybe call back next quarter?"
        },
        # Competition objection
        {
            "objection": "competition",
            "message": "We're already using Procore and it's working fine for us."
        },
        # Decision maker objection
        {
            "objection": "decision_maker",
            "message": "I'm not the one who makes these decisions. You'd need to talk to our operations director."
        },
        # Value objection
        {
            "objection": "value",
            "message": "I don't really see how this would help our business. We're doing fine with spreadsheets."
        },
    ]

    print("\n" + "-" * 50)
    print("Testing objection handling responses...")
    print("-" * 50)

    results = []
    for test in test_objections:
        print(f"\n[{test['objection'].upper()}]")
        print(f"Customer: {test['message']}")

        conversation = [
            {"role": "user", "content": test['message']},
        ]

        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        print(f"Sales Rep: {response}")
        print("-" * 40)

        results.append({
            "objection_type": test['objection'],
            "customer": test['message'],
            "response": response,
        })

    # Summary
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)
    print("\nEvaluation Checklist:")
    print("[ ] Responses acknowledge customer concerns?")
    print("[ ] Responses reframe value proposition?")
    print("[ ] Responses ask discovery questions?")
    print("[ ] Tone is professional and empathetic?")
    print("[ ] Responses are relevant to MEP/construction context?")
    print("\nIf mostly yes → SFT worked! Consider GRPO for refinement.")
    print("If mostly no → Review training data quality or train longer.")

    # Save results for review
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: test_results.json")


if __name__ == "__main__":
    main()
