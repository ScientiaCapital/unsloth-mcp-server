#!/usr/bin/env python3
"""
06-test-model.py - Test your trained model

WHAT THIS DOES:
- Loads the trained model
- Tests it on questions from training AND new questions
- Shows if the model actually learned

WHAT TO LOOK FOR:
- Questions from training: Should get correct answers
- New questions: Should generalize (maybe not perfect)
- If both fail: Check your training data

THIS IS THE MOMENT OF TRUTH:
If this works, your SFT foundation is solid.
ONLY THEN should you consider GRPO.
"""

from unsloth import FastLanguageModel
import torch

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"


def test_model():
    """Test the trained model."""
    print("=" * 50)
    print("Testing the fine-tuned model")
    print("=" * 50)

    # Load and prepare model
    print("\n[1/2] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        load_in_4bit=True,
    )

    # Add LoRA and train (same as 05)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Quick train
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    examples = [
        ("What is 2+2?", "4"), ("What is 3+3?", "6"),
        ("What is 5+5?", "10"), ("What is 10+10?", "20"),
        ("What is 7+3?", "10"), ("What is 8+2?", "10"),
    ]
    dataset = Dataset.from_list([
        {"text": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"}
        for q, a in examples
    ])

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=512,
        args=TrainingArguments(
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            warmup_steps=5, max_steps=30, learning_rate=2e-4, fp16=True,
            logging_steps=10, output_dir="outputs", seed=42,
        ),
    )
    trainer.train()

    # Enable inference mode
    print("\n[2/2] Testing...")
    FastLanguageModel.for_inference(model)

    # Test questions
    test_questions = [
        ("What is 2+2?", "From training"),
        ("What is 7+3?", "From training"),
        ("What is 8+8?", "NEW - not trained"),
        ("What is 14+6?", "NEW - not trained"),
    ]

    print("\nResults:")
    print("-" * 40)

    for question, note in test_questions:
        messages = [{"role": "user", "content": question}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=20,
            temperature=0.1,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("assistant")[-1].strip()

        print(f"Q: {question} ({note})")
        print(f"A: {answer}")
        print("-" * 40)

    print("\n" + "=" * 50)
    print("COMPLETE")
    print("If answers are mostly correct, SFT is working!")
    print("=" * 50)
    print("\nNext step: Run 07-save-model.py")


if __name__ == "__main__":
    test_model()
