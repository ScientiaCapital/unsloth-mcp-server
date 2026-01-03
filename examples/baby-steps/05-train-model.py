#!/usr/bin/env python3
"""
05-train-model.py - Train the model with SFT

WHAT THIS DOES:
- Runs Supervised Fine-Tuning (SFT) on your data
- Trains the LoRA adapters (not the whole model)
- Shows training progress with loss metrics

TRAINING PARAMETERS:
- max_steps=60: Short training (2-3 minutes) to test
- batch_size=2: How many examples at once
- gradient_accumulation=4: Effective batch = 2*4 = 8
- learning_rate=2e-4: Standard for LoRA

WATCH THE LOSS:
- Loss should DECREASE as training progresses
- Starting: ~2.0-3.0
- After training: ~0.5-1.0
- If loss stays high, check your data format

REMEMBER: SFT BEFORE GRPO!
This is the foundation - make it work first.
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_STEPS = 60  # Short training to prove it works
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4


def format_for_chatml(question: str, answer: str) -> str:
    """Format a Q&A pair for ChatML."""
    return f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""


def create_mini_dataset() -> Dataset:
    """Create a small test dataset."""
    examples = [
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
        ("What is 5+5?", "10"),
        ("What is 10+10?", "20"),
        ("What is 7+3?", "10"),
        ("What is 8+2?", "10"),
        ("What is 1+1?", "2"),
        ("What is 4+4?", "8"),
        ("What is 6+6?", "12"),
        ("What is 9+1?", "10"),
        ("What is 15+15?", "30"),
        ("What is 20+20?", "40"),
        ("What is 11+11?", "22"),
        ("What is 12+8?", "20"),
        ("What is 25+25?", "50"),
        ("What is 30+30?", "60"),
        ("What is 100+100?", "200"),
        ("What is 50+50?", "100"),
        ("What is 13+7?", "20"),
        ("What is 16+4?", "20"),
    ]
    formatted = [{"text": format_for_chatml(q, a)} for q, a in examples]
    return Dataset.from_list(formatted)


def train_model():
    """Train the model with SFT."""
    print("=" * 50)
    print("Starting SFT Training")
    print("=" * 50)

    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        load_in_4bit=True,
    )

    # Add LoRA
    print("\n[2/4] Adding LoRA adapters...")
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

    # Create dataset
    print("\n[3/4] Creating dataset...")
    dataset = create_mini_dataset()
    print(f"  Examples: {len(dataset)}")

    # Configure trainer
    print("\n[4/4] Training...")
    print(f"  Steps: {MAX_STEPS}")
    print(f"  Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print("\nWatch the loss - it should DECREASE:\n")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=5,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            output_dir="outputs",
            seed=42,
        ),
    )

    # Train!
    stats = trainer.train()

    print("\n" + "=" * 50)
    print("COMPLETE - Training finished!")
    print(f"  Final loss: {stats.training_loss:.4f}")
    print(f"  Time: {stats.metrics['train_runtime']:.0f} seconds")
    print("=" * 50)
    print("\nNext step: Run 06-test-model.py")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_model()
