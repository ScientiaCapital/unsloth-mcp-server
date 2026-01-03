#!/usr/bin/env python3
"""
04-create-dataset.py - Create training data

WHAT THIS DOES:
- Creates a simple training dataset
- Formats it for ChatML (the format Qwen uses)

THE KEY INSIGHT:
This file has HARDCODED math examples that WORK.
Once you see training succeed, SWAP THIS with your real data:
- Sales emails from coperniq-forge
- CRM extraction examples
- Your domain-specific data

DATA FORMAT (ChatML):
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>

SWAP YOUR DATA:
1. Keep this file working with math examples
2. Create a copy: 04-create-dataset-sales.py
3. Load your data from JSON instead of hardcoding
"""

from datasets import Dataset
import json
import os

# ============================================
# OPTION 1: Hardcoded examples (for testing)
# ============================================
MATH_EXAMPLES = [
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


def format_for_chatml(question: str, answer: str) -> str:
    """Format a Q&A pair for ChatML."""
    return f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""


def create_dataset_from_examples(examples: list) -> Dataset:
    """Create a HuggingFace Dataset from Q&A pairs."""
    formatted = [{"text": format_for_chatml(q, a)} for q, a in examples]
    return Dataset.from_list(formatted)


def create_dataset_from_jsonl(filepath: str) -> Dataset:
    """
    Load training data from JSONL file.

    SWAP YOUR DATA HERE!
    Expected format (one per line):
    {"instruction": "What is 2+2?", "output": "4"}

    OR ChatML format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            item = json.loads(line)

            # Handle different formats
            if "messages" in item:
                # ChatML format (from coperniq-forge)
                user_msg = next(m["content"] for m in item["messages"] if m["role"] == "user")
                asst_msg = next(m["content"] for m in item["messages"] if m["role"] == "assistant")
                text = format_for_chatml(user_msg, asst_msg)
            elif "instruction" in item and "output" in item:
                # Alpaca format
                text = format_for_chatml(item["instruction"], item["output"])
            elif "text" in item:
                # Already formatted
                text = item["text"]
            else:
                raise ValueError(f"Unknown format: {item.keys()}")

            data.append({"text": text})

    return Dataset.from_list(data)


def create_dataset():
    """Create training dataset."""
    print("=" * 50)
    print("Creating training dataset")
    print("=" * 50)

    # Check for external data file
    data_dir = os.path.dirname(__file__)
    jsonl_path = os.path.join(data_dir, "data", "training.jsonl")

    if os.path.exists(jsonl_path):
        print(f"\n[Using external data]: {jsonl_path}")
        dataset = create_dataset_from_jsonl(jsonl_path)
    else:
        print("\n[Using hardcoded math examples]")
        print("To use your own data, create: data/training.jsonl")
        dataset = create_dataset_from_examples(MATH_EXAMPLES)

    print("\n" + "=" * 50)
    print("COMPLETE - Dataset created")
    print(f"  Examples: {len(dataset)}")
    print(f"  Sample: {dataset[0]['text'][:80]}...")
    print("=" * 50)
    print("\nNext step: Run 05-train-model.py")

    return dataset


if __name__ == "__main__":
    dataset = create_dataset()

    # Save for inspection
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "data", "math_examples.json"), "w") as f:
        json.dump([{"instruction": q, "output": a} for q, a in MATH_EXAMPLES], f, indent=2)
    print("\nSaved math examples to data/math_examples.json")
