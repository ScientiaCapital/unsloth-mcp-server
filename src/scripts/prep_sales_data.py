#!/usr/bin/env python3
"""
prep_sales_data.py - Filter and prepare sales data for SFT training

PURPOSE:
    Start with your 100 BEST objection handling examples.
    Quality over quantity for initial SFT training.

LEARNING APPROACH:
    1. Filter for objection handling patterns
    2. Train on best 100 examples first
    3. Test if model learned the pattern
    4. Add more data only if needed

USAGE:
    python prep_sales_data.py '{"input": "data/coperniq_sales.jsonl", "output": "data/training_100.jsonl"}'

Or direct function call:
    python -c "from prep_sales_data import validate_and_filter_chatml; validate_and_filter_chatml('input.jsonl', 'output.jsonl')"
"""

import json
import sys
import os


# Common sales objection patterns
OBJECTION_KEYWORDS = [
    # Price objections
    "too expensive", "not in budget", "cost too much", "can't afford",
    "cheaper alternative", "price is high", "budget constraints",

    # Timing objections
    "not the right time", "maybe later", "next quarter", "next year",
    "too busy right now", "call back later", "not ready",

    # Decision maker objections
    "talk to partner", "need to check with", "not my decision",
    "run it by my team", "get approval from", "boss needs to approve",

    # Competition objections
    "already have", "using another", "happy with current",
    "competitor offers", "looking at other options",

    # Trust/Need objections
    "need to think", "not sure we need", "don't see the value",
    "how is this different", "what makes you special",

    # Follow-up resistance
    "stop calling", "not interested", "remove from list",
    "already told you no", "don't email me",
]


def validate_and_filter_chatml(chatml_file: str, output_file: str, max_samples: int = 100):
    """
    Load ChatML samples, filter for quality objection handling examples.

    Args:
        chatml_file: Path to input JSONL with ChatML format
        output_file: Path to write filtered samples
        max_samples: Maximum samples to output (default: 100)

    Returns:
        dict with stats about filtering
    """

    if not os.path.exists(chatml_file):
        return {"error": f"Input file not found: {chatml_file}"}

    with open(chatml_file, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(samples)} total samples", file=sys.stderr)

    quality_samples = []
    objection_counts = {keyword: 0 for keyword in OBJECTION_KEYWORDS[:10]}  # Track top 10

    for sample in samples:
        messages = sample.get("messages", [])

        # Skip very short conversations
        if len(messages) < 3:
            continue

        # Check for objection patterns in user messages
        found_objection = None
        for msg in messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "").lower()
            for keyword in OBJECTION_KEYWORDS:
                if keyword in content:
                    found_objection = keyword
                    break
            if found_objection:
                break

        if found_objection:
            quality_samples.append({
                "sample": sample,
                "objection_type": found_objection,
                "message_count": len(messages),
            })

            # Track common objections
            for key in objection_counts:
                if key in found_objection:
                    objection_counts[key] += 1

    # Sort by message count (prefer longer, more detailed examples)
    quality_samples.sort(key=lambda x: x["message_count"], reverse=True)

    # Take top N samples
    selected = quality_samples[:max_samples]

    print(f"Filtered {len(samples)} → {len(quality_samples)} objection samples", file=sys.stderr)
    print(f"Selected top {len(selected)} for training", file=sys.stderr)

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write filtered samples
    with open(output_file, 'w') as f:
        for item in selected:
            f.write(json.dumps(item["sample"]) + '\n')

    # Analyze what we got
    objection_distribution = {}
    for item in selected:
        obj_type = item["objection_type"]
        objection_distribution[obj_type] = objection_distribution.get(obj_type, 0) + 1

    result = {
        "success": True,
        "input_samples": len(samples),
        "objection_samples_found": len(quality_samples),
        "output_samples": len(selected),
        "output_file": output_file,
        "objection_distribution": dict(sorted(objection_distribution.items(),
                                              key=lambda x: -x[1])[:10]),
        "message": f"Filtered to {len(selected)} best objection handling examples. Ready for SFT!"
    }

    return result


def main():
    """Main entry point for CLI usage."""
    # Parse args
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        # Check for stdin
        import select
        if select.select([sys.stdin], [], [], 0.0)[0]:
            args = json.loads(sys.stdin.read())
        else:
            # Default behavior
            args = {
                "input": "data/coperniq_sales_chatml.jsonl",
                "output": "data/objection_training_100.jsonl",
                "max_samples": 100,
            }

    input_file = args.get("input", "data/coperniq_sales_chatml.jsonl")
    output_file = args.get("output", "data/objection_training_100.jsonl")
    max_samples = args.get("max_samples", 100)

    result = validate_and_filter_chatml(input_file, output_file, max_samples)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
