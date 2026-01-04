#!/usr/bin/env python3
"""
Add Gong Transcripts to Training Data
======================================
Easy copy-paste tool to add your best Gong calls to training data.

USAGE:
1. Create a text file with the transcript (see format below)
2. Run: python add_gong_transcript.py --input transcript.txt

FORMAT (Simple):
-----------------
# My best discovery call

USER: Tell me about your current process for managing projects.

AGENT: Great question. Let me understand your workflow. Are you currently using any software for project management, or mostly spreadsheets and email?

USER: We use spreadsheets and a shared Google Drive. It's a mess.

AGENT: I hear that a lot. What's the biggest pain point - is it finding information, or tracking where projects are in the pipeline?
-----------------

The script will:
1. Parse the conversation
2. Add Coperniq system prompt
3. Save to ChatML JSONL format
4. Append to your training data
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime

# System prompt for Coperniq sales agent
SYSTEM_PROMPT = """You are a top-performing sales development representative (SDR) for Coperniq, a construction management software built specifically for MEP contractors (HVAC, Plumbing, Electrical, Solar).

Key Coperniq differentiators:
1. ASSET LIFECYCLE TRACKING: When a job closes in other tools, the record goes cold. In Coperniq, the ASSET lives forever - track production, service history, and O&M billing for 25+ years.
2. REAL-TIME ENERGY MONITORING: Systems dashboard shows status (green/yellow/red), peak power, performance trends. Know when systems underperform BEFORE customers call.
3. AI-NATIVE FEATURES: Copilot for plain English queries, AI invoicing (saves 25+ hours/month), workflow builder.
4. UNIFIED PLATFORM: Sales + Service in one system, not separate modules.
5. 30-50% LOWER COST: Compared to ServiceTitan, Procore, etc.

Target customer: $5-50M multi-trade, asset-centric, self-performing contractors.

Your style: Consultative, not pushy. Ask discovery questions. Use the Challenger Sale approach when needed."""


def parse_transcript(text):
    """
    Parse transcript text into conversation turns.

    Supports formats:
    - USER: message
    - AGENT: message
    - PROSPECT: message
    - REP: message
    - SDR: message
    - Customer: message
    - Sales: message
    """
    # Normalize role names
    text = re.sub(r'\b(PROSPECT|CUSTOMER|CLIENT)\s*:', 'USER:', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(REP|SDR|SALES|AGENT)\s*:', 'AGENT:', text, flags=re.IGNORECASE)

    # Split by role markers
    pattern = r'(USER:|AGENT:)'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    messages = []
    current_role = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.upper() == 'USER:':
            current_role = 'user'
        elif part.upper() == 'AGENT:':
            current_role = 'assistant'
        elif current_role:
            # Clean up the message
            message = part.strip()
            if message:
                messages.append({
                    "role": current_role,
                    "content": message
                })

    return messages


def extract_title(text):
    """Extract title from # heading or first line."""
    lines = text.strip().split('\n')
    for line in lines:
        if line.startswith('#'):
            return line.lstrip('#').strip()
        if line.strip() and not line.startswith(('USER:', 'AGENT:', 'PROSPECT:', 'REP:')):
            return line.strip()[:50]
    return f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_training_examples(messages, title="", category="gong_real"):
    """
    Create training examples from conversation.

    For each user message followed by an agent response,
    create a training example.
    """
    examples = []

    # Add system prompt to first message set
    for i, msg in enumerate(messages):
        if msg['role'] == 'user' and i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
            # Create example: system + user + assistant
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg['content']},
                    {"role": "assistant", "content": messages[i + 1]['content']}
                ],
                "_metadata": {
                    "category": category,
                    "source": "gong_transcript",
                    "title": title,
                    "added": datetime.now().isoformat()
                }
            }
            examples.append(example)

    return examples


def main():
    parser = argparse.ArgumentParser(
        description='Add Gong transcripts to training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE TRANSCRIPT FORMAT:
--------------------------
# Discovery Call - ABC Electric

USER: We're looking at different project management tools.

AGENT: Great! What's driving that search? Is there something specific that's not working with your current setup?

USER: Yeah, we can't track equipment after we install it.

AGENT: That's a common pain point. Let me show you how Coperniq handles asset lifecycle tracking...
--------------------------

You can also paste directly using --paste flag:
  python add_gong_transcript.py --paste

Or provide content inline:
  python add_gong_transcript.py --content "USER: Hello\\nAGENT: Hi there!"
        """
    )

    parser.add_argument('--input', '-i', type=str, help='Input transcript file')
    parser.add_argument('--output', '-o', type=str, default='sft_training_data.jsonl',
                        help='Output JSONL file (default: sft_training_data.jsonl)')
    parser.add_argument('--category', '-c', type=str, default='gong_real',
                        help='Category tag (default: gong_real)')
    parser.add_argument('--paste', '-p', action='store_true',
                        help='Paste transcript interactively')
    parser.add_argument('--content', type=str, help='Provide transcript content directly')
    parser.add_argument('--preview', action='store_true', help='Preview only, don\'t save')

    args = parser.parse_args()

    # Get transcript content
    if args.paste:
        print("Paste your transcript below. When done, enter an empty line or Ctrl+D:\n")
        lines = []
        try:
            while True:
                line = input()
                if not line and lines:  # Empty line after content
                    break
                lines.append(line)
        except EOFError:
            pass
        transcript = '\n'.join(lines)
    elif args.content:
        transcript = args.content.replace('\\n', '\n')
    elif args.input:
        with open(args.input) as f:
            transcript = f.read()
    else:
        print("ERROR: Provide --input FILE, --paste, or --content TEXT")
        print("Run with --help for examples")
        return

    if not transcript.strip():
        print("ERROR: Empty transcript")
        return

    # Parse and create examples
    title = extract_title(transcript)
    messages = parse_transcript(transcript)

    if len(messages) < 2:
        print(f"ERROR: Found only {len(messages)} messages. Need at least USER + AGENT pair.")
        print("Make sure your transcript uses 'USER:' and 'AGENT:' (or similar) prefixes.")
        return

    examples = create_training_examples(messages, title, args.category)

    print(f"\n{'='*60}")
    print(f"TRANSCRIPT: {title}")
    print(f"{'='*60}")
    print(f"Messages found: {len(messages)}")
    print(f"Training examples created: {len(examples)}")

    # Preview
    if examples:
        print(f"\n--- Preview (first example) ---")
        ex = examples[0]
        print(f"User: {ex['messages'][1]['content'][:100]}...")
        print(f"Agent: {ex['messages'][2]['content'][:100]}...")
        print("---")

    if args.preview:
        print("\n[Preview mode - not saving]")
        print(json.dumps(examples, indent=2))
        return

    # Save
    output_path = Path(args.output)
    mode = 'a' if output_path.exists() else 'w'

    with open(output_path, mode) as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    # Count total examples
    with open(output_path) as f:
        total = sum(1 for _ in f)

    print(f"\n✅ Added {len(examples)} examples to {output_path}")
    print(f"📊 Total examples in file: {total}")
    print(f"\nNext: Re-run training with updated data:")
    print(f"  python train_coperniq_sft.py --data {output_path}")


if __name__ == '__main__':
    main()
