#!/usr/bin/env python3
"""
Pull Gong Call Transcripts for Training
========================================
Fetches call transcripts from Gong API and converts to ChatML format.

Usage:
    python pull_gong_calls.py --limit 50
    python pull_gong_calls.py --days 30 --limit 100
"""

import os
import json
import requests
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GONG_ACCESS_KEY = os.getenv("GONG_ACCESS_KEY")
GONG_SECRET_KEY = os.getenv("GONG_SECRET_KEY")

if not GONG_ACCESS_KEY or not GONG_SECRET_KEY:
    print("❌ Error: GONG_ACCESS_KEY and GONG_SECRET_KEY must be set in .env")
    exit(1)

# Gong API base URL
GONG_API_BASE = "https://api.gong.io/v2"

# Auth header
def get_auth_header():
    import base64
    credentials = f"{GONG_ACCESS_KEY}:{GONG_SECRET_KEY}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}

def get_calls(from_date=None, to_date=None, limit=50):
    """Fetch list of calls from Gong."""
    url = f"{GONG_API_BASE}/calls"

    params = {}
    if from_date:
        params["fromDateTime"] = from_date.isoformat() + "Z"
    if to_date:
        params["toDateTime"] = to_date.isoformat() + "Z"

    headers = get_auth_header()
    headers["Content-Type"] = "application/json"

    # Gong uses POST for listing calls with a filter body
    body = {
        "filter": {
            "fromDateTime": from_date.isoformat() + "Z" if from_date else None,
            "toDateTime": to_date.isoformat() + "Z" if to_date else None,
        },
        "cursor": None
    }

    # Remove None values
    body["filter"] = {k: v for k, v in body["filter"].items() if v is not None}

    try:
        response = requests.post(f"{url}/extensive", headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        calls = data.get("calls", [])
        print(f"📞 Found {len(calls)} calls")
        return calls[:limit]
    except requests.exceptions.HTTPError as e:
        print(f"❌ API Error: {e}")
        print(f"   Response: {e.response.text if e.response else 'No response'}")
        return []

def get_call_transcript(call_id):
    """Fetch transcript for a specific call."""
    url = f"{GONG_API_BASE}/calls/transcript"
    headers = get_auth_header()
    headers["Content-Type"] = "application/json"

    body = {"filter": {"callIds": [call_id]}}

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        transcripts = data.get("callTranscripts", [])
        if transcripts:
            return transcripts[0]
        return None
    except requests.exceptions.HTTPError as e:
        print(f"   ⚠️ Could not get transcript for {call_id}: {e}")
        return None

def transcript_to_chatml(transcript_data, call_info=None):
    """Convert Gong transcript to ChatML format.

    Gong transcript structure (per API docs):
    {
        "callId": "...",
        "transcript": [
            {
                "speakerId": "...",
                "topic": "...",
                "sentences": [{"start": 123, "end": 456, "text": "..."}]
            }
        ]
    }
    """

    SYSTEM_PROMPT = """You are a top-performing sales development representative (SDR) for Coperniq, a construction management software built specifically for MEP contractors (HVAC, Plumbing, Electrical, Solar).

Key Coperniq differentiators:
1. ASSET LIFECYCLE TRACKING: When a job closes in other tools, the record goes cold. In Coperniq, the ASSET lives forever.
2. REAL-TIME ENERGY MONITORING: Systems dashboard shows status (green/yellow/red).
3. AI-NATIVE FEATURES: Copilot for plain English queries, AI invoicing (saves 25+ hours/month).
4. 30-50% LOWER COST than ServiceTitan, Procore.

Your style: Consultative, not pushy. Ask discovery questions. Use the Challenger Sale approach."""

    examples = []

    # Get transcript segments (each segment has speakerId and sentences array)
    segments = transcript_data.get("transcript", [])
    if not segments:
        return examples

    # Flatten to list of (speakerId, text) from nested structure
    all_sentences = []
    for segment in segments:
        speaker_id = segment.get("speakerId", "unknown")
        sentences = segment.get("sentences", [])
        for sentence in sentences:
            text = sentence.get("text", "").strip()
            if text:
                all_sentences.append({"speakerId": speaker_id, "text": text})

    if not all_sentences:
        return examples

    # Group by speaker turns
    current_speaker = None
    current_text = []
    turns = []

    for sentence in all_sentences:
        speaker = sentence.get("speakerId", "unknown")
        text = sentence.get("text", "").strip()

        if speaker != current_speaker:
            if current_text:
                turns.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)

    # Don't forget last turn
    if current_text:
        turns.append({
            "speaker": current_speaker,
            "text": " ".join(current_text)
        })

    # Identify which speaker is the rep vs prospect
    # Usually the rep speaks first and more
    speaker_counts = {}
    for turn in turns:
        sid = turn["speaker"]
        speaker_counts[sid] = speaker_counts.get(sid, 0) + len(turn["text"])

    # Assume rep talks more (giving info)
    sorted_speakers = sorted(speaker_counts.items(), key=lambda x: -x[1])
    rep_speaker = sorted_speakers[0][0] if sorted_speakers else None

    # Create training examples from conversation pairs
    for i in range(len(turns) - 1):
        current = turns[i]
        next_turn = turns[i + 1]

        # If prospect asks, rep responds
        if current["speaker"] != rep_speaker and next_turn["speaker"] == rep_speaker:
            prospect_text = current["text"]
            rep_text = next_turn["text"]

            # Filter: need reasonable length
            if len(prospect_text) > 20 and len(rep_text) > 50:
                example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prospect_text},
                        {"role": "assistant", "content": rep_text}
                    ],
                    "_metadata": {
                        "source": "gong",
                        "call_id": transcript_data.get("callId", "unknown"),
                        "category": "gong_real"
                    }
                }
                examples.append(example)

    return examples

def main():
    parser = argparse.ArgumentParser(description="Pull Gong calls for training")
    parser.add_argument("--days", type=int, default=90, help="Days back to search")
    parser.add_argument("--limit", type=int, default=50, help="Max calls to process")
    parser.add_argument("--output", type=str, default="gong_training_data.jsonl", help="Output file")
    args = parser.parse_args()

    print("="*60)
    print("Gong Training Data Extractor")
    print("="*60)

    # Calculate date range
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=args.days)

    print(f"\n📅 Date range: {from_date.date()} to {to_date.date()}")
    print(f"📊 Max calls: {args.limit}")

    # Get calls
    print("\n🔍 Fetching calls from Gong...")
    calls = get_calls(from_date, to_date, args.limit)

    if not calls:
        print("❌ No calls found. Check your API credentials and date range.")
        return

    # Process each call
    all_examples = []
    for i, call in enumerate(calls):
        # Call ID is nested in metaData.id per Gong API spec
        meta = call.get("metaData", {})
        call_id = meta.get("id")
        title = meta.get("title", "Untitled")
        print(f"\n[{i+1}/{len(calls)}] Processing: {title[:50]}...")

        transcript = get_call_transcript(call_id)
        if transcript:
            examples = transcript_to_chatml(transcript, call)
            all_examples.extend(examples)
            print(f"   ✅ Extracted {len(examples)} training examples")
        else:
            print(f"   ⚠️ No transcript available")

    # Save to file
    output_path = Path(__file__).parent / args.output
    with open(output_path, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print("\n" + "="*60)
    print(f"✅ DONE! Extracted {len(all_examples)} training examples")
    print(f"📁 Saved to: {output_path}")
    print("="*60)

    print(f"\nNext: Merge with existing data and retrain:")
    print(f"  cat sft_training_data.jsonl {args.output} > combined_training.jsonl")

if __name__ == "__main__":
    main()
