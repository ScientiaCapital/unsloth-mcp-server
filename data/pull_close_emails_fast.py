#!/usr/bin/env python3
"""
Fast Close CRM Email Extraction (No Enrichment)
===============================================
Extracts emails WITHOUT lead/contact enrichment for speed.
Uses sender/recipient info directly from email data.

Usage:
    python pull_close_emails_fast.py --days 180
"""

import os
import json
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment variables from coperniq-forge
load_dotenv(Path(__file__).parent.parent.parent / 'coperniq-forge' / '.env')

CLOSE_API_KEY = os.getenv('CLOSE_API_KEY')
BASE_URL = 'https://api.close.com/api/v1'

# Users to filter for
TARGET_USERS = ['tim@coperniq.io', 'abdullah', 'max']

def get_auth():
    if not CLOSE_API_KEY:
        raise ValueError("CLOSE_API_KEY not found in .env")
    return (CLOSE_API_KEY, '')


def fetch_emails(days_back=180):
    """Fetch outgoing emails from Close CRM."""
    since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    params = {
        'date_created__gt': since_date,
        '_limit': 100,
        '_skip': 0,
        'direction': 'outgoing',
        '_fields': 'id,subject,body_text,sender,to,date_sent,opens,status'
    }

    all_emails = []
    page = 0

    while True:
        try:
            response = requests.get(
                f'{BASE_URL}/activity/email/',
                auth=get_auth(),
                params=params
            )
            response.raise_for_status()
            data = response.json()

            emails = data.get('data', [])
            if not emails:
                break

            all_emails.extend(emails)

            if not data.get('has_more', False):
                break

            params['_skip'] += 100
            page += 1
            time.sleep(0.1)  # Light rate limiting

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break

    return all_emails


def filter_by_sender_or_recipient(emails, target_users):
    """Filter emails by sender OR recipient name/email."""
    filtered = []
    for email in emails:
        sender = (email.get('sender') or '').lower()
        to_list = email.get('to') or []
        recipients = ' '.join(to_list).lower() if to_list else ''

        # Check if any target user is in sender OR recipient
        if any(t.lower() in sender or t.lower() in recipients for t in target_users):
            filtered.append(email)
    return filtered


def format_to_chatml(email):
    """Format email to ChatML training format (no enrichment needed)."""
    body = (email.get('body_text') or '').strip()
    subject = email.get('subject', '')
    to_addr = email.get('to', [''])[0] if email.get('to') else ''

    # Skip empty or short emails
    if not body or len(body) < 50:
        return None

    # Build user prompt
    context_parts = []
    if subject:
        context_parts.append(f"Subject: {subject}")
    if to_addr:
        context_parts.append(f"To: {to_addr}")

    context = '\n'.join(context_parts) if context_parts else ''

    user_prompt = f"""Write a sales email for Coperniq (construction management software for MEP contractors - HVAC, Plumbing, Electrical).

{context}""".strip()

    return {
        'messages': [
            {'role': 'system', 'content': 'You are a top-performing SDR for Coperniq, construction software for MEP contractors. Key differentiators: Asset lifecycle tracking, real-time monitoring, AI features, 30-50% lower cost than ServiceTitan.'},
            {'role': 'user', 'content': user_prompt},
            {'role': 'assistant', 'content': body}
        ],
        '_metadata': {
            'source': 'close_email',
            'email_id': email.get('id'),
            'subject': subject,
            'sender': email.get('sender'),
            'date_sent': email.get('date_sent'),
            'opens': email.get('opens', 0),
            'category': 'close_real'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Fast Close CRM email extraction')
    parser.add_argument('--days', type=int, default=180, help='Days to look back')
    parser.add_argument('--output', type=str, default='close_emails_training.jsonl', help='Output file')
    parser.add_argument('--all-senders', action='store_true', help='Include all senders (not just Tim/Abdullah/Max)')
    args = parser.parse_args()

    print("=" * 60)
    print("FAST CLOSE CRM EMAIL EXTRACTION")
    print("=" * 60)
    print(f"Days: {args.days}")
    print(f"Filter: {'All senders' if args.all_senders else 'Tim, Abdullah, Max only'}")
    print()

    # Fetch emails
    print("Fetching outgoing emails...")
    emails = fetch_emails(args.days)
    print(f"Found {len(emails)} outgoing emails")

    # Filter by sender/recipient if needed
    if not args.all_senders:
        emails = filter_by_sender_or_recipient(emails, TARGET_USERS)
        print(f"Filtered to {len(emails)} emails from/to target users")

    # Format to ChatML
    training_data = []
    for email in tqdm(emails, desc="Formatting"):
        formatted = format_to_chatml(email)
        if formatted:
            training_data.append(formatted)

    # Save
    output_path = Path(__file__).parent / args.output
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')

    print()
    print("=" * 60)
    print(f"DONE! Extracted {len(training_data)} training examples")
    print(f"Saved to: {output_path}")
    print("=" * 60)

    # Stats
    if training_data:
        avg_len = sum(len(d['messages'][2]['content']) for d in training_data) / len(training_data)
        # Opens can be a list or int
        total_opens = sum(
            len(d['_metadata'].get('opens', [])) if isinstance(d['_metadata'].get('opens'), list)
            else (d['_metadata'].get('opens') or 0)
            for d in training_data
        )
        print(f"\nStats:")
        print(f"  Avg email length: {avg_len:.0f} chars")
        print(f"  Emails with opens: {total_opens}")

    print(f"\nNext: Merge with combined_training.jsonl")
    print(f"  cat combined_training.jsonl {args.output} > all_training.jsonl")


if __name__ == '__main__':
    main()
