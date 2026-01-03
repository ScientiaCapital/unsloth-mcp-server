#!/usr/bin/env python3
"""
Generate SFT Training Dataset for Coperniq Sales Agent
========================================================
Combines data from:
- grpo_prompts.json (50 objection handling prompts)
- battle-cards.json (competitor battle cards)
- COMPETITIVE_ANALYSIS.md (detailed competitor info)
- objection-handlers.md (5 objection examples)

Output: ChatML format JSONL for Unsloth SFT training

Usage:
    python generate_sft_dataset.py --output training_data.jsonl
"""

import json
import random
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
GRPO_PROMPTS = BASE_DIR / "models/configs/grpo/grpo_prompts.json"
BATTLE_CARDS = Path("/Users/tmk/tk_projects/coperniq-battle-cards/battle-cards.json")

# Coperniq context for system prompt
SYSTEM_PROMPT = """You are a top-performing sales development representative (SDR) for Coperniq, a construction management software built specifically for MEP contractors (HVAC, Plumbing, Electrical, Solar).

Key Coperniq differentiators:
1. ASSET LIFECYCLE TRACKING: When a job closes in other tools, the record goes cold. In Coperniq, the ASSET lives forever - track production, service history, and O&M billing for 25+ years.
2. REAL-TIME ENERGY MONITORING: Systems dashboard shows status (green/yellow/red), peak power, performance trends. Know when systems underperform BEFORE customers call.
3. AI-NATIVE FEATURES: Copilot for plain English queries, AI invoicing (saves 25+ hours/month), workflow builder.
4. UNIFIED PLATFORM: Sales + Service in one system, not separate modules.
5. 30-50% LOWER COST: Compared to ServiceTitan, Procore, etc.

Target customer: $5-50M multi-trade, asset-centric, self-performing contractors.

Your style: Consultative, not pushy. Ask discovery questions. Use the Challenger Sale approach when needed."""

# Expert responses for GRPO prompts based on battle cards
EXPERT_RESPONSES = {
    "objection_handling": {
        "too_small": "What's your annual revenue? And how many trades do you perform? If you're in the $5-50M range with 2+ trades, that's actually our sweet spot. We're built for contractors your size who've outgrown spreadsheets but don't need enterprise tools that cost $500/user.",
        "competitor": "Got it - what are you using for asset tracking after project completion? That's usually the gap we fill. We're not replacing your whole stack - we're adding the asset-centric layer that tools like {competitor} don't have. In their system, when a job closes, the record goes cold. In Coperniq, the ASSET lives forever.",
        "expensive": "I understand cost is a factor. Let me ask - what's the cost of losing an O&M customer because you didn't know their system was underperforming? Or how much time does your team spend on manual invoicing? Our AI invoicing alone saves 25+ hours per month. That's real ROI.",
        "think_about_it": "Totally understand. What would need to be true for you to move forward today? [Listen carefully, then address that specific concern] Let me show you that specific feature - if that solves your concern, we can get you started this week.",
        "team_adoption": "That's a real concern I hear often. Here's how we solve it: Coperniq Academy. Your team goes through role-based training - field techs learn mobile, office staff learn invoicing, managers learn analytics. Certified in days, not weeks. Want me to show you what the training looks like?",
        "no_time": "I get it - you're busy running a business. Here's the thing: our implementation takes 2 weeks, not 3 months like ServiceTitan. And we designed Coperniq for contractors who don't have time for complicated software. What if I showed you just the one feature that would save you the most time?",
        "spreadsheets_work": "Spreadsheets work until they don't. What happens when you need to pull service history for a system you installed 2 years ago? Or when a customer calls about their solar production and you have to dig through emails? The question isn't whether spreadsheets work today - it's whether they'll support your growth to the next level.",
    },
    "discovery": {
        "pain_points": "Great question. MEP contractors typically struggle with: 1) Asset tracking after project closes - no way to see equipment health or service history, 2) Manual invoicing eating up 20+ hours a month, 3) No visibility into which systems are underperforming until customers complain, 4) Using 5+ different tools that don't talk to each other. Which of these hits closest to home for you?",
        "qualification": "To understand if we're a good fit, I'd ask: 1) What's your annual revenue? (sweet spot is $5-50M), 2) How many trades do you perform? (we're built for 2+), 3) Do you install equipment AND service it afterwards? (asset lifecycle is our superpower), 4) Are you self-performing or managing subs? (we're built for your own crews).",
        "cold_call": "Here's an opener that works: 'Hi [Name], I'm [Your Name] from Coperniq. We're the construction management platform built specifically for MEP contractors like [Company]. I noticed you do both installations and service - quick question: when you need to service equipment you installed 2 years ago, how do you find that info?' Then listen.",
        "mirroring": "Mirroring and labeling are powerful. Mirroring: repeat the last 3 words they said as a question. 'We've been looking at software...' → 'Looking at software?' Labeling: name their emotion. 'It sounds like you're frustrated with...' or 'It seems like you've been burned by software before...' These show you're listening and build trust.",
    },
    "closing": {
        "ready_signs": "Signs a prospect is ready: 1) They ask about pricing/contract terms unprompted, 2) They start using 'when' instead of 'if' (When we implement...), 3) They bring in additional stakeholders to the call, 4) They ask about implementation timeline, 5) They reference specific workflows they want to see. When you see 2+ of these, it's time to ask for the close.",
        "urgency": "Create urgency without being pushy: 1) Tie to their business events - 'You mentioned busy season starts in March, to be fully ramped up by then we'd need to start in January', 2) Quantify cost of inaction - 'Every month of manual invoicing is 25 hours. That's $1,250 in labor', 3) Share success stories - 'Similar contractors saw 30% efficiency gains in the first quarter'.",
        "pilot": "Great question on structuring pilots: 1) Define success metrics upfront - what would make you say 'yes this works', 2) Set a time limit - 30 days is ideal, enough to see value but short enough to maintain urgency, 3) Assign an internal champion who owns the pilot, 4) Schedule weekly check-ins, 5) Pre-agree on the path to full rollout if successful. Pilots work best when they're small but meaningful - one full project workflow rather than just 'poking around'.",
        "silent": "When a demo goes well but prospect goes silent: 1) Wait 2-3 business days before following up, 2) Lead with value, not 'just checking in' - share a relevant case study or new feature, 3) Reference something specific from the demo - 'You mentioned the asset tracking would help with your O&M contracts', 4) Propose next steps - 'Would a 15-minute call to review the proposal work this Thursday at 2?', 5) If still silent after 3 touchpoints, send a breakup email - sometimes that gets a response.",
    },
    "mep_specific": {
        "servicetitan": "Great question on positioning against ServiceTitan. Here's the key: ServiceTitan is built for pure service companies - residential HVAC/plumbing/electrical. But you install equipment AND maintain it. Ask them: 'When a solar system starts underperforming, how does ServiceTitan alert you?' Answer: It doesn't. You find out when the customer calls angry. In Coperniq, the ASSET lives forever - we track production, service history, and O&M billing for 25 years. Plus we're 30-50% less expensive.",
        "job_costing": "For plumbing contractors, job costing in Coperniq works like this: 1) Set up labor rates and material markups, 2) Track actual time vs estimated on each job phase, 3) See real-time margin as work progresses, 4) AI Copilot can show you 'projects losing money this month' with one question, 5) Integrate with QuickBooks in real-time (not batch like ServiceTitan). The key is catching margin problems DURING the job, not after you invoice.",
        "mobile_value": "The mobile app value for field crews: 1) Check in/out with GPS - no more disputed hours, 2) Access complete equipment history on-site - what was installed, past service, customer notes, 3) Take photos that auto-attach to the asset record (not buried in email), 4) Create service tickets in 30 seconds, 5) View today's schedule with turn-by-turn navigation. The pitch: 'Your techs spend 20% of time on paperwork. What if we cut that to 5%?'",
    }
}

def load_grpo_prompts():
    """Load GRPO prompts from JSON file."""
    with open(GRPO_PROMPTS) as f:
        data = json.load(f)
    return data.get("prompts", [])

def load_battle_cards():
    """Load battle cards from JSON file."""
    with open(BATTLE_CARDS) as f:
        return json.load(f)

def generate_response_for_prompt(prompt_data):
    """Generate an expert response for a given prompt based on category."""
    category = prompt_data.get("category", "general")
    prompt_text = prompt_data.get("prompt", "")

    # Find best matching response template
    response = None

    if category == "objection_handling":
        if "too small" in prompt_text.lower():
            response = EXPERT_RESPONSES["objection_handling"]["too_small"]
        elif "already have" in prompt_text.lower() or "currently" in prompt_text.lower():
            response = EXPERT_RESPONSES["objection_handling"]["competitor"]
        elif "expensive" in prompt_text.lower() or "price" in prompt_text.lower() or "cost" in prompt_text.lower():
            response = EXPERT_RESPONSES["objection_handling"]["expensive"]
        elif "think about it" in prompt_text.lower() or "think it over" in prompt_text.lower():
            response = EXPERT_RESPONSES["objection_handling"]["think_about_it"]
        elif "won't adopt" in prompt_text.lower() or "team" in prompt_text.lower():
            response = EXPERT_RESPONSES["objection_handling"]["team_adoption"]
        elif "too busy" in prompt_text.lower() or "no time" in prompt_text.lower():
            response = EXPERT_RESPONSES["objection_handling"]["no_time"]
        elif "spreadsheet" in prompt_text.lower() or "works fine" in prompt_text.lower():
            response = EXPERT_RESPONSES["objection_handling"]["spreadsheets_work"]
        else:
            # Generic objection handling
            response = """I hear you, and that's a valid concern. Let me address it directly: [specific concern].

Here's what I've seen work for contractors in similar situations: [relevant case study or feature].

What would help you feel more confident about this? Is it seeing a demo of that specific workflow, or talking to a reference customer who had the same concern?"""

    elif category == "discovery":
        if "pain point" in prompt_text.lower():
            response = EXPERT_RESPONSES["discovery"]["pain_points"]
        elif "qualif" in prompt_text.lower():
            response = EXPERT_RESPONSES["discovery"]["qualification"]
        elif "cold call" in prompt_text.lower() or "opening" in prompt_text.lower():
            response = EXPERT_RESPONSES["discovery"]["cold_call"]
        elif "mirror" in prompt_text.lower() or "label" in prompt_text.lower():
            response = EXPERT_RESPONSES["discovery"]["mirroring"]
        else:
            response = EXPERT_RESPONSES["discovery"]["pain_points"]

    elif category == "closing":
        if "ready" in prompt_text.lower() or "signs" in prompt_text.lower():
            response = EXPERT_RESPONSES["closing"]["ready_signs"]
        elif "urgency" in prompt_text.lower():
            response = EXPERT_RESPONSES["closing"]["urgency"]
        elif "pilot" in prompt_text.lower():
            response = EXPERT_RESPONSES["closing"]["pilot"]
        elif "silent" in prompt_text.lower() or "ghost" in prompt_text.lower():
            response = EXPERT_RESPONSES["closing"]["silent"]
        else:
            response = EXPERT_RESPONSES["closing"]["ready_signs"]

    elif category == "mep_specific":
        if "servicetitan" in prompt_text.lower():
            response = EXPERT_RESPONSES["mep_specific"]["servicetitan"]
        elif "job cost" in prompt_text.lower() or "plumbing" in prompt_text.lower():
            response = EXPERT_RESPONSES["mep_specific"]["job_costing"]
        elif "mobile" in prompt_text.lower() or "field" in prompt_text.lower():
            response = EXPERT_RESPONSES["mep_specific"]["mobile_value"]
        else:
            response = EXPERT_RESPONSES["mep_specific"]["servicetitan"]

    elif category == "gong_real":
        # Real Gong transcript scenarios - generate contextual responses
        if "salesforce" in prompt_text.lower():
            response = """You've hit on exactly why contractors move to us. Salesforce is a general CRM - it's designed for selling, not for what happens after the sale. The 'band-aid' PM capabilities you've built are costing you time and probably causing data silos.

Here's what Coperniq does differently:
1. Sales and Operations in ONE system - no sync issues
2. Project workflows designed for MEP from day one - permits, inspections, NTP, PTO
3. Asset lifecycle after install - track equipment for 25 years
4. AI Copilot for instant queries vs running reports

Would it help to see a side-by-side of how a project flows through both systems?"""
        elif "pricing" in prompt_text.lower() or "per user" in prompt_text.lower() or "expensive" in prompt_text.lower():
            response = """Great pricing question. Let me be direct: we charge per user, not per project or per square foot. A project is an active job with its own timeline and milestones. Lead records are separate and don't count toward limits.

For your use case with clients who have many properties, here's how it works: you set up the CLIENT once, then add SITES underneath, and PROJECTS under each site. The hierarchy is flexible. We've worked with portfolio managers who have thousands of assets - it doesn't blow up the cost.

Want me to show you the hierarchy setup? It takes about 2 minutes to understand."""
        elif "implementation" in prompt_text.lower() or "how long" in prompt_text.lower():
            response = """Implementation timeline: 2-3 weeks for most teams, not months. Here's the breakdown:

Week 1: Data migration, user setup, basic workflow config
Week 2: Training (Coperniq Academy, role-based), testing
Week 3: Go-live with support, refinements

We assign you a dedicated CSM who handles onboarding. Unlike ServiceTitan (3+ months) or Procore (enterprise sales cycle), we're designed for fast deployment.

The key is we don't try to customize everything before you start - we get you using the core product quickly, then iterate. What's your target go-live date?"""
        else:
            # Generic gong response
            response = """That's a great question. Let me address it directly based on what I've seen with similar contractors.

[Specific answer tailored to their situation]

The key insight is: Coperniq is built for contractors who install AND service equipment. We track the ASSET for its entire lifecycle - not just until the project closes.

Would it help to see how that specific workflow works in a demo? Or would you prefer to connect with a reference customer who had a similar question?"""

    else:
        # Generic fallback
        response = """Great question. Let me share what I've seen work well.

[Specific answer based on Coperniq's approach]

The key differentiator is our asset-centric architecture. Where other tools end at project completion, Coperniq tracks equipment for its entire lifecycle.

What would be most helpful - a demo of that specific feature, or connecting you with a customer in a similar situation?"""

    return response

def create_chatml_entry(user_content, assistant_content, metadata=None):
    """Create a ChatML format training entry."""
    entry = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }
    if metadata:
        entry["_metadata"] = metadata
    return entry

def generate_competitor_qa(battle_cards):
    """Generate Q&A pairs from battle card data."""
    examples = []

    for competitor, data in battle_cards.get("competitors", {}).items():
        # Opener Q&A
        opener = data.get("opener", "")
        if opener:
            examples.append(create_chatml_entry(
                f"How do I start a conversation with a prospect who uses {competitor.title()}?",
                opener,
                {"category": "competitor", "competitor": competitor}
            ))

        # Killer question Q&A
        killer_q = data.get("killer_question", "")
        killer_a = data.get("killer_answer", "")
        if killer_q and killer_a:
            examples.append(create_chatml_entry(
                f"What's the killer question to ask a {competitor.title()} user?",
                f"Ask them: \"{killer_q}\"\n\nTheir answer: {killer_a}\n\nThis exposes the key gap that Coperniq fills - asset lifecycle tracking and proactive monitoring.",
                {"category": "competitor", "competitor": competitor}
            ))

        # Value props
        for prop_name, prop_value in data.get("value_props", {}).items():
            examples.append(create_chatml_entry(
                f"How is Coperniq's {prop_name.replace('_', ' ')} better than {competitor.title()}?",
                prop_value,
                {"category": "value_prop", "competitor": competitor}
            ))

        # Gaps
        gaps = data.get("gaps", [])
        if gaps:
            gaps_text = ", ".join(gaps)
            examples.append(create_chatml_entry(
                f"What are {competitor.title()}'s main limitations compared to Coperniq?",
                f"{competitor.title()}'s key gaps that Coperniq addresses:\n\n" + "\n".join([f"- {gap}" for gap in gaps]) + "\n\nThese are the areas where we consistently win deals against them.",
                {"category": "gaps", "competitor": competitor}
            ))

    return examples

def generate_objection_qa(battle_cards):
    """Generate objection handling Q&A pairs."""
    examples = []

    for trigger, data in battle_cards.get("objection_handlers", {}).items():
        trigger_text = data.get("trigger", "")
        response = data.get("response", "")

        if trigger_text and response:
            examples.append(create_chatml_entry(
                f"The prospect says they're \"{trigger_text}\". How do I respond?",
                response,
                {"category": "objection_handling", "trigger": trigger}
            ))

    return examples

def generate_ai_features_qa(battle_cards):
    """Generate AI features Q&A pairs."""
    examples = []

    for feature, data in battle_cards.get("ai_features", {}).items():
        pitch = data.get("pitch", "")
        demo = data.get("demo", [])
        roi = data.get("roi", "")

        if pitch:
            demo_text = f"\n\nDemo example: {demo[0]}" if demo else ""
            roi_text = f"\n\nROI: {roi}" if roi else ""
            examples.append(create_chatml_entry(
                f"How do I pitch Coperniq's {feature.replace('_', ' ')} feature?",
                pitch + demo_text + roi_text,
                {"category": "ai_features", "feature": feature}
            ))

    return examples

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate SFT training dataset")
    parser.add_argument("--output", type=str, default="sft_training_data.jsonl", help="Output file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    args = parser.parse_args()

    print("="*60)
    print("Coperniq SFT Training Data Generator")
    print("="*60)

    all_examples = []

    # 1. Load and process GRPO prompts
    print("\n1. Processing GRPO prompts...")
    grpo_prompts = load_grpo_prompts()
    for prompt_data in grpo_prompts:
        prompt_text = prompt_data.get("prompt", "")
        response = generate_response_for_prompt(prompt_data)

        example = create_chatml_entry(
            prompt_text,
            response,
            {"category": prompt_data.get("category"), "source": "grpo_prompts"}
        )
        all_examples.append(example)
    print(f"   Generated {len(grpo_prompts)} examples from GRPO prompts")

    # 2. Load and process battle cards
    print("\n2. Processing battle cards...")
    battle_cards = load_battle_cards()

    competitor_examples = generate_competitor_qa(battle_cards)
    all_examples.extend(competitor_examples)
    print(f"   Generated {len(competitor_examples)} competitor Q&A examples")

    objection_examples = generate_objection_qa(battle_cards)
    all_examples.extend(objection_examples)
    print(f"   Generated {len(objection_examples)} objection handling examples")

    ai_examples = generate_ai_features_qa(battle_cards)
    all_examples.extend(ai_examples)
    print(f"   Generated {len(ai_examples)} AI features examples")

    # 3. Core differentiator example
    core_diff = battle_cards.get("core_differentiator", "")
    if core_diff:
        all_examples.append(create_chatml_entry(
            "What's Coperniq's main differentiator against competitors?",
            core_diff + "\n\nThis is the key insight: project-centric tools treat every job as a one-time event. Coperniq is asset-centric - we understand you're building a 25-year relationship with every piece of equipment you install.",
            {"category": "differentiator", "source": "battle_cards"}
        ))

    # Shuffle for training variety
    random.shuffle(all_examples)

    # Apply limit if specified
    if args.limit:
        all_examples = all_examples[:args.limit]

    # Save to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print("\n" + "="*60)
    print(f"TOTAL: {len(all_examples)} training examples generated")
    print(f"Output: {output_path}")
    print("="*60)

    # Category breakdown
    categories = {}
    for ex in all_examples:
        cat = ex.get("_metadata", {}).get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nNext step: Use this data for SFT training with Unsloth")
    print(f"  python train_sft.py --data {output_path}")

if __name__ == "__main__":
    main()
