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
        "solar": "For solar contractors, Coperniq is purpose-built for your workflow: 1) Track system performance in real-time - see production vs expected, 2) Automated alerts when systems underperform BEFORE customers notice, 3) O&M billing tied to actual production data, 4) 25-year asset lifecycle tracking - know when inverters need replacement, 5) NTP-to-PTO milestone tracking with inspection scheduling. The pitch: 'How do you currently know when a system you installed 3 years ago starts underperforming?' We solve that.",
        "hvac": "For HVAC contractors, here's how Coperniq differs: 1) Equipment-centric, not job-centric - track every unit you install for its lifetime, 2) Service history attached to the ASSET, not buried in job records, 3) Preventive maintenance scheduling based on equipment age and usage, 4) Real-time alerts for commercial clients when systems show issues, 5) O&M contract management with automated invoicing. The killer question: 'When a compressor fails on a unit you installed 5 years ago, how fast can you pull up the full history?'",
    },
    "email_templates": {
        "cold_outreach": "Subject: Quick question about your [HVAC/Solar/Plumbing] service operations\n\nHi [Name],\n\nI noticed [Company] does both installations and ongoing service work. Quick question:\n\nWhen you need to service equipment you installed 2-3 years ago, how quickly can your team pull up the full history - what was installed, past service calls, any issues?\n\nIf the answer involves digging through emails or spreadsheets, that's exactly the problem Coperniq solves. We're the first platform built specifically for MEP contractors who install AND service equipment.\n\nWorth a 15-minute call this week?\n\n[Your name]",
        "follow_up": "Subject: Re: Following up on our conversation\n\nHi [Name],\n\nWanted to follow up on our conversation about [specific topic discussed]. You mentioned [their key pain point] - I pulled together a quick case study of how [similar company] solved that exact problem.\n\n[Key result they achieved in 1 sentence]\n\nWould it help to do a quick 15-minute deep dive on that specific workflow? I can show you exactly how it would work for [their company].\n\nFree this [day/time]?",
        "demo_recap": "Subject: Coperniq demo recap + next steps\n\nHi [Name],\n\nThanks for taking time today to see Coperniq. Here's a quick recap of what we covered:\n\n✅ [Feature 1 they were excited about]\n✅ [Feature 2 that solved their pain point]\n✅ [Feature 3 - the 'aha' moment]\n\nNext steps we discussed:\n1. [Specific next step - trial, pricing review, intro to team]\n2. [Timeline they mentioned]\n\nI'm attaching [relevant resource - case study, pricing, etc.]. Let me know if you have any questions before our next call on [date].\n\nLooking forward to getting you started!",
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
        elif "solar" in prompt_text.lower():
            response = EXPERT_RESPONSES["mep_specific"]["solar"]
        elif "hvac" in prompt_text.lower():
            response = EXPERT_RESPONSES["mep_specific"]["hvac"]
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

    # 4. Email template examples
    print("\n3. Adding email template examples...")
    email_examples = [
        create_chatml_entry(
            "Write a cold outreach email to an HVAC contractor",
            EXPERT_RESPONSES["email_templates"]["cold_outreach"].replace("[HVAC/Solar/Plumbing]", "HVAC"),
            {"category": "email", "type": "cold_outreach"}
        ),
        create_chatml_entry(
            "Write a cold email to a solar installer",
            EXPERT_RESPONSES["email_templates"]["cold_outreach"].replace("[HVAC/Solar/Plumbing]", "Solar"),
            {"category": "email", "type": "cold_outreach"}
        ),
        create_chatml_entry(
            "How do I follow up after a demo?",
            EXPERT_RESPONSES["email_templates"]["demo_recap"],
            {"category": "email", "type": "demo_recap"}
        ),
        create_chatml_entry(
            "Write a follow-up email for a prospect who hasn't responded",
            EXPERT_RESPONSES["email_templates"]["follow_up"],
            {"category": "email", "type": "follow_up"}
        ),
    ]
    all_examples.extend(email_examples)
    print(f"   Generated {len(email_examples)} email template examples")

    # 5. Additional scenario-based examples
    print("\n4. Adding scenario-based examples...")
    scenario_examples = [
        create_chatml_entry(
            "A prospect says 'We tried software before and it was a disaster.' How do I respond?",
            "I totally understand that concern - bad software implementations can set a team back months. Let me ask: what specifically went wrong? [Listen carefully]\n\nBased on what you're describing, here's how Coperniq is different:\n1. 2-week implementation, not 3 months - we get you using core features fast\n2. Dedicated CSM who handles onboarding personally\n3. Role-based training through Coperniq Academy - your team gets certified in days\n4. We don't try to customize everything upfront - start simple, add complexity as needed\n\nWhat if I showed you our onboarding process? You could judge whether it's different enough to give it another shot.",
            {"category": "objection_handling", "scenario": "past_failure"}
        ),
        create_chatml_entry(
            "How do I handle a prospect who keeps rescheduling demos?",
            "After 2 reschedules, change your approach:\n\n1. Acknowledge their busy schedule: 'I know you're juggling a lot right now'\n2. Reduce the ask: 'Instead of a full demo, would 10 minutes work to see if there's even a fit?'\n3. Add urgency with value: 'I have a case study from [similar company] I think you'd find interesting - happy to send that over either way'\n4. Set a soft deadline: 'I'll check back next week - if the timing isn't right, no problem at all'\n\nIf they keep rescheduling, they're probably not a real opportunity. Send a breakup email: 'It seems like the timing isn't right. I'll check back in 6 months unless you tell me otherwise.' Often this gets a response.",
            {"category": "closing", "scenario": "rescheduling"}
        ),
        create_chatml_entry(
            "What questions should I ask on a discovery call?",
            "Here's my discovery framework for MEP contractors:\n\n1. SITUATION questions:\n   - 'Walk me through how a project flows from lead to completion today'\n   - 'What tools are you using currently?'\n   - 'How many people touch a typical project?'\n\n2. PAIN questions:\n   - 'What happens when you need to service equipment you installed 2 years ago?'\n   - 'How much time does your office spend on invoicing each month?'\n   - 'When was the last time a customer called upset about something you should have caught first?'\n\n3. IMPACT questions:\n   - 'If you could get those 25 hours back each month, what would you do with them?'\n   - 'What would it mean for your business to know about system issues before customers do?'\n\n4. FUTURE questions:\n   - 'Where do you want the business to be in 2 years?'\n   - 'What needs to be true for you to get there?'\n\nListen more than you talk. Their answers tell you exactly how to position Coperniq.",
            {"category": "discovery", "scenario": "framework"}
        ),
        create_chatml_entry(
            "How do I position Coperniq against Procore?",
            "Procore is for large general contractors managing subs across big commercial projects. Here's how to position against them:\n\n1. Their blind spot: 'Procore is project-centric. When the project ends, the record goes cold. Do you track equipment after installation?' (They don't)\n\n2. Size mismatch: 'Procore is built for $100M+ GCs. At your size, you're paying for features you'll never use. We're built specifically for $5-50M specialty contractors.'\n\n3. Price advantage: 'We're typically 30-50% less than Procore, and you get features they don't have - like real-time system monitoring.'\n\n4. Killer question: 'When a piece of equipment you installed starts underperforming, how does Procore alert you?' (It doesn't)\n\nThe key: Procore is great at project management. We're great at asset lifecycle. If they install AND service equipment, we win.",
            {"category": "competitor", "competitor": "procore"}
        ),
        create_chatml_entry(
            "How do I demo Coperniq to an MEP contractor?",
            "Here's my winning demo flow for MEP contractors:\n\n1. START WITH THEIR PAIN (5 min):\n   'You mentioned [their specific pain point from discovery]. Let me show you exactly how we solve that.'\n\n2. ASSET LIFECYCLE DEMO (10 min):\n   - Create an equipment record\n   - Show how it connects to projects, service calls, O&M billing\n   - 'This record lives for 25 years - every touchpoint tracked'\n\n3. REAL-TIME MONITORING (5 min):\n   - Show the systems dashboard (green/yellow/red status)\n   - 'This is how you know about problems before customers call'\n\n4. AI COPILOT (5 min):\n   - Ask natural language questions: 'Which projects are losing money this month?'\n   - Show AI invoicing: 'This saves 25+ hours per month'\n\n5. THEIR WORKFLOW (10 min):\n   - 'Let me show you how [their specific process] would work'\n   - Customize based on discovery\n\n6. CLOSE (5 min):\n   - 'What would need to be true for this to work for your team?'\n   - Address concerns, propose next steps",
            {"category": "demo", "scenario": "mep_contractor"}
        ),
        create_chatml_entry(
            "A prospect asks about integrations with QuickBooks. What do I say?",
            "Great question - QuickBooks integration is one of our most popular features. Here's how it works:\n\n1. REAL-TIME SYNC: Unlike ServiceTitan (batch sync), we sync with QuickBooks in real-time. Invoice in Coperniq, it's in QB immediately.\n\n2. TWO-WAY SYNC: Changes flow both directions. Update a customer in QB, it's updated in Coperniq.\n\n3. WHAT SYNCS:\n   - Customers/vendors\n   - Invoices and payments\n   - Chart of accounts\n   - Job costing data\n\n4. SETUP TIME: About 15 minutes. Our CSM walks you through it during onboarding.\n\n5. COMMON QUESTION: 'Do I have to re-enter everything?' No - we pull your existing data during setup.\n\nWant me to show you the integration settings? Takes about 2 minutes to see how it works.",
            {"category": "features", "feature": "quickbooks"}
        ),
        create_chatml_entry(
            "How do I explain Coperniq's pricing?",
            "Here's how I explain our pricing:\n\n1. STRUCTURE: Per user, per month. Simple, predictable.\n\n2. WHAT'S INCLUDED:\n   - All core features (no feature gating)\n   - Unlimited projects and assets\n   - QuickBooks integration\n   - Coperniq Academy training\n   - Dedicated CSM for onboarding\n\n3. COMPARISON:\n   - 30-50% less than ServiceTitan\n   - Significantly less than Procore\n   - More value than Jobber/Housecall Pro at similar price\n\n4. ROI CONVERSATION:\n   - 'AI invoicing saves 25+ hours/month - what's that worth?'\n   - 'Proactive monitoring prevents 1 angry customer call per week - what's that worth?'\n   - 'Unified system means no double entry - how much time does that save?'\n\n5. HANDLE STICKER SHOCK:\n   - 'What are you paying now?' (Often multiple tools adding up to more)\n   - 'What's the cost of the problems we're solving?' (Lost customers, wasted time)\n\nNever lead with price - lead with value, then price makes sense.",
            {"category": "pricing", "scenario": "explanation"}
        ),
        create_chatml_entry(
            "What's the best way to get a referral from a happy customer?",
            "Here's my referral playbook:\n\n1. TIMING: Ask when they're experiencing a 'win moment' - just saved time, caught an issue early, got a compliment on professionalism.\n\n2. THE ASK:\n   'I'm glad that's working well for you. We're looking to help more contractors like you. Who's one person in your network who's dealing with [the problem we solved for them]?'\n\n3. MAKE IT EASY:\n   - 'Can I draft an intro email for you to send?'\n   - 'Would it be easier if I sent you a link to share?'\n   - 'Would you be open to a 2-minute case study we could share?'\n\n4. RECIPROCATE:\n   - Offer something in return (extended support, feature preview, etc.)\n   - 'If they become a customer, we'll give you a month free'\n\n5. FOLLOW UP:\n   - If they say yes but don't act, gentle reminder in 1 week\n   - 'Did you get a chance to think about who might benefit?'\n\nBest referrals come from customers who are genuinely excited, not just satisfied. Wait for the excitement moment.",
            {"category": "referral", "scenario": "happy_customer"}
        ),
        create_chatml_entry(
            "How do I re-engage a cold lead from 6 months ago?",
            "Re-engagement strategy for cold leads:\n\n1. RESEARCH FIRST:\n   - Check their LinkedIn for company updates\n   - Look for news about their business\n   - Review your notes from last conversation\n\n2. OPENING EMAIL:\n   Subject: 'Quick update since we last spoke'\n   'Hi [Name], it's been about 6 months since we connected. A few things have changed at Coperniq that I thought you'd find interesting: [1-2 new features relevant to them]. Also curious - has anything changed on your end with [the pain point they mentioned]?'\n\n3. IF NO RESPONSE:\n   - Try a different channel (phone, LinkedIn)\n   - Send something of value (case study, industry report)\n   - Reference something specific from your past conversation\n\n4. THE 'BREAKUP' EMAIL:\n   'I've reached out a few times without hearing back. I'm guessing the timing isn't right, which is totally fine. I'll remove you from my follow-up list, but if things change, here's my calendar link: [link]. Best of luck!'\n\nOften the breakup email gets a response when nothing else did.",
            {"category": "outreach", "scenario": "re_engagement"}
        ),
        create_chatml_entry(
            "A prospect wants to see the ROI calculation. What do I show them?",
            "Here's how I build the ROI case:\n\n1. TIME SAVINGS:\n   - AI Invoicing: 25+ hours/month → $1,250/month at $50/hour\n   - No double entry: 10 hours/month → $500/month\n   - Faster service scheduling: 5 hours/month → $250/month\n   - TOTAL TIME: ~$2,000/month in labor savings\n\n2. REVENUE PROTECTION:\n   - Proactive monitoring catches issues before complaints\n   - Every prevented angry customer call = potential $5K+ in retained business\n   - Better O&M billing = 15-20% increase in service revenue\n\n3. OPERATIONAL EFFICIENCY:\n   - One system vs 3-5 tools = fewer subscription costs\n   - Faster onboarding for new hires (days, not weeks)\n   - Better data for bidding = win more profitable jobs\n\n4. THE CALCULATION:\n   Cost of Coperniq: $X/month\n   Time savings: $2,000/month\n   Revenue protection: $5,000+/month (conservative)\n   ROI: 3-10x depending on company size\n\n5. CASE STUDY:\n   '[Similar contractor] saw 30% efficiency gains in first quarter and caught 3 underperforming systems before customers noticed.'\n\nAlways tie ROI to THEIR specific pain points from discovery.",
            {"category": "roi", "scenario": "calculation"}
        ),
        create_chatml_entry(
            "How do I handle a multi-stakeholder deal?",
            "Multi-stakeholder deals require strategy:\n\n1. MAP THE DECISION:\n   - Who has budget authority? (Usually owner/CFO)\n   - Who has operational influence? (Operations manager, project manager)\n   - Who will use it daily? (Field techs, office staff)\n   - Who can block the deal? (IT, resistant users)\n\n2. CHAMPION STRATEGY:\n   - Identify your internal champion early\n   - Arm them with materials to sell internally\n   - Coach them on objections others might raise\n\n3. STAKEHOLDER-SPECIFIC DEMOS:\n   - Owner: ROI, competitive advantage, time to value\n   - Operations: Workflow efficiency, real-time visibility\n   - Field crews: Mobile app, ease of use, fewer headaches\n   - Finance: QuickBooks integration, job costing\n\n4. MEETING MANAGEMENT:\n   - Try to get all stakeholders in one meeting (harder to misalign)\n   - If separate meetings, send recap to everyone\n   - Address concerns immediately (don't let them fester)\n\n5. THE CLOSE:\n   - 'What would each of you need to see to feel confident moving forward?'\n   - Address each concern, then ask for commitment\n\nMulti-stakeholder deals take longer but are stickier once closed.",
            {"category": "sales_process", "scenario": "multi_stakeholder"}
        ),
        create_chatml_entry(
            "What's the elevator pitch for Coperniq?",
            "Here are elevator pitches for different audiences:\n\n30-SECOND VERSION:\n'Coperniq is construction management software built specifically for MEP contractors. Unlike tools that end at project completion, we track equipment for its entire 25-year lifecycle. That means you know when systems underperform before customers call, and you can bill O&M services tied to actual performance data. We're 30-50% less than ServiceTitan and purpose-built for contractors who install AND service equipment.'\n\n10-SECOND VERSION:\n'We're the first construction platform that tracks equipment forever - not just until the project closes. That means you catch problems before customers do and maximize service revenue.'\n\nONE-LINER:\n'Coperniq: Because your job isn't done when the project closes.'\n\nCUSTOMIZED FOR SOLAR:\n'We're how solar contractors track system performance and O&M billing for 25 years - so you know about underperformance before customers do.'",
            {"category": "pitch", "scenario": "elevator"}
        ),
        create_chatml_entry(
            "How do I handle a prospect who says 'Just send me information'?",
            "The 'just send me info' brush-off - here's how to handle it:\n\n1. ACKNOWLEDGE + PIVOT:\n   'Happy to send information. To make sure I send the right stuff, quick question: what's the main thing you're hoping to solve right now?'\n\n2. IF THEY ENGAGE:\n   - Ask 1-2 discovery questions\n   - Then: 'Based on that, let me send you [specific case study]. And would a 15-minute call to walk through it be helpful?'\n\n3. IF THEY DON'T:\n   - Send a concise one-pager, not a 20-page deck\n   - Include a case study relevant to their trade\n   - Add a calendar link: 'If you have questions after reviewing, grab 15 minutes here'\n\n4. FOLLOW UP:\n   - 2-3 days later: 'Did the info I sent answer your questions? Happy to clarify anything over a quick call.'\n   - Reference something specific in what you sent\n\n5. KNOW WHEN TO MOVE ON:\n   - If they don't engage after 2 follow-ups, they weren't a real prospect\n   - Add to nurture sequence, check back in 3 months\n\nThe goal is to turn 'send info' into a real conversation, not just dump documents.",
            {"category": "objection_handling", "scenario": "send_info"}
        ),
        create_chatml_entry(
            "What should I say in my first cold call to an MEP contractor?",
            "Here's my cold call script for MEP contractors:\n\n[OPENER - 10 seconds]\n'Hi [Name], this is [Your Name] from Coperniq. Got 30 seconds?'\n\n[IF YES - THE HOOK]\n'Great. I'm calling because we work with MEP contractors like [Company], and I had a quick question: When you need to service equipment you installed 2 years ago, how do you pull up the history?'\n\n[LISTEN - This is the key]\n- If they say 'spreadsheets/emails/CRM' → 'Yeah, that's exactly what we help with...'\n- If they say 'we have that handled' → 'Good to hear. What are you using?'\n\n[BRIDGE TO VALUE]\n'We're the first platform built specifically for contractors who install AND service equipment. The asset record lives forever - so you know when systems underperform before customers call. Worth a 15-minute look?'\n\n[IF NOT NOW]\n'No problem. When's a better time this week?'\n\n[IF NO]\n'Understood. If I could send one thing that might be helpful, what would it be?'\n\nTips:\n- Stand up while calling (more energy)\n- Smile (they can hear it)\n- Talk slower than feels natural\n- Goal is conversation, not pitch",
            {"category": "cold_calling", "scenario": "first_call"}
        ),
        create_chatml_entry(
            "A prospect says 'I need to talk to my partner/team first.' What do I do?",
            "This is often a soft 'no' or a legitimate need. Here's how to handle both:\n\n1. VALIDATE:\n   'Absolutely - this is a team decision. It's important everyone is aligned.'\n\n2. UNDERSTAND:\n   'Help me understand - what are the main things your partner/team will want to know?'\n\n3. OFFER TO HELP:\n   - 'Would it help if I put together a summary of what we discussed for them?'\n   - 'Would they want to join a quick call? I can answer questions directly.'\n   - 'What concerns do you think they'll have?' (Then address them)\n\n4. SET NEXT STEPS:\n   - 'When are you planning to talk to them?'\n   - 'Let's schedule a follow-up for [2 days after]. That way I can answer any questions that come up.'\n   - Get it on the calendar before you hang up\n\n5. ARM YOUR CHAMPION:\n   - Send a 1-pager they can share\n   - Include the 3 key points that resonated with them\n   - Add a 'frequently asked questions' section\n\n6. FOLLOW UP:\n   - Day after their meeting: 'How did the conversation go?'\n   - Be ready to address new objections\n\nThe goal is to make their internal selling easier and keep momentum.",
            {"category": "objection_handling", "scenario": "need_partner"}
        ),
        create_chatml_entry(
            "What are the top 3 features I should demo for a solar contractor?",
            "For solar contractors, lead with these 3 features:\n\n1. REAL-TIME SYSTEM MONITORING (The 'Wow' Feature)\n   - Show the systems dashboard with green/yellow/red status\n   - Demo: 'See this yellow? This system is producing 15% below expected. You'd know today, not when the customer calls in 3 months.'\n   - Tie to O&M: 'Imagine proactively calling them before they notice'\n\n2. ASSET LIFECYCLE TRACKING (The Differentiator)\n   - Create a solar system record with all components\n   - Show 25-year timeline: 'Install date, inspections, service calls, inverter replacement - all tracked'\n   - Demo: 'In 5 years when the inverter fails, your tech has full history on their phone'\n\n3. O&M BILLING TIED TO PRODUCTION (The ROI)\n   - Show how billing links to actual production data\n   - Demo: 'Your O&M contracts can now be based on performance, not just time'\n   - Tie to revenue: 'Contractors using this see 15-20% higher service revenue'\n\nBONUS IF TIME:\n- NTP-to-PTO milestone tracking\n- AI Copilot: 'Show me systems underperforming this week'\n- Mobile app for field techs\n\nEnd with: 'Which of these would have the biggest impact on your business?'",
            {"category": "demo", "scenario": "solar_contractor"}
        ),
        create_chatml_entry(
            "How do I compete against a lower-priced competitor?",
            "When competing against cheaper alternatives:\n\n1. DON'T LEAD WITH PRICE:\n   'I hear you - price is important. Let me make sure we're comparing apples to apples.'\n\n2. EXPAND THE COMPARISON:\n   - 'What's included in their price? Users? Features? Support?'\n   - 'Do they charge for integrations? Training? Additional modules?'\n   - 'What's their implementation cost and timeline?'\n\n3. QUANTIFY TOTAL COST OF OWNERSHIP:\n   - Cheaper tools often have: Limited features, paid integrations, poor support, longer implementation\n   - Add it up: 'Their $50/user + QB integration + training + support = our $75/user all-in'\n\n4. SHIFT TO VALUE:\n   - 'What's the cost of the problem we're solving?'\n   - 'If Coperniq saves 25 hours/month, that's $1,250 in labor. Does the $25/month difference matter?'\n   - 'What happens if the cheaper tool can't scale with you?'\n\n5. USE THE 'THREE PRICES' FRAME:\n   - 'There's the price you pay today'\n   - 'The price you pay when it doesn't do what you need'\n   - 'And the price you pay to switch again'\n   - 'Which price matters most?'\n\n6. KNOW WHEN TO WALK:\n   If they're purely price shopping, they may not be your customer. Focus on value-conscious buyers.",
            {"category": "objection_handling", "scenario": "price_competition"}
        ),
    ]
    all_examples.extend(scenario_examples)
    print(f"   Generated {len(scenario_examples)} scenario-based examples")

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
