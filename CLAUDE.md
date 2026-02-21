# Unsloth MCP Server

**Updated:** 2026-01-18
**Status:** 33 tools working, 180 tests passing, training has environment issues

## Critical Rules

- **NO OpenAI** - Use Anthropic Claude or Google Gemini only
- **API keys in .env only** - Never hardcode
- RunPod API key stored in .env

## What This Project Is

MCP server giving Claude access to Unsloth fine-tuning. 33 tools for model training, RunPod GPU management, knowledge capture, and cost tracking.

## Current Status

| Component | Status |
|-----------|--------|
| MCP tools (33) | Working |
| Test suite (180) | Passing |
| RunPod pod management | Working |
| Knowledge capture | Working |
| GRPO training | Dtype issues |
| SFT training | Environment-dependent |

## Key Files

```
src/index.ts          # Main MCP server (33 tools)
src/utils/runpod.ts   # RunPod API client
src/utils/checkpoint.ts # Training checkpoints
src/utils/cost-tracker.ts # GPU cost tracking
src/knowledge/        # OCR + training data pipeline
src/__tests__/        # 180 tests
```

## Tool Categories

- **Core (12)**: finetune_model, load_model, export_model, etc.
- **Knowledge (10)**: OCR, enhancement, training pair generation
- **RunPod (11)**: Pod lifecycle, training jobs, cost estimation

## Development

```bash
npm run build    # Build TypeScript
npm run test     # 180 tests
npm run start    # Start server
```

## Environment

```bash
RUNPOD_API_KEY=rpa_XXX
HUGGINGFACE_TOKEN=hf_XXX  # Optional
```

---

## MANDATORY: Observer Protocol

**You MUST follow this protocol before writing ANY code.** No exceptions. No rationalizing.

### Step 1: Classify Task Scope

| Scope | Criteria | Observer Required |
|-------|----------|-------------------|
| **MINIMAL** | Typos, comments, single config tweak | None |
| **SMALL** | 1-3 files changed, no new dependencies | observer-lite (Haiku) |
| **STANDARD** | 4-10 files, or any new dependency | observer-full (Sonnet) |
| **FULL** | >10 files, new architecture, new patterns | observer-full + feature contract |

### Step 2: Spawn Observer (if SMALL or above)

```
# For SMALL scope:
Task tool -> subagent_type: "observer-lite"
  prompt: "Run quality checks on the unsloth-mcp-server codebase. Focus on [relevant area]."

# For STANDARD/FULL scope:
Task tool -> subagent_type: "observer-full"
  prompt: "Run full drift detection on unsloth-mcp-server. The current task is: [describe task]."
```

### Step 3: For FULL scope — Create Feature Contract First

Before coding, create `.claude/contracts/[feature-name].md`:
- Define IN SCOPE and OUT OF SCOPE boundaries
- List success criteria
- Get observer approval before writing code

### Step 4: Verify Observer Ran

Before making your first code change, confirm:
- [ ] `.claude/OBSERVER_QUALITY.md` has a real date (not `_not yet run_`)
- [ ] Scope classification matches the task complexity

**If the PreToolUse hook prints `** OBSERVER NOT ACTIVE **`, STOP and spawn the observer.**

### Scope Escalation Rule

If during work you hit ANY of these triggers, upgrade from Lite to Full:
- **>5 files modified** (the PostToolUse hook will remind you)
- **New dependency added** to package.json or pyproject.toml
- **Task scope expanded** beyond original description

---

## Dual-Team Workflow

This project uses the **TK Dual-Team Daily Workflow**.

### Quality Gates

| Gate | Check | Enforced By |
|------|-------|-------------|
| Pre-code | Observer spawned | PreToolUse hook |
| During code | Scope escalation | PostToolUse hook |
| Pre-merge | No open BLOCKERs | OBSERVER_ALERTS.md |

### Observer Cost Guide

| Observer | Model | Cost | When |
|----------|-------|------|------|
| observer-lite | Haiku 4.5 | ~$0.03-0.05 | SMALL scope |
| observer-full | Sonnet 4.6 | ~$0.50-2.00 | STANDARD/FULL scope |

### Copy-Paste Prompts

**START DAY:** Start day — project is unsloth-mcp-server. Path: ~/Desktop/tk_projects/unsloth-mcp-server
**FEATURE BUILD:** Feature build — [FEATURE_NAME]
**END DAY:** End day — project is unsloth-mcp-server
