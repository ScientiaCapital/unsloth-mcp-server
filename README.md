# Unsloth MCP Server

> **33 MCP tools** for LLM fine-tuning via Claude. 180 tests passing.
> Integrates Unsloth (2x faster, 80% less memory) with RunPod GPU management.

---

## What It Does

- **33 MCP tools** for model fine-tuning, tokenizer training, GPU management
- **RunPod integration** - create, start, stop pods programmatically
- **Knowledge capture pipeline** - OCR, AI enhancement, training data generation
- **Checkpoint management** - save/load/resume training sessions
- **Cost tracking** - budgets, alerts, expense reporting

## Goals

Enable Claude to orchestrate LLM fine-tuning workflows without manual Python scripting.

## Quick Start

```bash
cd unsloth-mcp-server
npm install && npm run build
npm test  # 180 tests passing
```

Add to Claude Code MCP settings:

```json
{
  "mcpServers": {
    "unsloth-server": {
      "command": "node",
      "args": ["/path/to/build/index.js"]
    }
  }
}
```

## Current Status

| Component             | Status                     |
| --------------------- | -------------------------- |
| MCP tools (33)        | Working                    |
| Test suite (180)      | Passing                    |
| RunPod pod management | Working                    |
| Knowledge capture     | Working                    |
| Checkpoint management | Working                    |
| GRPO training         | Dtype compatibility issues |
| SFT training          | Environment-dependent      |

## Known Issues

- GRPO: Dtype mismatches in some configurations
- Requires Python 3.10-3.12 (not 3.13)
- GPU required for fine-tuning

## GTME Skills Developed

Building toward Go-To-Market Engineer through hands-on projects:

| Skill Area                    | What I Learned                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------ |
| **Developer tooling**         | Built MCP server that developers actually use - learned API design for DX      |
| **Cost optimization**         | Implemented budget tracking, alerts - understand unit economics of GPU compute |
| **Infrastructure automation** | RunPod API integration - programmatic cloud GPU provisioning                   |
| **Testing discipline**        | 180 tests with Jest - shipping quality that earns trust                        |
| **Technical documentation**   | Writing docs that reduce support burden                                        |
| **Product iteration**         | v2.0 → v2.3.0 based on real usage feedback                                     |

## Tech Stack

Node.js/TypeScript, MCP SDK, Python/Unsloth, Jest, RunPod API

## Requirements

- Node.js 18+
- Python 3.10-3.12
- NVIDIA GPU with CUDA (for training)
