# Unsloth MCP Server

## Overview

MCP (Model Context Protocol) server for Unsloth - enabling Claude Code to fine-tune LLMs 2x faster with 80% less memory. Production-ready v2.1.0 with 12 tools, comprehensive testing, and enterprise infrastructure.

## Tech Stack

- **Runtime**: Node.js 18+ / TypeScript 5.2
- **MCP SDK**: @modelcontextprotocol/sdk 1.6.1
- **Backend**: Python 3.10-3.12 + Unsloth + PyTorch
- **Testing**: Jest 30 (43 tests)
- **GPU**: NVIDIA CUDA 11.8+ / RunPod
- **Docker**: Multi-stage build with CUDA support

## Key Files

```
src/index.ts          # Main MCP server (1427 lines, 12 tools)
src/cli.ts            # CLI interface
src/utils/            # Production utilities:
  - config.ts         # Configuration system
  - cache.ts          # Caching layer
  - logger.ts         # Winston logging
  - metrics.ts        # Performance tracking
  - progress.ts       # Progress reporting
  - security.ts       # Input sanitization
  - validation.ts     # Tool input validation
```

## Available Tools

1. `check_installation` - Verify Unsloth setup
2. `list_supported_models` - Get supported models
3. `load_model` - Load with Unsloth optimizations
4. `finetune_model` - Fine-tune with LoRA/QLoRA
5. `generate_text` - Text generation
6. `export_model` - Export to GGUF/HuggingFace
7. `train_superbpe_tokenizer` - Train efficient tokenizers
8. `get_model_info` - Model architecture details
9. `compare_tokenizers` - Compare tokenization efficiency
10. `benchmark_model` - Performance benchmarking
11. `list_datasets` - Search HuggingFace datasets
12. `prepare_dataset` - Format datasets for training

## Development Commands

```bash
npm run build    # Build TypeScript
npm run start    # Run server
npm run test     # Run 43 tests
npm run lint     # TypeScript + ESLint
npm run cli      # Run CLI tool
npm run bench    # Run benchmarks
```

## Active Skills

Check ~/.claude/skills/ for available skills:

- `runpod-deployment-skill` - GPU deployment (next phase)
- `langgraph-agents-skill` - Multi-agent systems
- `supabase-sql-skill` - SQL migrations

## Critical Rules

- **NO OpenAI** - Use Anthropic Claude or Google Gemini only
- **API keys in .env only** - Never hardcode
- RunPod API key available in ai-development-cockpit/.env
