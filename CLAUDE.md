# Unsloth MCP Server

## Overview

MCP (Model Context Protocol) server for Unsloth - enabling Claude Code to fine-tune LLMs 2x faster with 80% less memory. Production-ready v2.3.0 with 33 tools, comprehensive testing, and RunPod GPU integration.

## Tech Stack

- **Runtime**: Node.js 18+ / TypeScript 5.2
- **MCP SDK**: @modelcontextprotocol/sdk 1.6.1
- **Backend**: Python 3.10-3.12 + Unsloth + PyTorch
- **Testing**: Jest 30 (180 tests)
- **GPU**: NVIDIA CUDA 11.8+ / RunPod API
- **Database**: Supabase PostgreSQL (shared with ai-development-cockpit)
- **OCR**: Tesseract, EasyOCR, Claude Vision

## Key Files

```
src/index.ts          # Main MCP server (33 tools)
src/cli.ts            # CLI interface
src/utils/
  - runpod.ts         # RunPod API client (pod management, training jobs)
  - checkpoint.ts     # Training checkpoint management (save/load/resume)
  - cost-tracker.ts   # GPU cost tracking dashboard with budgets
  - config.ts         # Configuration system
  - cache.ts          # Caching layer
  - logger.ts         # Winston logging
  - metrics.ts        # Performance tracking
  - progress.ts       # Progress reporting
  - security.ts       # Input sanitization
  - validation.ts     # Tool input validation
src/knowledge/        # Knowledge capture pipeline
  - ocr.ts            # Multi-engine OCR
  - database.ts       # SQLite storage
  - ai-enhancer.ts    # Claude-powered enhancement
  - training-generator.ts # Training pair generation
src/__tests__/        # Test suite (180 tests)
  - knowledge.test.ts # Knowledge module tests (57 tests)
  - checkpoint.test.ts # Checkpoint manager tests
  - cost-tracker.test.ts # Cost tracker tests
```

## Available Tools (33 total)

### Core Tools (12)

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

### Knowledge Capture Tools (10)

13. `knowledge_process_image` - OCR with multi-engine support
14. `knowledge_enhance_entry` - AI-powered text enhancement
15. `knowledge_create_entry` - Create knowledge entries
16. `knowledge_get_stats` - Database statistics
17. `knowledge_list_entries` - Browse entries
18. `knowledge_generate_pairs` - Generate training pairs
19. `knowledge_export_dataset` - Export to Alpaca/ShareGPT/ChatML
20. `knowledge_batch_process` - Batch image processing
21. `knowledge_search` - Full-text search
22. `knowledge_quality_review` - Quality assessment

### RunPod Tools (11)

23. `runpod_list_pods` - List all pods
24. `runpod_get_pod` - Get pod details
25. `runpod_check_gpus` - Check GPU availability
26. `runpod_create_pod` - Create new pod
27. `runpod_start_pod` - Start stopped pod
28. `runpod_stop_pod` - Stop running pod
29. `runpod_terminate_pod` - Terminate pod
30. `runpod_start_training` - Start fine-tuning job
31. `runpod_get_training_status` - Check training progress
32. `runpod_get_training_logs` - Get training logs
33. `runpod_estimate_cost` - Estimate training cost

## Development Commands

```bash
npm run build    # Build TypeScript
npm run start    # Run server
npm run test     # Run 180 tests
npm run lint     # TypeScript + ESLint
npm run cli      # Run CLI tool
npm run bench    # Run benchmarks
```

## Environment Variables

```bash
# RunPod Configuration
RUNPOD_API_KEY=rpa_XXX
RUNPOD_API_ENDPOINT=https://api.runpod.io/v2
RUNPOD_POD_ID=your_pod_id

# Supabase (shared with ai-development-cockpit)
SUPABASE_URL=https://xucngysrzjtwqzgcutqf.supabase.co
SUPABASE_ANON_KEY=eyJhbG...
SUPABASE_SERVICE_ROLE_KEY=eyJhbG...
ORGANIZATION_ID=scientia-capital
```

## Active Skills

Check ~/.claude/skills/ for available skills:

- `runpod-deployment-skill` - GPU deployment
- `langgraph-agents-skill` - Multi-agent systems
- `supabase-sql-skill` - SQL migrations

## Data-Forge Integration

**This project works in tandem with `data-forge` (~/Desktop/tk_projects/data-forge)**

| Project                | Role                                         |
| ---------------------- | -------------------------------------------- |
| **data-forge**         | Finds, processes, and prepares training data |
| **unsloth-mcp-server** | Fine-tunes models using that data            |

### Data Pipeline Flow

```
data-forge (data acquisition)           unsloth-mcp-server (fine-tuning)
├── Gong transcripts → ChatML      →   training-data/grpo_prompts.json
├── Sales books (Chris Voss, etc.) →   Reward function keywords
├── Coperniq docs (notion, miro)   →   Domain knowledge signals
└── Continuous updates             →   Iterative training improvement
```

### Available Training Data (from data-forge)

- **Gong transcripts**: 169 ChatML examples (8 calls, Abdullah's demos)
- **Never Split the Difference**: 101 samples, 89K words
- **The Challenger Sale**: 72 samples, 8.8K words
- **Blue Ocean Strategy**: ~100 samples
- **Jobs to be Done**: ~80 samples
- **Coperniq sales docs**: 430KB domain knowledge

### Workflow

1. **data-forge** extracts and processes new data sources
2. Update `training-data/grpo_prompts.json` with real scenarios
3. Enhance reward function keywords based on patterns
4. Run GRPO training via `train_grpo.py`
5. Iterate with Ralph Wiggum loop

## Cross-Project Training

Shared infrastructure for all 70 TK projects:

- **Docs**: `~/.shared/training-infrastructure.md`
- **Ready data**: coperniq-forge (1,803 samples)
- **Projects**: FieldVault, NetZero, Signal Siphon, Solar\*

## Critical Rules

- **NO OpenAI** - Use Anthropic Claude or Google Gemini only
- **API keys in .env only** - Never hardcode
- RunPod API key stored in .env
