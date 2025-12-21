# Unsloth MCP Server - Planning

## Project Vision

Enable Claude Code users to fine-tune LLMs efficiently through an MCP interface, leveraging Unsloth's 2x speed and 80% memory reduction.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Claude Code   │────▶│  MCP Server      │────▶│  Python/Unsloth │
│   (Client)      │     │  (TypeScript)    │     │  (GPU Backend)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                         │
                               ▼                         ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  RunPod API      │────▶│  GPU Pods       │
                        │  (Orchestration) │     │  (A5000/A100)   │
                        └──────────────────┘     └─────────────────┘
```

## Completed Phases

### Phase 1: Core MCP Server (v1.0.0) [DONE]

- [x] Basic MCP server structure
- [x] 6 core tools (check, list, load, finetune, generate, export)
- [x] Python script execution
- [x] Error handling

### Phase 2: Enhanced Tools (v2.0.0) [DONE]

- [x] SuperBPE tokenizer training
- [x] Model information tool
- [x] Tokenizer comparison
- [x] Performance benchmarking
- [x] Dataset discovery & preparation
- [x] 6 additional tools

### Phase 3: Production Infrastructure (v2.1.0) [DONE]

- [x] Configuration system (JSON/env)
- [x] Caching layer (memory + disk)
- [x] Winston structured logging
- [x] Input validation (all 12 tools)
- [x] Security (sanitization, rate limiting)
- [x] Performance metrics
- [x] Progress reporting
- [x] CLI tool
- [x] 43 comprehensive tests
- [x] Pre-commit hooks (Husky)

## Current Phase

### Phase 4: RunPod Integration [IN PROGRESS]

- [ ] RunPod API client
- [ ] Pod lifecycle management (create/start/stop/terminate)
- [ ] Training job submission
- [ ] Real-time training logs
- [ ] Model artifact retrieval
- [ ] Cost tracking

### Phase 5: Advanced Features [PLANNED]

- [ ] Multi-GPU training support
- [ ] Checkpoint management
- [ ] Training visualization
- [ ] Hyperparameter optimization
- [ ] Model versioning

## RunPod Integration Design

### New Tools to Add

1. `runpod_create_pod` - Create GPU pod for training
2. `runpod_list_pods` - List active/available pods
3. `runpod_start_training` - Submit training job to pod
4. `runpod_get_training_logs` - Stream training output
5. `runpod_download_model` - Retrieve trained model
6. `runpod_terminate_pod` - Clean up resources

### Available Resources

- **Pod**: coperniq-ft2 (yopwr0r7v1pf9j)
- **GPU**: RTX A5000 (24GB VRAM)
- **Storage**: 30GB volume at /runpod
- **Image**: runpod/pytorch:2.2.0-py3.10-cuda12.1.1
- **Cost**: $0.16/hr compute + $0.008/hr storage

## Decision Log

| Date       | Decision                   | Rationale                    |
| ---------- | -------------------------- | ---------------------------- |
| 2025-11-06 | TypeScript MCP server      | Best MCP SDK support         |
| 2025-11-06 | Python backend for Unsloth | Native Unsloth integration   |
| 2025-11-07 | Winston for logging        | Production-grade, structured |
| 2025-11-07 | Jest for testing           | Best TS/JS testing framework |
| TBD        | RunPod for GPU             | Cost-effective, API-friendly |
