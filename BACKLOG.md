# Unsloth MCP Server - Backlog

## High Priority (Phase 4: RunPod)

### RunPod Pod Management

- [ ] Create RunPod API client (`src/utils/runpod.ts`)
- [ ] `runpod_list_pods` tool - List all pods with status
- [ ] `runpod_create_pod` tool - Create new GPU pod
- [ ] `runpod_start_pod` tool - Start stopped pod
- [ ] `runpod_stop_pod` tool - Stop running pod
- [ ] `runpod_terminate_pod` tool - Delete pod
- [ ] Cost estimation before pod creation

### RunPod Training Jobs

- [ ] `runpod_start_training` tool - Submit fine-tuning job
- [ ] Real-time log streaming via WebSocket
- [ ] `runpod_get_training_logs` tool - Fetch logs
- [ ] `runpod_download_model` tool - Download trained model
- [ ] Training progress webhooks
- [ ] Auto-terminate after training completes

### Testing & Validation

- [ ] Unit tests for RunPod client (mock API)
- [ ] Integration tests with test pod
- [ ] End-to-end fine-tuning workflow test
- [ ] Cost tracking validation

## Medium Priority (Phase 5: Advanced)

### Multi-GPU Training

- [ ] Distributed training configuration
- [ ] Multi-pod orchestration
- [ ] FSDP/DeepSpeed integration
- [ ] Automatic GPU scaling

### Checkpoint Management

- [ ] Auto-save checkpoints to cloud storage
- [ ] Resume training from checkpoint
- [ ] Checkpoint versioning
- [ ] Cleanup old checkpoints

### Training Visualization

- [ ] TensorBoard integration
- [ ] Loss/metric graphs via MCP
- [ ] Training dashboard tool

### Hyperparameter Optimization

- [ ] Grid search support
- [ ] Optuna integration
- [ ] Best model selection
- [ ] Experiment tracking

## Low Priority (Future)

### Model Registry

- [ ] HuggingFace Hub integration
- [ ] Private model registry
- [ ] Model versioning and tagging
- [ ] Model comparison tools

### Advanced Export

- [ ] Ollama direct push
- [ ] vLLM optimization
- [ ] ONNX export
- [ ] MLX export (Apple Silicon)

### Enterprise Features

- [ ] Team collaboration
- [ ] Usage quotas
- [ ] Audit logging
- [ ] SSO integration

## Technical Debt

- [ ] Refactor tool handlers into separate files
- [ ] Add OpenAPI/Swagger docs for tools
- [ ] Improve Python script caching
- [ ] Add retry logic for HuggingFace API calls

## Ideas & Research

- [ ] Evaluate PEFT alternatives
- [ ] Research training on consumer GPUs (RTX 4090)
- [ ] Investigate serverless GPU options
- [ ] Explore model merging techniques

---

_Last updated: 2025-12-21_
