# Unsloth MCP Server - Current Tasks

## Active Sprint: RunPod Integration

### In Progress

- [ ] **Add RunPod API key to project .env**
  - Source: ai-development-cockpit/.env
  - Verify API connectivity

### Up Next

- [ ] **Create RunPod API client module**
  - `src/utils/runpod.ts`
  - Pod management functions
  - Error handling for API calls

- [ ] **Add pod management tools**
  - `runpod_list_pods` - List available pods
  - `runpod_create_pod` - Spin up new pod
  - `runpod_start_pod` / `runpod_stop_pod`
  - `runpod_terminate_pod`

- [ ] **Add training job tools**
  - `runpod_start_training` - Submit job
  - `runpod_get_training_logs` - Stream output
  - `runpod_download_model` - Get artifacts

### Testing Requirements

- [ ] Unit tests for RunPod client
- [ ] Integration tests with mock API
- [ ] End-to-end test with real pod

## Recently Completed

- [x] MCP server v2.1.0 (12 tools)
- [x] Production infrastructure
- [x] 43 test cases
- [x] Claude Skills integration
- [x] Pre-commit hooks
- [x] Docker configuration

## Blocked

None currently

## Quick Reference

```bash
# Build and test
npm run build && npm test

# Run server
npm run start

# CLI dashboard
npm run cli

# Check pod status (once integrated)
curl -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.io/v2/pod/yopwr0r7v1pf9j
```
