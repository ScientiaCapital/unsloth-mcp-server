# Project Context - Unsloth MCP Server

Last updated: 2025-12-31

## Current State

**Version:** v2.3.0
**Status:** Production-ready
**Tests:** 180/180 passing
**Dependencies:** 0 vulnerabilities

## Done (This Session - 2025-12-31)

### Documentation & Transfer

- [x] Updated `.env.example` with RunPod and Supabase configuration
- [x] Created `WORK_COMPUTER_SETUP.md` guide for M2 MacBook sync
- [x] Created `.env.coperniq` with actual keys for secure transfer
- [x] Added `.env.*` patterns to `.gitignore`
- [x] Created transfer zip (558KB, excludes node_modules)

### Security & Quality

- [x] Fixed 2 HIGH severity npm vulnerabilities (qs DoS)
- [x] Ran full security sweep (secrets, git history, API audit)
- [x] All 180 tests passing
- [x] Linting: 0 errors, 41 warnings (all `any` type warnings)

### Git Commits

1. `f7babfe` - chore: Add .env.coperniq and .env.\* to gitignore
2. `6348bb9` - docs: Add RunPod configuration + work computer setup guide

## Blockers Encountered

None this session.

## Decisions Made

1. **Key Transfer Method:** Created `.env.coperniq` in zip file (bypasses gitignore) rather than using git for secrets transfer
2. **Security Fix:** Applied `npm audit fix` to resolve qs vulnerability immediately rather than defer

## Tomorrow's Priorities

1. **Test RunPod Training E2E**
   - Connect to existing pod (yopwr0r7v1pf9j)
   - Submit training job with coperniq-forge data
   - Verify model artifact retrieval

2. **langgraph-voice-agents Fine-tuning**
   - Set up training data for voice agent model
   - Configure training parameters
   - Run first training iteration

3. **M2 MacBook Setup**
   - Extract zip and verify RunPod connectivity
   - Test MCP server tools

## Architecture Notes

```
unsloth-mcp-server/
├── src/
│   ├── index.ts        # Main MCP server (33 tools)
│   ├── cli.ts          # CLI interface
│   ├── utils/
│   │   ├── runpod.ts   # RunPod API (pods, training, logs)
│   │   ├── checkpoint.ts # Training checkpoint mgmt
│   │   └── cost-tracker.ts # GPU cost tracking
│   └── knowledge/      # OCR + training data pipeline
└── .claude/skills/     # Claude Code skills (8 skills)
```

## Environment Variables Required

```bash
RUNPOD_API_KEY=rpa_...
RUNPOD_POD_ID=yopwr0r7v1pf9j
SUPABASE_URL=https://xucngysrzjtwqzgcutqf.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
ORGANIZATION_ID=scientia-capital
```

## Related Projects

| Project                | Status  | Training Data |
| ---------------------- | ------- | ------------- |
| coperniq-forge         | Ready   | 1,803 samples |
| langgraph-voice-agents | Next up | TBD           |
| FieldVault             | Pending | TBD           |
| NetZero                | Pending | TBD           |
