# Coding Rules — unsloth-mcp-server

## Stack

TypeScript (Node.js), MCP SDK, Jest, RunPod API

## Rules

- No OpenAI — use Anthropic Claude or Google Gemini only
- API keys in .env only, never hardcode
- TypeScript strict mode; no `any` types; explicit return types on all functions
- Use ES modules with named exports
- All 33 MCP tools must have corresponding Jest tests
- Run `npm run build` before running tests to compile TypeScript
- RunPod API calls must handle errors gracefully with structured error types
- Cost tracking must log all GPU usage with timestamps and job IDs
