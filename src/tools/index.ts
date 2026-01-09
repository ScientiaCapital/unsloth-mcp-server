/**
 * tools/index.ts - Tool Module Aggregator
 *
 * Combines all tool modules into a single export for the main server.
 * This allows index.ts to import all tools from one place.
 *
 * Tool Categories:
 *   - Core (6 tools): Installation, model loading, training, export
 *   - Training (6 tools): Datasets, benchmarking, model info
 *   - Tokenizer (2 tools): SuperBPE training, comparison
 *   - Knowledge (10 tools): OCR, cataloguing, training data generation
 *   - RunPod (11 tools): GPU pod management, remote training
 *
 * Total: 35 tools
 */

import { ToolDefinition, ToolHandler, ToolModule } from './types.js';

// Import individual modules
import { coreModule } from './core.js';
import { trainingModule } from './training.js';
import { tokenizerModule } from './tokenizer.js';
import { knowledgeModule } from './knowledge.js';
import { runpodModule } from './runpod.js';

// Re-export types for convenience
export * from './types.js';

// Re-export individual modules
export { coreModule } from './core.js';
export { trainingModule } from './training.js';
export { tokenizerModule } from './tokenizer.js';
export { knowledgeModule } from './knowledge.js';
export { runpodModule } from './runpod.js';

/**
 * All tool definitions aggregated
 */
export const allTools: ToolDefinition[] = [
  ...coreModule.tools,
  ...trainingModule.tools,
  ...tokenizerModule.tools,
  ...knowledgeModule.tools,
  ...runpodModule.tools,
];

/**
 * All tool handlers aggregated
 */
export const allHandlers: Record<string, ToolHandler> = {
  ...coreModule.handlers,
  ...trainingModule.handlers,
  ...tokenizerModule.handlers,
  ...knowledgeModule.handlers,
  ...runpodModule.handlers,
};

/**
 * Get a list of all tool names
 */
export function getToolNames(): string[] {
  return allTools.map((t) => t.name);
}

/**
 * Get tool count by category
 */
export function getToolCounts(): Record<string, number> {
  return {
    core: coreModule.tools.length,
    training: trainingModule.tools.length,
    tokenizer: tokenizerModule.tools.length,
    knowledge: knowledgeModule.tools.length,
    runpod: runpodModule.tools.length,
    total: allTools.length,
  };
}

/**
 * Combined module export
 */
export const toolsModule: ToolModule = {
  tools: allTools,
  handlers: allHandlers,
};

export default toolsModule;
