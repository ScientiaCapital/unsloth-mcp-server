#!/usr/bin/env node
/**
 * Unsloth MCP Server - Main Entry Point
 *
 * A Model Context Protocol server for fine-tuning LLMs with Unsloth.
 * Tools are organized in modular files under src/tools/.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import { exec } from 'child_process';
import { promisify } from 'util';

// Utilities
import logger from './utils/logger.js';
import { validateToolInputs, ValidationError } from './utils/validation.js';
import { safeExecute, SecurityError, TimeoutError } from './utils/security.js';
import { metricsCollector } from './utils/metrics.js';
import { config } from './utils/config.js';
import { cache } from './utils/cache.js';

// Knowledge module
import {
  knowledgeDb,
  processImage,
  processImageBatch,
  checkOCRBackends,
  generateTrainingPairs,
  CATEGORY_DEFINITIONS,
} from './knowledge/index.js';

// RunPod client
import { getRunPodClient } from './utils/runpod.js';

// Tool modules
import { allTools, allHandlers, ToolContext, ToolResult } from './tools/index.js';

const execPromise = promisify(exec);

/**
 * Unsloth MCP Server
 */
class UnslothServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      { name: 'unsloth-server', version: '2.4.0' },
      { capabilities: { tools: {} } }
    );

    this.setupToolHandlers();
    this.setupErrorHandling();

    const serverConfig = config.get();
    logger.info('Unsloth MCP Server initialized', {
      version: '2.4.0',
      environment: serverConfig.server.environment,
      toolCount: allTools.length,
    });
  }

  /**
   * Set up error handling
   */
  private setupErrorHandling() {
    this.server.onerror = (error) => {
      logger.error('[MCP Error]', { error: error.message, stack: error.stack });
    };

    process.on('SIGINT', async () => {
      logger.info('Received SIGINT, shutting down gracefully...');
      await this.server.close();
      process.exit(0);
    });
  }

  /**
   * Execute a Python/Unsloth script safely
   */
  private async executeScript(script: string): Promise<string> {
    try {
      logger.debug('Executing script', { length: script.length });
      const result = await safeExecute(script);
      return result;
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      if (error instanceof TimeoutError) {
        throw new Error(`Operation timed out: ${msg}`);
      }
      if (error instanceof SecurityError) {
        throw new Error(`Security error: ${msg}`);
      }
      throw new Error(`Script execution failed: ${msg}`);
    }
  }

  /**
   * Create tool context for handlers
   */
  private createContext(toolName: string, startTime: number): ToolContext {
    const runpodClient = getRunPodClient();
    return {
      logger,
      cache: {
        get: <T>(key: string) => cache.get(key) as T | undefined,
        set: <T>(key: string, value: T, ttl?: number) => cache.set(key, value, ttl),
        delete: (key: string) => cache.delete(key),
      },
      executeScript: this.executeScript.bind(this),
      execCommand: execPromise,
      startTime,
      toolName,
      knowledgeDb: {
        searchEntries: (query: string, limit?: number) =>
          knowledgeDb.searchEntries(query, limit) as Promise<unknown[]>,
        listByCategory: (category: string, limit?: number) =>
          knowledgeDb.listByCategory(category as never, limit) as Promise<unknown[]>,
        getEntry: (id: string) => knowledgeDb.getEntry(id) as Promise<unknown | null>,
        addEntry: (entry: unknown) => knowledgeDb.addEntry(entry as never) as Promise<string>,
        getStats: () => knowledgeDb.getStats() as Promise<unknown>,
        getAllTrainingPairs: (minQuality?: number) =>
          knowledgeDb.getAllTrainingPairs(minQuality) as Promise<unknown[]>,
        exportTrainingData: (path: string, format: string, minQuality?: number) =>
          knowledgeDb
            .exportTrainingData(path, format as 'alpaca' | 'sharegpt' | 'chatml', minQuality)
            .then((r) => r.count),
      },
      runpodClient: runpodClient
        ? {
            listPods: () => runpodClient.listPods() as Promise<unknown[]>,
            getPod: (id: string) => runpodClient.getPod(id) as Promise<unknown>,
            getAvailableGpus: () => runpodClient.getGpuTypes() as Promise<unknown[]>,
            createPod: (options: unknown) =>
              runpodClient.createPod(options as never) as Promise<unknown>,
            startPod: (id: string) => runpodClient.startPod(id) as Promise<unknown>,
            stopPod: (id: string) => runpodClient.stopPod(id) as Promise<unknown>,
            terminatePod: (id: string) => runpodClient.terminatePod(id) as Promise<unknown>,
            startTraining: (podId: string, config: unknown) =>
              runpodClient.startTrainingJob(podId, config as never) as Promise<unknown>,
            getTrainingStatus: (podId: string, _jobId: string) =>
              runpodClient.getTrainingStatus(podId) as Promise<unknown>,
            getTrainingLogs: (podId: string, _jobId: string, lines?: number) =>
              runpodClient.getTrainingLogs(podId, lines),
            estimateTrainingCost: (config: unknown) =>
              Promise.resolve(
                runpodClient.estimateTrainingCost(
                  (config as { datasetTokens?: number }).datasetTokens || 1000000,
                  (config as { baseModel?: string }).baseModel || 'llama-3b',
                  (config as { gpuCostPerHour?: number }).gpuCostPerHour || 0.5,
                  (config as { epochs?: number }).epochs || 1
                )
              ),
          }
        : null,
      processImage: (path: string, options?: unknown) =>
        processImage(path, options as never) as Promise<unknown>,
      processImageBatch: (paths: string[], options?: unknown) =>
        processImageBatch(paths, options as never) as Promise<unknown[]>,
      checkOCRBackends: () => checkOCRBackends() as Promise<unknown>,
      generateTrainingPairs: (content: string, category: string, _options?: unknown) =>
        Promise.resolve(generateTrainingPairs({ cleaned_text: content, category } as never)),
      categoryDefinitions: Object.fromEntries(
        Object.entries(CATEGORY_DEFINITIONS).map(([k, v]) => [
          k,
          { name: k, description: v.description },
        ])
      ),
    };
  }

  /**
   * Set up MCP tool handlers
   */
  private setupToolHandlers() {
    // List all available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: allTools,
    }));

    // Handle tool calls
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this.server.setRequestHandler(CallToolRequestSchema, async (request): Promise<any> => {
      const { name, arguments: args } = request.params;
      const startTime = metricsCollector.startTool(name);

      logger.info(`Tool called: ${name}`, { args });

      try {
        // Validate inputs
        validateToolInputs(name, args || {});

        // Find handler
        const handler = allHandlers[name];
        if (!handler) {
          throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
        }

        // Create context and execute handler
        const context = this.createContext(name, startTime);
        const result = await handler(args || {}, context);

        // Record success metrics
        metricsCollector.endTool(name, startTime, !result.isError);

        return result;
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        logger.error(`Tool error: ${name}`, { error: message });
        metricsCollector.endTool(name, startTime, false);

        if (error instanceof ValidationError) {
          throw new McpError(ErrorCode.InvalidParams, message);
        }
        if (error instanceof McpError) {
          throw error;
        }

        return {
          content: [{ type: 'text', text: `Error: ${message}` }],
          isError: true,
        };
      }
    });
  }

  /**
   * Start the server
   */
  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    logger.info('Unsloth MCP Server running on stdio');
  }
}

// Start server
const server = new UnslothServer();
server.run().catch((error) => {
  logger.error('Fatal error', { error });
  process.exit(1);
});
