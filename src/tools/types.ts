/**
 * types.ts - Shared types for tool handlers
 *
 * This file defines the common interfaces used across all tool modules.
 */

/**
 * Tool definition as returned to MCP clients
 */
export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, PropertyDefinition>;
    required: string[];
  };
}

/**
 * Property definition for tool input schemas
 */
export interface PropertyDefinition {
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  description?: string;
  default?: unknown;
  enum?: string[];
  items?: PropertyDefinition;
  properties?: Record<string, PropertyDefinition>;
  required?: string[];
}

/**
 * Tool call result returned to MCP clients
 */
export interface ToolResult {
  content: Array<{
    type: 'text' | 'image' | 'resource';
    text?: string;
    data?: string;
    mimeType?: string;
    resource?: {
      uri: string;
      mimeType: string;
      text?: string;
      blob?: string;
    };
  }>;
  isError?: boolean;
}

/**
 * Context passed to tool handlers
 */
export interface ToolContext {
  /** Logger instance */
  logger: {
    info: (msg: string, meta?: Record<string, unknown>) => void;
    error: (msg: string, meta?: Record<string, unknown>) => void;
    debug: (msg: string, meta?: Record<string, unknown>) => void;
    warn: (msg: string, meta?: Record<string, unknown>) => void;
  };
  /** Cache instance */
  cache: {
    get: <T>(key: string) => T | undefined;
    set: <T>(key: string, value: T, ttl?: number) => void;
    delete: (key: string) => void;
  };
  /** Execute Python/Unsloth script */
  executeScript: (script: string) => Promise<string>;
  /** Execute shell command */
  execCommand: (cmd: string) => Promise<{ stdout: string; stderr: string }>;
  /** Tool start time for metrics */
  startTime: number;
  /** Tool name being executed */
  toolName: string;
  /** Knowledge database instance */
  knowledgeDb: {
    searchEntries: (query: string, limit?: number) => Promise<unknown[]>;
    listByCategory: (category: string, limit?: number) => Promise<unknown[]>;
    getEntry: (id: string) => Promise<unknown | null>;
    addEntry: (entry: unknown) => Promise<string>;
    getStats: () => Promise<unknown>;
    getAllTrainingPairs: (minQuality?: number) => Promise<unknown[]>;
    exportTrainingData: (path: string, format: string, minQuality?: number) => Promise<number>;
  };
  /** RunPod client for GPU operations */
  runpodClient: {
    listPods: () => Promise<unknown[]>;
    getPod: (id: string) => Promise<unknown>;
    getAvailableGpus: () => Promise<unknown[]>;
    createPod: (options: unknown) => Promise<unknown>;
    startPod: (id: string) => Promise<unknown>;
    stopPod: (id: string) => Promise<unknown>;
    terminatePod: (id: string) => Promise<unknown>;
    startTraining: (podId: string, config: unknown) => Promise<unknown>;
    getTrainingStatus: (podId: string, jobId: string) => Promise<unknown>;
    getTrainingLogs: (podId: string, jobId: string, lines?: number) => Promise<string>;
    estimateTrainingCost: (config: unknown) => Promise<unknown>;
  } | null;
  /** OCR and image processing */
  processImage: (path: string, options?: unknown) => Promise<unknown>;
  processImageBatch: (paths: string[], options?: unknown) => Promise<unknown[]>;
  checkOCRBackends: () => Promise<unknown>;
  /** Training pair generation */
  generateTrainingPairs: (
    content: string,
    category: string,
    options?: unknown
  ) => Promise<unknown[]>;
  /** Category definitions */
  categoryDefinitions: Record<string, { name: string; description: string }>;
}

/**
 * Tool handler function signature
 */
export type ToolHandler = (
  args: Record<string, unknown>,
  context: ToolContext
) => Promise<ToolResult>;

/**
 * Tool module interface - what each tool file exports
 */
export interface ToolModule {
  /** Tool definitions for listing */
  tools: ToolDefinition[];
  /** Handler implementations keyed by tool name */
  handlers: Record<string, ToolHandler>;
}

/**
 * Success response helper
 */
export function successResponse(text: string): ToolResult {
  return {
    content: [{ type: 'text', text }],
  };
}

/**
 * Error response helper
 */
export function errorResponse(message: string): ToolResult {
  return {
    content: [{ type: 'text', text: `Error: ${message}` }],
    isError: true,
  };
}

/**
 * JSON response helper
 */
export function jsonResponse(data: unknown, prefix?: string): ToolResult {
  const text = prefix
    ? `${prefix}\n\n${JSON.stringify(data, null, 2)}`
    : JSON.stringify(data, null, 2);
  return {
    content: [{ type: 'text', text }],
  };
}
