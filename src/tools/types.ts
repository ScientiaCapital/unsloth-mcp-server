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
  logger: {
    info: (msg: string, meta?: Record<string, unknown>) => void;
    error: (msg: string, meta?: Record<string, unknown>) => void;
    debug: (msg: string, meta?: Record<string, unknown>) => void;
    warn: (msg: string, meta?: Record<string, unknown>) => void;
  };
  cache: {
    get: <T>(key: string) => T | undefined;
    set: <T>(key: string, value: T, ttl?: number) => void;
    delete: (key: string) => void;
  };
  executeUnslothScript: (script: string) => Promise<string>;
  startTime: number;
  name: string;
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
