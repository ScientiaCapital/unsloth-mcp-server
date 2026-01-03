#!/usr/bin/env node
/**
 * minimal-server.ts - A simplified MCP server with 6 core tools
 *
 * PURPOSE:
 *   Learning-focused server for understanding Unsloth fine-tuning.
 *   Start here before using the full 33-tool server.
 *
 * TOOLS (6 total):
 *   1. check_installation - Verify Unsloth setup
 *   2. list_supported_models - Get available models
 *   3. load_model - Load with Unsloth optimizations
 *   4. finetune_model - Fine-tune with LoRA/SFT
 *   5. generate_text - Text generation
 *   6. export_model - Export to GGUF/HuggingFace
 *
 * LEARNING PATH:
 *   CRAWL: Use this server with baby-steps examples
 *   WALK: Train on your own data
 *   JOG: Move to full server for more features
 *   RUN: Add GRPO (only after SFT works!)
 *
 * USAGE:
 *   npm run start:minimal
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ErrorCode,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import { exec } from 'child_process';
import { promisify } from 'util';
import {
  executePythonScript,
  executePythonScriptWithProgress,
  checkPythonEnvironment,
} from './utils/python-executor.js';

const execPromise = promisify(exec);

// Server metadata
const SERVER_NAME = 'unsloth-minimal';
const SERVER_VERSION = '1.0.0';

/**
 * Tool definitions - just 6 core tools for learning
 */
const TOOLS = [
  {
    name: 'check_installation',
    description:
      'Verify that Unsloth, PyTorch, and CUDA are properly installed. Run this first!',
    inputSchema: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'list_supported_models',
    description:
      'List all models supported by Unsloth for fine-tuning. Includes Llama, Mistral, Qwen, and more.',
    inputSchema: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'load_model',
    description:
      'Load a model with Unsloth optimizations (4-bit quantization, gradient checkpointing).',
    inputSchema: {
      type: 'object',
      properties: {
        model_name: {
          type: 'string',
          description: 'HuggingFace model name (e.g., unsloth/Qwen2.5-0.5B-Instruct)',
        },
        max_seq_length: {
          type: 'number',
          description: 'Maximum sequence length (default: 512 for learning, 2048 for production)',
          default: 512,
        },
        load_in_4bit: {
          type: 'boolean',
          description: 'Use 4-bit quantization to save memory (default: true)',
          default: true,
        },
      },
      required: ['model_name'],
    },
  },
  {
    name: 'finetune_model',
    description:
      'Fine-tune a model using SFT (Supervised Fine-Tuning). REMEMBER: SFT before GRPO!',
    inputSchema: {
      type: 'object',
      properties: {
        model_name: {
          type: 'string',
          description: 'Model to fine-tune (e.g., unsloth/Qwen2.5-0.5B-Instruct)',
        },
        dataset_name: {
          type: 'string',
          description: 'Dataset path (JSONL file) or HuggingFace dataset name',
        },
        output_dir: {
          type: 'string',
          description: 'Where to save the fine-tuned model',
          default: 'outputs',
        },
        max_steps: {
          type: 'number',
          description: 'Training steps (60 for learning, 200+ for production)',
          default: 60,
        },
        learning_rate: {
          type: 'number',
          description: 'Learning rate (2e-4 works well for most cases)',
          default: 0.0002,
        },
        lora_rank: {
          type: 'number',
          description: 'LoRA rank - higher = more capacity but slower (16 is good)',
          default: 16,
        },
      },
      required: ['model_name', 'dataset_name'],
    },
  },
  {
    name: 'generate_text',
    description: 'Generate text using a fine-tuned model. Test your training results!',
    inputSchema: {
      type: 'object',
      properties: {
        model_path: {
          type: 'string',
          description: 'Path to fine-tuned model (e.g., outputs/ or my_first_finetune/)',
        },
        prompt: {
          type: 'string',
          description: 'Input prompt to generate from',
        },
        max_new_tokens: {
          type: 'number',
          description: 'Maximum tokens to generate',
          default: 256,
        },
        temperature: {
          type: 'number',
          description: 'Sampling temperature (0.1=focused, 1.0=creative)',
          default: 0.7,
        },
      },
      required: ['model_path', 'prompt'],
    },
  },
  {
    name: 'export_model',
    description:
      'Export fine-tuned model to GGUF (for Ollama) or merged format (for HuggingFace).',
    inputSchema: {
      type: 'object',
      properties: {
        model_path: {
          type: 'string',
          description: 'Path to fine-tuned model',
        },
        format: {
          type: 'string',
          description: 'Export format: gguf, merged_16bit, merged_4bit, lora_only',
          enum: ['gguf', 'merged_16bit', 'merged_4bit', 'lora_only'],
          default: 'gguf',
        },
        quantization: {
          type: 'string',
          description: 'GGUF quantization method (q8_0 for quality, q4_k_m for size)',
          default: 'q8_0',
        },
        output_path: {
          type: 'string',
          description: 'Output path (default: model_path_format)',
        },
      },
      required: ['model_path'],
    },
  },
];

/**
 * Create and configure the MCP server
 */
function createServer(): Server {
  const server = new Server(
    { name: SERVER_NAME, version: SERVER_VERSION },
    { capabilities: { tools: {} } }
  );

  // List available tools
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOLS,
  }));

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args = {} } = request.params;

    console.error(`[${SERVER_NAME}] Tool called: ${name}`);

    try {
      switch (name) {
        case 'check_installation': {
          const env = await checkPythonEnvironment();
          let status = '✅ Ready for training!\n\n';

          if (!env.pythonVersion) {
            status = '❌ Python not found. Install Python 3.10-3.12.\n\n';
          } else if (!env.unslothAvailable) {
            status = '❌ Unsloth not installed. Run: pip install unsloth\n\n';
          } else if (!env.cudaAvailable) {
            status = '⚠️ CUDA not available. Training will be slow on CPU.\n\n';
          }

          return {
            content: [
              {
                type: 'text',
                text:
                  status +
                  JSON.stringify(
                    {
                      python: env.pythonVersion,
                      unsloth: env.unslothAvailable ? 'installed' : 'not found',
                      torch: env.torchAvailable ? 'installed' : 'not found',
                      cuda: env.cudaAvailable ? 'available' : 'not available',
                    },
                    null,
                    2
                  ),
              },
            ],
          };
        }

        case 'list_supported_models': {
          const result = await executePythonScript('list_models');
          if (!result.success) {
            throw new Error(result.error);
          }
          return {
            content: [
              {
                type: 'text',
                text: `Supported Models:\n\n${JSON.stringify(result.data, null, 2)}\n\n💡 Tip: Start with unsloth/Qwen2.5-0.5B-Instruct for learning!`,
              },
            ],
          };
        }

        case 'load_model': {
          const result = await executePythonScriptWithProgress(
            'load_model',
            args as Record<string, unknown>,
            (msg) => console.error(`[load] ${msg}`)
          );
          if (!result.success) {
            throw new Error(result.error);
          }
          return {
            content: [
              {
                type: 'text',
                text: `Model loaded successfully!\n\n${JSON.stringify(result.data, null, 2)}`,
              },
            ],
          };
        }

        case 'finetune_model': {
          console.error('[finetune] Starting SFT training...');
          console.error('[finetune] REMEMBER: SFT before GRPO!');

          const result = await executePythonScriptWithProgress(
            'finetune_model',
            args as Record<string, unknown>,
            (msg) => console.error(`[train] ${msg}`)
          );

          if (!result.success) {
            throw new Error(result.error);
          }

          return {
            content: [
              {
                type: 'text',
                text: `🎉 SFT Training Complete!\n\n${JSON.stringify(result.data, null, 2)}\n\nNext: Test with generate_text, then export_model to GGUF.`,
              },
            ],
          };
        }

        case 'generate_text': {
          const result = await executePythonScriptWithProgress(
            'generate_text',
            args as Record<string, unknown>,
            (msg) => console.error(`[generate] ${msg}`)
          );

          if (!result.success) {
            throw new Error(result.error);
          }

          const data = result.data as { prompt: string; response: string };
          return {
            content: [
              {
                type: 'text',
                text: `Prompt: ${data.prompt}\n\nResponse: ${data.response}`,
              },
            ],
          };
        }

        case 'export_model': {
          const result = await executePythonScriptWithProgress(
            'export_model',
            args as Record<string, unknown>,
            (msg) => console.error(`[export] ${msg}`)
          );

          if (!result.success) {
            throw new Error(result.error);
          }

          return {
            content: [
              {
                type: 'text',
                text: `Model exported!\n\n${JSON.stringify(result.data, null, 2)}\n\n💡 For Ollama: ollama create mymodel -f Modelfile`,
              },
            ],
          };
        }

        default:
          throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
      }
    } catch (error: any) {
      console.error(`[${SERVER_NAME}] Error in ${name}:`, error.message);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error.message}`,
          },
        ],
        isError: true,
      };
    }
  });

  return server;
}

/**
 * Main entry point
 */
async function main() {
  console.error('='.repeat(50));
  console.error('Unsloth MCP Server (Minimal - 6 Tools)');
  console.error('='.repeat(50));
  console.error('');
  console.error('Learning Path:');
  console.error('  1. check_installation - Verify setup');
  console.error('  2. list_supported_models - See available models');
  console.error('  3. load_model - Load a model');
  console.error('  4. finetune_model - Train with SFT');
  console.error('  5. generate_text - Test your model');
  console.error('  6. export_model - Export to GGUF');
  console.error('');
  console.error('⚠️  REMEMBER: SFT before GRPO!');
  console.error('='.repeat(50));

  const server = createServer();
  const transport = new StdioServerTransport();

  await server.connect(transport);

  console.error('[minimal-server] Connected and ready!');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
