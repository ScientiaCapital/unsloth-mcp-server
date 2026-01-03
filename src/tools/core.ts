/**
 * core.ts - Core Unsloth tools (6 tools)
 *
 * The essential tools for fine-tuning LLMs with Unsloth.
 * Start here for learning, then explore other modules.
 *
 * TOOLS:
 *   1. check_installation - Verify Unsloth setup
 *   2. list_supported_models - Get available models
 *   3. load_model - Load with optimizations
 *   4. finetune_model - Fine-tune with SFT
 *   5. generate_text - Text generation
 *   6. export_model - Export to GGUF
 */

import { ToolDefinition, ToolModule, ToolHandler, successResponse, jsonResponse, errorResponse } from './types.js';

/**
 * Tool definitions
 */
export const CORE_TOOLS: ToolDefinition[] = [
  {
    name: 'check_installation',
    description: 'Verify that Unsloth, PyTorch, and CUDA are properly installed. Run this first!',
    inputSchema: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'list_supported_models',
    description: 'List all models supported by Unsloth. Includes Llama, Mistral, Qwen, DeepSeek, and more.',
    inputSchema: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'load_model',
    description: 'Load a model with Unsloth optimizations (4-bit quantization, gradient checkpointing).',
    inputSchema: {
      type: 'object',
      properties: {
        model_name: {
          type: 'string',
          description: 'HuggingFace model name (e.g., unsloth/Qwen2.5-0.5B-Instruct)',
        },
        max_seq_length: {
          type: 'number',
          description: 'Maximum sequence length (default: 2048)',
          default: 2048,
        },
        load_in_4bit: {
          type: 'boolean',
          description: 'Use 4-bit quantization (default: true)',
          default: true,
        },
        use_gradient_checkpointing: {
          type: 'boolean',
          description: 'Enable gradient checkpointing (default: true)',
          default: true,
        },
      },
      required: ['model_name'],
    },
  },
  {
    name: 'finetune_model',
    description: 'Fine-tune a model using SFT (Supervised Fine-Tuning) with LoRA. REMEMBER: SFT before GRPO!',
    inputSchema: {
      type: 'object',
      properties: {
        model_name: {
          type: 'string',
          description: 'Model to fine-tune',
        },
        dataset_name: {
          type: 'string',
          description: 'Dataset path or HuggingFace dataset name',
        },
        output_dir: {
          type: 'string',
          description: 'Output directory (default: outputs)',
          default: 'outputs',
        },
        max_seq_length: {
          type: 'number',
          description: 'Maximum sequence length (default: 512)',
          default: 512,
        },
        lora_rank: {
          type: 'number',
          description: 'LoRA rank (default: 16)',
          default: 16,
        },
        lora_alpha: {
          type: 'number',
          description: 'LoRA alpha (default: 16)',
          default: 16,
        },
        learning_rate: {
          type: 'number',
          description: 'Learning rate (default: 2e-4)',
          default: 0.0002,
        },
        batch_size: {
          type: 'number',
          description: 'Per-device batch size (default: 2)',
          default: 2,
        },
        gradient_accumulation_steps: {
          type: 'number',
          description: 'Gradient accumulation steps (default: 4)',
          default: 4,
        },
        max_steps: {
          type: 'number',
          description: 'Max training steps (default: 60)',
          default: 60,
        },
        dataset_text_field: {
          type: 'string',
          description: 'Dataset text field name (default: text)',
          default: 'text',
        },
        load_in_4bit: {
          type: 'boolean',
          description: 'Use 4-bit quantization (default: true)',
          default: true,
        },
      },
      required: ['model_name', 'dataset_name'],
    },
  },
  {
    name: 'generate_text',
    description: 'Generate text using a loaded or fine-tuned model.',
    inputSchema: {
      type: 'object',
      properties: {
        model_path: {
          type: 'string',
          description: 'Path to the model or adapter',
        },
        prompt: {
          type: 'string',
          description: 'Input prompt',
        },
        max_new_tokens: {
          type: 'number',
          description: 'Maximum new tokens (default: 256)',
          default: 256,
        },
        temperature: {
          type: 'number',
          description: 'Sampling temperature (default: 0.7)',
          default: 0.7,
        },
        top_p: {
          type: 'number',
          description: 'Top-p sampling (default: 0.9)',
          default: 0.9,
        },
      },
      required: ['model_path', 'prompt'],
    },
  },
  {
    name: 'export_model',
    description: 'Export a fine-tuned model to GGUF (Ollama), merged 16-bit, or merged 4-bit format.',
    inputSchema: {
      type: 'object',
      properties: {
        model_path: {
          type: 'string',
          description: 'Path to the fine-tuned model',
        },
        format: {
          type: 'string',
          description: 'Export format',
          enum: ['gguf', 'merged_16bit', 'merged_4bit', 'lora_only'],
          default: 'gguf',
        },
        quantization: {
          type: 'string',
          description: 'GGUF quantization method (default: q8_0)',
          default: 'q8_0',
        },
        output_path: {
          type: 'string',
          description: 'Output path (optional)',
        },
      },
      required: ['model_path'],
    },
  },
];

/**
 * Handler implementations
 * Note: These are stubs - the actual implementations use executeUnslothScript
 * which is defined in index.ts. This module is for organization and typing.
 */
export const CORE_HANDLERS: Record<string, ToolHandler> = {
  check_installation: async (_args, ctx) => {
    ctx.logger.info('Checking Unsloth installation...');
    // Actual implementation delegates to Python script
    return successResponse('Use executeUnslothScript for implementation');
  },

  list_supported_models: async (_args, ctx) => {
    ctx.logger.info('Listing supported models...');
    return successResponse('Use executeUnslothScript for implementation');
  },

  load_model: async (args, ctx) => {
    const { model_name } = args as { model_name: string };
    ctx.logger.info(`Loading model: ${model_name}`);
    return successResponse('Use executeUnslothScript for implementation');
  },

  finetune_model: async (args, ctx) => {
    const { model_name, dataset_name } = args as { model_name: string; dataset_name: string };
    ctx.logger.info(`Fine-tuning ${model_name} with ${dataset_name}`);
    ctx.logger.warn('REMEMBER: SFT before GRPO!');
    return successResponse('Use executeUnslothScript for implementation');
  },

  generate_text: async (args, ctx) => {
    const { model_path, prompt } = args as { model_path: string; prompt: string };
    ctx.logger.info(`Generating from ${model_path}`);
    return successResponse('Use executeUnslothScript for implementation');
  },

  export_model: async (args, ctx) => {
    const { model_path, format = 'gguf' } = args as { model_path: string; format?: string };
    ctx.logger.info(`Exporting ${model_path} to ${format}`);
    return successResponse('Use executeUnslothScript for implementation');
  },
};

/**
 * Module export
 */
export const coreModule: ToolModule = {
  tools: CORE_TOOLS,
  handlers: CORE_HANDLERS,
};

export default coreModule;
