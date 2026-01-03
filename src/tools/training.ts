/**
 * training.ts - Model info, benchmark, and GRPO tools (6 tools)
 *
 * Tools for understanding models, measuring performance, and advanced training.
 *
 * TOOLS:
 *   1. get_model_info - Get model architecture details
 *   2. benchmark_model - Performance benchmarking
 *   3. list_datasets - Search HuggingFace datasets
 *   4. prepare_dataset - Format datasets for training
 *   5. compare_tokenizers - Compare tokenization efficiency
 *   6. train_grpo_model - GRPO preference optimization (ONLY after SFT!)
 */

import { ToolDefinition, ToolModule, ToolHandler, successResponse } from './types.js';

/**
 * Tool definitions
 */
export const TRAINING_TOOLS: ToolDefinition[] = [
  {
    name: 'get_model_info',
    description: 'Get detailed information about a model including architecture, parameters, and config.',
    inputSchema: {
      type: 'object',
      properties: {
        model_name: {
          type: 'string',
          description: 'HuggingFace model name or local path',
        },
      },
      required: ['model_name'],
    },
  },
  {
    name: 'benchmark_model',
    description: 'Benchmark model inference performance (tokens/second, memory usage).',
    inputSchema: {
      type: 'object',
      properties: {
        model_name: {
          type: 'string',
          description: 'Model to benchmark',
        },
        prompt: {
          type: 'string',
          description: 'Test prompt for benchmarking',
        },
        num_iterations: {
          type: 'number',
          description: 'Number of benchmark iterations (default: 5)',
          default: 5,
        },
        max_new_tokens: {
          type: 'number',
          description: 'Tokens per generation (default: 128)',
          default: 128,
        },
      },
      required: ['model_name', 'prompt'],
    },
  },
  {
    name: 'list_datasets',
    description: 'Search HuggingFace datasets for training data.',
    inputSchema: {
      type: 'object',
      properties: {
        search_query: {
          type: 'string',
          description: 'Search query (e.g., "instruction", "chat", "sales")',
          default: '',
        },
        limit: {
          type: 'number',
          description: 'Maximum results (default: 20)',
          default: 20,
        },
      },
      required: [],
    },
  },
  {
    name: 'prepare_dataset',
    description: 'Format a dataset for training (Alpaca, ShareGPT, ChatML formats).',
    inputSchema: {
      type: 'object',
      properties: {
        dataset_name: {
          type: 'string',
          description: 'Dataset name or path',
        },
        format: {
          type: 'string',
          description: 'Target format',
          enum: ['alpaca', 'sharegpt', 'chatml'],
          default: 'chatml',
        },
        output_path: {
          type: 'string',
          description: 'Output file path',
        },
        max_samples: {
          type: 'number',
          description: 'Maximum samples to process',
        },
      },
      required: ['dataset_name', 'output_path'],
    },
  },
  {
    name: 'compare_tokenizers',
    description: 'Compare tokenization efficiency between different tokenizers.',
    inputSchema: {
      type: 'object',
      properties: {
        tokenizer_a: {
          type: 'string',
          description: 'First tokenizer (model name or path)',
        },
        tokenizer_b: {
          type: 'string',
          description: 'Second tokenizer (model name or path)',
        },
        test_texts: {
          type: 'array',
          description: 'Test texts to compare',
          items: { type: 'string' },
        },
      },
      required: ['tokenizer_a', 'tokenizer_b'],
    },
  },
  {
    name: 'train_grpo_model',
    description: `GRPO (Group Relative Policy Optimization) training for preference alignment.

CRITICAL: Only use AFTER successful SFT training! GRPO optimizes preferences, not basic skills.

Learning path:
1. FIRST: finetune_model (SFT) → Model learns FORMAT and DOMAIN
2. THEN: train_grpo_model → Model learns PREFERENCES (what responses are better)

GRPO works by:
- Generating multiple responses to each prompt
- Scoring with reward functions (length, keywords, structure, sales techniques)
- Learning to prefer higher-reward responses

Config location: models/configs/grpo/grpo_config.yaml`,
    inputSchema: {
      type: 'object',
      properties: {
        config_path: {
          type: 'string',
          description: 'Path to GRPO config YAML (default: models/configs/grpo/grpo_config.yaml)',
        },
        prompts_path: {
          type: 'string',
          description: 'Path to training prompts JSON (default: models/configs/grpo/grpo_prompts.json)',
        },
        iteration: {
          type: 'number',
          description: 'Resume from specific iteration (default: use config)',
        },
        output_dir: {
          type: 'string',
          description: 'Output directory for trained model (default: grpo_experiments)',
        },
        sft_adapter_path: {
          type: 'string',
          description: 'REQUIRED: Path to SFT-trained adapter to improve upon',
        },
        // GRPO hyperparameters (overrides config)
        learning_rate: {
          type: 'number',
          description: 'Learning rate (default: 2e-5, lower than SFT)',
        },
        beta: {
          type: 'number',
          description: 'KL penalty coefficient (default: 0.1)',
        },
        num_generations: {
          type: 'number',
          description: 'Responses per prompt for comparison (default: 4)',
        },
        num_epochs: {
          type: 'number',
          description: 'Training epochs (default: 1, GRPO converges fast)',
        },
      },
      required: ['sft_adapter_path'],
    },
  },
];

/**
 * Handler stubs
 */
export const TRAINING_HANDLERS: Record<string, ToolHandler> = {
  get_model_info: async (args, ctx) => {
    const { model_name } = args as { model_name: string };
    ctx.logger.info(`Getting info for: ${model_name}`);
    return successResponse('Use executeUnslothScript for implementation');
  },

  benchmark_model: async (args, ctx) => {
    const { model_name, prompt } = args as { model_name: string; prompt: string };
    ctx.logger.info(`Benchmarking: ${model_name}`);
    return successResponse('Use executeUnslothScript for implementation');
  },

  list_datasets: async (args, ctx) => {
    const { search_query = '' } = args as { search_query?: string };
    ctx.logger.info(`Searching datasets: ${search_query}`);
    return successResponse('Use executeUnslothScript for implementation');
  },

  prepare_dataset: async (args, ctx) => {
    const { dataset_name, format } = args as { dataset_name: string; format: string };
    ctx.logger.info(`Preparing ${dataset_name} in ${format} format`);
    return successResponse('Use executeUnslothScript for implementation');
  },

  compare_tokenizers: async (args, ctx) => {
    const { tokenizer_a, tokenizer_b } = args as { tokenizer_a: string; tokenizer_b: string };
    ctx.logger.info(`Comparing ${tokenizer_a} vs ${tokenizer_b}`);
    return successResponse('Use executeUnslothScript for implementation');
  },

  train_grpo_model: async (args, ctx) => {
    const {
      sft_adapter_path,
      config_path = 'models/configs/grpo/grpo_config.yaml',
      prompts_path = 'models/configs/grpo/grpo_prompts.json',
      iteration,
      output_dir = 'grpo_experiments',
      learning_rate,
      beta,
      num_generations,
      num_epochs,
    } = args as {
      sft_adapter_path: string;
      config_path?: string;
      prompts_path?: string;
      iteration?: number;
      output_dir?: string;
      learning_rate?: number;
      beta?: number;
      num_generations?: number;
      num_epochs?: number;
    };

    ctx.logger.info('='.repeat(60));
    ctx.logger.info('GRPO Training - Preference Optimization');
    ctx.logger.info('='.repeat(60));
    ctx.logger.info(`SFT adapter: ${sft_adapter_path}`);
    ctx.logger.info(`Config: ${config_path}`);

    // Validate SFT adapter exists
    const fs = await import('fs');
    if (!fs.existsSync(sft_adapter_path)) {
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            success: false,
            error: `SFT adapter not found: ${sft_adapter_path}`,
            hint: 'You must train with finetune_model (SFT) first before using GRPO!',
            learning_path: [
              '1. FIRST: Use finetune_model to train on your data',
              '2. WAIT: Verify SFT model generates correct format',
              '3. THEN: Use train_grpo_model with the SFT adapter path',
            ],
          }, null, 2),
        }],
      };
    }

    // Build command for GRPO training script
    const scriptArgs: string[] = [];
    if (iteration !== undefined) {
      scriptArgs.push(`--iteration=${iteration}`);
    }

    // Log hyperparameter overrides
    if (learning_rate) ctx.logger.info(`LR override: ${learning_rate}`);
    if (beta) ctx.logger.info(`Beta override: ${beta}`);
    if (num_generations) ctx.logger.info(`Generations: ${num_generations}`);

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          success: true,
          message: 'GRPO training initialized',
          sft_adapter: sft_adapter_path,
          config: config_path,
          prompts: prompts_path,
          output_dir: output_dir,
          hyperparameters: {
            learning_rate: learning_rate || '2e-5 (default)',
            beta: beta || '0.1 (default)',
            num_generations: num_generations || '4 (default)',
            num_epochs: num_epochs || '1 (default)',
          },
          next_steps: [
            'Run the GRPO training script:',
            `  python models/configs/grpo/train_grpo.py`,
            '',
            'Or use RunPod for GPU acceleration:',
            '  Use runpod_start_training with script=train_grpo.py',
          ],
          note: 'GRPO training requires GPU. Use RunPod or Colab for best results.',
        }, null, 2),
      }],
    };
  },
};

/**
 * Module export
 */
export const trainingModule: ToolModule = {
  tools: TRAINING_TOOLS,
  handlers: TRAINING_HANDLERS,
};

export default trainingModule;
