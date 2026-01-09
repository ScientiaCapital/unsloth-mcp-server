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

import { ToolDefinition, ToolModule, ToolHandler } from './types.js';

/**
 * Tool definitions
 */
export const TRAINING_TOOLS: ToolDefinition[] = [
  {
    name: 'get_model_info',
    description:
      'Get detailed information about a model including architecture, parameters, and config.',
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
          description:
            'Path to training prompts JSON (default: models/configs/grpo/grpo_prompts.json)',
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
 * Handler implementations
 */
export const TRAINING_HANDLERS: Record<string, ToolHandler> = {
  get_model_info: async (args, ctx) => {
    const { model_name } = args as { model_name: string };
    ctx.logger.info(`Getting info for: ${model_name}`);

    const script = `
import json
try:
    from transformers import AutoConfig, AutoTokenizer
    import torch

    # Load model config
    config = AutoConfig.from_pretrained("${model_name}")

    # Try to load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("${model_name}")
        vocab_size = tokenizer.vocab_size
        model_max_length = tokenizer.model_max_length
    except:
        vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else "Unknown"
        model_max_length = "Unknown"

    # Get parameter count estimate
    def estimate_parameters(config):
        if hasattr(config, 'num_parameters'):
            return config.num_parameters

        # Estimate based on architecture
        hidden_size = getattr(config, 'hidden_size', 0)
        num_layers = getattr(config, 'num_hidden_layers', 0)
        vocab = getattr(config, 'vocab_size', 0)

        if hidden_size and num_layers and vocab:
            # Rough estimate: vocab * hidden + layers * (4 * hidden^2)
            embedding_params = vocab * hidden_size
            layer_params = num_layers * (4 * hidden_size * hidden_size)
            return embedding_params + layer_params
        return "Unknown"

    model_info = {
        "model_name": "${model_name}",
        "architecture": config.architectures[0] if hasattr(config, 'architectures') else config.model_type,
        "model_type": config.model_type,
        "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else "Unknown",
        "num_layers": config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else "Unknown",
        "num_attention_heads": config.num_attention_heads if hasattr(config, 'num_attention_heads') else "Unknown",
        "vocab_size": vocab_size,
        "max_position_embeddings": config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else "Unknown",
        "model_max_length": model_max_length,
        "estimated_parameters": estimate_parameters(config),
        "torch_dtype": str(config.torch_dtype) if hasattr(config, 'torch_dtype') else "Unknown",
        "success": True
    }

    print(json.dumps(model_info))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
    const result = await ctx.executeScript(script);

    try {
      const modelInfo = JSON.parse(result);
      if (!modelInfo.success) {
        throw new Error(modelInfo.error);
      }

      return {
        content: [
          {
            type: 'text',
            text: `Model Information for ${model_name}:\n\n${JSON.stringify(modelInfo, null, 2)}`,
          },
        ],
      };
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      throw new Error(`Error getting model info: ${msg}`);
    }
  },

  benchmark_model: async (args, ctx) => {
    const {
      model_name,
      prompt,
      num_iterations = 10,
      max_new_tokens = 128,
    } = args as {
      model_name: string;
      prompt: string;
      num_iterations?: number;
      max_new_tokens?: number;
    };

    ctx.logger.info(`Benchmarking: ${model_name}`);

    const script = `
import json
import time
try:
    from unsloth import FastLanguageModel
    import torch
    import psutil
    import os

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="${model_name}",
        max_seq_length=2048,
        load_in_4bit=True
    )

    # Prepare model for inference
    FastLanguageModel.for_inference(model)

    # Warm-up run
    inputs = tokenizer("${prompt.replace(/"/g, '\\"')}", return_tensors="pt").to(model.device)
    _ = model.generate(**inputs, max_new_tokens=10)

    # Get initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Benchmark runs
    times = []
    tokens_per_second = []

    for i in range(${num_iterations}):
        inputs = tokenizer("${prompt.replace(/"/g, '\\"')}", return_tensors="pt").to(model.device)

        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=${max_new_tokens})
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)
        tokens_per_second.append(${max_new_tokens} / elapsed)

    # Get final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB

    benchmark_results = {
        "model_name": "${model_name}",
        "num_iterations": ${num_iterations},
        "max_new_tokens": ${max_new_tokens},
        "avg_time_seconds": round(sum(times) / len(times), 3),
        "min_time_seconds": round(min(times), 3),
        "max_time_seconds": round(max(times), 3),
        "avg_tokens_per_second": round(sum(tokens_per_second) / len(tokens_per_second), 2),
        "memory_used_mb": round(final_memory - initial_memory, 2),
        "total_memory_mb": round(final_memory, 2),
        "success": True
    }

    print(json.dumps(benchmark_results))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
    const result = await ctx.executeScript(script);

    try {
      const benchmarkResults = JSON.parse(result);
      if (!benchmarkResults.success) {
        throw new Error(benchmarkResults.error);
      }

      return {
        content: [
          {
            type: 'text',
            text: `Benchmark Results for ${model_name}:\n\n${JSON.stringify(benchmarkResults, null, 2)}`,
          },
        ],
      };
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      throw new Error(`Error benchmarking model: ${msg}`);
    }
  },

  list_datasets: async (args, ctx) => {
    const { search_query = '', limit = 20 } = args as {
      search_query?: string;
      limit?: number;
    };

    ctx.logger.info(`Searching datasets: ${search_query}`);

    const script = `
import json
try:
    from huggingface_hub import list_datasets

    # List datasets
    datasets = list_datasets(
        search="${search_query}",
        limit=${limit},
        sort="downloads",
        direction=-1
    )

    dataset_list = []
    for dataset in datasets:
        dataset_list.append({
            "id": dataset.id,
            "author": dataset.author if hasattr(dataset, 'author') else "Unknown",
            "downloads": dataset.downloads if hasattr(dataset, 'downloads') else 0,
            "likes": dataset.likes if hasattr(dataset, 'likes') else 0,
            "tags": dataset.tags[:5] if hasattr(dataset, 'tags') else []
        })

    result = {
        "query": "${search_query}",
        "count": len(dataset_list),
        "datasets": dataset_list,
        "success": True
    }

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
    const result = await ctx.executeScript(script);

    try {
      const datasetList = JSON.parse(result);
      if (!datasetList.success) {
        throw new Error(datasetList.error);
      }

      return {
        content: [
          {
            type: 'text',
            text: `Available Datasets${search_query ? ` (search: "${search_query}")` : ''}:\n\n${JSON.stringify(datasetList, null, 2)}`,
          },
        ],
      };
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      throw new Error(`Error listing datasets: ${msg}`);
    }
  },

  prepare_dataset: async (args, ctx) => {
    const {
      dataset_name,
      output_path,
      text_field = 'text',
      format = 'jsonl',
    } = args as {
      dataset_name: string;
      output_path: string;
      text_field?: string;
      format?: 'json' | 'jsonl' | 'csv';
    };

    ctx.logger.info(`Preparing ${dataset_name} in ${format} format`);

    const script = `
import json
import os
try:
    from datasets import load_dataset
    import pandas as pd

    # Load dataset
    dataset = load_dataset("${dataset_name}")

    # Get training split
    train_data = dataset["train"]

    # Create output directory if needed
    os.makedirs(os.path.dirname("${output_path}") if os.path.dirname("${output_path}") else ".", exist_ok=True)

    # Prepare data with proper formatting
    if "${format}" == "jsonl":
        with open("${output_path}", "w") as f:
            for item in train_data:
                if "${text_field}" in item:
                    f.write(json.dumps({"text": item["${text_field}"]}) + "\\n")
    elif "${format}" == "json":
        prepared_data = []
        for item in train_data:
            if "${text_field}" in item:
                prepared_data.append({"text": item["${text_field}"]})
        with open("${output_path}", "w") as f:
            json.dump(prepared_data, f, indent=2)
    elif "${format}" == "csv":
        df_data = []
        for item in train_data:
            if "${text_field}" in item:
                df_data.append({"text": item["${text_field}"]})
        df = pd.DataFrame(df_data)
        df.to_csv("${output_path}", index=False)

    result = {
        "dataset_name": "${dataset_name}",
        "output_path": "${output_path}",
        "format": "${format}",
        "num_examples": len(train_data),
        "text_field": "${text_field}",
        "success": True
    }

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
    const result = await ctx.executeScript(script);

    try {
      const prepareResult = JSON.parse(result);
      if (!prepareResult.success) {
        throw new Error(prepareResult.error);
      }

      return {
        content: [
          {
            type: 'text',
            text: `Successfully prepared dataset:\n\n${JSON.stringify(prepareResult, null, 2)}`,
          },
        ],
      };
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      throw new Error(`Error preparing dataset: ${msg}`);
    }
  },

  compare_tokenizers: async (args, ctx) => {
    const {
      tokenizer_a,
      tokenizer_b,
      test_texts = [],
    } = args as {
      tokenizer_a: string;
      tokenizer_b: string;
      test_texts?: string[];
    };

    ctx.logger.info(`Comparing ${tokenizer_a} vs ${tokenizer_b}`);

    const textsJson = JSON.stringify(
      test_texts.length > 0 ? test_texts : ['Hello world!', 'This is a test sentence.']
    );

    const script = `
import json
try:
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer

    # Try loading as HuggingFace tokenizer first, then as raw tokenizer
    def load_tokenizer(path):
        try:
            return AutoTokenizer.from_pretrained(path)
        except:
            return Tokenizer.from_file(path)

    tokenizer1 = load_tokenizer("${tokenizer_a}")
    tokenizer2 = load_tokenizer("${tokenizer_b}")

    test_texts = ${textsJson}

    results = []
    for text in test_texts:
        # Get token counts
        if hasattr(tokenizer1, 'encode'):
            tokens1 = tokenizer1.encode(text)
            count1 = len(tokens1) if isinstance(tokens1, list) else len(tokens1.ids)
        else:
            tokens1 = tokenizer1.encode(text)
            count1 = len(tokens1.ids)

        if hasattr(tokenizer2, 'encode'):
            tokens2 = tokenizer2.encode(text)
            count2 = len(tokens2) if isinstance(tokens2, list) else len(tokens2.ids)
        else:
            tokens2 = tokenizer2.encode(text)
            count2 = len(tokens2.ids)

        results.append({
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "tokenizer_a_tokens": count1,
            "tokenizer_b_tokens": count2,
            "difference": count1 - count2,
            "efficiency_gain": round((count1 - count2) / count1 * 100, 2) if count1 > 0 else 0
        })

    # Calculate averages
    avg_efficiency = sum(r["efficiency_gain"] for r in results) / len(results) if results else 0

    print(json.dumps({
        "success": True,
        "tokenizer_a": "${tokenizer_a}",
        "tokenizer_b": "${tokenizer_b}",
        "comparisons": results,
        "avg_efficiency_gain": round(avg_efficiency, 2),
        "summary": f"Tokenizer B is {abs(round(avg_efficiency, 1))}% {'more' if avg_efficiency > 0 else 'less'} efficient than Tokenizer A"
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
    const result = await ctx.executeScript(script);

    try {
      const compareResult = JSON.parse(result);
      if (!compareResult.success) {
        throw new Error(compareResult.error);
      }

      return {
        content: [
          {
            type: 'text',
            text: `Tokenizer Comparison:\n\n${JSON.stringify(compareResult, null, 2)}`,
          },
        ],
      };
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      throw new Error(`Error comparing tokenizers: ${msg}`);
    }
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
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: false,
                error: `SFT adapter not found: ${sft_adapter_path}`,
                hint: 'You must train with finetune_model (SFT) first before using GRPO!',
                learning_path: [
                  '1. FIRST: Use finetune_model to train on your data',
                  '2. WAIT: Verify SFT model generates correct format',
                  '3. THEN: Use train_grpo_model with the SFT adapter path',
                ],
              },
              null,
              2
            ),
          },
        ],
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
      content: [
        {
          type: 'text',
          text: JSON.stringify(
            {
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
            },
            null,
            2
          ),
        },
      ],
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
