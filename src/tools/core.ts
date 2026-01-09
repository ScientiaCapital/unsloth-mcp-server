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

import {
  ToolDefinition,
  ToolModule,
  ToolHandler,
  ToolContext,
  successResponse,
  errorResponse,
  jsonResponse,
} from './types.js';

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
    description:
      'List all models supported by Unsloth. Includes Llama, Mistral, Qwen, DeepSeek, and more.',
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
    description:
      'Fine-tune a model using SFT (Supervised Fine-Tuning) with LoRA. REMEMBER: SFT before GRPO!',
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
    description:
      'Export a fine-tuned model to GGUF (Ollama), merged 16-bit, or merged 4-bit format.',
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
 */
export const CORE_HANDLERS: Record<string, ToolHandler> = {
  check_installation: async (_args, ctx: ToolContext) => {
    ctx.logger.info('Checking Unsloth installation...');

    try {
      await ctx.execCommand('python -c "import unsloth"');
      return successResponse('Unsloth is properly installed.');
    } catch {
      return errorResponse('Unsloth is not installed. Please install it with: pip install unsloth');
    }
  },

  list_supported_models: async (_args, ctx: ToolContext) => {
    // Check cache first
    const cached = ctx.cache.get<string[]>('supported_models');
    if (cached) {
      ctx.logger.debug('Returning cached supported models');
      return jsonResponse(cached);
    }

    const script = `
import json
try:
    from unsloth import FastLanguageModel
    models = [
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-1B-bnb-4bit",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Llama-3.1-8B-bnb-4bit",
        "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
        "unsloth/Mistral-Small-Instruct-2409",
        "unsloth/Phi-3.5-mini-instruct",
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",
        "unsloth/Qwen-2.5-7B"
    ]
    print(json.dumps(models))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`;

    const result = await ctx.executeScript(script);
    const models = JSON.parse(result);

    if (models.error) {
      throw new Error(models.error);
    }

    // Cache for 1 hour
    ctx.cache.set('supported_models', models, 3600);
    return jsonResponse(models);
  },

  load_model: async (args, ctx: ToolContext) => {
    const {
      model_name,
      max_seq_length = 2048,
      load_in_4bit = true,
      use_gradient_checkpointing = true,
    } = args as {
      model_name: string;
      max_seq_length?: number;
      load_in_4bit?: boolean;
      use_gradient_checkpointing?: boolean;
    };

    ctx.logger.info(`Loading model: ${model_name}`);

    const script = `
import json
try:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="${model_name}",
        max_seq_length=${max_seq_length},
        load_in_4bit=${load_in_4bit ? 'True' : 'False'},
        use_gradient_checkpointing=${use_gradient_checkpointing ? '"unsloth"' : 'False'}
    )

    model_info = {
        "model_name": "${model_name}",
        "max_seq_length": ${max_seq_length},
        "load_in_4bit": ${load_in_4bit},
        "use_gradient_checkpointing": ${use_gradient_checkpointing},
        "vocab_size": tokenizer.vocab_size,
        "model_type": model.config.model_type,
        "success": True
    }

    print(json.dumps(model_info))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;

    const result = await ctx.executeScript(script);
    const modelInfo = JSON.parse(result);

    if (!modelInfo.success) {
      throw new Error(modelInfo.error);
    }

    return jsonResponse(modelInfo, `Successfully loaded model: ${model_name}`);
  },

  finetune_model: async (args, ctx: ToolContext) => {
    const {
      model_name,
      dataset_name,
      output_dir = 'outputs',
      max_seq_length = 2048,
      lora_rank = 16,
      lora_alpha = 16,
      batch_size = 2,
      gradient_accumulation_steps = 4,
      learning_rate = 2e-4,
      max_steps = 100,
      dataset_text_field = 'text',
      load_in_4bit = true,
    } = args as {
      model_name: string;
      dataset_name: string;
      output_dir?: string;
      max_seq_length?: number;
      lora_rank?: number;
      lora_alpha?: number;
      batch_size?: number;
      gradient_accumulation_steps?: number;
      learning_rate?: number;
      max_steps?: number;
      dataset_text_field?: string;
      load_in_4bit?: boolean;
    };

    ctx.logger.info(`Fine-tuning ${model_name} with ${dataset_name}`);
    ctx.logger.warn('REMEMBER: SFT before GRPO!');

    const script = `
import json
import os
try:
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    os.makedirs("${output_dir}", exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="${model_name}",
        max_seq_length=${max_seq_length},
        load_in_4bit=${load_in_4bit ? 'True' : 'False'},
        use_gradient_checkpointing="unsloth"
    )

    dataset = load_dataset("${dataset_name}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=${lora_rank},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=${lora_alpha},
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=${max_seq_length},
    )

    sft_config = SFTConfig(
        output_dir="${output_dir}",
        per_device_train_batch_size=${batch_size},
        gradient_accumulation_steps=${gradient_accumulation_steps},
        warmup_steps=5,
        max_steps=${max_steps},
        learning_rate=${learning_rate},
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="steps",
        save_steps=25,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"] if "train" in dataset else dataset,
        args=sft_config,
        dataset_text_field="${dataset_text_field}",
        max_seq_length=${max_seq_length},
    )

    stats = trainer.train()
    model.save_pretrained("${output_dir}")
    tokenizer.save_pretrained("${output_dir}")

    result = {
        "success": True,
        "output_dir": "${output_dir}",
        "training_loss": float(stats.training_loss) if hasattr(stats, 'training_loss') else None,
        "steps": ${max_steps},
    }
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;

    const result = await ctx.executeScript(script);
    const trainResult = JSON.parse(result);

    if (!trainResult.success) {
      throw new Error(trainResult.error);
    }

    return jsonResponse(trainResult, 'Fine-tuning completed successfully!');
  },

  generate_text: async (args, ctx: ToolContext) => {
    const {
      model_path,
      prompt,
      max_new_tokens = 256,
      temperature = 0.7,
      top_p = 0.9,
    } = args as {
      model_path: string;
      prompt: string;
      max_new_tokens?: number;
      temperature?: number;
      top_p?: number;
    };

    ctx.logger.info(`Generating text from ${model_path}`);

    const escapedPrompt = prompt.replace(/"/g, '\\"').replace(/\n/g, '\\n');

    const script = `
import json
try:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="${model_path}",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    inputs = tokenizer("${escapedPrompt}", return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=${max_new_tokens},
        temperature=${temperature},
        top_p=${top_p},
        do_sample=True,
        use_cache=True,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(json.dumps({
        "success": True,
        "generated_text": generated_text,
        "tokens_generated": len(outputs[0]) - len(inputs.input_ids[0])
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;

    const result = await ctx.executeScript(script);
    const genResult = JSON.parse(result);

    if (!genResult.success) {
      throw new Error(genResult.error);
    }

    return successResponse(genResult.generated_text);
  },

  export_model: async (args, ctx: ToolContext) => {
    const {
      model_path,
      format = 'gguf',
      output_path,
      quantization = 'q8_0',
    } = args as {
      model_path: string;
      format?: string;
      output_path?: string;
      quantization?: string;
    };

    ctx.logger.info(`Exporting ${model_path} to ${format}`);

    const outputDir = output_path || `${model_path}_${format}`;

    const script = `
import json
import os
try:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="${model_path}",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    os.makedirs("${outputDir}", exist_ok=True)

    format_type = "${format}"

    if format_type == "gguf":
        model.save_pretrained_gguf("${outputDir}", tokenizer, quantization_method="${quantization}")
        export_type = "GGUF (${quantization})"
    elif format_type == "merged_16bit":
        model.save_pretrained_merged("${outputDir}", tokenizer, save_method="merged_16bit")
        export_type = "Merged 16-bit"
    elif format_type == "merged_4bit":
        model.save_pretrained_merged("${outputDir}", tokenizer, save_method="merged_4bit")
        export_type = "Merged 4-bit"
    else:
        model.save_pretrained("${outputDir}")
        tokenizer.save_pretrained("${outputDir}")
        export_type = "LoRA only"

    print(json.dumps({
        "success": True,
        "output_path": "${outputDir}",
        "format": export_type,
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;

    const result = await ctx.executeScript(script);
    const exportResult = JSON.parse(result);

    if (!exportResult.success) {
      throw new Error(exportResult.error);
    }

    return jsonResponse(exportResult, `Model exported successfully to ${outputDir}`);
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
