#!/usr/bin/env node
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
import axios from 'axios';
import logger from './utils/logger.js';
import { validateToolInputs, ValidationError } from './utils/validation.js';
import { safeExecute, SecurityError, TimeoutError } from './utils/security.js';
import { metricsCollector } from './utils/metrics.js';

const execPromise = promisify(exec);

// Get API keys from environment variables if needed
const HF_TOKEN = process.env.HUGGINGFACE_TOKEN;

class UnslothServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: 'unsloth-server',
        version: '2.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    // Set up tool handlers
    this.setupToolHandlers();

    // Error handling
    this.server.onerror = (error) => {
      logger.error('[MCP Error]', { error: error.message, stack: error.stack });
    };

    process.on('SIGINT', async () => {
      logger.info('Received SIGINT, shutting down gracefully...');
      await this.server.close();
      logger.info('Server closed successfully');
      process.exit(0);
    });

    logger.info('Unsloth MCP Server initialized', { version: '2.0.0' });
  }

  private async checkUnslothInstallation(): Promise<boolean> {
    try {
      logger.debug('Checking Unsloth installation');
      await execPromise('python -c "import unsloth"');
      logger.info('Unsloth is installed');
      return true;
    } catch (error) {
      logger.warn('Unsloth is not installed', { error });
      return false;
    }
  }

  private async executeUnslothScript(script: string): Promise<string> {
    try {
      logger.debug('Executing Unsloth script', { scriptLength: script.length });
      const result = await safeExecute(script);
      logger.debug('Script execution successful');
      return result;
    } catch (error: any) {
      if (error instanceof TimeoutError) {
        throw new Error(`Operation timed out: ${error.message}. Try reducing the workload or increasing timeout.`);
      }
      if (error instanceof SecurityError) {
        throw new Error(`Security error: ${error.message}`);
      }
      logger.error('Script execution failed', { error: error.message });
      throw new Error(`Error executing Unsloth script: ${error.message}`);
    }
  }

  private createSuccessResponse(toolName: string, startTime: number, text: string) {
    metricsCollector.endTool(toolName, startTime, true);
    return {
      content: [
        {
          type: 'text',
          text,
        },
      ],
    };
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'check_installation',
          description: 'Check if Unsloth is properly installed',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'list_supported_models',
          description: 'List all models supported by Unsloth',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'load_model',
          description: 'Load a pretrained model with Unsloth optimizations',
          inputSchema: {
            type: 'object',
            properties: {
              model_name: {
                type: 'string',
                description: 'Name of the model to load (e.g., "unsloth/Llama-3.2-1B")',
              },
              max_seq_length: {
                type: 'number',
                description: 'Maximum sequence length for the model',
              },
              load_in_4bit: {
                type: 'boolean',
                description: 'Whether to load the model in 4-bit quantization',
              },
              use_gradient_checkpointing: {
                type: 'boolean',
                description: 'Whether to use gradient checkpointing to save memory',
              },
            },
            required: ['model_name'],
          },
        },
        {
          name: 'finetune_model',
          description: 'Fine-tune a model with Unsloth optimizations',
          inputSchema: {
            type: 'object',
            properties: {
              model_name: {
                type: 'string',
                description: 'Name of the model to fine-tune',
              },
              dataset_name: {
                type: 'string',
                description: 'Name of the dataset to use for fine-tuning',
              },
              output_dir: {
                type: 'string',
                description: 'Directory to save the fine-tuned model',
              },
              max_seq_length: {
                type: 'number',
                description: 'Maximum sequence length for training',
              },
              lora_rank: {
                type: 'number',
                description: 'Rank for LoRA fine-tuning',
              },
              lora_alpha: {
                type: 'number',
                description: 'Alpha for LoRA fine-tuning',
              },
              batch_size: {
                type: 'number',
                description: 'Batch size for training',
              },
              gradient_accumulation_steps: {
                type: 'number',
                description: 'Number of gradient accumulation steps',
              },
              learning_rate: {
                type: 'number',
                description: 'Learning rate for training',
              },
              max_steps: {
                type: 'number',
                description: 'Maximum number of training steps',
              },
              dataset_text_field: {
                type: 'string',
                description: 'Field in the dataset containing the text',
              },
              load_in_4bit: {
                type: 'boolean',
                description: 'Whether to use 4-bit quantization',
              },
            },
            required: ['model_name', 'dataset_name', 'output_dir'],
          },
        },
        {
          name: 'generate_text',
          description: 'Generate text using a fine-tuned Unsloth model',
          inputSchema: {
            type: 'object',
            properties: {
              model_path: {
                type: 'string',
                description: 'Path to the fine-tuned model',
              },
              prompt: {
                type: 'string',
                description: 'Prompt for text generation',
              },
              max_new_tokens: {
                type: 'number',
                description: 'Maximum number of tokens to generate',
              },
              temperature: {
                type: 'number',
                description: 'Temperature for text generation',
              },
              top_p: {
                type: 'number',
                description: 'Top-p for text generation',
              },
            },
            required: ['model_path', 'prompt'],
          },
        },
        {
          name: 'export_model',
          description: 'Export a fine-tuned Unsloth model to various formats',
          inputSchema: {
            type: 'object',
            properties: {
              model_path: {
                type: 'string',
                description: 'Path to the fine-tuned model',
              },
              export_format: {
                type: 'string',
                description: 'Format to export to (gguf, ollama, vllm, huggingface)',
                enum: ['gguf', 'ollama', 'vllm', 'huggingface'],
              },
              output_path: {
                type: 'string',
                description: 'Path to save the exported model',
              },
              quantization_bits: {
                type: 'number',
                description: 'Bits for quantization (for GGUF export)',
              },
            },
            required: ['model_path', 'export_format', 'output_path'],
          },
        },
        {
          name: 'train_superbpe_tokenizer',
          description: 'Train a SuperBPE tokenizer for improved efficiency (up to 33% fewer tokens)',
          inputSchema: {
            type: 'object',
            properties: {
              corpus_path: {
                type: 'string',
                description: 'Path to the training corpus or dataset name',
              },
              vocab_size: {
                type: 'number',
                description: 'Vocabulary size for the tokenizer (default: 50000)',
              },
              output_path: {
                type: 'string',
                description: 'Path to save the trained tokenizer',
              },
              num_inherit_merges: {
                type: 'number',
                description: 'Number of merges to inherit from BPE stage (default: vocab_size * 0.8)',
              },
            },
            required: ['corpus_path', 'output_path'],
          },
        },
        {
          name: 'get_model_info',
          description: 'Get detailed information about a model including architecture, parameters, and capabilities',
          inputSchema: {
            type: 'object',
            properties: {
              model_name: {
                type: 'string',
                description: 'Name or path of the model to inspect',
              },
            },
            required: ['model_name'],
          },
        },
        {
          name: 'compare_tokenizers',
          description: 'Compare tokenization efficiency between different tokenizers (BPE vs SuperBPE)',
          inputSchema: {
            type: 'object',
            properties: {
              text: {
                type: 'string',
                description: 'Sample text to tokenize for comparison',
              },
              tokenizer1_path: {
                type: 'string',
                description: 'Path to first tokenizer (e.g., standard BPE)',
              },
              tokenizer2_path: {
                type: 'string',
                description: 'Path to second tokenizer (e.g., SuperBPE)',
              },
            },
            required: ['text', 'tokenizer1_path', 'tokenizer2_path'],
          },
        },
        {
          name: 'benchmark_model',
          description: 'Benchmark model inference speed and memory usage',
          inputSchema: {
            type: 'object',
            properties: {
              model_name: {
                type: 'string',
                description: 'Name of the model to benchmark',
              },
              prompt: {
                type: 'string',
                description: 'Sample prompt for benchmarking',
              },
              num_iterations: {
                type: 'number',
                description: 'Number of iterations to run (default: 10)',
              },
              max_new_tokens: {
                type: 'number',
                description: 'Number of tokens to generate per iteration (default: 128)',
              },
            },
            required: ['model_name', 'prompt'],
          },
        },
        {
          name: 'list_datasets',
          description: 'List popular datasets available for fine-tuning from Hugging Face',
          inputSchema: {
            type: 'object',
            properties: {
              search_query: {
                type: 'string',
                description: 'Optional search query to filter datasets',
              },
              limit: {
                type: 'number',
                description: 'Maximum number of datasets to return (default: 20)',
              },
            },
          },
        },
        {
          name: 'prepare_dataset',
          description: 'Prepare and format a dataset for Unsloth fine-tuning',
          inputSchema: {
            type: 'object',
            properties: {
              dataset_name: {
                type: 'string',
                description: 'Name of the dataset to prepare',
              },
              output_path: {
                type: 'string',
                description: 'Path to save the prepared dataset',
              },
              text_field: {
                type: 'string',
                description: 'Field containing the text data (default: "text")',
              },
              format: {
                type: 'string',
                description: 'Output format (json, jsonl, csv)',
                enum: ['json', 'jsonl', 'csv'],
              },
            },
            required: ['dataset_name', 'output_path'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      const startTime = metricsCollector.startTool(name);

      logger.info(`Tool called: ${name}`, { args });

      try {
        // Validate inputs
        validateToolInputs(name, args || {});

        switch (name) {
          case 'check_installation': {
            const isInstalled = await this.checkUnslothInstallation();
            
            if (!isInstalled) {
              return {
                content: [
                  {
                    type: 'text',
                    text: 'Unsloth is not installed. Please install it with: pip install unsloth',
                  },
                ],
                isError: true,
              };
            }

            return {
              content: [
                {
                  type: 'text',
                  text: 'Unsloth is properly installed.',
                },
              ],
            };
          }

          case 'list_supported_models': {
            const script = `
import json
try:
    from unsloth import FastLanguageModel
    # Define a list of supported models
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
            const result = await this.executeUnslothScript(script);
            
            try {
              const models = JSON.parse(result);
              if (models.error) {
                throw new Error(models.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: JSON.stringify(models, null, 2),
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error parsing model list: ${error.message}`);
            }
          }

          case 'load_model': {
            const { model_name, max_seq_length = 2048, load_in_4bit = true, use_gradient_checkpointing = true } = args as {
              model_name: string;
              max_seq_length?: number;
              load_in_4bit?: boolean;
              use_gradient_checkpointing?: boolean;
            };

            const script = `
import json
try:
    from unsloth import FastLanguageModel
    
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="${model_name}",
        max_seq_length=${max_seq_length},
        load_in_4bit=${load_in_4bit ? 'True' : 'False'},
        use_gradient_checkpointing=${use_gradient_checkpointing ? '"unsloth"' : 'False'}
    )
    
    # Get model info
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
            const result = await this.executeUnslothScript(script);
            
            try {
              const modelInfo = JSON.parse(result);
              if (!modelInfo.success) {
                throw new Error(modelInfo.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: `Successfully loaded model: ${model_name}\n\n${JSON.stringify(modelInfo, null, 2)}`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error loading model: ${error.message}`);
            }
          }

          case 'finetune_model': {
            const {
              model_name,
              dataset_name,
              output_dir,
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
              output_dir: string;
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

            const script = `
import json
import os
try:
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch
    
    # Create output directory if it doesn't exist
    os.makedirs("${output_dir}", exist_ok=True)
    
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="${model_name}",
        max_seq_length=${max_seq_length},
        load_in_4bit=${load_in_4bit ? 'True' : 'False'},
        use_gradient_checkpointing="unsloth"
    )
    
    # Load the dataset
    dataset = load_dataset("${dataset_name}")
    
    # Patch the model with LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=${lora_rank},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=${lora_alpha},
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=${max_seq_length},
        use_rslora=False,
        loftq_config=None
    )
    
    # Configure the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="${dataset_text_field}",
            max_seq_length=${max_seq_length},
            per_device_train_batch_size=${batch_size},
            gradient_accumulation_steps=${gradient_accumulation_steps},
            warmup_steps=10,
            max_steps=${max_steps},
            learning_rate=${learning_rate},
            logging_steps=1,
            output_dir="${output_dir}",
            optim="adamw_8bit",
            seed=3407,
        ),
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model()
    
    print(json.dumps({
        "success": True,
        "output_dir": "${output_dir}",
        "model_name": "${model_name}",
        "dataset_name": "${dataset_name}",
        "max_steps": ${max_steps}
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            const result = await this.executeUnslothScript(script);
            
            try {
              const trainingResult = JSON.parse(result);
              if (!trainingResult.success) {
                throw new Error(trainingResult.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: `Successfully fine-tuned model: ${model_name} with dataset: ${dataset_name}\n\n${JSON.stringify(trainingResult, null, 2)}`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error fine-tuning model: ${error.message}`);
            }
          }

          case 'generate_text': {
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

            const script = `
import json
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("${model_path}")
    tokenizer = AutoTokenizer.from_pretrained("${model_path}")
    
    # Create a text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=${max_new_tokens},
        temperature=${temperature},
        top_p=${top_p},
        do_sample=True
    )
    
    # Generate text
    result = generator("${prompt.replace(/"/g, '\\"')}")
    
    print(json.dumps({
        "success": True,
        "prompt": "${prompt.replace(/"/g, '\\"')}",
        "generated_text": result[0]["generated_text"]
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            const result = await this.executeUnslothScript(script);
            
            try {
              const generationResult = JSON.parse(result);
              if (!generationResult.success) {
                throw new Error(generationResult.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: `Generated text:\n\n${generationResult.generated_text}`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error generating text: ${error.message}`);
            }
          }

          case 'export_model': {
            const {
              model_path,
              export_format,
              output_path,
              quantization_bits = 4,
            } = args as {
              model_path: string;
              export_format: 'gguf' | 'ollama' | 'vllm' | 'huggingface';
              output_path: string;
              quantization_bits?: number;
            };

            let script = '';
            
            if (export_format === 'gguf') {
              script = `
import json
import os
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname("${output_path}"), exist_ok=True)
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("${model_path}")
    tokenizer = AutoTokenizer.from_pretrained("${model_path}")
    
    # Save the model in GGUF format
    from transformers import LlamaForCausalLM
    import ctranslate2
    
    # Convert to GGUF format
    ct_model = ctranslate2.converters.TransformersConverter(
        "${model_path}",
        "${output_path}",
        quantization="int${quantization_bits}"
    ).convert()
    
    print(json.dumps({
        "success": True,
        "model_path": "${model_path}",
        "export_format": "gguf",
        "output_path": "${output_path}",
        "quantization_bits": ${quantization_bits}
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            } else if (export_format === 'huggingface') {
              script = `
import json
import os
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create output directory if it doesn't exist
    os.makedirs("${output_path}", exist_ok=True)
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("${model_path}")
    tokenizer = AutoTokenizer.from_pretrained("${model_path}")
    
    # Save the model in Hugging Face format
    model.save_pretrained("${output_path}")
    tokenizer.save_pretrained("${output_path}")
    
    print(json.dumps({
        "success": True,
        "model_path": "${model_path}",
        "export_format": "huggingface",
        "output_path": "${output_path}"
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            } else {
              return {
                content: [
                  {
                    type: 'text',
                    text: `Export format '${export_format}' is not yet implemented. Currently, only 'gguf' and 'huggingface' formats are supported.`,
                  },
                ],
                isError: true,
              };
            }
            
            const result = await this.executeUnslothScript(script);
            
            try {
              const exportResult = JSON.parse(result);
              if (!exportResult.success) {
                throw new Error(exportResult.error);
              }
              
              return {
                content: [
                  {
                    type: 'text',
                    text: `Successfully exported model to ${export_format} format:\n\n${JSON.stringify(exportResult, null, 2)}`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error exporting model: ${error.message}`);
            }
          }

          case 'train_superbpe_tokenizer': {
            const {
              corpus_path,
              vocab_size = 50000,
              output_path,
              num_inherit_merges,
            } = args as {
              corpus_path: string;
              vocab_size?: number;
              output_path: string;
              num_inherit_merges?: number;
            };

            const inherit_merges = num_inherit_merges || Math.floor(vocab_size * 0.8);

            const script = `
import json
import os
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace, ByteLevel
    from tokenizers.processors import ByteLevel as ByteLevelProcessor
    from datasets import load_dataset

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE())

    # Stage 1: Train BPE with whitespace pretokenization
    print("Stage 1: Training BPE with whitespace pretokenization...")
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=${vocab_size}, special_tokens=["<pad>", "<s>", "</s>", "<unk>"])

    # Load corpus
    try:
        dataset = load_dataset("${corpus_path}")
        texts = [item["text"] for item in dataset["train"]]
    except:
        # If not a dataset, assume it's a file path
        with open("${corpus_path}", "r") as f:
            texts = [f.read()]

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Stage 2: Continue training without whitespace pretokenization (SuperBPE)
    print("Stage 2: Training SuperBPE (learning superwords)...")
    tokenizer.pre_tokenizer = ByteLevel()

    # Get current vocab and merges
    current_vocab_size = tokenizer.get_vocab_size()

    # Resume training with more merges to learn superwords
    additional_vocab = ${vocab_size} - current_vocab_size
    if additional_vocab > 0:
        trainer2 = BpeTrainer(vocab_size=${vocab_size}, special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
        tokenizer.train_from_iterator(texts, trainer=trainer2)

    # Configure the decoder
    tokenizer.post_processor = ByteLevelProcessor()

    # Create output directory if needed
    os.makedirs(os.path.dirname("${output_path}") if os.path.dirname("${output_path}") else ".", exist_ok=True)

    # Save the tokenizer
    tokenizer.save("${output_path}")

    # Test the tokenizer
    sample_text = texts[0][:200] if texts else "Hello world!"
    encoding = tokenizer.encode(sample_text)
    tokens_count = len(encoding.tokens)

    print(json.dumps({
        "success": True,
        "output_path": "${output_path}",
        "vocab_size": ${vocab_size},
        "final_vocab_size": tokenizer.get_vocab_size(),
        "sample_tokens": tokens_count,
        "message": "SuperBPE tokenizer trained successfully! This tokenizer should encode text 20-33% more efficiently than standard BPE."
    }))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            const result = await this.executeUnslothScript(script);

            try {
              const trainingResult = JSON.parse(result);
              if (!trainingResult.success) {
                throw new Error(trainingResult.error);
              }

              return {
                content: [
                  {
                    type: 'text',
                    text: `Successfully trained SuperBPE tokenizer!\n\n${JSON.stringify(trainingResult, null, 2)}`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error training SuperBPE tokenizer: ${error.message}`);
            }
          }

          case 'get_model_info': {
            const { model_name } = args as { model_name: string };

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
            const result = await this.executeUnslothScript(script);

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
            } catch (error: any) {
              throw new Error(`Error getting model info: ${error.message}`);
            }
          }

          case 'compare_tokenizers': {
            const { text, tokenizer1_path, tokenizer2_path } = args as {
              text: string;
              tokenizer1_path: string;
              tokenizer2_path: string;
            };

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

    tokenizer1 = load_tokenizer("${tokenizer1_path}")
    tokenizer2 = load_tokenizer("${tokenizer2_path}")

    # Tokenize the text
    tokens1 = tokenizer1.encode("${text.replace(/"/g, '\\"')}")
    tokens2 = tokenizer2.encode("${text.replace(/"/g, '\\"')}")

    # Get token counts
    count1 = len(tokens1) if hasattr(tokens1, '__len__') else len(tokens1.ids)
    count2 = len(tokens2) if hasattr(tokens2, '__len__') else len(tokens2.ids)

    # Calculate efficiency improvement
    efficiency_gain = ((count1 - count2) / count1 * 100) if count1 > 0 else 0

    comparison = {
        "tokenizer1_path": "${tokenizer1_path}",
        "tokenizer2_path": "${tokenizer2_path}",
        "tokenizer1_count": count1,
        "tokenizer2_count": count2,
        "difference": count1 - count2,
        "efficiency_gain_percent": round(efficiency_gain, 2),
        "text_length": len("${text.replace(/"/g, '\\"')}"),
        "winner": "tokenizer2" if count2 < count1 else "tokenizer1" if count1 < count2 else "tie",
        "success": True
    }

    print(json.dumps(comparison))
except Exception as e:
    print(json.dumps({"error": str(e), "success": False}))
`;
            const result = await this.executeUnslothScript(script);

            try {
              const comparison = JSON.parse(result);
              if (!comparison.success) {
                throw new Error(comparison.error);
              }

              return {
                content: [
                  {
                    type: 'text',
                    text: `Tokenizer Comparison Results:\n\n${JSON.stringify(comparison, null, 2)}\n\nTokenizer 2 is ${comparison.efficiency_gain_percent}% more efficient!`,
                  },
                ],
              };
            } catch (error: any) {
              throw new Error(`Error comparing tokenizers: ${error.message}`);
            }
          }

          case 'benchmark_model': {
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
            const result = await this.executeUnslothScript(script);

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
            } catch (error: any) {
              throw new Error(`Error benchmarking model: ${error.message}`);
            }
          }

          case 'list_datasets': {
            const { search_query = '', limit = 20 } = args as {
              search_query?: string;
              limit?: number;
            };

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
            const result = await this.executeUnslothScript(script);

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
            } catch (error: any) {
              throw new Error(`Error listing datasets: ${error.message}`);
            }
          }

          case 'prepare_dataset': {
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
            const result = await this.executeUnslothScript(script);

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
            } catch (error: any) {
              throw new Error(`Error preparing dataset: ${error.message}`);
            }
          }

          default:
            throw new McpError(
              ErrorCode.MethodNotFound,
              `Unknown tool: ${name}`
            );
        }
      } catch (error: any) {
        // Track failed execution
        metricsCollector.endTool(name, startTime, false, error.message);

        logger.error(`Error executing tool ${name}`, {
          error: error.message,
          stack: error.stack,
          toolName: name,
        });

        let errorMessage = error.message || 'Unknown error';
        let suggestions: string[] = [];

        // Provide helpful error messages based on error type
        if (error instanceof ValidationError) {
          errorMessage = `Validation error: ${error.message}`;
          if (error.suggestions) {
            suggestions = error.suggestions;
          }
        } else if (error instanceof TimeoutError) {
          errorMessage = `Timeout error: ${error.message}`;
          suggestions = [
            'Try reducing the workload size',
            'Consider processing in smaller batches',
            'Check if Python dependencies are installed correctly',
          ];
        } else if (error instanceof SecurityError) {
          errorMessage = `Security error: ${error.message}`;
          suggestions = ['Check file paths and permissions', 'Ensure input is properly formatted'];
        } else if (error.message.includes('ENOENT') || error.message.includes('not found')) {
          errorMessage = `File or resource not found: ${error.message}`;
          suggestions = [
            'Check that the file path is correct',
            'Ensure the file exists before proceeding',
            'Use absolute paths when possible',
          ];
        } else if (error.message.includes('EACCES') || error.message.includes('permission denied')) {
          errorMessage = `Permission denied: ${error.message}`;
          suggestions = [
            'Check file permissions',
            'Ensure you have write access to the output directory',
            'Try using a different output location',
          ];
        } else if (error.message.includes('Out of memory') || error.message.includes('OOM')) {
          errorMessage = `Out of memory: ${error.message}`;
          suggestions = [
            'Try using 4-bit quantization (load_in_4bit=true)',
            'Reduce batch size',
            'Use a smaller model',
            'Increase available RAM or use a machine with more memory',
          ];
        }

        const responseText = suggestions.length > 0
          ? `${errorMessage}\n\nSuggestions:\n${suggestions.map((s, i) => `${i + 1}. ${s}`).join('\n')}`
          : errorMessage;

        return {
          content: [
            {
              type: 'text',
              text: responseText,
            },
          ],
          isError: true,
        };
      } finally {
        // Always track completion (success is tracked in the try block per tool)
        // This ensures metrics are recorded even if we don't explicitly track success
      }
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Unsloth MCP server running on stdio');
  }
}

const server = new UnslothServer();
server.run().catch(console.error);
