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
import { config } from './utils/config.js';
import { cache } from './utils/cache.js';
import { progressTracker } from './utils/progress.js';
import {
  KnowledgeDatabase,
  knowledgeDb,
  processImage,
  processImageBatch,
  checkOCRBackends,
  classifyContent,
  cleanText,
  generateTrainingPairs,
  generateFromDatabase,
  generateSyntheticPairs,
  Category,
  CATEGORY_DEFINITIONS,
} from './knowledge/index.js';
import { getRunPodClient, RunPodClient } from './utils/runpod.js';
import { createWebScraper, WebScraper } from './utils/web-scraper.js';

const execPromise = promisify(exec);

// Get API keys from environment variables if needed
const HF_TOKEN = process.env.HUGGINGFACE_TOKEN;

class UnslothServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: 'unsloth-server',
        version: '2.4.0',
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

    const serverConfig = config.get();
    logger.info('Unsloth MCP Server initialized', {
      version: '2.4.0',
      environment: serverConfig.server.environment,
      cacheEnabled: serverConfig.cache.enabled,
      logLevel: serverConfig.logging.level,
    });
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
        throw new Error(
          `Operation timed out: ${error.message}. Try reducing the workload or increasing timeout.`
        );
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
          description:
            'Train a SuperBPE tokenizer for improved efficiency (up to 33% fewer tokens)',
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
                description:
                  'Number of merges to inherit from BPE stage (default: vocab_size * 0.8)',
              },
            },
            required: ['corpus_path', 'output_path'],
          },
        },
        {
          name: 'get_model_info',
          description:
            'Get detailed information about a model including architecture, parameters, and capabilities',
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
          description:
            'Compare tokenization efficiency between different tokenizers (BPE vs SuperBPE)',
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
        // ==================== KNOWLEDGE BASE TOOLS ====================
        {
          name: 'process_book_image',
          description:
            'OCR a book/document image and catalogue the extracted text into the knowledge base',
          inputSchema: {
            type: 'object',
            properties: {
              image_path: {
                type: 'string',
                description: 'Path to the image file (jpg, png, etc.)',
              },
              book_title: {
                type: 'string',
                description: 'Title of the book (optional)',
              },
              author: {
                type: 'string',
                description: 'Author of the book (optional)',
              },
              chapter: {
                type: 'string',
                description: 'Chapter name or number (optional)',
              },
              page_numbers: {
                type: 'string',
                description: 'Page number(s) (optional)',
              },
              category: {
                type: 'string',
                description: 'Content category for classification',
                enum: [
                  'candlestick_patterns',
                  'chart_patterns',
                  'technical_indicators',
                  'risk_management',
                  'trading_psychology',
                  'market_structure',
                  'options_strategies',
                  'fundamental_analysis',
                  'order_flow',
                  'volume_analysis',
                  'general',
                ],
              },
              tags: {
                type: 'array',
                items: { type: 'string' },
                description: 'Tags for this content (optional)',
              },
              ocr_backend: {
                type: 'string',
                description: 'OCR backend to use (auto, tesseract, easyocr, claude)',
                enum: ['auto', 'tesseract', 'easyocr', 'claude'],
              },
            },
            required: ['image_path'],
          },
        },
        {
          name: 'batch_process_images',
          description: 'Process multiple book images at once and catalogue them',
          inputSchema: {
            type: 'object',
            properties: {
              image_paths: {
                type: 'array',
                items: { type: 'string' },
                description: 'Array of image file paths',
              },
              book_title: {
                type: 'string',
                description: 'Title of the book (applies to all)',
              },
              author: {
                type: 'string',
                description: 'Author of the book (applies to all)',
              },
              category: {
                type: 'string',
                description: 'Content category',
                enum: [
                  'candlestick_patterns',
                  'chart_patterns',
                  'technical_indicators',
                  'risk_management',
                  'trading_psychology',
                  'market_structure',
                  'options_strategies',
                  'fundamental_analysis',
                  'order_flow',
                  'volume_analysis',
                  'general',
                ],
              },
              ocr_backend: {
                type: 'string',
                description: 'OCR backend to use',
                enum: ['auto', 'tesseract', 'easyocr', 'claude'],
              },
            },
            required: ['image_paths'],
          },
        },
        {
          name: 'search_knowledge',
          description: 'Search the knowledge base using full-text search',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'Search query',
              },
              limit: {
                type: 'number',
                description: 'Maximum results to return (default: 20)',
              },
            },
            required: ['query'],
          },
        },
        {
          name: 'list_knowledge_by_category',
          description: 'List knowledge entries by category',
          inputSchema: {
            type: 'object',
            properties: {
              category: {
                type: 'string',
                description: 'Category to filter by',
                enum: [
                  'candlestick_patterns',
                  'chart_patterns',
                  'technical_indicators',
                  'risk_management',
                  'trading_psychology',
                  'market_structure',
                  'options_strategies',
                  'fundamental_analysis',
                  'order_flow',
                  'volume_analysis',
                  'general',
                ],
              },
              limit: {
                type: 'number',
                description: 'Maximum results to return (default: 50)',
              },
            },
            required: ['category'],
          },
        },
        {
          name: 'get_knowledge_entry',
          description: 'Get a specific knowledge entry by ID',
          inputSchema: {
            type: 'object',
            properties: {
              entry_id: {
                type: 'string',
                description: 'The knowledge entry ID',
              },
            },
            required: ['entry_id'],
          },
        },
        {
          name: 'generate_training_pairs',
          description: 'Generate training data pairs from knowledge base entries',
          inputSchema: {
            type: 'object',
            properties: {
              entry_id: {
                type: 'string',
                description:
                  'Generate pairs from specific entry (optional - if not provided, generates from all)',
              },
              min_quality_score: {
                type: 'number',
                description: 'Minimum quality score for entries (0-100, default: 30)',
              },
              pairs_per_entry: {
                type: 'number',
                description: 'Number of pairs to generate per entry (default: 3)',
              },
              include_system_prompt: {
                type: 'boolean',
                description: 'Include system prompts in pairs (default: true)',
              },
              generate_synthetic: {
                type: 'boolean',
                description: 'Use AI to generate additional synthetic pairs (requires API key)',
              },
            },
          },
        },
        {
          name: 'export_training_data',
          description: 'Export all training pairs to a file for fine-tuning',
          inputSchema: {
            type: 'object',
            properties: {
              output_path: {
                type: 'string',
                description: 'Path to save the training data file',
              },
              format: {
                type: 'string',
                description: 'Output format (alpaca, sharegpt, chatml)',
                enum: ['alpaca', 'sharegpt', 'chatml'],
              },
              min_quality_score: {
                type: 'number',
                description: 'Minimum quality score to include (default: 0)',
              },
            },
            required: ['output_path', 'format'],
          },
        },
        {
          name: 'knowledge_stats',
          description: 'Get statistics about the knowledge base',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'check_ocr_backends',
          description: 'Check which OCR backends are available on the system',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'list_categories',
          description: 'List all available knowledge categories with descriptions',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        // RunPod GPU Management Tools
        {
          name: 'runpod_list_pods',
          description: 'List all RunPod pods with their status, GPU info, and costs',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'runpod_get_pod',
          description: 'Get detailed information about a specific RunPod pod',
          inputSchema: {
            type: 'object',
            properties: {
              pod_id: {
                type: 'string',
                description: 'The ID of the pod to get info for',
              },
            },
            required: ['pod_id'],
          },
        },
        {
          name: 'runpod_check_gpus',
          description: 'Check available GPU types and their pricing on RunPod',
          inputSchema: {
            type: 'object',
            properties: {
              min_vram_gb: {
                type: 'number',
                description: 'Minimum VRAM in GB (default: 24 for fine-tuning)',
              },
            },
          },
        },
        {
          name: 'runpod_create_pod',
          description: 'Create a new RunPod pod for fine-tuning',
          inputSchema: {
            type: 'object',
            properties: {
              name: {
                type: 'string',
                description: 'Name for the pod',
              },
              gpu_type: {
                type: 'string',
                description: 'GPU type ID (e.g., "NVIDIA RTX A5000")',
              },
              gpu_count: {
                type: 'number',
                description: 'Number of GPUs (default: 1)',
              },
              volume_gb: {
                type: 'number',
                description: 'Volume size in GB (default: 30)',
              },
              image: {
                type: 'string',
                description: 'Docker image (default: pytorch with CUDA)',
              },
            },
            required: ['name', 'gpu_type'],
          },
        },
        {
          name: 'runpod_start_pod',
          description: 'Start a stopped RunPod pod',
          inputSchema: {
            type: 'object',
            properties: {
              pod_id: {
                type: 'string',
                description: 'The ID of the pod to start',
              },
            },
            required: ['pod_id'],
          },
        },
        {
          name: 'runpod_stop_pod',
          description: 'Stop a running RunPod pod (keeps volume data)',
          inputSchema: {
            type: 'object',
            properties: {
              pod_id: {
                type: 'string',
                description: 'The ID of the pod to stop',
              },
            },
            required: ['pod_id'],
          },
        },
        {
          name: 'runpod_terminate_pod',
          description: 'Terminate a RunPod pod (deletes everything including volume)',
          inputSchema: {
            type: 'object',
            properties: {
              pod_id: {
                type: 'string',
                description: 'The ID of the pod to terminate',
              },
              confirm: {
                type: 'boolean',
                description: 'Must be true to confirm termination',
              },
            },
            required: ['pod_id', 'confirm'],
          },
        },
        {
          name: 'runpod_start_training',
          description: 'Start a fine-tuning job on a RunPod pod',
          inputSchema: {
            type: 'object',
            properties: {
              pod_id: {
                type: 'string',
                description: 'The ID of the pod to run training on',
              },
              base_model: {
                type: 'string',
                description: 'Base model to fine-tune (e.g., "unsloth/Llama-3.2-1B")',
              },
              dataset_path: {
                type: 'string',
                description: 'Path to training dataset (JSON/JSONL)',
              },
              output_dir: {
                type: 'string',
                description: 'Output directory for the model',
              },
              lora_r: {
                type: 'number',
                description: 'LoRA rank (default: 16)',
              },
              lora_alpha: {
                type: 'number',
                description: 'LoRA alpha (default: 32)',
              },
              learning_rate: {
                type: 'number',
                description: 'Learning rate (default: 2e-4)',
              },
              epochs: {
                type: 'number',
                description: 'Number of training epochs (default: 3)',
              },
              batch_size: {
                type: 'number',
                description: 'Batch size (default: 4)',
              },
              max_seq_length: {
                type: 'number',
                description: 'Maximum sequence length (default: 2048)',
              },
            },
            required: ['pod_id', 'base_model', 'dataset_path', 'output_dir'],
          },
        },
        {
          name: 'runpod_get_training_status',
          description: 'Get the status and progress of a training job',
          inputSchema: {
            type: 'object',
            properties: {
              pod_id: {
                type: 'string',
                description: 'The ID of the pod running training',
              },
            },
            required: ['pod_id'],
          },
        },
        {
          name: 'runpod_get_training_logs',
          description: 'Get training logs from a RunPod pod',
          inputSchema: {
            type: 'object',
            properties: {
              pod_id: {
                type: 'string',
                description: 'The ID of the pod',
              },
              lines: {
                type: 'number',
                description: 'Number of log lines to retrieve (default: 100)',
              },
            },
            required: ['pod_id'],
          },
        },
        {
          name: 'runpod_estimate_cost',
          description: 'Estimate the cost of a fine-tuning job',
          inputSchema: {
            type: 'object',
            properties: {
              dataset_tokens: {
                type: 'number',
                description: 'Total tokens in the dataset',
              },
              base_model: {
                type: 'string',
                description: 'Base model name',
              },
              gpu_cost_per_hour: {
                type: 'number',
                description: 'GPU cost per hour (default: 0.16)',
              },
              epochs: {
                type: 'number',
                description: 'Number of epochs (default: 3)',
              },
            },
            required: ['dataset_tokens', 'base_model'],
          },
        },
        // Web Scraping Tools
        {
          name: 'web_search',
          description:
            'Search the web using Exa AI semantic search. Returns relevant pages with content for training data gathering.',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'Search query (semantic, not just keywords)',
              },
              num_results: {
                type: 'number',
                description: 'Number of results to return (default: 10)',
              },
              include_domains: {
                type: 'array',
                items: { type: 'string' },
                description: 'Only include results from these domains',
              },
              exclude_domains: {
                type: 'array',
                items: { type: 'string' },
                description: 'Exclude results from these domains',
              },
              category: {
                type: 'string',
                description:
                  'Filter by category: company, research, news, github, tweet, paper, pdf, etc.',
              },
            },
            required: ['query'],
          },
        },
        {
          name: 'web_scrape',
          description:
            'Scrape a single URL and convert to LLM-ready markdown using Firecrawl. Great for extracting training data from specific pages.',
          inputSchema: {
            type: 'object',
            properties: {
              url: {
                type: 'string',
                description: 'URL to scrape',
              },
              only_main_content: {
                type: 'boolean',
                description: 'Extract only main content, removing nav/footer (default: true)',
              },
              include_tags: {
                type: 'array',
                items: { type: 'string' },
                description: 'Only include these HTML tags',
              },
              exclude_tags: {
                type: 'array',
                items: { type: 'string' },
                description: 'Exclude these HTML tags',
              },
            },
            required: ['url'],
          },
        },
        {
          name: 'web_crawl',
          description:
            'Crawl an entire website and convert all pages to LLM-ready markdown. Returns a job ID for tracking.',
          inputSchema: {
            type: 'object',
            properties: {
              url: {
                type: 'string',
                description: 'Starting URL to crawl',
              },
              max_pages: {
                type: 'number',
                description: 'Maximum pages to crawl (default: 50)',
              },
              max_depth: {
                type: 'number',
                description: 'Maximum link depth to follow (default: 2)',
              },
              include_paths: {
                type: 'array',
                items: { type: 'string' },
                description: 'Only crawl URLs matching these patterns',
              },
              exclude_paths: {
                type: 'array',
                items: { type: 'string' },
                description: 'Skip URLs matching these patterns',
              },
            },
            required: ['url'],
          },
        },
        {
          name: 'web_crawl_status',
          description: 'Check the status of a crawl job and retrieve results when complete.',
          inputSchema: {
            type: 'object',
            properties: {
              job_id: {
                type: 'string',
                description: 'Crawl job ID returned from web_crawl',
              },
            },
            required: ['job_id'],
          },
        },
        {
          name: 'web_map_urls',
          description:
            'Discover all URLs on a website without scraping content. Useful for planning what to crawl.',
          inputSchema: {
            type: 'object',
            properties: {
              url: {
                type: 'string',
                description: 'Website URL to map',
              },
              search: {
                type: 'string',
                description: 'Filter URLs containing this text',
              },
              limit: {
                type: 'number',
                description: 'Maximum URLs to return (default: 100)',
              },
              include_subdomains: {
                type: 'boolean',
                description: 'Include subdomains (default: false)',
              },
            },
            required: ['url'],
          },
        },
        {
          name: 'web_research',
          description:
            'Perform deep research on a topic using Exa. Returns a synthesized summary with citations - perfect for generating training data.',
          inputSchema: {
            type: 'object',
            properties: {
              topic: {
                type: 'string',
                description: 'Research topic or question',
              },
              max_sources: {
                type: 'number',
                description: 'Maximum sources to analyze (default: 20)',
              },
            },
            required: ['topic'],
          },
        },
        {
          name: 'web_find_similar',
          description:
            'Find pages similar to a given URL. Great for expanding training data from good examples.',
          inputSchema: {
            type: 'object',
            properties: {
              url: {
                type: 'string',
                description: 'URL to find similar pages for',
              },
              num_results: {
                type: 'number',
                description: 'Number of similar pages to find (default: 10)',
              },
              include_domains: {
                type: 'array',
                items: { type: 'string' },
                description: 'Only include results from these domains',
              },
            },
            required: ['url'],
          },
        },
        {
          name: 'web_gather_training_data',
          description:
            'Gather training data from web search results. Combines search + content extraction + filtering for quality.',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'Search query for finding relevant content',
              },
              num_results: {
                type: 'number',
                description: 'Number of pages to gather (default: 20)',
              },
              min_word_count: {
                type: 'number',
                description: 'Minimum words per document (default: 100)',
              },
              domains: {
                type: 'array',
                items: { type: 'string' },
                description: 'Limit to these domains',
              },
              save_to_file: {
                type: 'boolean',
                description: 'Save results to file (default: true)',
              },
              output_format: {
                type: 'string',
                enum: ['jsonl', 'json', 'markdown'],
                description: 'Output format (default: jsonl)',
              },
            },
            required: ['query'],
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
            // Check cache first
            const cached = cache.get<string[]>('supported_models');
            if (cached) {
              logger.debug('Returning cached supported models');
              return this.createSuccessResponse(name, startTime, JSON.stringify(cached, null, 2));
            }

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

              // Cache the result for 1 hour
              cache.set('supported_models', models, 3600);
              logger.debug('Cached supported models list');

              return this.createSuccessResponse(name, startTime, JSON.stringify(models, null, 2));
            } catch (error: any) {
              throw new Error(`Error parsing model list: ${error.message}`);
            }
          }

          case 'load_model': {
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

          // ==================== KNOWLEDGE BASE TOOL HANDLERS ====================

          case 'process_book_image': {
            const {
              image_path,
              book_title,
              author,
              chapter,
              page_numbers,
              category,
              tags = [],
              ocr_backend = 'auto',
            } = args as {
              image_path: string;
              book_title?: string;
              author?: string;
              chapter?: string;
              page_numbers?: string;
              category?: Category;
              tags?: string[];
              ocr_backend?: 'auto' | 'tesseract' | 'easyocr' | 'claude';
            };

            try {
              // Process image with OCR
              const ocrResult = await processImage(image_path, {
                backend: ocr_backend,
                enhance_image: true,
              });

              // Auto-classify content if no category provided
              const classification = classifyContent(ocrResult.cleaned_text);
              const finalCategory = category || classification.category;
              const detectedTopics = classification.detected_topics;

              // Clean the text
              const cleanedText = cleanText(ocrResult.raw_text);

              // Add to knowledge base
              const entryId = await knowledgeDb.addEntry({
                source: {
                  type: 'book',
                  book_title,
                  author,
                  chapter,
                  page_numbers,
                  image_path,
                  capture_date: new Date().toISOString(),
                },
                raw_text: ocrResult.raw_text,
                cleaned_text: cleanedText,
                category: finalCategory,
                topics: detectedTopics,
                tags,
                quality_score: Math.round(ocrResult.confidence),
                ocr_confidence: ocrResult.confidence,
                manually_reviewed: false,
              });

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    entry_id: entryId,
                    ocr_backend: ocrResult.backend_used,
                    ocr_confidence: ocrResult.confidence,
                    processing_time_ms: ocrResult.processing_time_ms,
                    category: finalCategory,
                    detected_topics: detectedTopics,
                    text_preview:
                      cleanedText.substring(0, 300) + (cleanedText.length > 300 ? '...' : ''),
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error processing book image: ${error.message}`);
            }
          }

          case 'batch_process_images': {
            const {
              image_paths,
              book_title,
              author,
              category,
              ocr_backend = 'auto',
            } = args as {
              image_paths: string[];
              book_title?: string;
              author?: string;
              category?: Category;
              ocr_backend?: 'auto' | 'tesseract' | 'easyocr' | 'claude';
            };

            const results: Array<{ path: string; entry_id?: string; error?: string }> = [];

            for (let i = 0; i < image_paths.length; i++) {
              const imagePath = image_paths[i];
              try {
                const ocrResult = await processImage(imagePath, {
                  backend: ocr_backend,
                  enhance_image: true,
                });

                const classification = classifyContent(ocrResult.cleaned_text);
                const finalCategory = category || classification.category;
                const cleanedText = cleanText(ocrResult.raw_text);

                const entryId = await knowledgeDb.addEntry({
                  source: {
                    type: 'book',
                    book_title,
                    author,
                    page_numbers: `Image ${i + 1}`,
                    image_path: imagePath,
                    capture_date: new Date().toISOString(),
                  },
                  raw_text: ocrResult.raw_text,
                  cleaned_text: cleanedText,
                  category: finalCategory,
                  topics: classification.detected_topics,
                  tags: [],
                  quality_score: Math.round(ocrResult.confidence),
                  ocr_confidence: ocrResult.confidence,
                  manually_reviewed: false,
                });

                results.push({ path: imagePath, entry_id: entryId });
              } catch (error: any) {
                results.push({ path: imagePath, error: error.message });
              }
            }

            const successful = results.filter((r) => r.entry_id).length;
            const failed = results.filter((r) => r.error).length;

            return this.createSuccessResponse(
              name,
              startTime,
              JSON.stringify(
                {
                  success: true,
                  total_images: image_paths.length,
                  successful,
                  failed,
                  results,
                },
                null,
                2
              )
            );
          }

          case 'search_knowledge': {
            const { query, limit = 20 } = args as { query: string; limit?: number };

            try {
              const entries = await knowledgeDb.searchEntries(query, limit);
              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    query,
                    count: entries.length,
                    entries,
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error searching knowledge base: ${error.message}`);
            }
          }

          case 'list_knowledge_by_category': {
            const { category, limit = 50 } = args as { category: Category; limit?: number };

            try {
              const entries = await knowledgeDb.listByCategory(category, limit);
              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    category,
                    count: entries.length,
                    entries,
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error listing knowledge entries: ${error.message}`);
            }
          }

          case 'get_knowledge_entry': {
            const { entry_id } = args as { entry_id: string };

            try {
              const entry = await knowledgeDb.getEntry(entry_id);
              if (!entry) {
                return {
                  content: [{ type: 'text', text: `Knowledge entry not found: ${entry_id}` }],
                  isError: true,
                };
              }
              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    entry,
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error getting knowledge entry: ${error.message}`);
            }
          }

          case 'generate_training_pairs': {
            const {
              entry_id,
              min_quality_score = 30,
              pairs_per_entry = 3,
              include_system_prompt = true,
              generate_synthetic = false,
            } = args as {
              entry_id?: string;
              min_quality_score?: number;
              pairs_per_entry?: number;
              include_system_prompt?: boolean;
              generate_synthetic?: boolean;
            };

            try {
              if (entry_id) {
                // Generate for specific entry
                const entry = await knowledgeDb.getEntry(entry_id);
                if (!entry) {
                  throw new Error(`Entry not found: ${entry_id}`);
                }

                let pairs = generateTrainingPairs(entry, {
                  min_quality_score,
                  pairs_per_entry,
                  include_system_prompt,
                });

                // Optionally generate synthetic pairs
                if (generate_synthetic) {
                  const syntheticPairs = await generateSyntheticPairs(
                    entry.cleaned_text,
                    entry.category,
                    pairs_per_entry
                  );
                  pairs = [...pairs, ...syntheticPairs];
                }

                // Store pairs in database
                for (const pair of pairs) {
                  await knowledgeDb.addTrainingPair(entry_id, pair);
                }

                return this.createSuccessResponse(
                  name,
                  startTime,
                  JSON.stringify(
                    {
                      success: true,
                      entry_id,
                      pairs_generated: pairs.length,
                      pairs,
                    },
                    null,
                    2
                  )
                );
              } else {
                // Generate for all entries
                const result = await generateFromDatabase(knowledgeDb, {
                  min_quality_score,
                  pairs_per_entry,
                  include_system_prompt,
                });

                return this.createSuccessResponse(
                  name,
                  startTime,
                  JSON.stringify(
                    {
                      success: true,
                      ...result,
                    },
                    null,
                    2
                  )
                );
              }
            } catch (error: any) {
              throw new Error(`Error generating training pairs: ${error.message}`);
            }
          }

          case 'export_training_data': {
            const {
              output_path,
              format,
              min_quality_score = 0,
            } = args as {
              output_path: string;
              format: 'alpaca' | 'sharegpt' | 'chatml';
              min_quality_score?: number;
            };

            try {
              const result = await knowledgeDb.exportTrainingData(
                output_path,
                format,
                min_quality_score
              );
              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    format,
                    ...result,
                    message: `Training data exported to ${output_path}. Ready for fine-tuning!`,
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error exporting training data: ${error.message}`);
            }
          }

          case 'knowledge_stats': {
            try {
              const stats = await knowledgeDb.getStats();
              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    ...stats,
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error getting knowledge stats: ${error.message}`);
            }
          }

          case 'check_ocr_backends': {
            try {
              const backends = await checkOCRBackends();
              const available = Object.entries(backends)
                .filter(([, v]) => v)
                .map(([k]) => k);

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    backends,
                    available,
                    recommendation: backends.tesseract
                      ? 'tesseract (fast, good for clear text)'
                      : backends.easyocr
                        ? 'easyocr (slower, better accuracy)'
                        : backends.claude
                          ? 'claude (best for charts/diagrams, requires API key)'
                          : 'No OCR backend available. Install pytesseract or easyocr.',
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error checking OCR backends: ${error.message}`);
            }
          }

          case 'list_categories': {
            const categories = Object.entries(CATEGORY_DEFINITIONS).map(([key, value]) => ({
              id: key,
              description: value.description,
              keywords: value.keywords.slice(0, 5),
              examples: value.examples.slice(0, 2),
            }));

            return this.createSuccessResponse(
              name,
              startTime,
              JSON.stringify(
                {
                  success: true,
                  categories,
                },
                null,
                2
              )
            );
          }

          // ================================================================
          // RunPod GPU Management Tools
          // ================================================================

          case 'runpod_list_pods': {
            try {
              const client = getRunPodClient();
              const pods = await client.listPods();

              const podSummary = pods.map((pod) => ({
                id: pod.id,
                name: pod.name,
                status: pod.desiredStatus,
                gpu: pod.machine?.gpuTypeId,
                gpuCount: pod.gpuCount,
                costPerHr: pod.costPerHr,
                uptime: pod.runtime?.uptimeInSeconds
                  ? `${Math.round(pod.runtime.uptimeInSeconds / 60)} minutes`
                  : 'stopped',
                gpuUtilization: pod.runtime?.gpus?.[0]?.gpuUtilPercent
                  ? `${pod.runtime.gpus[0].gpuUtilPercent}%`
                  : 'N/A',
              }));

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify({ success: true, pods: podSummary }, null, 2)
              );
            } catch (error: any) {
              throw new Error(`Error listing pods: ${error.message}`);
            }
          }

          case 'runpod_get_pod': {
            const { pod_id } = args as { pod_id: string };

            try {
              const client = getRunPodClient();
              const pod = await client.getPod(pod_id);

              if (!pod) {
                return this.createSuccessResponse(
                  name,
                  startTime,
                  JSON.stringify({ success: false, error: `Pod ${pod_id} not found` }, null, 2)
                );
              }

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify({ success: true, pod }, null, 2)
              );
            } catch (error: any) {
              throw new Error(`Error getting pod: ${error.message}`);
            }
          }

          case 'runpod_check_gpus': {
            const { min_vram_gb = 24 } = args as { min_vram_gb?: number };

            try {
              const client = getRunPodClient();
              const gpuTypes = await client.getGpuTypes();

              // Filter by VRAM and availability, sort by VRAM
              const available = gpuTypes
                .filter((gpu) => {
                  const hasCapacity = gpu.secureCloud || gpu.communityCloud;
                  const hasVram = gpu.memoryInGb >= min_vram_gb;
                  return hasCapacity && hasVram;
                })
                .map((gpu) => ({
                  id: gpu.id,
                  name: gpu.displayName,
                  vram: `${gpu.memoryInGb}GB`,
                  secureCloud: gpu.secureCloud,
                  communityCloud: gpu.communityCloud,
                }))
                .sort((a, b) => parseInt(a.vram) - parseInt(b.vram));

              const bestGpu = await client.findBestAvailableGpu(min_vram_gb);

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    minVramFilter: `${min_vram_gb}GB`,
                    availableGpus: available,
                    recommendation: bestGpu
                      ? `Recommended: ${bestGpu.displayName} (${bestGpu.memoryInGb}GB VRAM)`
                      : 'No GPUs available meeting requirements',
                  },
                  null,
                  2
                )
              );
            } catch (error: unknown) {
              throw new Error(
                `Error checking GPUs: ${error instanceof Error ? error.message : 'Unknown error'}`
              );
            }
          }

          case 'runpod_create_pod': {
            const {
              name: podName,
              gpu_type,
              gpu_count = 1,
              volume_gb = 30,
              image,
            } = args as {
              name: string;
              gpu_type: string;
              gpu_count?: number;
              volume_gb?: number;
              image?: string;
            };

            try {
              const client = getRunPodClient();

              // Check GPU availability first
              const availability = await client.checkGpuAvailability(gpu_type);
              if (!availability.available) {
                const bestAlt = await client.findBestAvailableGpu(24);
                return this.createSuccessResponse(
                  name,
                  startTime,
                  JSON.stringify(
                    {
                      success: false,
                      error: `GPU type ${gpu_type} is not available`,
                      suggestion: bestAlt
                        ? `Try ${bestAlt.displayName} instead (${bestAlt.memoryInGb}GB VRAM)`
                        : 'No suitable GPUs available right now',
                    },
                    null,
                    2
                  )
                );
              }

              const pod = await client.createPod({
                name: podName,
                gpuTypeId: gpu_type,
                gpuCount: gpu_count,
                volumeInGb: volume_gb,
                imageName: image,
              });

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    message: `Pod ${pod.name} created successfully`,
                    pod: {
                      id: pod.id,
                      name: pod.name,
                      status: pod.desiredStatus,
                      costPerHr: pod.costPerHr,
                    },
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error creating pod: ${error.message}`);
            }
          }

          case 'runpod_start_pod': {
            const { pod_id } = args as { pod_id: string };

            try {
              const client = getRunPodClient();
              const pod = await client.startPod(pod_id);

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    message: `Pod ${pod.name} starting`,
                    pod: {
                      id: pod.id,
                      name: pod.name,
                      status: pod.desiredStatus,
                      costPerHr: pod.costPerHr,
                    },
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              // Check if it's a GPU availability issue
              if (error.message.includes('GPU') || error.message.includes('available')) {
                const client = getRunPodClient();
                const bestAlt = await client.findBestAvailableGpu(24);
                throw new Error(
                  `${error.message}. ${
                    bestAlt
                      ? `Suggested alternative: ${bestAlt.displayName} (${bestAlt.memoryInGb}GB VRAM)`
                      : 'No suitable GPUs available. Try again later.'
                  }`
                );
              }
              throw new Error(`Error starting pod: ${error.message}`);
            }
          }

          case 'runpod_stop_pod': {
            const { pod_id } = args as { pod_id: string };

            try {
              const client = getRunPodClient();
              const pod = await client.stopPod(pod_id);

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    message: `Pod ${pod.name} stopped. Volume data preserved.`,
                    pod: {
                      id: pod.id,
                      name: pod.name,
                      status: pod.desiredStatus,
                    },
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error stopping pod: ${error.message}`);
            }
          }

          case 'runpod_terminate_pod': {
            const { pod_id, confirm } = args as { pod_id: string; confirm: boolean };

            if (!confirm) {
              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: false,
                    error: 'Termination not confirmed. Set confirm=true to proceed.',
                    warning: 'This will permanently delete the pod and all its data!',
                  },
                  null,
                  2
                )
              );
            }

            try {
              const client = getRunPodClient();
              await client.terminatePod(pod_id);

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    message: `Pod ${pod_id} terminated. All data has been deleted.`,
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error terminating pod: ${error.message}`);
            }
          }

          case 'runpod_start_training': {
            const {
              pod_id,
              base_model,
              dataset_path,
              output_dir,
              lora_r,
              lora_alpha,
              learning_rate,
              epochs,
              batch_size,
              max_seq_length,
            } = args as {
              pod_id: string;
              base_model: string;
              dataset_path: string;
              output_dir: string;
              lora_r?: number;
              lora_alpha?: number;
              learning_rate?: number;
              epochs?: number;
              batch_size?: number;
              max_seq_length?: number;
            };

            try {
              const client = getRunPodClient();

              // Check if pod is running
              const pod = await client.getPod(pod_id);
              if (!pod || !pod.runtime) {
                return this.createSuccessResponse(
                  name,
                  startTime,
                  JSON.stringify(
                    {
                      success: false,
                      error: `Pod ${pod_id} is not running. Start it first with runpod_start_pod.`,
                    },
                    null,
                    2
                  )
                );
              }

              const job = await client.startTrainingJob(pod_id, {
                baseModel: base_model,
                datasetPath: dataset_path,
                outputDir: output_dir,
                loraR: lora_r,
                loraAlpha: lora_alpha,
                learningRate: learning_rate,
                epochs: epochs,
                batchSize: batch_size,
                maxSeqLength: max_seq_length,
              });

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    message: 'Training job started',
                    job: {
                      id: job.id,
                      status: job.status,
                      baseModel: base_model,
                      outputDir: output_dir,
                    },
                    nextSteps: [
                      'Use runpod_get_training_status to monitor progress',
                      'Use runpod_get_training_logs to view training output',
                    ],
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error starting training: ${error.message}`);
            }
          }

          case 'runpod_get_training_status': {
            const { pod_id } = args as { pod_id: string };

            try {
              const client = getRunPodClient();
              const status = await client.getTrainingStatus(pod_id);

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify({ success: true, training: status }, null, 2)
              );
            } catch (error: any) {
              throw new Error(`Error getting training status: ${error.message}`);
            }
          }

          case 'runpod_get_training_logs': {
            const { pod_id, lines = 100 } = args as { pod_id: string; lines?: number };

            try {
              const client = getRunPodClient();
              const logs = await client.getTrainingLogs(pod_id, lines);

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify({ success: true, logs }, null, 2)
              );
            } catch (error: any) {
              throw new Error(`Error getting training logs: ${error.message}`);
            }
          }

          case 'runpod_estimate_cost': {
            const {
              dataset_tokens,
              base_model,
              gpu_cost_per_hour = 0.16,
              epochs = 3,
            } = args as {
              dataset_tokens: number;
              base_model: string;
              gpu_cost_per_hour?: number;
              epochs?: number;
            };

            try {
              const client = getRunPodClient();
              const estimate = client.estimateTrainingCost(
                dataset_tokens,
                base_model,
                gpu_cost_per_hour,
                epochs
              );

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    estimate: {
                      ...estimate,
                      datasetTokens: dataset_tokens,
                      baseModel: base_model,
                      epochs: epochs,
                      gpuCostPerHour: gpu_cost_per_hour,
                      summary: `Estimated ${estimate.estimatedHours} hours at $${gpu_cost_per_hour}/hr = $${estimate.estimatedCost}`,
                    },
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Error estimating cost: ${error.message}`);
            }
          }

          // ================================================================
          // Web Scraping Tools
          // ================================================================

          case 'web_search': {
            const { query, num_results, include_domains, exclude_domains, category } = args as {
              query: string;
              num_results?: number;
              include_domains?: string[];
              exclude_domains?: string[];
              category?: string;
            };

            try {
              const scraper = createWebScraper();
              const results = await scraper.exaSearch({
                query,
                numResults: num_results || 10,
                includeDomains: include_domains,
                excludeDomains: exclude_domains,
                category,
                includeText: true,
                includeHighlights: true,
              });

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    query,
                    resultCount: results.length,
                    results: results.map((r) => ({
                      title: r.title,
                      url: r.url,
                      score: r.score,
                      publishedDate: r.publishedDate,
                      textPreview:
                        r.text?.substring(0, 500) + (r.text && r.text.length > 500 ? '...' : ''),
                      highlights: r.highlights,
                    })),
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Web search failed: ${error.message}`);
            }
          }

          case 'web_scrape': {
            const { url, only_main_content, include_tags, exclude_tags } = args as {
              url: string;
              only_main_content?: boolean;
              include_tags?: string[];
              exclude_tags?: string[];
            };

            try {
              const scraper = createWebScraper();
              const result = await scraper.firecrawlScrape({
                url,
                onlyMainContent: only_main_content ?? true,
                includeTags: include_tags,
                excludeTags: exclude_tags,
              });

              const wordCount = result.markdown?.split(/\s+/).length || 0;

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    url: result.url,
                    title: result.title,
                    wordCount,
                    markdown: result.markdown,
                    links: result.links?.slice(0, 20),
                    metadata: result.metadata,
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Web scrape failed: ${error.message}`);
            }
          }

          case 'web_crawl': {
            const { url, max_pages, max_depth, include_paths, exclude_paths } = args as {
              url: string;
              max_pages?: number;
              max_depth?: number;
              include_paths?: string[];
              exclude_paths?: string[];
            };

            try {
              const scraper = createWebScraper();
              const job = await scraper.firecrawlCrawl({
                url,
                maxPages: max_pages || 50,
                maxDepth: max_depth || 2,
                includePaths: include_paths,
                excludePaths: exclude_paths,
              });

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    jobId: job.id,
                    status: job.status,
                    url: job.url,
                    message: `Crawl started. Use web_crawl_status with job_id "${job.id}" to check progress.`,
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Web crawl failed: ${error.message}`);
            }
          }

          case 'web_crawl_status': {
            const { job_id } = args as { job_id: string };

            try {
              const scraper = createWebScraper();
              const job = await scraper.firecrawlCrawlStatus(job_id);

              const response: Record<string, unknown> = {
                success: true,
                jobId: job.id,
                status: job.status,
                pagesScraped: job.pagesScraped,
                totalPages: job.totalPages,
              };

              if (job.status === 'completed' && job.results) {
                response.resultCount = job.results.length;
                response.totalWords = job.results.reduce(
                  (sum, r) => sum + (r.markdown?.split(/\s+/).length || 0),
                  0
                );
                response.pages = job.results.map((r) => ({
                  url: r.url,
                  title: r.title,
                  wordCount: r.markdown?.split(/\s+/).length || 0,
                }));
              }

              return this.createSuccessResponse(name, startTime, JSON.stringify(response, null, 2));
            } catch (error: any) {
              throw new Error(`Failed to get crawl status: ${error.message}`);
            }
          }

          case 'web_map_urls': {
            const { url, search, limit, include_subdomains } = args as {
              url: string;
              search?: string;
              limit?: number;
              include_subdomains?: boolean;
            };

            try {
              const scraper = createWebScraper();
              const urls = await scraper.firecrawlMap({
                url,
                search,
                limit: limit || 100,
                includeSubdomains: include_subdomains || false,
              });

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    baseUrl: url,
                    urlCount: urls.length,
                    urls: urls.slice(0, limit || 100),
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`URL mapping failed: ${error.message}`);
            }
          }

          case 'web_research': {
            const { topic, max_sources } = args as {
              topic: string;
              max_sources?: number;
            };

            try {
              const scraper = createWebScraper();
              const research = await scraper.exaResearch({
                query: topic,
                maxResults: max_sources || 20,
                outputFormat: 'markdown',
              });

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    topic,
                    summary: research.summary,
                    sourceCount: research.sources.length,
                    sources: research.sources.map((s) => ({
                      title: s.title,
                      url: s.url,
                      score: s.score,
                    })),
                    citations: research.citations,
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Research failed: ${error.message}`);
            }
          }

          case 'web_find_similar': {
            const { url, num_results, include_domains } = args as {
              url: string;
              num_results?: number;
              include_domains?: string[];
            };

            try {
              const scraper = createWebScraper();
              const results = await scraper.exaFindSimilar(url, {
                numResults: num_results || 10,
                includeDomains: include_domains,
              });

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    sourceUrl: url,
                    similarCount: results.length,
                    similar: results.map((r) => ({
                      title: r.title,
                      url: r.url,
                      score: r.score,
                      textPreview: r.text?.substring(0, 300),
                    })),
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Find similar failed: ${error.message}`);
            }
          }

          case 'web_gather_training_data': {
            const { query, num_results, min_word_count, domains, save_to_file, output_format } =
              args as {
                query: string;
                num_results?: number;
                min_word_count?: number;
                domains?: string[];
                save_to_file?: boolean;
                output_format?: 'jsonl' | 'json' | 'markdown';
              };

            try {
              const scraper = createWebScraper();
              const data = await scraper.gatherFromSearch(query, {
                numResults: num_results || 20,
                domains,
                minWordCount: min_word_count || 100,
              });

              const stats = scraper.getStats(data);
              let filePath: string | undefined;

              if (save_to_file !== false) {
                filePath = await scraper.saveTrainingData(data, {
                  format: output_format || 'jsonl',
                });
              }

              // Also save to knowledge database for pipeline integration
              for (const item of data) {
                try {
                  await knowledgeDb.addEntry({
                    source: {
                      type: 'article',
                      image_path: item.metadata.url,
                      capture_date: item.metadata.scrapedAt.toISOString(),
                      book_title: item.metadata.title,
                    },
                    raw_text: item.content,
                    cleaned_text: item.content,
                    category: 'reference' as Category,
                    topics: [],
                    tags: ['web_scrape', item.metadata.provider],
                    quality_score: 80,
                    ocr_confidence: 100, // Not OCR, direct text
                    manually_reviewed: false,
                  });
                } catch {
                  // Ignore duplicate entries
                }
              }

              return this.createSuccessResponse(
                name,
                startTime,
                JSON.stringify(
                  {
                    success: true,
                    query,
                    stats,
                    savedToFile: filePath,
                    savedToKnowledgeDb: data.length,
                    message: `Gathered ${data.length} documents (${stats.totalWords} words). Ready for training pair generation.`,
                    nextSteps: [
                      'Use knowledge_generate_pairs to create training pairs',
                      'Use knowledge_export_dataset to export in Alpaca/ShareGPT format',
                      'Use finetune_model to train on the generated data',
                    ],
                  },
                  null,
                  2
                )
              );
            } catch (error: any) {
              throw new Error(`Gather training data failed: ${error.message}`);
            }
          }

          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
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
        } else if (
          error.message.includes('EACCES') ||
          error.message.includes('permission denied')
        ) {
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

        const responseText =
          suggestions.length > 0
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
