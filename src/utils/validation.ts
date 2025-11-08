import logger from './logger.js';

export class ValidationError extends Error {
  constructor(
    message: string,
    public suggestions?: string[]
  ) {
    super(message);
    this.name = 'ValidationError';
  }
}

export const validators = {
  // Validate model name format
  modelName: (name: string): void => {
    if (!name || typeof name !== 'string') {
      throw new ValidationError('Model name is required and must be a string');
    }
    if (name.length === 0) {
      throw new ValidationError('Model name cannot be empty');
    }
    if (name.length > 500) {
      throw new ValidationError('Model name is too long (max 500 characters)');
    }
  },

  // Validate file path
  filePath: (path: string, fieldName: string = 'file path'): void => {
    if (!path || typeof path !== 'string') {
      throw new ValidationError(`${fieldName} is required and must be a string`);
    }

    // Prevent directory traversal
    if (path.includes('..')) {
      throw new ValidationError(`${fieldName} contains invalid characters (..)`, [
        'Use absolute paths or paths without .. components',
      ]);
    }

    // Prevent absolute paths outside safe directories (optional, can be configured)
    const dangerousPaths = ['/etc', '/sys', '/proc', '/root'];
    if (dangerousPaths.some((dp) => path.startsWith(dp))) {
      throw new ValidationError(`${fieldName} points to a restricted directory`, [
        'Use paths in your home directory or project directory',
      ]);
    }
  },

  // Validate dataset name
  datasetName: (name: string): void => {
    if (!name || typeof name !== 'string') {
      throw new ValidationError('Dataset name is required and must be a string');
    }
    if (name.length === 0) {
      throw new ValidationError('Dataset name cannot be empty');
    }
    if (name.length > 500) {
      throw new ValidationError('Dataset name is too long (max 500 characters)');
    }
  },

  // Validate text input
  text: (text: string, fieldName: string = 'text', maxLength: number = 100000): void => {
    if (typeof text !== 'string') {
      throw new ValidationError(`${fieldName} must be a string`);
    }
    if (text.length > maxLength) {
      throw new ValidationError(`${fieldName} is too long (max ${maxLength} characters)`, [
        `Current length: ${text.length}`,
        'Consider splitting into smaller chunks',
      ]);
    }
  },

  // Validate numeric range
  numericRange: (
    value: number | undefined,
    fieldName: string,
    min: number,
    max: number,
    defaultValue?: number
  ): number => {
    if (value === undefined) {
      if (defaultValue !== undefined) {
        return defaultValue;
      }
      throw new ValidationError(`${fieldName} is required`);
    }

    if (typeof value !== 'number' || isNaN(value)) {
      throw new ValidationError(`${fieldName} must be a valid number`);
    }

    if (value < min || value > max) {
      throw new ValidationError(`${fieldName} must be between ${min} and ${max}`, [
        `Current value: ${value}`,
      ]);
    }

    return value;
  },

  // Validate enum value
  enumValue: <T extends string>(
    value: T | undefined,
    fieldName: string,
    allowedValues: T[],
    defaultValue?: T
  ): T => {
    if (value === undefined) {
      if (defaultValue !== undefined) {
        return defaultValue;
      }
      throw new ValidationError(`${fieldName} is required`);
    }

    if (!allowedValues.includes(value)) {
      throw new ValidationError(`${fieldName} must be one of: ${allowedValues.join(', ')}`, [
        `Current value: ${value}`,
      ]);
    }

    return value;
  },

  // Sanitize text for Python script execution
  sanitizeForPython: (text: string): string => {
    // Escape backslashes and quotes
    return text
      .replace(/\\/g, '\\\\')
      .replace(/"/g, '\\"')
      .replace(/\n/g, '\\n')
      .replace(/\r/g, '\\r')
      .replace(/\t/g, '\\t');
  },

  // Validate search query
  searchQuery: (query: string | undefined, maxLength: number = 200): string => {
    if (query === undefined || query === '') {
      return '';
    }

    if (typeof query !== 'string') {
      throw new ValidationError('Search query must be a string');
    }

    if (query.length > maxLength) {
      throw new ValidationError(`Search query is too long (max ${maxLength} characters)`, [
        `Current length: ${query.length}`,
      ]);
    }

    return query;
  },
};

// Helper to validate and sanitize all inputs
export function validateToolInputs(toolName: string, args: any): void {
  logger.debug(`Validating inputs for tool: ${toolName}`, { args });

  try {
    switch (toolName) {
      case 'load_model':
      case 'get_model_info':
        validators.modelName(args.model_name);
        if (args.max_seq_length !== undefined) {
          validators.numericRange(args.max_seq_length, 'max_seq_length', 128, 131072);
        }
        break;

      case 'finetune_model':
        validators.modelName(args.model_name);
        validators.datasetName(args.dataset_name);
        validators.filePath(args.output_dir, 'output_dir');
        if (args.max_seq_length !== undefined) {
          validators.numericRange(args.max_seq_length, 'max_seq_length', 128, 131072);
        }
        if (args.lora_rank !== undefined) {
          validators.numericRange(args.lora_rank, 'lora_rank', 1, 256);
        }
        if (args.batch_size !== undefined) {
          validators.numericRange(args.batch_size, 'batch_size', 1, 128);
        }
        if (args.max_steps !== undefined) {
          validators.numericRange(args.max_steps, 'max_steps', 1, 1000000);
        }
        break;

      case 'generate_text':
        validators.filePath(args.model_path, 'model_path');
        validators.text(args.prompt, 'prompt', 10000);
        if (args.max_new_tokens !== undefined) {
          validators.numericRange(args.max_new_tokens, 'max_new_tokens', 1, 4096);
        }
        if (args.temperature !== undefined) {
          validators.numericRange(args.temperature, 'temperature', 0, 2);
        }
        if (args.top_p !== undefined) {
          validators.numericRange(args.top_p, 'top_p', 0, 1);
        }
        break;

      case 'export_model':
        validators.filePath(args.model_path, 'model_path');
        validators.filePath(args.output_path, 'output_path');
        validators.enumValue(args.export_format, 'export_format', [
          'gguf',
          'ollama',
          'vllm',
          'huggingface',
        ]);
        if (args.quantization_bits !== undefined) {
          validators.numericRange(args.quantization_bits, 'quantization_bits', 2, 16);
        }
        break;

      case 'train_superbpe_tokenizer':
        validators.filePath(args.corpus_path, 'corpus_path');
        validators.filePath(args.output_path, 'output_path');
        if (args.vocab_size !== undefined) {
          validators.numericRange(args.vocab_size, 'vocab_size', 1000, 1000000);
        }
        break;

      case 'compare_tokenizers':
        validators.text(args.text, 'text', 50000);
        validators.filePath(args.tokenizer1_path, 'tokenizer1_path');
        validators.filePath(args.tokenizer2_path, 'tokenizer2_path');
        break;

      case 'benchmark_model':
        validators.modelName(args.model_name);
        validators.text(args.prompt, 'prompt', 1000);
        if (args.num_iterations !== undefined) {
          validators.numericRange(args.num_iterations, 'num_iterations', 1, 100);
        }
        if (args.max_new_tokens !== undefined) {
          validators.numericRange(args.max_new_tokens, 'max_new_tokens', 1, 2048);
        }
        break;

      case 'list_datasets':
        if (args.search_query !== undefined) {
          validators.searchQuery(args.search_query);
        }
        if (args.limit !== undefined) {
          validators.numericRange(args.limit, 'limit', 1, 100);
        }
        break;

      case 'prepare_dataset':
        validators.datasetName(args.dataset_name);
        validators.filePath(args.output_path, 'output_path');
        if (args.format !== undefined) {
          validators.enumValue(args.format, 'format', ['json', 'jsonl', 'csv']);
        }
        break;
    }

    logger.debug(`Validation successful for tool: ${toolName}`);
  } catch (error) {
    logger.warn(`Validation failed for tool: ${toolName}`, { error });
    throw error;
  }
}
