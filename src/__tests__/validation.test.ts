import { describe, test, expect } from '@jest/globals';
import { validators, ValidationError, validateToolInputs } from '../utils/validation.js';

describe('validators', () => {
  describe('modelName', () => {
    test('should accept valid model names', () => {
      expect(() => validators.modelName('unsloth/Llama-3.2-1B')).not.toThrow();
      expect(() => validators.modelName('meta-llama/Llama-2-7b')).not.toThrow();
    });

    test('should reject empty model names', () => {
      expect(() => validators.modelName('')).toThrow(ValidationError);
    });

    test('should reject non-string model names', () => {
      expect(() => validators.modelName(123 as unknown as string)).toThrow(ValidationError);
    });

    test('should reject very long model names', () => {
      const longName = 'a'.repeat(501);
      expect(() => validators.modelName(longName)).toThrow(ValidationError);
    });
  });

  describe('filePath', () => {
    test('should accept valid file paths', () => {
      expect(() => validators.filePath('/home/user/model.bin')).not.toThrow();
      expect(() => validators.filePath('./data/dataset.json')).not.toThrow();
    });

    test('should reject paths with directory traversal', () => {
      expect(() => validators.filePath('../../../etc/passwd')).toThrow(ValidationError);
      expect(() => validators.filePath('/home/../root')).toThrow(ValidationError);
    });

    test('should reject paths to restricted directories', () => {
      expect(() => validators.filePath('/etc/passwd')).toThrow(ValidationError);
      expect(() => validators.filePath('/root/secret')).toThrow(ValidationError);
    });

    test('should provide suggestions for invalid paths', () => {
      try {
        validators.filePath('../../../etc/passwd');
        fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(ValidationError);
        expect((error as ValidationError).suggestions).toBeDefined();
        expect((error as ValidationError).suggestions!.length).toBeGreaterThan(0);
      }
    });
  });

  describe('numericRange', () => {
    test('should accept values within range', () => {
      expect(validators.numericRange(50, 'test', 0, 100)).toBe(50);
      expect(validators.numericRange(0, 'test', 0, 100)).toBe(0);
      expect(validators.numericRange(100, 'test', 0, 100)).toBe(100);
    });

    test('should reject values outside range', () => {
      expect(() => validators.numericRange(-1, 'test', 0, 100)).toThrow(ValidationError);
      expect(() => validators.numericRange(101, 'test', 0, 100)).toThrow(ValidationError);
    });

    test('should use default value when undefined', () => {
      expect(validators.numericRange(undefined, 'test', 0, 100, 50)).toBe(50);
    });

    test('should throw when required and undefined', () => {
      expect(() => validators.numericRange(undefined, 'test', 0, 100)).toThrow(ValidationError);
    });

    test('should reject NaN', () => {
      expect(() => validators.numericRange(NaN, 'test', 0, 100)).toThrow(ValidationError);
    });
  });

  describe('enumValue', () => {
    test('should accept valid enum values', () => {
      expect(validators.enumValue('json', 'format', ['json', 'jsonl', 'csv'])).toBe('json');
      expect(validators.enumValue('csv', 'format', ['json', 'jsonl', 'csv'])).toBe('csv');
    });

    test('should reject invalid enum values', () => {
      expect(() => validators.enumValue('xml', 'format', ['json', 'jsonl', 'csv'])).toThrow(
        ValidationError
      );
    });

    test('should use default value when undefined', () => {
      expect(validators.enumValue(undefined, 'format', ['json', 'jsonl', 'csv'], 'json')).toBe(
        'json'
      );
    });
  });

  describe('sanitizeForPython', () => {
    test('should escape quotes', () => {
      expect(validators.sanitizeForPython('Hello "world"')).toBe('Hello \\"world\\"');
    });

    test('should escape backslashes', () => {
      expect(validators.sanitizeForPython('C:\\path\\to\\file')).toBe('C:\\\\path\\\\to\\\\file');
    });

    test('should escape newlines and tabs', () => {
      expect(validators.sanitizeForPython('Hello\nWorld\tTest')).toBe('Hello\\nWorld\\tTest');
    });
  });

  describe('text', () => {
    test('should accept valid text', () => {
      expect(() => validators.text('Hello world', 'prompt')).not.toThrow();
    });

    test('should reject text that is too long', () => {
      const longText = 'a'.repeat(100001);
      expect(() => validators.text(longText, 'prompt')).toThrow(ValidationError);
    });

    test('should accept custom max length', () => {
      const text = 'a'.repeat(100);
      expect(() => validators.text(text, 'prompt', 50)).toThrow(ValidationError);
      expect(() => validators.text(text, 'prompt', 150)).not.toThrow();
    });
  });
});

describe('validateToolInputs', () => {
  test('should validate load_model inputs', () => {
    expect(() =>
      validateToolInputs('load_model', {
        model_name: 'unsloth/Llama-3.2-1B',
        max_seq_length: 2048,
      })
    ).not.toThrow();
  });

  test('should reject invalid load_model inputs', () => {
    expect(() =>
      validateToolInputs('load_model', {
        model_name: '',
      })
    ).toThrow(ValidationError);

    expect(() =>
      validateToolInputs('load_model', {
        model_name: 'unsloth/Llama-3.2-1B',
        max_seq_length: 200000, // Too large
      })
    ).toThrow(ValidationError);
  });

  test('should validate finetune_model inputs', () => {
    expect(() =>
      validateToolInputs('finetune_model', {
        model_name: 'unsloth/Llama-3.2-1B',
        dataset_name: 'tatsu-lab/alpaca',
        output_dir: './output',
      })
    ).not.toThrow();
  });

  test('should validate export_model inputs', () => {
    expect(() =>
      validateToolInputs('export_model', {
        model_path: './model',
        export_format: 'gguf',
        output_path: './output.gguf',
      })
    ).not.toThrow();

    expect(() =>
      validateToolInputs('export_model', {
        model_path: './model',
        export_format: 'invalid_format',
        output_path: './output',
      })
    ).toThrow(ValidationError);
  });

  test('should validate compare_tokenizers inputs', () => {
    expect(() =>
      validateToolInputs('compare_tokenizers', {
        text: 'Sample text',
        tokenizer1_path: './tok1.json',
        tokenizer2_path: './tok2.json',
      })
    ).not.toThrow();
  });

  test('should validate benchmark_model inputs', () => {
    expect(() =>
      validateToolInputs('benchmark_model', {
        model_name: 'unsloth/Llama-3.2-1B',
        prompt: 'Test prompt',
        num_iterations: 10,
      })
    ).not.toThrow();

    expect(() =>
      validateToolInputs('benchmark_model', {
        model_name: 'unsloth/Llama-3.2-1B',
        prompt: 'Test prompt',
        num_iterations: 1000, // Too many
      })
    ).toThrow(ValidationError);
  });
});
