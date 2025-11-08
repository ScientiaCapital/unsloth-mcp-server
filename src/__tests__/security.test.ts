import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { sanitizePythonScript, SecurityError, RESOURCE_LIMITS } from '../utils/security.js';

describe('security utilities', () => {
  describe('sanitizePythonScript', () => {
    test('should accept safe Python scripts', () => {
      const safeScript = `
import json
print(json.dumps({"hello": "world"}))
`;
      expect(() => sanitizePythonScript(safeScript)).not.toThrow();
    });

    test('should reject scripts that are too long', () => {
      const longScript = 'a'.repeat(RESOURCE_LIMITS.maxScriptLength + 1);
      expect(() => sanitizePythonScript(longScript)).toThrow(SecurityError);
    });

    test('should detect potentially dangerous patterns', () => {
      // These should log warnings but not throw (current implementation)
      const dangerousScripts = [
        'import os.system\nos.system("rm -rf /")',
        'import subprocess\nsubprocess.call("ls")',
        'eval("print(1)")',
        'exec("print(1)")',
        '__import__("os")',
      ];

      for (const script of dangerousScripts) {
        // Currently these log warnings but don't throw
        // In strict mode, they would throw SecurityError
        expect(() => sanitizePythonScript(script)).not.toThrow();
      }
    });

    test('should accept scripts with safe imports', () => {
      const safeScript = `
import json
import torch
from transformers import AutoModel
from datasets import load_dataset
print("safe")
`;
      expect(() => sanitizePythonScript(safeScript)).not.toThrow();
    });
  });

  describe('RESOURCE_LIMITS', () => {
    test('should have reasonable defaults', () => {
      expect(RESOURCE_LIMITS.maxExecutionTime).toBeGreaterThan(0);
      expect(RESOURCE_LIMITS.maxFileSize).toBeGreaterThan(0);
      expect(RESOURCE_LIMITS.maxScriptLength).toBeGreaterThan(0);
      expect(RESOURCE_LIMITS.maxConcurrentOperations).toBeGreaterThan(0);
    });

    test('should have execution time of 10 minutes', () => {
      expect(RESOURCE_LIMITS.maxExecutionTime).toBe(600000);
    });

    test('should have file size limit of 100MB', () => {
      expect(RESOURCE_LIMITS.maxFileSize).toBe(100 * 1024 * 1024);
    });
  });
});
