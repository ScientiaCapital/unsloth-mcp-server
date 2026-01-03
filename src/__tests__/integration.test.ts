import { describe, it, expect } from '@jest/globals';
import { validateToolInputs } from '../utils/validation.js';
import { cache } from '../utils/cache.js';
import { config } from '../utils/config.js';
import { metricsCollector } from '../utils/metrics.js';
import logger from '../utils/logger.js';

describe('Integration Tests - Utilities Working Together', () => {
  describe('Validation + Logging Integration', () => {
    it('should validate and log successful validation', () => {
      expect(() => {
        validateToolInputs('check_installation', {});
        logger.debug('Validation passed for check_installation');
      }).not.toThrow();
    });

    it('should validate complex tool with multiple parameters', () => {
      expect(() => {
        validateToolInputs('finetune_model', {
          model_name: 'unsloth/Llama-3.2-1B-bnb-4bit',
          dataset_name: 'tatsu-lab/alpaca',
          output_dir: './output',
          max_steps: 100,
          batch_size: 2,
        });
      }).not.toThrow();
    });
  });

  describe('Cache + Config Integration', () => {
    it('should use config settings for cache operations', () => {
      const cacheConfig = config.getCache();
      expect(cacheConfig.enabled).toBeDefined();

      if (cacheConfig.enabled) {
        cache.set('integration-test', { data: 'test' });
        const result = cache.get('integration-test');
        expect(result).toEqual({ data: 'test' });
      }
    });

    it('should respect cache TTL from config', () => {
      const cacheConfig = config.getCache();
      expect(cacheConfig.ttl).toBeGreaterThan(0);

      cache.set('ttl-test', 'value', 1);
      expect(cache.has('ttl-test')).toBe(true);
    });
  });

  describe('Metrics + Validation Integration', () => {
    it('should track validation performance', () => {
      const startTime = Date.now();

      validateToolInputs('load_model', {
        model_name: 'unsloth/Llama-3.2-1B-bnb-4bit',
      });

      metricsCollector.endTool('load_model', startTime, true);

      const stats = metricsCollector.getStats('load_model');
      expect(stats.totalCalls).toBeGreaterThan(0);
    });
  });

  describe('Full Workflow Integration', () => {
    it('should handle complete tool execution workflow', () => {
      // 1. Get config
      const serverConfig = config.get();
      expect(serverConfig).toBeDefined();

      // 2. Check cache for cached result
      const cacheKey = 'workflow-test';
      let result = cache.get(cacheKey);

      if (!result) {
        // 3. Validate inputs
        validateToolInputs('list_supported_models', {});

        // 4. Simulate result
        result = ['model1', 'model2'];

        // 5. Cache result
        cache.set(cacheKey, result, 3600);
      }

      // 6. Track metrics
      metricsCollector.endTool('list_supported_models', Date.now() - 100, true);

      // 7. Verify result
      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('Error Handling Integration', () => {
    it('should handle validation errors with metrics', () => {
      const startTime = Date.now();

      try {
        validateToolInputs('load_model', {
          // Missing required model_name
        });
      } catch (error) {
        metricsCollector.endTool('load_model', startTime, false);
        expect(error).toBeDefined();
      }

      const stats = metricsCollector.getStats('load_model');
      expect(stats.failedCalls).toBeGreaterThan(0);
    });
  });

  describe('Performance Under Load', () => {
    it('should handle rapid cache operations', () => {
      for (let i = 0; i < 100; i++) {
        cache.set(`perf-test-${i}`, { value: i });
      }

      for (let i = 0; i < 100; i++) {
        const result = cache.get(`perf-test-${i}`);
        expect(result).toEqual({ value: i });
      }
    });

    it('should handle rapid validation calls', () => {
      for (let i = 0; i < 100; i++) {
        validateToolInputs('check_installation', {});
      }

      // Should complete without errors
      expect(true).toBe(true);
    });

    it('should handle rapid metrics recording', () => {
      for (let i = 0; i < 100; i++) {
        metricsCollector.endTool('test-tool', Date.now() - 1, true);
      }

      const stats = metricsCollector.getStats('test-tool');
      expect(stats.totalCalls).toBeGreaterThanOrEqual(100);
    });
  });
});
