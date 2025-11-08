import { describe, it, expect } from '@jest/globals';
import { cache } from '../utils/cache.js';

describe('Cache', () => {
  describe('basic operations', () => {
    it('should store and retrieve a value', () => {
      cache.set('test-key', 'test-value');
      const result = cache.get<string>('test-key');
      expect(result).toBe('test-value');
    });

    it('should return null for non-existent keys', () => {
      const result = cache.get<string>('non-existent-key-xyz');
      expect(result).toBeNull();
    });

    it('should check if key exists', () => {
      cache.set('exists-test', 'value');
      expect(cache.has('exists-test')).toBe(true);
      expect(cache.has('does-not-exist')).toBe(false);
    });

    it('should delete a key', () => {
      cache.set('delete-test', 'value');
      expect(cache.has('delete-test')).toBe(true);
      cache.delete('delete-test');
      expect(cache.has('delete-test')).toBe(false);
    });

    it('should get stats', () => {
      const stats = cache.getStats();
      expect(stats).toHaveProperty('enabled');
      expect(stats).toHaveProperty('memoryEntries');
      expect(stats).toHaveProperty('maxSize');
      expect(stats).toHaveProperty('defaultTTL');
    });
  });
});
