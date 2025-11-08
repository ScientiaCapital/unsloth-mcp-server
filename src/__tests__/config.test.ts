import { describe, it, expect } from '@jest/globals';
import { config } from '../utils/config.js';

describe('ConfigManager', () => {
  describe('basic operations', () => {
    it('should return configuration', () => {
      const cfg = config.get();
      expect(cfg).toHaveProperty('server');
      expect(cfg).toHaveProperty('logging');
      expect(cfg).toHaveProperty('python');
      expect(cfg).toHaveProperty('limits');
      expect(cfg).toHaveProperty('cache');
      expect(cfg).toHaveProperty('security');
    });

    it('should have valid server config', () => {
      const serverConfig = config.getServer();
      expect(serverConfig).toHaveProperty('name');
      expect(serverConfig).toHaveProperty('version');
      expect(serverConfig).toHaveProperty('environment');
    });

    it('should have valid logging config', () => {
      const loggingConfig = config.getLogging();
      expect(loggingConfig).toHaveProperty('level');
      expect(loggingConfig).toHaveProperty('filePath');
      expect(loggingConfig).toHaveProperty('maxSize');
      expect(loggingConfig).toHaveProperty('maxFiles');
    });

    it('should have valid cache config', () => {
      const cacheConfig = config.getCache();
      expect(cacheConfig).toHaveProperty('enabled');
      expect(cacheConfig).toHaveProperty('ttl');
      expect(cacheConfig).toHaveProperty('maxSize');
      expect(cacheConfig).toHaveProperty('directory');
    });

    it('should have valid security config', () => {
      const securityConfig = config.getSecurity();
      expect(securityConfig).toHaveProperty('validateInput');
      expect(securityConfig).toHaveProperty('sanitizeScripts');
      expect(securityConfig).toHaveProperty('allowedPaths');
      expect(securityConfig).toHaveProperty('blockedPaths');
    });
  });
});
