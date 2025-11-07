import { existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync, readdirSync } from 'fs';
import { join } from 'path';
import crypto from 'crypto';
import logger from './logger.js';
import config from './config.js';

interface CacheEntry<T> {
  key: string;
  value: T;
  timestamp: number;
  ttl: number;
}

class Cache {
  private memoryCache: Map<string, CacheEntry<any>>;
  private cacheDir: string;
  private enabled: boolean;
  private defaultTTL: number;
  private maxSize: number;

  constructor() {
    const cacheConfig = config.getCache();
    this.memoryCache = new Map();
    this.cacheDir = cacheConfig.directory;
    this.enabled = cacheConfig.enabled;
    this.defaultTTL = cacheConfig.ttl;
    this.maxSize = cacheConfig.maxSize;

    if (this.enabled) {
      this.initializeCacheDir();
      this.cleanupExpiredEntries();
    }

    logger.info('Cache initialized', {
      enabled: this.enabled,
      ttl: this.defaultTTL,
      maxSize: this.maxSize,
      directory: this.cacheDir,
    });
  }

  private initializeCacheDir(): void {
    if (!existsSync(this.cacheDir)) {
      mkdirSync(this.cacheDir, { recursive: true });
      logger.debug('Cache directory created', { directory: this.cacheDir });
    }
  }

  private generateKey(input: string | object): string {
    const data = typeof input === 'string' ? input : JSON.stringify(input);
    return crypto.createHash('sha256').update(data).digest('hex');
  }

  private isExpired(entry: CacheEntry<any>): boolean {
    const age = (Date.now() - entry.timestamp) / 1000; // age in seconds
    return age > entry.ttl;
  }

  private cleanupExpiredEntries(): void {
    if (!this.enabled) return;

    // Clean memory cache
    for (const [key, entry] of this.memoryCache.entries()) {
      if (this.isExpired(entry)) {
        this.memoryCache.delete(key);
        logger.debug('Removed expired cache entry from memory', { key });
      }
    }

    // Clean disk cache
    try {
      const files = readdirSync(this.cacheDir);
      for (const file of files) {
        if (file.endsWith('.cache')) {
          const filePath = join(this.cacheDir, file);
          try {
            const content = readFileSync(filePath, 'utf-8');
            const entry = JSON.parse(content);
            if (this.isExpired(entry)) {
              unlinkSync(filePath);
              logger.debug('Removed expired cache file', { file });
            }
          } catch (error) {
            // Invalid cache file, remove it
            unlinkSync(filePath);
          }
        }
      }
    } catch (error: any) {
      logger.warn('Failed to cleanup cache directory', { error: error.message });
    }
  }

  private enforceMaxSize(): void {
    if (this.memoryCache.size > this.maxSize) {
      // Remove oldest entries
      const entries = Array.from(this.memoryCache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);

      const toRemove = entries.slice(0, this.memoryCache.size - this.maxSize);
      for (const [key] of toRemove) {
        this.memoryCache.delete(key);
        logger.debug('Removed old cache entry to enforce max size', { key });
      }
    }
  }

  set<T>(key: string, value: T, ttl?: number): void {
    if (!this.enabled) return;

    const cacheKey = this.generateKey(key);
    const entry: CacheEntry<T> = {
      key: cacheKey,
      value,
      timestamp: Date.now(),
      ttl: ttl || this.defaultTTL,
    };

    // Store in memory
    this.memoryCache.set(cacheKey, entry);
    this.enforceMaxSize();

    // Store on disk
    try {
      const filePath = join(this.cacheDir, `${cacheKey}.cache`);
      writeFileSync(filePath, JSON.stringify(entry), 'utf-8');
      logger.debug('Cache entry stored', { key: cacheKey, ttl: entry.ttl });
    } catch (error: any) {
      logger.warn('Failed to write cache to disk', { key: cacheKey, error: error.message });
    }
  }

  get<T>(key: string): T | null {
    if (!this.enabled) return null;

    const cacheKey = this.generateKey(key);

    // Check memory cache first
    const memEntry = this.memoryCache.get(cacheKey);
    if (memEntry && !this.isExpired(memEntry)) {
      logger.debug('Cache hit (memory)', { key: cacheKey });
      return memEntry.value as T;
    }

    // Check disk cache
    try {
      const filePath = join(this.cacheDir, `${cacheKey}.cache`);
      if (existsSync(filePath)) {
        const content = readFileSync(filePath, 'utf-8');
        const entry: CacheEntry<T> = JSON.parse(content);

        if (!this.isExpired(entry)) {
          // Restore to memory cache
          this.memoryCache.set(cacheKey, entry);
          logger.debug('Cache hit (disk)', { key: cacheKey });
          return entry.value;
        } else {
          // Expired, remove it
          unlinkSync(filePath);
          logger.debug('Cache entry expired', { key: cacheKey });
        }
      }
    } catch (error: any) {
      logger.warn('Failed to read cache from disk', { key: cacheKey, error: error.message });
    }

    logger.debug('Cache miss', { key: cacheKey });
    return null;
  }

  has(key: string): boolean {
    return this.get(key) !== null;
  }

  delete(key: string): void {
    if (!this.enabled) return;

    const cacheKey = this.generateKey(key);

    // Remove from memory
    this.memoryCache.delete(cacheKey);

    // Remove from disk
    try {
      const filePath = join(this.cacheDir, `${cacheKey}.cache`);
      if (existsSync(filePath)) {
        unlinkSync(filePath);
      }
      logger.debug('Cache entry deleted', { key: cacheKey });
    } catch (error: any) {
      logger.warn('Failed to delete cache file', { key: cacheKey, error: error.message });
    }
  }

  clear(): void {
    if (!this.enabled) return;

    // Clear memory
    this.memoryCache.clear();

    // Clear disk
    try {
      const files = readdirSync(this.cacheDir);
      for (const file of files) {
        if (file.endsWith('.cache')) {
          unlinkSync(join(this.cacheDir, file));
        }
      }
      logger.info('Cache cleared');
    } catch (error: any) {
      logger.warn('Failed to clear cache directory', { error: error.message });
    }
  }

  getStats() {
    return {
      enabled: this.enabled,
      memoryEntries: this.memoryCache.size,
      maxSize: this.maxSize,
      defaultTTL: this.defaultTTL,
    };
  }
}

export const cache = new Cache();
export default cache;
