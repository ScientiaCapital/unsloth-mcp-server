import path from 'path';
import { existsSync, readFileSync } from 'fs';
import logger from './logger.js';

export interface ServerConfig {
  // Server settings
  server: {
    name: string;
    version: string;
    environment: 'development' | 'production' | 'test';
  };

  // Logging settings
  logging: {
    level: string;
    filePath: string;
    maxSize: number;
    maxFiles: number;
  };

  // Python settings
  python: {
    path: string;
    timeout: number;
    maxRetries: number;
  };

  // Resource limits
  limits: {
    maxExecutionTime: number;
    maxFileSize: number;
    maxScriptLength: number;
    maxConcurrentOperations: number;
  };

  // Cache settings
  cache: {
    enabled: boolean;
    ttl: number; // Time to live in seconds
    maxSize: number; // Max number of cached items
    directory: string;
  };

  // Security settings
  security: {
    validateInput: boolean;
    sanitizeScripts: boolean;
    allowedPaths: string[];
    blockedPaths: string[];
  };
}

const DEFAULT_CONFIG: ServerConfig = {
  server: {
    name: 'unsloth-server',
    version: '2.0.1',
    environment: (process.env.NODE_ENV as any) || 'development',
  },

  logging: {
    level: process.env.LOG_LEVEL || 'info',
    filePath: process.env.LOG_PATH || './logs',
    maxSize: 5242880, // 5MB
    maxFiles: 5,
  },

  python: {
    path: process.env.PYTHON_PATH || 'python',
    timeout: parseInt(process.env.PYTHON_TIMEOUT || '600000', 10), // 10 minutes
    maxRetries: parseInt(process.env.MAX_RETRIES || '2', 10),
  },

  limits: {
    maxExecutionTime: parseInt(process.env.MAX_EXECUTION_TIME || '600000', 10),
    maxFileSize: parseInt(process.env.MAX_FILE_SIZE || String(100 * 1024 * 1024), 10),
    maxScriptLength: parseInt(process.env.MAX_SCRIPT_LENGTH || '50000', 10),
    maxConcurrentOperations: parseInt(process.env.MAX_CONCURRENT_OPERATIONS || '3', 10),
  },

  cache: {
    enabled: process.env.CACHE_ENABLED !== 'false',
    ttl: parseInt(process.env.CACHE_TTL || '3600', 10), // 1 hour
    maxSize: parseInt(process.env.CACHE_MAX_SIZE || '1000', 10),
    directory: process.env.CACHE_DIR || './.cache',
  },

  security: {
    validateInput: process.env.VALIDATE_INPUT !== 'false',
    sanitizeScripts: process.env.SANITIZE_SCRIPTS !== 'false',
    allowedPaths: process.env.ALLOWED_PATHS?.split(',') || [],
    blockedPaths: process.env.BLOCKED_PATHS?.split(',') || ['/etc', '/sys', '/proc', '/root'],
  },
};

class ConfigManager {
  private config: ServerConfig;

  constructor() {
    this.config = { ...DEFAULT_CONFIG };
    this.loadConfigFile();
    this.validateConfig();
  }

  private loadConfigFile(): void {
    const configPaths = [
      process.env.CONFIG_FILE,
      './config.json',
      './unsloth-config.json',
      path.join(process.env.HOME || '', '.unsloth-mcp-config.json'),
    ].filter(Boolean) as string[];

    for (const configPath of configPaths) {
      if (existsSync(configPath)) {
        try {
          const fileContent = readFileSync(configPath, 'utf-8');
          const fileConfig = JSON.parse(fileContent);
          this.mergeConfig(fileConfig);
          logger.info(`Loaded configuration from ${configPath}`);
          break;
        } catch (error: any) {
          logger.warn(`Failed to load config from ${configPath}`, { error: error.message });
        }
      }
    }
  }

  private mergeConfig(fileConfig: Partial<ServerConfig>): void {
    // Deep merge configuration
    if (fileConfig.server) {
      this.config.server = { ...this.config.server, ...fileConfig.server };
    }
    if (fileConfig.logging) {
      this.config.logging = { ...this.config.logging, ...fileConfig.logging };
    }
    if (fileConfig.python) {
      this.config.python = { ...this.config.python, ...fileConfig.python };
    }
    if (fileConfig.limits) {
      this.config.limits = { ...this.config.limits, ...fileConfig.limits };
    }
    if (fileConfig.cache) {
      this.config.cache = { ...this.config.cache, ...fileConfig.cache };
    }
    if (fileConfig.security) {
      this.config.security = { ...this.config.security, ...fileConfig.security };
    }
  }

  private validateConfig(): void {
    // Validate limits are positive
    if (this.config.limits.maxExecutionTime <= 0) {
      logger.warn('maxExecutionTime must be positive, using default');
      this.config.limits.maxExecutionTime = DEFAULT_CONFIG.limits.maxExecutionTime;
    }

    if (this.config.limits.maxConcurrentOperations <= 0) {
      logger.warn('maxConcurrentOperations must be positive, using default');
      this.config.limits.maxConcurrentOperations = DEFAULT_CONFIG.limits.maxConcurrentOperations;
    }

    // Validate cache settings
    if (this.config.cache.ttl < 0) {
      logger.warn('cache TTL cannot be negative, using default');
      this.config.cache.ttl = DEFAULT_CONFIG.cache.ttl;
    }

    logger.debug('Configuration validated', { config: this.config });
  }

  get(): ServerConfig {
    return { ...this.config };
  }

  getServer() {
    return this.config.server;
  }

  getLogging() {
    return this.config.logging;
  }

  getPython() {
    return this.config.python;
  }

  getLimits() {
    return this.config.limits;
  }

  getCache() {
    return this.config.cache;
  }

  getSecurity() {
    return this.config.security;
  }

  set(key: keyof ServerConfig, value: any): void {
    this.config[key] = { ...this.config[key], ...value };
    this.validateConfig();
  }

  reload(): void {
    this.config = { ...DEFAULT_CONFIG };
    this.loadConfigFile();
    this.validateConfig();
    logger.info('Configuration reloaded');
  }
}

// Export both the class and a singleton instance
export { ConfigManager, DEFAULT_CONFIG };
export const config = new ConfigManager();
export default config;
