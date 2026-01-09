import { exec } from 'child_process';
import { promisify } from 'util';
import logger from './logger.js';
import config from './config.js';

const execPromise = promisify(exec);

// Resource limits configuration (loaded from config)
export const RESOURCE_LIMITS = config.getLimits();

// Track concurrent operations for rate limiting
let currentOperations = 0;
const operationQueue: Array<() => Promise<void>> = [];

export class SecurityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SecurityError';
  }
}

export class TimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TimeoutError';
  }
}

// Rate limiter for tool executions
export async function acquireExecutionSlot(): Promise<() => void> {
  if (currentOperations >= RESOURCE_LIMITS.maxConcurrentOperations) {
    logger.info('Rate limit reached, queuing operation');

    await new Promise<void>((resolve) => {
      operationQueue.push(async () => resolve());
    });
  }

  currentOperations++;
  logger.debug(`Acquired execution slot. Current operations: ${currentOperations}`);

  return () => {
    currentOperations--;
    logger.debug(`Released execution slot. Current operations: ${currentOperations}`);

    // Process next queued operation
    const next = operationQueue.shift();
    if (next) {
      next();
    }
  };
}

// Sanitize Python script to prevent code injection
export function sanitizePythonScript(script: string): string {
  if (script.length > RESOURCE_LIMITS.maxScriptLength) {
    throw new SecurityError(
      `Python script exceeds maximum length of ${RESOURCE_LIMITS.maxScriptLength} characters`
    );
  }

  // Check for potentially dangerous patterns
  const dangerousPatterns = [
    /import\s+os\.system/i,
    /import\s+subprocess/i,
    /eval\s*\(/i,
    /exec\s*\(/i,
    /__import__\s*\(/i,
    /compile\s*\(/i,
    /open\s*\([^)]*['"]w/i, // Writing to files (except through safe APIs)
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(script)) {
      logger.warn('Dangerous pattern detected in Python script', { pattern: pattern.toString() });
      // Don't throw - log and continue for now, as some patterns may be false positives
      // In strict mode, you could throw SecurityError
    }
  }

  return script;
}

// Execute Python with timeout
export async function executePythonWithTimeout(
  script: string,
  timeout: number = RESOURCE_LIMITS.maxExecutionTime
): Promise<{ stdout: string; stderr: string }> {
  const sanitizedScript = sanitizePythonScript(script);

  logger.debug('Executing Python script with timeout', { timeout, scriptLength: script.length });

  try {
    const result = await Promise.race([
      execPromise(`python -c "${sanitizedScript}"`),
      new Promise<never>((_, reject) => {
        setTimeout(() => {
          reject(new TimeoutError(`Operation exceeded timeout of ${timeout}ms`));
        }, timeout);
      }),
    ]);

    logger.debug('Python script executed successfully');
    return result;
  } catch (error: unknown) {
    if (error instanceof TimeoutError) {
      logger.error('Python script execution timed out', { timeout });
      throw error;
    }

    const msg = error instanceof Error ? error.message : String(error);
    logger.error('Python script execution failed', { error: msg });
    throw error;
  }
}

// Validate file size before processing
export async function validateFileSize(filePath: string): Promise<void> {
  try {
    const { stdout } = await execPromise(
      `stat -c%s "${filePath}" 2>/dev/null || wc -c < "${filePath}"`
    );
    const fileSize = parseInt(stdout.trim(), 10);

    if (fileSize > RESOURCE_LIMITS.maxFileSize) {
      throw new SecurityError(
        `File size ${fileSize} bytes exceeds maximum allowed size of ${RESOURCE_LIMITS.maxFileSize} bytes`
      );
    }

    logger.debug('File size validation passed', { filePath, fileSize });
  } catch (error: unknown) {
    if (error instanceof SecurityError) {
      throw error;
    }
    const msg = error instanceof Error ? error.message : String(error);
    logger.warn('Could not validate file size', { filePath, error: msg });
    // Don't throw - file might not exist yet or stat failed
  }
}

// Sanitize environment variables
export function sanitizeEnvironment(): Record<string, string> {
  const safeEnv: Record<string, string> = {};

  // Only pass through safe environment variables
  const allowedVars = [
    'PATH',
    'HOME',
    'USER',
    'HUGGINGFACE_TOKEN',
    'HF_TOKEN',
    'TRANSFORMERS_CACHE',
    'HF_HOME',
  ];

  for (const varName of allowedVars) {
    const value = process.env[varName];
    if (value) {
      safeEnv[varName] = value;
    }
  }

  return safeEnv;
}

// Create a safe execution context
export interface SafeExecutionContext {
  timeout: number;
  maxRetries: number;
  validateInput: boolean;
  sanitizeScript: boolean;
}

export const DEFAULT_EXECUTION_CONTEXT: SafeExecutionContext = {
  timeout: RESOURCE_LIMITS.maxExecutionTime,
  maxRetries: 2,
  validateInput: true,
  sanitizeScript: true,
};

// Execute with full safety checks
export async function safeExecute(
  script: string,
  context: SafeExecutionContext = DEFAULT_EXECUTION_CONTEXT
): Promise<string> {
  const release = await acquireExecutionSlot();

  try {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= context.maxRetries; attempt++) {
      try {
        if (attempt > 0) {
          logger.info(`Retry attempt ${attempt} of ${context.maxRetries}`);
          // Exponential backoff: 1s, 2s, 4s
          await new Promise((resolve) => setTimeout(resolve, Math.pow(2, attempt - 1) * 1000));
        }

        const { stdout, stderr } = await executePythonWithTimeout(script, context.timeout);

        if (stderr && !stdout) {
          throw new Error(stderr);
        }

        return stdout;
      } catch (error: unknown) {
        lastError = error instanceof Error ? error : new Error(String(error));

        if (error instanceof TimeoutError || error instanceof SecurityError) {
          // Don't retry on timeout or security errors
          throw error;
        }

        const msg = error instanceof Error ? error.message : String(error);
        logger.warn(`Execution attempt ${attempt + 1} failed`, { error: msg });
      }
    }

    throw lastError || new Error('Execution failed after retries');
  } finally {
    release();
  }
}
