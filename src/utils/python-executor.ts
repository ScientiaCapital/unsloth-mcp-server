/**
 * python-executor.ts - Execute Python scripts with JSON I/O
 *
 * This utility allows TypeScript to call the extracted Python scripts
 * with structured arguments and receive structured responses.
 *
 * USAGE:
 *   const result = await executePythonScript('finetune_model', {
 *     model_name: 'unsloth/Qwen2.5-0.5B-Instruct',
 *     dataset_name: 'data/training.jsonl',
 *   });
 */

import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import * as fs from 'fs';

const execPromise = promisify(exec);

// Scripts directory relative to this file
const SCRIPTS_DIR = path.resolve(__dirname, '../scripts');

/**
 * Available Python scripts
 */
export const SCRIPTS = {
  list_models: 'list_models.py',
  load_model: 'load_model.py',
  finetune_model: 'finetune_model.py',
  generate_text: 'generate_text.py',
  export_model: 'export_model.py',
  benchmark_model: 'benchmark_model.py',
  train_tokenizer: 'train_tokenizer.py',
  get_model_info: 'get_model_info.py',
} as const;

export type ScriptName = keyof typeof SCRIPTS;

/**
 * Result from a Python script execution
 */
export interface ScriptResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  stderr?: string;
}

/**
 * Execute a Python script with JSON arguments
 *
 * @param scriptName - Name of the script (without .py)
 * @param args - Arguments to pass as JSON
 * @returns Parsed JSON result from the script
 */
export async function executePythonScript<T = unknown>(
  scriptName: ScriptName,
  args: Record<string, unknown> = {}
): Promise<ScriptResult<T>> {
  const scriptFile = SCRIPTS[scriptName];
  if (!scriptFile) {
    return {
      success: false,
      error: `Unknown script: ${scriptName}. Available: ${Object.keys(SCRIPTS).join(', ')}`,
    };
  }

  const scriptPath = path.join(SCRIPTS_DIR, scriptFile);

  if (!fs.existsSync(scriptPath)) {
    return {
      success: false,
      error: `Script not found: ${scriptPath}`,
    };
  }

  const argsJson = JSON.stringify(args);

  return new Promise((resolve) => {
    const process = spawn('python', [scriptPath, argsJson], {
      cwd: SCRIPTS_DIR,
    });

    let stdout = '';
    let stderr = '';

    process.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    process.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    process.on('close', (code) => {
      if (code !== 0 && !stdout.trim()) {
        resolve({
          success: false,
          error: stderr || `Script exited with code ${code}`,
          stderr,
        });
        return;
      }

      try {
        const data = JSON.parse(stdout.trim()) as T;
        resolve({
          success: true,
          data,
          stderr: stderr || undefined,
        });
      } catch {
        resolve({
          success: false,
          error: `Failed to parse output: ${stdout}`,
          stderr,
        });
      }
    });

    process.on('error', (err) => {
      resolve({
        success: false,
        error: `Failed to execute script: ${err.message}`,
      });
    });
  });
}

/**
 * Execute a Python script and stream output (for long-running tasks)
 *
 * @param scriptName - Name of the script
 * @param args - Arguments to pass as JSON
 * @param onProgress - Callback for stderr progress messages
 * @returns Final JSON result
 */
export async function executePythonScriptWithProgress<T = unknown>(
  scriptName: ScriptName,
  args: Record<string, unknown> = {},
  onProgress?: (message: string) => void
): Promise<ScriptResult<T>> {
  const scriptFile = SCRIPTS[scriptName];
  if (!scriptFile) {
    return {
      success: false,
      error: `Unknown script: ${scriptName}`,
    };
  }

  const scriptPath = path.join(SCRIPTS_DIR, scriptFile);
  const argsJson = JSON.stringify(args);

  return new Promise((resolve) => {
    const process = spawn('python', [scriptPath, argsJson], {
      cwd: SCRIPTS_DIR,
    });

    let stdout = '';
    let stderr = '';

    process.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    process.stderr.on('data', (data) => {
      const message = data.toString();
      stderr += message;
      if (onProgress) {
        // Split by lines and call progress for each
        message
          .split('\n')
          .filter((l: string) => l.trim())
          .forEach(onProgress);
      }
    });

    process.on('close', (code) => {
      if (code !== 0 && !stdout.trim()) {
        resolve({
          success: false,
          error: stderr || `Script exited with code ${code}`,
          stderr,
        });
        return;
      }

      try {
        const data = JSON.parse(stdout.trim()) as T;
        resolve({
          success: true,
          data,
          stderr: stderr || undefined,
        });
      } catch {
        resolve({
          success: false,
          error: `Failed to parse output: ${stdout}`,
          stderr,
        });
      }
    });

    process.on('error', (err) => {
      resolve({
        success: false,
        error: `Failed to execute script: ${err.message}`,
      });
    });
  });
}

/**
 * Check if Python and required packages are available
 */
export async function checkPythonEnvironment(): Promise<{
  pythonVersion: string | null;
  unslothAvailable: boolean;
  torchAvailable: boolean;
  cudaAvailable: boolean;
}> {
  const result = {
    pythonVersion: null as string | null,
    unslothAvailable: false,
    torchAvailable: false,
    cudaAvailable: false,
  };

  try {
    const { stdout: version } = await execPromise('python --version');
    result.pythonVersion = version.trim();
  } catch {
    return result;
  }

  try {
    await execPromise('python -c "import unsloth"');
    result.unslothAvailable = true;
  } catch {
    // Unsloth not installed
  }

  try {
    await execPromise('python -c "import torch"');
    result.torchAvailable = true;
  } catch {
    // PyTorch not installed
  }

  try {
    const { stdout } = await execPromise(
      'python -c "import torch; print(torch.cuda.is_available())"'
    );
    result.cudaAvailable = stdout.trim() === 'True';
  } catch {
    // CUDA check failed
  }

  return result;
}

/**
 * Get the path to a script file
 */
export function getScriptPath(scriptName: ScriptName): string {
  return path.join(SCRIPTS_DIR, SCRIPTS[scriptName]);
}

/**
 * List all available scripts
 */
export function listScripts(): { name: string; path: string }[] {
  return Object.entries(SCRIPTS).map(([name, file]) => ({
    name,
    path: path.join(SCRIPTS_DIR, file),
  }));
}
