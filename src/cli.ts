#!/usr/bin/env node
import { exec } from 'child_process';
import { promisify } from 'util';
import config from './utils/config.js';
import cache from './utils/cache.js';
import { metricsCollector } from './utils/metrics.js';

const execPromise = promisify(exec);

const COLORS = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
};

function colorize(text: string, color: keyof typeof COLORS): string {
  return `${COLORS[color]}${text}${COLORS.reset}`;
}

function printHeader(title: string): void {
  console.log('\n' + colorize('='.repeat(60), 'cyan'));
  console.log(colorize(`  ${title}`, 'bright'));
  console.log(colorize('='.repeat(60), 'cyan') + '\n');
}

function printSection(title: string): void {
  console.log(colorize(`\n${title}`, 'yellow'));
  console.log(colorize('-'.repeat(40), 'yellow'));
}

async function checkInstallation(): Promise<void> {
  printHeader('Unsloth Installation Check');

  console.log('Checking Python...');
  try {
    const { stdout: pythonVersion } = await execPromise('python --version');
    console.log(colorize(`✓ Python: ${pythonVersion.trim()}`, 'green'));
  } catch {
    console.log(colorize('✗ Python not found', 'red'));
    return;
  }

  console.log('\nChecking Unsloth...');
  try {
    await execPromise('python -c "import unsloth; print(unsloth.__version__)"');
    console.log(colorize('✓ Unsloth is installed', 'green'));
  } catch {
    console.log(colorize('✗ Unsloth not found', 'red'));
    console.log('\nInstall with: pip install unsloth');
    return;
  }

  console.log('\nChecking dependencies...');
  const deps = ['torch', 'transformers', 'datasets', 'tokenizers', 'trl'];
  for (const dep of deps) {
    try {
      const { stdout } = await execPromise(`python -c "import ${dep}; print(${dep}.__version__)"`);
      console.log(colorize(`✓ ${dep}: ${stdout.trim()}`, 'green'));
    } catch {
      console.log(colorize(`✗ ${dep} not found`, 'red'));
    }
  }

  console.log(colorize('\n✓ All checks passed!', 'green'));
}

function listModels(): void {
  printHeader('Supported Unsloth Models');

  const models = [
    { name: 'Llama 3.3 70B Instruct', id: 'unsloth/Llama-3.3-70B-Instruct-bnb-4bit' },
    { name: 'Llama 3.2 1B', id: 'unsloth/Llama-3.2-1B-bnb-4bit' },
    { name: 'Llama 3.2 1B Instruct', id: 'unsloth/Llama-3.2-1B-Instruct-bnb-4bit' },
    { name: 'Llama 3.2 3B', id: 'unsloth/Llama-3.2-3B-bnb-4bit' },
    { name: 'Llama 3.2 3B Instruct', id: 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit' },
    { name: 'Llama 3.1 8B', id: 'unsloth/Llama-3.1-8B-bnb-4bit' },
    { name: 'Mistral 7B Instruct v0.3', id: 'unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit' },
    { name: 'Mistral Small Instruct', id: 'unsloth/Mistral-Small-Instruct-2409' },
    { name: 'Phi 3.5 Mini Instruct', id: 'unsloth/Phi-3.5-mini-instruct' },
    { name: 'Phi 3 Medium 4K Instruct', id: 'unsloth/Phi-3-medium-4k-instruct' },
    { name: 'Gemma 2 9B', id: 'unsloth/gemma-2-9b-bnb-4bit' },
    { name: 'Gemma 2 27B', id: 'unsloth/gemma-2-27b-bnb-4bit' },
    { name: 'Qwen 2.5 7B', id: 'unsloth/Qwen-2.5-7B' },
  ];

  models.forEach((model, i) => {
    console.log(colorize(`${i + 1}. ${model.name}`, 'cyan'));
    console.log(`   ${model.id}\n`);
  });
}

function showConfig(): void {
  printHeader('Current Configuration');

  const cfg = config.get();

  printSection('Server');
  console.log(`  Name: ${cfg.server.name}`);
  console.log(`  Version: ${cfg.server.version}`);
  console.log(`  Environment: ${cfg.server.environment}`);

  printSection('Logging');
  console.log(`  Level: ${cfg.logging.level}`);
  console.log(`  Path: ${cfg.logging.filePath}`);
  console.log(`  Max Size: ${cfg.logging.maxSize} bytes`);
  console.log(`  Max Files: ${cfg.logging.maxFiles}`);

  printSection('Python');
  console.log(`  Path: ${cfg.python.path}`);
  console.log(`  Timeout: ${cfg.python.timeout}ms`);
  console.log(`  Max Retries: ${cfg.python.maxRetries}`);

  printSection('Resource Limits');
  console.log(`  Max Execution Time: ${cfg.limits.maxExecutionTime}ms`);
  console.log(`  Max File Size: ${cfg.limits.maxFileSize} bytes`);
  console.log(`  Max Script Length: ${cfg.limits.maxScriptLength} chars`);
  console.log(`  Max Concurrent Ops: ${cfg.limits.maxConcurrentOperations}`);

  printSection('Cache');
  console.log(`  Enabled: ${cfg.cache.enabled}`);
  console.log(`  TTL: ${cfg.cache.ttl}s`);
  console.log(`  Max Size: ${cfg.cache.maxSize} entries`);
  console.log(`  Directory: ${cfg.cache.directory}`);

  printSection('Security');
  console.log(`  Validate Input: ${cfg.security.validateInput}`);
  console.log(`  Sanitize Scripts: ${cfg.security.sanitizeScripts}`);
  console.log(`  Blocked Paths: ${cfg.security.blockedPaths.join(', ')}`);
}

function showMetrics(): void {
  printHeader('Performance Metrics');

  const stats = metricsCollector.getStats();

  if (stats.totalCalls === 0) {
    console.log('No metrics available yet.\n');
    return;
  }

  printSection('Overall Statistics');
  console.log(`  Total Calls: ${stats.totalCalls}`);
  console.log(`  Successful: ${colorize(String(stats.successfulCalls), 'green')} (${Math.round((stats.successfulCalls / stats.totalCalls) * 100)}%)`);
  console.log(`  Failed: ${colorize(String(stats.failedCalls), 'red')} (${Math.round((stats.failedCalls / stats.totalCalls) * 100)}%)`);

  if (stats.averageDuration > 0) {
    console.log(`\n  Average Duration: ${stats.averageDuration.toFixed(2)}ms`);
    console.log(`  Min Duration: ${stats.minDuration.toFixed(2)}ms`);
    console.log(`  Max Duration: ${stats.maxDuration.toFixed(2)}ms`);
  }

  console.log();
}

function showCacheStats(): void {
  printHeader('Cache Statistics');

  const stats = cache.getStats();

  console.log(`  Enabled: ${stats.enabled ? colorize('Yes', 'green') : colorize('No', 'red')}`);
  console.log(`  Memory Entries: ${stats.memoryEntries}`);
  console.log(`  Max Size: ${stats.maxSize}`);
  console.log(`  Default TTL: ${stats.defaultTTL}s`);

  console.log();
}

function clearCache(): void {
  printHeader('Clear Cache');

  cache.clear();
  console.log(colorize('✓ Cache cleared successfully!', 'green'));
  console.log();
}

function showHelp(): void {
  printHeader('Unsloth MCP CLI');

  console.log('Usage: unsloth-mcp <command>\n');
  console.log('Commands:');
  console.log(colorize('  check', 'cyan').padEnd(25) + 'Check Unsloth installation');
  console.log(colorize('  models', 'cyan').padEnd(25) + 'List supported models');
  console.log(colorize('  config', 'cyan').padEnd(25) + 'Show current configuration');
  console.log(colorize('  metrics', 'cyan').padEnd(25) + 'Show performance metrics');
  console.log(colorize('  cache', 'cyan').padEnd(25) + 'Show cache statistics');
  console.log(colorize('  cache-clear', 'cyan').padEnd(25) + 'Clear the cache');
  console.log(colorize('  help', 'cyan').padEnd(25) + 'Show this help message');

  console.log('\nEnvironment Variables:');
  console.log('  LOG_LEVEL              Logging level (debug, info, warn, error)');
  console.log('  NODE_ENV               Environment (development, production)');
  console.log('  PYTHON_PATH            Path to Python executable');
  console.log('  CACHE_ENABLED          Enable/disable cache (true/false)');
  console.log('  MAX_CONCURRENT_OPERATIONS   Max concurrent operations');

  console.log('\nConfiguration File:');
  console.log('  Create config.json or ~/.unsloth-mcp-config.json');
  console.log('  See config.example.json for template');

  console.log();
}

// Main CLI
async function main() {
  const command = process.argv[2];

  switch (command) {
    case 'check':
      await checkInstallation();
      break;
    case 'models':
      listModels();
      break;
    case 'config':
      showConfig();
      break;
    case 'metrics':
      showMetrics();
      break;
    case 'cache':
      showCacheStats();
      break;
    case 'cache-clear':
      clearCache();
      break;
    case 'help':
    case undefined:
      showHelp();
      break;
    default:
      console.log(colorize(`Unknown command: ${command}`, 'red'));
      console.log('Run "unsloth-mcp help" for usage information.\n');
      process.exit(1);
  }
}

main().catch(error => {
  console.error(colorize('Error:', 'red'), error.message);
  process.exit(1);
});
