/**
 * Model Registry - Track trained and deployed models
 *
 * Maintains a registry of all models trained, their configurations,
 * performance metrics, and deployment status.
 *
 * @module utils/model-registry
 */

import * as fs from 'fs';
import * as path from 'path';
import logger from './logger.js';

// ============================================================================
// Types & Interfaces
// ============================================================================

export type ModelStatus = 'training' | 'trained' | 'deployed' | 'archived' | 'failed';
export type ModelProvider = 'unsloth' | 'huggingface' | 'ollama' | 'vllm' | 'gguf';

export interface TrainingConfig {
  base_model: string;
  dataset: string;
  dataset_size: number;
  lora_rank: number;
  lora_alpha: number;
  learning_rate: number;
  batch_size: number;
  gradient_accumulation: number;
  max_seq_length: number;
  epochs: number;
  max_steps: number;
  load_in_4bit: boolean;
  use_gradient_checkpointing: boolean;
}

export interface TrainingMetrics {
  final_loss: number;
  best_loss: number;
  total_steps: number;
  training_time_hours: number;
  tokens_processed: number;
  gpu_hours: number;
  cost_usd: number;
}

export interface DeploymentInfo {
  provider: ModelProvider;
  endpoint?: string;
  huggingface_repo?: string;
  ollama_model_name?: string;
  gguf_path?: string;
  deployed_at: string;
  last_used?: string;
  inference_count?: number;
}

export interface ModelEntry {
  id: string;
  name: string;
  description?: string;
  version: string;
  status: ModelStatus;

  // Base model info
  base_model: string;
  base_model_params: string; // e.g., "1B", "7B", "70B"

  // Training info
  training_config: TrainingConfig;
  training_metrics?: TrainingMetrics;
  training_started_at?: string;
  training_completed_at?: string;
  checkpoint_path?: string;

  // Deployment info
  deployments: DeploymentInfo[];

  // Data lineage
  dataset_sources: string[]; // URLs or paths
  knowledge_entries?: string[]; // Knowledge DB entry IDs

  // Quality
  evaluation_scores?: {
    task: string;
    score: number;
    benchmark?: string;
  }[];

  // Metadata
  tags: string[];
  created_at: string;
  updated_at: string;
  created_by?: string;
  project?: string;
}

export interface ModelRegistryConfig {
  registry_path: string;
  auto_save: boolean;
  backup_enabled: boolean;
  max_backups: number;
}

export interface RegistryStats {
  total_models: number;
  by_status: Record<ModelStatus, number>;
  by_base_model: Record<string, number>;
  total_training_hours: number;
  total_cost: number;
  total_tokens_processed: number;
}

// ============================================================================
// Model Registry Class
// ============================================================================

export class ModelRegistry {
  private config: ModelRegistryConfig;
  private models: Map<string, ModelEntry> = new Map();
  private registryFile: string;

  constructor(config?: Partial<ModelRegistryConfig>) {
    this.config = {
      registry_path: config?.registry_path || path.join(process.cwd(), 'data'),
      auto_save: config?.auto_save ?? true,
      backup_enabled: config?.backup_enabled ?? true,
      max_backups: config?.max_backups || 10,
    };

    this.registryFile = path.join(this.config.registry_path, 'model_registry.json');
    this.load();
  }

  // ==========================================================================
  // Core Operations
  // ==========================================================================

  /**
   * Register a new model (typically when starting training)
   */
  registerModel(entry: Omit<ModelEntry, 'id' | 'created_at' | 'updated_at'>): ModelEntry {
    const id = this.generateId();
    const now = new Date().toISOString();

    const model: ModelEntry = {
      ...entry,
      id,
      created_at: now,
      updated_at: now,
    };

    this.models.set(id, model);
    logger.info('Model registered', { id, name: model.name, base_model: model.base_model });

    if (this.config.auto_save) {
      this.save();
    }

    return model;
  }

  /**
   * Update an existing model entry
   */
  updateModel(id: string, updates: Partial<ModelEntry>): ModelEntry | null {
    const model = this.models.get(id);
    if (!model) {
      logger.warn('Model not found for update', { id });
      return null;
    }

    const updated: ModelEntry = {
      ...model,
      ...updates,
      id: model.id, // Preserve original ID
      created_at: model.created_at, // Preserve creation time
      updated_at: new Date().toISOString(),
    };

    this.models.set(id, updated);
    logger.info('Model updated', { id, name: updated.name });

    if (this.config.auto_save) {
      this.save();
    }

    return updated;
  }

  /**
   * Get a model by ID
   */
  getModel(id: string): ModelEntry | null {
    return this.models.get(id) || null;
  }

  /**
   * Get a model by name (returns most recent version)
   */
  getModelByName(name: string): ModelEntry | null {
    const models = Array.from(this.models.values())
      .filter((m) => m.name === name)
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

    return models[0] || null;
  }

  /**
   * List all models with optional filters
   */
  listModels(filters?: {
    status?: ModelStatus;
    base_model?: string;
    project?: string;
    tags?: string[];
  }): ModelEntry[] {
    let models = Array.from(this.models.values());

    if (filters?.status) {
      models = models.filter((m) => m.status === filters.status);
    }

    if (filters?.base_model) {
      models = models.filter((m) => m.base_model.includes(filters.base_model!));
    }

    if (filters?.project) {
      models = models.filter((m) => m.project === filters.project);
    }

    if (filters?.tags && filters.tags.length > 0) {
      models = models.filter((m) => filters.tags!.some((tag) => m.tags.includes(tag)));
    }

    return models.sort(
      (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
    );
  }

  /**
   * Delete a model from the registry
   */
  deleteModel(id: string): boolean {
    const existed = this.models.delete(id);

    if (existed) {
      logger.info('Model deleted from registry', { id });
      if (this.config.auto_save) {
        this.save();
      }
    }

    return existed;
  }

  // ==========================================================================
  // Training Lifecycle
  // ==========================================================================

  /**
   * Start training - creates a new model entry with 'training' status
   */
  startTraining(params: {
    name: string;
    description?: string;
    base_model: string;
    training_config: TrainingConfig;
    dataset_sources: string[];
    project?: string;
    tags?: string[];
  }): ModelEntry {
    // Determine version based on existing models with same name
    const existingModels = this.listModels().filter((m) => m.name === params.name);
    const version = `v${existingModels.length + 1}`;

    // Extract param count from model name
    const paramMatch = params.base_model.match(/(\d+)B/i);
    const baseModelParams = paramMatch ? `${paramMatch[1]}B` : 'unknown';

    return this.registerModel({
      name: params.name,
      description: params.description,
      version,
      status: 'training',
      base_model: params.base_model,
      base_model_params: baseModelParams,
      training_config: params.training_config,
      training_started_at: new Date().toISOString(),
      dataset_sources: params.dataset_sources,
      deployments: [],
      tags: params.tags || [],
      project: params.project,
    });
  }

  /**
   * Complete training - update model with metrics and 'trained' status
   */
  completeTraining(
    id: string,
    metrics: TrainingMetrics,
    checkpoint_path: string
  ): ModelEntry | null {
    return this.updateModel(id, {
      status: 'trained',
      training_metrics: metrics,
      training_completed_at: new Date().toISOString(),
      checkpoint_path,
    });
  }

  /**
   * Mark training as failed
   */
  failTraining(id: string, error?: string): ModelEntry | null {
    return this.updateModel(id, {
      status: 'failed',
      training_completed_at: new Date().toISOString(),
      tags: error ? ['failed', `error:${error.substring(0, 50)}`] : ['failed'],
    });
  }

  // ==========================================================================
  // Deployment Operations
  // ==========================================================================

  /**
   * Add a deployment to a model
   */
  addDeployment(id: string, deployment: Omit<DeploymentInfo, 'deployed_at'>): ModelEntry | null {
    const model = this.models.get(id);
    if (!model) return null;

    const newDeployment: DeploymentInfo = {
      ...deployment,
      deployed_at: new Date().toISOString(),
    };

    return this.updateModel(id, {
      status: 'deployed',
      deployments: [...model.deployments, newDeployment],
    });
  }

  /**
   * Update deployment usage stats
   */
  recordInference(id: string, provider: ModelProvider): ModelEntry | null {
    const model = this.models.get(id);
    if (!model) return null;

    const deployments = model.deployments.map((d) => {
      if (d.provider === provider) {
        return {
          ...d,
          last_used: new Date().toISOString(),
          inference_count: (d.inference_count || 0) + 1,
        };
      }
      return d;
    });

    return this.updateModel(id, { deployments });
  }

  // ==========================================================================
  // Analytics
  // ==========================================================================

  /**
   * Get registry statistics
   */
  getStats(): RegistryStats {
    const models = Array.from(this.models.values());

    const by_status: Record<ModelStatus, number> = {
      training: 0,
      trained: 0,
      deployed: 0,
      archived: 0,
      failed: 0,
    };

    const by_base_model: Record<string, number> = {};
    let total_training_hours = 0;
    let total_cost = 0;
    let total_tokens_processed = 0;

    for (const model of models) {
      by_status[model.status]++;

      // Group by base model family
      const baseFamily = model.base_model.split('/').pop()?.split('-')[0] || 'unknown';
      by_base_model[baseFamily] = (by_base_model[baseFamily] || 0) + 1;

      if (model.training_metrics) {
        total_training_hours += model.training_metrics.training_time_hours;
        total_cost += model.training_metrics.cost_usd;
        total_tokens_processed += model.training_metrics.tokens_processed;
      }
    }

    return {
      total_models: models.length,
      by_status,
      by_base_model,
      total_training_hours: Math.round(total_training_hours * 100) / 100,
      total_cost: Math.round(total_cost * 100) / 100,
      total_tokens_processed,
    };
  }

  /**
   * Get training history for a project
   */
  getProjectHistory(project: string): {
    models: ModelEntry[];
    total_cost: number;
    total_hours: number;
    success_rate: number;
  } {
    const models = this.listModels({ project });
    const completed = models.filter((m) => m.status !== 'training');
    const successful = completed.filter((m) => m.status !== 'failed');

    let total_cost = 0;
    let total_hours = 0;

    for (const model of models) {
      if (model.training_metrics) {
        total_cost += model.training_metrics.cost_usd;
        total_hours += model.training_metrics.training_time_hours;
      }
    }

    return {
      models,
      total_cost: Math.round(total_cost * 100) / 100,
      total_hours: Math.round(total_hours * 100) / 100,
      success_rate: completed.length > 0 ? successful.length / completed.length : 0,
    };
  }

  /**
   * Find best performing model for a task
   */
  findBestModel(task: string, benchmark?: string): ModelEntry | null {
    const models = this.listModels({ status: 'deployed' }).filter((m) => m.evaluation_scores);

    let bestModel: ModelEntry | null = null;
    let bestScore = -1;

    for (const model of models) {
      const score = model.evaluation_scores?.find(
        (e) => e.task === task && (!benchmark || e.benchmark === benchmark)
      );

      if (score && score.score > bestScore) {
        bestScore = score.score;
        bestModel = model;
      }
    }

    return bestModel;
  }

  // ==========================================================================
  // Export & Reporting
  // ==========================================================================

  /**
   * Export registry to various formats
   */
  export(format: 'json' | 'csv' | 'markdown' = 'json'): string {
    const models = this.listModels();

    switch (format) {
      case 'csv': {
        const headers = [
          'id',
          'name',
          'version',
          'status',
          'base_model',
          'final_loss',
          'training_hours',
          'cost_usd',
          'created_at',
        ];
        const rows = models.map((m) => [
          m.id,
          m.name,
          m.version,
          m.status,
          m.base_model,
          m.training_metrics?.final_loss || '',
          m.training_metrics?.training_time_hours || '',
          m.training_metrics?.cost_usd || '',
          m.created_at,
        ]);
        return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
      }

      case 'markdown': {
        let md = '# Model Registry\n\n';
        const stats = this.getStats();

        md += '## Summary\n\n';
        md += `- **Total Models**: ${stats.total_models}\n`;
        md += `- **Total Training Hours**: ${stats.total_training_hours}\n`;
        md += `- **Total Cost**: $${stats.total_cost}\n\n`;

        md += '## Models\n\n';
        md += '| Name | Version | Status | Base Model | Loss | Cost |\n';
        md += '|------|---------|--------|------------|------|------|\n';

        for (const m of models.slice(0, 50)) {
          md += `| ${m.name} | ${m.version} | ${m.status} | ${m.base_model} | `;
          md += `${m.training_metrics?.final_loss?.toFixed(4) || '-'} | `;
          md += `$${m.training_metrics?.cost_usd?.toFixed(2) || '-'} |\n`;
        }

        return md;
      }

      default:
        return JSON.stringify(models, null, 2);
    }
  }

  // ==========================================================================
  // Persistence
  // ==========================================================================

  /**
   * Save registry to disk
   */
  save(): void {
    try {
      // Ensure directory exists
      const dir = path.dirname(this.registryFile);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      // Backup existing file
      if (this.config.backup_enabled && fs.existsSync(this.registryFile)) {
        const backupPath = `${this.registryFile}.backup.${Date.now()}`;
        fs.copyFileSync(this.registryFile, backupPath);
        this.cleanupBackups();
      }

      // Write new file
      const data = {
        version: '1.0',
        updated_at: new Date().toISOString(),
        models: Array.from(this.models.values()),
      };

      fs.writeFileSync(this.registryFile, JSON.stringify(data, null, 2), 'utf-8');
      logger.debug('Model registry saved', { path: this.registryFile, count: this.models.size });
    } catch (error) {
      logger.error('Failed to save model registry', { error });
    }
  }

  /**
   * Load registry from disk
   */
  private load(): void {
    try {
      if (!fs.existsSync(this.registryFile)) {
        logger.info('Model registry not found, starting fresh', { path: this.registryFile });
        return;
      }

      const content = fs.readFileSync(this.registryFile, 'utf-8');
      const data = JSON.parse(content);

      this.models.clear();
      for (const model of data.models || []) {
        this.models.set(model.id, model);
      }

      logger.info('Model registry loaded', { count: this.models.size });
    } catch (error) {
      logger.error('Failed to load model registry', { error });
    }
  }

  /**
   * Clean up old backups
   */
  private cleanupBackups(): void {
    try {
      const dir = path.dirname(this.registryFile);
      const basename = path.basename(this.registryFile);
      const backups = fs
        .readdirSync(dir)
        .filter((f) => f.startsWith(`${basename}.backup.`))
        .sort()
        .reverse();

      // Remove old backups beyond max
      for (const backup of backups.slice(this.config.max_backups)) {
        fs.unlinkSync(path.join(dir, backup));
      }
    } catch (error) {
      logger.warn('Failed to cleanup backups', { error });
    }
  }

  /**
   * Generate unique ID
   */
  private generateId(): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 8);
    return `model_${timestamp}_${random}`;
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let registryInstance: ModelRegistry | null = null;

export function getModelRegistry(config?: Partial<ModelRegistryConfig>): ModelRegistry {
  if (!registryInstance) {
    registryInstance = new ModelRegistry(config);
  }
  return registryInstance;
}

export default ModelRegistry;
