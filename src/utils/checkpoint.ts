/**
 * Checkpoint Management System
 *
 * Provides save/load/resume capabilities for training jobs.
 * Supports versioning, metadata tracking, and cloud storage integration.
 */

import * as fs from 'fs';
import * as path from 'path';
import { logger } from './logger.js';

// ============================================================================
// Types
// ============================================================================

export interface CheckpointMetadata {
  id: string;
  version: number;
  created_at: string;
  training_job_id: string;
  pod_id?: string;

  // Training State
  current_step: number;
  total_steps: number;
  current_epoch: number;
  total_epochs: number;

  // Metrics
  training_loss?: number;
  eval_loss?: number;
  learning_rate?: number;

  // Model Info
  base_model: string;
  lora_r: number;
  lora_alpha: number;
  max_seq_length: number;

  // Dataset Info
  dataset_path: string;
  dataset_size: number;

  // Storage
  checkpoint_path: string;
  checkpoint_size_mb: number;

  // Resume Info
  is_resumable: boolean;
  resume_command?: string;
}

export interface CheckpointConfig {
  checkpoint_dir: string;
  max_checkpoints: number;
  save_interval_steps: number;
  save_interval_minutes: number;
  auto_cleanup: boolean;
  cloud_backup?: {
    enabled: boolean;
    provider: 'runpod' | 's3' | 'gcs' | 'local';
    bucket?: string;
    prefix?: string;
  };
}

export interface CheckpointSaveResult {
  success: boolean;
  checkpoint_id: string;
  version: number;
  path: string;
  size_mb: number;
  message: string;
}

export interface CheckpointListItem {
  id: string;
  version: number;
  created_at: string;
  training_job_id: string;
  current_step: number;
  total_steps: number;
  progress_percent: number;
  training_loss?: number;
  size_mb: number;
  is_resumable: boolean;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: CheckpointConfig = {
  checkpoint_dir: '/runpod-volume/checkpoints',
  max_checkpoints: 5,
  save_interval_steps: 100,
  save_interval_minutes: 15,
  auto_cleanup: true,
  cloud_backup: {
    enabled: false,
    provider: 'local',
  },
};

// ============================================================================
// Checkpoint Manager
// ============================================================================

export class CheckpointManager {
  private config: CheckpointConfig;
  private metadataCache: Map<string, CheckpointMetadata> = new Map();

  constructor(config: Partial<CheckpointConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.ensureCheckpointDir();
  }

  /**
   * Ensure checkpoint directory exists
   */
  private ensureCheckpointDir(): void {
    if (!fs.existsSync(this.config.checkpoint_dir)) {
      fs.mkdirSync(this.config.checkpoint_dir, { recursive: true });
      logger.info('Created checkpoint directory', { path: this.config.checkpoint_dir });
    }
  }

  /**
   * Generate checkpoint ID
   */
  private generateCheckpointId(jobId: string, step: number): string {
    const timestamp = Date.now();
    return `ckpt_${jobId}_step${step}_${timestamp}`;
  }

  /**
   * Get checkpoint path
   */
  private getCheckpointPath(checkpointId: string): string {
    return path.join(this.config.checkpoint_dir, checkpointId);
  }

  /**
   * Get metadata file path
   */
  private getMetadataPath(checkpointId: string): string {
    return path.join(this.getCheckpointPath(checkpointId), 'metadata.json');
  }

  /**
   * Calculate directory size in MB
   */
  private getDirSizeMB(dirPath: string): number {
    if (!fs.existsSync(dirPath)) return 0;

    let totalSize = 0;
    const files = fs.readdirSync(dirPath);

    for (const file of files) {
      const filePath = path.join(dirPath, file);
      const stat = fs.statSync(filePath);

      if (stat.isDirectory()) {
        totalSize += this.getDirSizeMB(filePath);
      } else {
        totalSize += stat.size;
      }
    }

    return Math.round((totalSize / (1024 * 1024)) * 100) / 100;
  }

  /**
   * Save a checkpoint
   */
  async saveCheckpoint(
    jobId: string,
    state: {
      current_step: number;
      total_steps: number;
      current_epoch: number;
      total_epochs: number;
      training_loss?: number;
      eval_loss?: number;
      learning_rate?: number;
    },
    modelInfo: {
      base_model: string;
      lora_r: number;
      lora_alpha: number;
      max_seq_length: number;
      dataset_path: string;
      dataset_size: number;
    },
    modelDir: string,
    podId?: string
  ): Promise<CheckpointSaveResult> {
    const checkpointId = this.generateCheckpointId(jobId, state.current_step);
    const checkpointPath = this.getCheckpointPath(checkpointId);

    try {
      // Create checkpoint directory
      fs.mkdirSync(checkpointPath, { recursive: true });

      // Copy model files to checkpoint (skip if modelDir is same as checkpoint dir to avoid recursion)
      const resolvedModelDir = path.resolve(modelDir);
      const resolvedCheckpointDir = path.resolve(this.config.checkpoint_dir);
      if (fs.existsSync(modelDir) && !resolvedModelDir.startsWith(resolvedCheckpointDir)) {
        this.copyDirectory(modelDir, path.join(checkpointPath, 'model'));
      }

      // Get existing checkpoints for this job to determine version
      const existingCheckpoints = await this.listCheckpoints(jobId);
      const version = existingCheckpoints.length + 1;

      // Create metadata
      const metadata: CheckpointMetadata = {
        id: checkpointId,
        version,
        created_at: new Date().toISOString(),
        training_job_id: jobId,
        pod_id: podId,
        ...state,
        ...modelInfo,
        checkpoint_path: checkpointPath,
        checkpoint_size_mb: this.getDirSizeMB(checkpointPath),
        is_resumable: true,
        resume_command: this.generateResumeCommand(checkpointId),
      };

      // Save metadata
      fs.writeFileSync(this.getMetadataPath(checkpointId), JSON.stringify(metadata, null, 2));
      this.metadataCache.set(checkpointId, metadata);

      logger.info('Checkpoint saved', {
        checkpointId,
        version,
        step: state.current_step,
        sizeMB: metadata.checkpoint_size_mb,
      });

      // Cleanup old checkpoints if needed
      if (this.config.auto_cleanup) {
        await this.cleanupOldCheckpoints(jobId);
      }

      return {
        success: true,
        checkpoint_id: checkpointId,
        version,
        path: checkpointPath,
        size_mb: metadata.checkpoint_size_mb,
        message: `Checkpoint saved at step ${state.current_step}`,
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error('Failed to save checkpoint', { checkpointId, error: errorMessage });

      return {
        success: false,
        checkpoint_id: checkpointId,
        version: 0,
        path: '',
        size_mb: 0,
        message: `Failed to save checkpoint: ${errorMessage}`,
      };
    }
  }

  /**
   * Load checkpoint metadata
   */
  async loadCheckpoint(checkpointId: string): Promise<CheckpointMetadata | null> {
    // Check cache first
    if (this.metadataCache.has(checkpointId)) {
      return this.metadataCache.get(checkpointId)!;
    }

    const metadataPath = this.getMetadataPath(checkpointId);

    if (!fs.existsSync(metadataPath)) {
      logger.warn('Checkpoint not found', { checkpointId });
      return null;
    }

    try {
      const content = fs.readFileSync(metadataPath, 'utf-8');
      const metadata = JSON.parse(content) as CheckpointMetadata;
      this.metadataCache.set(checkpointId, metadata);
      return metadata;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error('Failed to load checkpoint', { checkpointId, error: errorMessage });
      return null;
    }
  }

  /**
   * Get the latest checkpoint for a job
   */
  async getLatestCheckpoint(jobId: string): Promise<CheckpointMetadata | null> {
    const checkpoints = await this.listCheckpoints(jobId);

    if (checkpoints.length === 0) {
      return null;
    }

    // Sort by version descending and get the latest
    checkpoints.sort((a, b) => b.version - a.version);
    return this.loadCheckpoint(checkpoints[0].id);
  }

  /**
   * List all checkpoints for a training job
   */
  async listCheckpoints(jobId?: string): Promise<CheckpointListItem[]> {
    const checkpoints: CheckpointListItem[] = [];

    if (!fs.existsSync(this.config.checkpoint_dir)) {
      return checkpoints;
    }

    const dirs = fs.readdirSync(this.config.checkpoint_dir);

    for (const dir of dirs) {
      if (!dir.startsWith('ckpt_')) continue;

      // Filter by job ID if provided
      // Format: ckpt_${jobId}_step${step}_${timestamp}
      if (jobId) {
        const expectedPrefix = `ckpt_${jobId}_step`;
        if (!dir.startsWith(expectedPrefix)) continue;
      }

      const metadata = await this.loadCheckpoint(dir);
      if (!metadata) continue;

      checkpoints.push({
        id: metadata.id,
        version: metadata.version,
        created_at: metadata.created_at,
        training_job_id: metadata.training_job_id,
        current_step: metadata.current_step,
        total_steps: metadata.total_steps,
        progress_percent: Math.round((metadata.current_step / metadata.total_steps) * 100),
        training_loss: metadata.training_loss,
        size_mb: metadata.checkpoint_size_mb,
        is_resumable: metadata.is_resumable,
      });
    }

    // Sort by creation time descending
    checkpoints.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

    return checkpoints;
  }

  /**
   * Delete a checkpoint
   */
  async deleteCheckpoint(checkpointId: string): Promise<boolean> {
    const checkpointPath = this.getCheckpointPath(checkpointId);

    if (!fs.existsSync(checkpointPath)) {
      logger.warn('Checkpoint not found for deletion', { checkpointId });
      return false;
    }

    try {
      fs.rmSync(checkpointPath, { recursive: true, force: true });
      this.metadataCache.delete(checkpointId);
      logger.info('Checkpoint deleted', { checkpointId });
      return true;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error('Failed to delete checkpoint', { checkpointId, error: errorMessage });
      return false;
    }
  }

  /**
   * Cleanup old checkpoints, keeping only the most recent N
   */
  async cleanupOldCheckpoints(jobId: string): Promise<number> {
    const checkpoints = await this.listCheckpoints(jobId);

    if (checkpoints.length <= this.config.max_checkpoints) {
      return 0;
    }

    // Sort by version descending
    checkpoints.sort((a, b) => b.version - a.version);

    // Delete oldest checkpoints
    const toDelete = checkpoints.slice(this.config.max_checkpoints);
    let deleted = 0;

    for (const checkpoint of toDelete) {
      if (await this.deleteCheckpoint(checkpoint.id)) {
        deleted++;
      }
    }

    logger.info('Cleaned up old checkpoints', {
      jobId,
      deleted,
      remaining: this.config.max_checkpoints,
    });
    return deleted;
  }

  /**
   * Generate resume command for a checkpoint
   */
  generateResumeCommand(checkpointId: string): string {
    const checkpointPath = this.getCheckpointPath(checkpointId);
    return `python3 train.py --resume_from_checkpoint "${path.join(checkpointPath, 'model')}"`;
  }

  /**
   * Generate resume script content
   */
  generateResumeScript(metadata: CheckpointMetadata): string {
    return `#!/usr/bin/env python3
"""
Resume Training Script
Checkpoint: ${metadata.id}
Created: ${metadata.created_at}
"""

import os
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# Resume from checkpoint
CHECKPOINT_PATH = "${metadata.checkpoint_path}/model"
OUTPUT_DIR = "${path.dirname(metadata.checkpoint_path)}/resumed_${Date.now()}"
DATASET_PATH = "${metadata.dataset_path}"

# Load model from checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHECKPOINT_PATH,
    max_seq_length=${metadata.max_seq_length},
    load_in_4bit=True,
)

# Load dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Training arguments - resume from step ${metadata.current_step}
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=${metadata.total_epochs},
    learning_rate=${metadata.learning_rate || 2e-4},
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="adamw_8bit",
    resume_from_checkpoint=True,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=${metadata.max_seq_length},
    args=training_args,
)

# Resume training
print(f"Resuming from step ${metadata.current_step}...")
trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)

# Save final model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete! Model saved to {OUTPUT_DIR}")
`;
  }

  /**
   * Copy directory recursively
   */
  private copyDirectory(src: string, dest: string): void {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }

    const entries = fs.readdirSync(src, { withFileTypes: true });

    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);

      if (entry.isDirectory()) {
        this.copyDirectory(srcPath, destPath);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }

  /**
   * Get checkpoint statistics
   */
  async getStats(): Promise<{
    total_checkpoints: number;
    total_size_mb: number;
    checkpoints_by_job: Record<string, number>;
    oldest_checkpoint?: string;
    newest_checkpoint?: string;
  }> {
    const allCheckpoints = await this.listCheckpoints();

    const stats = {
      total_checkpoints: allCheckpoints.length,
      total_size_mb: 0,
      checkpoints_by_job: {} as Record<string, number>,
      oldest_checkpoint: undefined as string | undefined,
      newest_checkpoint: undefined as string | undefined,
    };

    if (allCheckpoints.length === 0) {
      return stats;
    }

    for (const checkpoint of allCheckpoints) {
      stats.total_size_mb += checkpoint.size_mb;

      if (!stats.checkpoints_by_job[checkpoint.training_job_id]) {
        stats.checkpoints_by_job[checkpoint.training_job_id] = 0;
      }
      stats.checkpoints_by_job[checkpoint.training_job_id]++;
    }

    // Sort by date
    allCheckpoints.sort(
      (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );

    stats.oldest_checkpoint = allCheckpoints[0].created_at;
    stats.newest_checkpoint = allCheckpoints[allCheckpoints.length - 1].created_at;
    stats.total_size_mb = Math.round(stats.total_size_mb * 100) / 100;

    return stats;
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let checkpointManager: CheckpointManager | null = null;

export function getCheckpointManager(config?: Partial<CheckpointConfig>): CheckpointManager {
  if (!checkpointManager) {
    checkpointManager = new CheckpointManager(config);
  }
  return checkpointManager;
}

export function resetCheckpointManager(): void {
  checkpointManager = null;
}
