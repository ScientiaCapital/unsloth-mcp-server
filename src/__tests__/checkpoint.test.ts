/**
 * Checkpoint Manager Tests
 */

import { describe, test, expect, beforeAll, afterAll } from '@jest/globals';
import * as fs from 'fs';
import { CheckpointManager } from '../utils/checkpoint.js';

describe('CheckpointManager', () => {
  // Use a unique directory for the entire test suite
  const testDir = `/tmp/test_checkpoints_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  let manager: CheckpointManager;

  beforeAll(() => {
    manager = new CheckpointManager({
      checkpoint_dir: testDir,
      max_checkpoints: 10,
      auto_cleanup: false,
    });
  });

  afterAll(() => {
    // Cleanup test directory
    try {
      if (fs.existsSync(testDir)) {
        fs.rmSync(testDir, { recursive: true, force: true });
      }
    } catch {
      // Ignore cleanup errors
    }
  });

  describe('Checkpoint Directory', () => {
    test('should create checkpoint directory on init', () => {
      expect(fs.existsSync(testDir)).toBe(true);
    });
  });

  describe('saveCheckpoint', () => {
    test('should save a checkpoint and return success', async () => {
      const result = await manager.saveCheckpoint(
        'test_job_1',
        {
          current_step: 100,
          total_steps: 1000,
          current_epoch: 1,
          total_epochs: 3,
          training_loss: 0.5,
        },
        {
          base_model: 'unsloth/Llama-3.2-1B',
          lora_r: 16,
          lora_alpha: 32,
          max_seq_length: 2048,
          dataset_path: '/data/train.json',
          dataset_size: 1000,
        },
        testDir, // Use testDir as model dir since it exists
        'pod123'
      );

      expect(result.success).toBe(true);
      expect(result.checkpoint_id).toContain('ckpt_test_job_1');
      expect(result.version).toBeGreaterThanOrEqual(1);
      expect(result.path).toContain(testDir);
    });

    test('should generate checkpoint ID with step number', async () => {
      const result = await manager.saveCheckpoint(
        'test_job_2',
        { current_step: 250, total_steps: 1000, current_epoch: 1, total_epochs: 3 },
        {
          base_model: 'model',
          lora_r: 16,
          lora_alpha: 32,
          max_seq_length: 2048,
          dataset_path: '/data',
          dataset_size: 100,
        },
        testDir
      );

      expect(result.success).toBe(true);
      expect(result.checkpoint_id).toContain('step250');
    });
  });

  describe('loadCheckpoint', () => {
    test('should return null for non-existent checkpoint', async () => {
      const loaded = await manager.loadCheckpoint('nonexistent_checkpoint_id');
      expect(loaded).toBeNull();
    });

    test('should load a previously saved checkpoint', async () => {
      // First save a checkpoint
      const saveResult = await manager.saveCheckpoint(
        'load_test_job',
        {
          current_step: 150,
          total_steps: 500,
          current_epoch: 2,
          total_epochs: 5,
          training_loss: 0.3,
        },
        {
          base_model: 'test/model',
          lora_r: 8,
          lora_alpha: 16,
          max_seq_length: 1024,
          dataset_path: '/test/data',
          dataset_size: 500,
        },
        testDir
      );

      expect(saveResult.success).toBe(true);

      // Then load it
      const loaded = await manager.loadCheckpoint(saveResult.checkpoint_id);

      expect(loaded).not.toBeNull();
      expect(loaded?.training_job_id).toBe('load_test_job');
      expect(loaded?.current_step).toBe(150);
      expect(loaded?.training_loss).toBe(0.3);
      expect(loaded?.base_model).toBe('test/model');
    });
  });

  describe('listCheckpoints', () => {
    test('should return empty array for non-existent job', async () => {
      const checkpoints = await manager.listCheckpoints('completely_nonexistent_job_xyz');
      expect(checkpoints).toEqual([]);
    });

    test('should list saved checkpoints', async () => {
      // Save a checkpoint with a unique job ID for this test
      const jobId = `list_test_${Date.now()}`;
      const saveResult = await manager.saveCheckpoint(
        jobId,
        { current_step: 100, total_steps: 1000, current_epoch: 1, total_epochs: 3 },
        {
          base_model: 'model',
          lora_r: 16,
          lora_alpha: 32,
          max_seq_length: 2048,
          dataset_path: '/data',
          dataset_size: 100,
        },
        testDir
      );

      expect(saveResult.success).toBe(true);

      const checkpoints = await manager.listCheckpoints(jobId);
      expect(checkpoints.length).toBeGreaterThanOrEqual(1);
      expect(checkpoints[0].training_job_id).toBe(jobId);
    });
  });

  describe('getLatestCheckpoint', () => {
    test('should return null when no checkpoints exist for job', async () => {
      const latest = await manager.getLatestCheckpoint('nonexistent_job_for_latest');
      expect(latest).toBeNull();
    });
  });

  describe('deleteCheckpoint', () => {
    test('should return false for non-existent checkpoint', async () => {
      const deleted = await manager.deleteCheckpoint('nonexistent_checkpoint_to_delete');
      expect(deleted).toBe(false);
    });

    test('should delete an existing checkpoint', async () => {
      // First save
      const saveResult = await manager.saveCheckpoint(
        'delete_test_job',
        { current_step: 100, total_steps: 1000, current_epoch: 1, total_epochs: 3 },
        {
          base_model: 'model',
          lora_r: 16,
          lora_alpha: 32,
          max_seq_length: 2048,
          dataset_path: '/data',
          dataset_size: 100,
        },
        testDir
      );

      expect(saveResult.success).toBe(true);

      // Then delete
      const deleted = await manager.deleteCheckpoint(saveResult.checkpoint_id);
      expect(deleted).toBe(true);

      // Verify it's gone
      const loaded = await manager.loadCheckpoint(saveResult.checkpoint_id);
      expect(loaded).toBeNull();
    });
  });

  describe('cleanupOldCheckpoints', () => {
    test('should return 0 when no checkpoints to clean', async () => {
      const deleted = await manager.cleanupOldCheckpoints('job_with_no_checkpoints');
      expect(deleted).toBe(0);
    });
  });

  describe('generateResumeCommand', () => {
    test('should generate a python resume command', async () => {
      const saveResult = await manager.saveCheckpoint(
        'resume_test_job',
        { current_step: 100, total_steps: 1000, current_epoch: 1, total_epochs: 3 },
        {
          base_model: 'model',
          lora_r: 16,
          lora_alpha: 32,
          max_seq_length: 2048,
          dataset_path: '/data',
          dataset_size: 100,
        },
        testDir
      );

      expect(saveResult.success).toBe(true);

      const command = manager.generateResumeCommand(saveResult.checkpoint_id);

      expect(command).toContain('python3');
      expect(command).toContain('--resume_from_checkpoint');
      expect(command).toContain(saveResult.checkpoint_id);
    });
  });

  describe('getStats', () => {
    test('should return stats object with correct structure', async () => {
      const stats = await manager.getStats();

      expect(stats).toHaveProperty('total_checkpoints');
      expect(stats).toHaveProperty('total_size_mb');
      expect(stats).toHaveProperty('checkpoints_by_job');
      expect(typeof stats.total_checkpoints).toBe('number');
      expect(typeof stats.total_size_mb).toBe('number');
    });
  });
});
