import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { RunPodClient } from '../utils/runpod.js';

// Mock fetch globally
const mockFetch = jest.fn() as jest.MockedFunction<typeof fetch>;
global.fetch = mockFetch;

describe('RunPod Client', () => {
  let client: RunPodClient;

  beforeEach(() => {
    // Reset mocks
    mockFetch.mockReset();
    // Create client with test credentials
    client = new RunPodClient({
      apiKey: 'test-api-key',
      apiEndpoint: 'https://api.runpod.io/graphql',
    });
  });

  describe('Pod Management', () => {
    it('should list pods successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            myself: {
              pods: [
                {
                  id: 'pod-1',
                  name: 'test-pod',
                  desiredStatus: 'RUNNING',
                  runtime: {
                    uptimeInSeconds: 3600,
                    gpus: [{ id: 'gpu-1', gpuUtilPercent: 50, memoryUtilPercent: 60 }],
                    ports: [],
                  },
                  machine: { gpuTypeId: 'NVIDIA RTX A4000', gpuCount: 1 },
                  imageName: 'runpod/pytorch:2.2.0',
                  volumeInGb: 30,
                  containerDiskInGb: 20,
                  costPerHr: 0.25,
                  vcpuCount: 8,
                  memoryInGb: 32,
                  volumeMountPath: '/runpod-volume',
                },
              ],
            },
          },
        }),
      } as Response);

      const pods = await client.listPods();
      expect(pods).toHaveLength(1);
      expect(pods[0].id).toBe('pod-1');
      expect(pods[0].name).toBe('test-pod');
      expect(pods[0].desiredStatus).toBe('RUNNING');
    });

    it('should get a specific pod', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            pod: {
              id: 'pod-1',
              name: 'test-pod',
              desiredStatus: 'RUNNING',
              runtime: {
                uptimeInSeconds: 7200,
                gpus: [{ id: 'gpu-1', gpuUtilPercent: 75, memoryUtilPercent: 80 }],
                ports: [],
              },
              machine: { gpuTypeId: 'NVIDIA RTX A4000', gpuCount: 1 },
              imageName: 'runpod/pytorch:2.2.0',
              volumeInGb: 30,
              containerDiskInGb: 20,
              costPerHr: 0.25,
              vcpuCount: 8,
              memoryInGb: 32,
              volumeMountPath: '/runpod-volume',
            },
          },
        }),
      } as Response);

      const pod = await client.getPod('pod-1');
      expect(pod).not.toBeNull();
      expect(pod?.id).toBe('pod-1');
      expect(pod?.runtime?.uptimeInSeconds).toBe(7200);
    });

    it('should return null for non-existent pod', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Not found'));

      const pod = await client.getPod('non-existent');
      expect(pod).toBeNull();
    });

    it('should start a pod', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            podResume: {
              id: 'pod-1',
              name: 'test-pod',
              desiredStatus: 'RUNNING',
              costPerHr: 0.25,
            },
          },
        }),
      } as Response);

      const pod = await client.startPod('pod-1');
      expect(pod.id).toBe('pod-1');
      expect(pod.desiredStatus).toBe('RUNNING');
    });

    it('should stop a pod', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            podStop: {
              id: 'pod-1',
              name: 'test-pod',
              desiredStatus: 'STOPPED',
            },
          },
        }),
      } as Response);

      const pod = await client.stopPod('pod-1');
      expect(pod.id).toBe('pod-1');
      expect(pod.desiredStatus).toBe('STOPPED');
    });

    it('should terminate a pod', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            podTerminate: null,
          },
        }),
      } as Response);

      await expect(client.terminatePod('pod-1')).resolves.not.toThrow();
    });
  });

  describe('GPU Availability', () => {
    it('should check GPU availability', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            gpuTypes: [
              {
                id: 'NVIDIA RTX A4000',
                displayName: 'RTX A4000',
                memoryInGb: 16,
                secureCloud: true,
                communityCloud: true,
              },
            ],
          },
        }),
      } as Response);

      const result = await client.checkGpuAvailability('NVIDIA RTX A4000');
      expect(result.available).toBe(true);
      expect(result.secureCloud).toBe(true);
      expect(result.communityCloud).toBe(true);
      expect(result.memoryInGb).toBe(16);
    });

    it('should return unavailable for non-existent GPU', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            gpuTypes: [],
          },
        }),
      } as Response);

      const result = await client.checkGpuAvailability('NONEXISTENT-GPU');
      expect(result.available).toBe(false);
      expect(result.memoryInGb).toBe(0);
    });

    it('should find best available GPU for VRAM requirement', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            gpuTypes: [
              {
                id: 'NVIDIA RTX A4000',
                displayName: 'RTX A4000',
                memoryInGb: 16,
                secureCloud: true,
                communityCloud: true,
              },
              {
                id: 'NVIDIA RTX A5000',
                displayName: 'RTX A5000',
                memoryInGb: 24,
                secureCloud: true,
                communityCloud: true,
              },
              {
                id: 'NVIDIA RTX 4090',
                displayName: 'RTX 4090',
                memoryInGb: 24,
                secureCloud: true,
                communityCloud: true,
              },
            ],
          },
        }),
      } as Response);

      const gpu = await client.findBestAvailableGpu(20);
      expect(gpu).not.toBeNull();
      expect(gpu?.memoryInGb).toBeGreaterThanOrEqual(20);
      // Should pick smallest sufficient VRAM (24GB in this case)
      expect(gpu?.memoryInGb).toBe(24);
    });

    it('should return null if no GPU meets VRAM requirement', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: {
            gpuTypes: [
              {
                id: 'NVIDIA RTX A4000',
                displayName: 'RTX A4000',
                memoryInGb: 16,
                secureCloud: false,
                communityCloud: false, // Not available
              },
            ],
          },
        }),
      } as Response);

      const gpu = await client.findBestAvailableGpu(80);
      expect(gpu).toBeNull();
    });
  });

  describe('Cost Estimation', () => {
    it('should estimate training cost for small 1B model', () => {
      const estimate = client.estimateTrainingCost(
        500000, // 500k tokens (larger to avoid rounding to zero)
        'unsloth/Llama-3.2-1B-bnb-4bit',
        0.25, // $0.25/hr
        3
      );

      expect(estimate.estimatedHours).toBeGreaterThan(0);
      expect(estimate.estimatedCost).toBeGreaterThan(0);
      expect(estimate.tokensPerSecond).toBe(8000); // 1B model is fast
    });

    it('should estimate training cost for larger 8B model', () => {
      const estimate = client.estimateTrainingCost(
        100000, // 100k tokens
        'unsloth/Llama-3.2-8B-bnb-4bit',
        0.44, // $0.44/hr
        3
      );

      expect(estimate.estimatedHours).toBeGreaterThan(0);
      expect(estimate.tokensPerSecond).toBe(2500); // 8B model is slower
    });

    it('should estimate higher cost for larger datasets', () => {
      const smallEstimate = client.estimateTrainingCost(50000, 'model', 0.25, 3);
      const largeEstimate = client.estimateTrainingCost(500000, 'model', 0.25, 3);

      expect(largeEstimate.estimatedHours).toBeGreaterThan(smallEstimate.estimatedHours);
      expect(largeEstimate.estimatedCost).toBeGreaterThan(smallEstimate.estimatedCost);
    });

    it('should scale with epochs', () => {
      const oneEpoch = client.estimateTrainingCost(50000, 'model', 0.25, 1);
      const threeEpochs = client.estimateTrainingCost(50000, 'model', 0.25, 3);

      expect(threeEpochs.estimatedHours).toBeCloseTo(oneEpoch.estimatedHours * 3, 1);
    });

    it('should estimate 70B models as slowest', () => {
      const estimate = client.estimateTrainingCost(100000, 'model-70B', 0.8, 1);
      expect(estimate.tokensPerSecond).toBe(400);
    });
  });

  describe('Training Script Generation', () => {
    it('should generate valid Unsloth training script', () => {
      const script = client.generateTrainingScript({
        baseModel: 'unsloth/Llama-3.2-1B-bnb-4bit',
        datasetPath: '/workspace/data/train.jsonl',
        outputDir: '/workspace/output',
        epochs: 3,
        batchSize: 4,
        learningRate: 2e-4,
        loraR: 16,
        loraAlpha: 32,
      });

      expect(script).toContain('from unsloth import FastLanguageModel');
      expect(script).toContain('unsloth/Llama-3.2-1B-bnb-4bit');
      expect(script).toContain('EPOCHS = 3');
      expect(script).toContain('BATCH_SIZE = 4');
      expect(script).toContain('LEARNING_RATE = 0.0002');
      expect(script).toContain('LORA_R = 16');
      expect(script).toContain('/workspace/data/train.jsonl');
      expect(script).toContain('/workspace/output');
    });

    it('should use default values for optional parameters', () => {
      const script = client.generateTrainingScript({
        baseModel: 'unsloth/Mistral-7B-bnb-4bit',
        datasetPath: '/workspace/data.jsonl',
        outputDir: '/workspace/out',
      });

      expect(script).toContain('EPOCHS = 3'); // default
      expect(script).toContain('BATCH_SIZE = 4'); // default
      expect(script).toContain('LEARNING_RATE = 0.0002'); // default
      expect(script).toContain('LORA_R = 16'); // default
      expect(script).toContain('LORA_ALPHA = 32'); // default
    });
  });

  describe('Error Handling', () => {
    it('should throw on GraphQL error response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          errors: [{ message: 'API Error' }],
        }),
      } as Response);

      await expect(client.listPods()).rejects.toThrow('RunPod GraphQL error: API Error');
    });

    it('should throw on network error', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(client.listPods()).rejects.toThrow('Network error');
    });

    it('should throw on HTTP error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
      } as Response);

      await expect(client.listPods()).rejects.toThrow('RunPod API error: 401 Unauthorized');
    });
  });

  describe('Singleton Pattern', () => {
    it('should throw when RUNPOD_API_KEY is not set', async () => {
      // Import fresh to test singleton
      const { getRunPodClient, resetRunPodClient } = await import('../utils/runpod.js');

      // Reset to clear any existing instance
      resetRunPodClient();

      // Clear env var
      const originalKey = process.env.RUNPOD_API_KEY;
      delete process.env.RUNPOD_API_KEY;

      try {
        expect(() => getRunPodClient()).toThrow('RUNPOD_API_KEY environment variable is not set');
      } finally {
        // Restore
        if (originalKey) process.env.RUNPOD_API_KEY = originalKey;
        resetRunPodClient();
      }
    });
  });
});
