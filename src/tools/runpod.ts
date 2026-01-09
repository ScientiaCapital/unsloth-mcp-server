/**
 * runpod.ts - RunPod GPU cloud tools (11 tools)
 *
 * Manage cloud GPU pods for training when local resources aren't enough.
 *
 * TOOLS:
 *   1. runpod_list_pods - List all pods
 *   2. runpod_get_pod - Get pod details
 *   3. runpod_check_gpus - Check GPU availability
 *   4. runpod_create_pod - Create new pod
 *   5. runpod_start_pod - Start stopped pod
 *   6. runpod_stop_pod - Stop running pod
 *   7. runpod_terminate_pod - Terminate pod
 *   8. runpod_start_training - Start fine-tuning job
 *   9. runpod_get_training_status - Check training progress
 *   10. runpod_get_training_logs - Get training logs
 *   11. runpod_estimate_cost - Estimate training cost
 */

import { ToolDefinition, ToolModule, ToolHandler } from './types.js';

/**
 * Tool definitions
 */
export const RUNPOD_TOOLS: ToolDefinition[] = [
  {
    name: 'runpod_list_pods',
    description: 'List all RunPod pods in your account.',
    inputSchema: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'runpod_get_pod',
    description: 'Get detailed information about a specific pod.',
    inputSchema: {
      type: 'object',
      properties: {
        pod_id: {
          type: 'string',
          description: 'The pod ID',
        },
      },
      required: ['pod_id'],
    },
  },
  {
    name: 'runpod_check_gpus',
    description: 'Check available GPU types and pricing.',
    inputSchema: {
      type: 'object',
      properties: {
        gpu_type: {
          type: 'string',
          description: 'Filter by GPU type (e.g., "RTX 4090", "A100")',
        },
      },
      required: [],
    },
  },
  {
    name: 'runpod_create_pod',
    description: 'Create a new RunPod pod for training.',
    inputSchema: {
      type: 'object',
      properties: {
        name: {
          type: 'string',
          description: 'Pod name',
        },
        gpu_type: {
          type: 'string',
          description: 'GPU type (e.g., "NVIDIA RTX 4090")',
        },
        image_name: {
          type: 'string',
          description: 'Docker image (default: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel)',
        },
        gpu_count: {
          type: 'number',
          description: 'Number of GPUs (default: 1)',
          default: 1,
        },
        volume_size: {
          type: 'number',
          description: 'Volume size in GB (default: 50)',
          default: 50,
        },
      },
      required: ['name', 'gpu_type'],
    },
  },
  {
    name: 'runpod_start_pod',
    description: 'Start a stopped pod.',
    inputSchema: {
      type: 'object',
      properties: {
        pod_id: {
          type: 'string',
          description: 'The pod ID to start',
        },
      },
      required: ['pod_id'],
    },
  },
  {
    name: 'runpod_stop_pod',
    description: 'Stop a running pod (preserves data).',
    inputSchema: {
      type: 'object',
      properties: {
        pod_id: {
          type: 'string',
          description: 'The pod ID to stop',
        },
      },
      required: ['pod_id'],
    },
  },
  {
    name: 'runpod_terminate_pod',
    description: 'Terminate a pod (WARNING: deletes all data).',
    inputSchema: {
      type: 'object',
      properties: {
        pod_id: {
          type: 'string',
          description: 'The pod ID to terminate',
        },
      },
      required: ['pod_id'],
    },
  },
  {
    name: 'runpod_start_training',
    description: 'Start a fine-tuning job on a RunPod pod.',
    inputSchema: {
      type: 'object',
      properties: {
        pod_id: {
          type: 'string',
          description: 'The pod ID to use',
        },
        model_name: {
          type: 'string',
          description: 'Model to fine-tune',
        },
        dataset_path: {
          type: 'string',
          description: 'Path to training dataset',
        },
        output_dir: {
          type: 'string',
          description: 'Output directory',
          default: '/workspace/output',
        },
        max_steps: {
          type: 'number',
          description: 'Training steps',
          default: 200,
        },
      },
      required: ['pod_id', 'model_name', 'dataset_path'],
    },
  },
  {
    name: 'runpod_get_training_status',
    description: 'Check the status of a training job.',
    inputSchema: {
      type: 'object',
      properties: {
        pod_id: {
          type: 'string',
          description: 'The pod ID',
        },
        job_id: {
          type: 'string',
          description: 'The training job ID',
        },
      },
      required: ['pod_id'],
    },
  },
  {
    name: 'runpod_get_training_logs',
    description: 'Get logs from a training job.',
    inputSchema: {
      type: 'object',
      properties: {
        pod_id: {
          type: 'string',
          description: 'The pod ID',
        },
        lines: {
          type: 'number',
          description: 'Number of log lines (default: 100)',
          default: 100,
        },
      },
      required: ['pod_id'],
    },
  },
  {
    name: 'runpod_estimate_cost',
    description: 'Estimate training cost based on model and dataset size.',
    inputSchema: {
      type: 'object',
      properties: {
        gpu_type: {
          type: 'string',
          description: 'GPU type',
        },
        training_hours: {
          type: 'number',
          description: 'Estimated training hours',
        },
        gpu_count: {
          type: 'number',
          description: 'Number of GPUs',
          default: 1,
        },
      },
      required: ['gpu_type', 'training_hours'],
    },
  },
];

/**
 * Helper to get error message from unknown error
 */
function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Handler implementations
 */
export const RUNPOD_HANDLERS: Record<string, ToolHandler> = {
  runpod_list_pods: async (_args, ctx) => {
    ctx.logger.info('Listing RunPod pods...');

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured. Set RUNPOD_API_KEY.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      const pods = (await ctx.runpodClient.listPods()) as Array<{
        id: string;
        name: string;
        desiredStatus: string;
        machine?: { gpuTypeId: string };
        gpuCount: number;
        costPerHr: number;
        runtime?: { uptimeInSeconds?: number; gpus?: Array<{ gpuUtilPercent?: number }> };
      }>;

      const podSummary = pods.map((pod) => ({
        id: pod.id,
        name: pod.name,
        status: pod.desiredStatus,
        gpu: pod.machine?.gpuTypeId,
        gpuCount: pod.gpuCount,
        costPerHr: pod.costPerHr,
        uptime: pod.runtime?.uptimeInSeconds
          ? `${Math.round(pod.runtime.uptimeInSeconds / 60)} minutes`
          : 'stopped',
        gpuUtilization: pod.runtime?.gpus?.[0]?.gpuUtilPercent
          ? `${pod.runtime.gpus[0].gpuUtilPercent}%`
          : 'N/A',
      }));

      return {
        content: [
          { type: 'text', text: JSON.stringify({ success: true, pods: podSummary }, null, 2) },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error listing pods: ${getErrorMessage(error)}`);
    }
  },

  runpod_get_pod: async (args, ctx) => {
    const { pod_id } = args as { pod_id: string };
    ctx.logger.info(`Getting pod: ${pod_id}`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      const pod = await ctx.runpodClient.getPod(pod_id);

      if (!pod) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({ success: false, error: `Pod ${pod_id} not found` }, null, 2),
            },
          ],
        };
      }

      return {
        content: [{ type: 'text', text: JSON.stringify({ success: true, pod }, null, 2) }],
      };
    } catch (error: unknown) {
      throw new Error(`Error getting pod: ${getErrorMessage(error)}`);
    }
  },

  runpod_check_gpus: async (args, ctx) => {
    const { gpu_type } = args as { gpu_type?: string };
    ctx.logger.info(`Checking GPUs: ${gpu_type || 'all'}`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      const gpuTypes = (await ctx.runpodClient.getAvailableGpus()) as Array<{
        id: string;
        displayName: string;
        memoryInGb: number;
        secureCloud: boolean;
        communityCloud: boolean;
      }>;

      const minVramGb = 24;
      const available = gpuTypes
        .filter((gpu) => {
          const hasCapacity = gpu.secureCloud || gpu.communityCloud;
          const hasVram = gpu.memoryInGb >= minVramGb;
          return hasCapacity && hasVram;
        })
        .map((gpu) => ({
          id: gpu.id,
          name: gpu.displayName,
          vram: `${gpu.memoryInGb}GB`,
          secureCloud: gpu.secureCloud,
          communityCloud: gpu.communityCloud,
        }))
        .sort((a, b) => parseInt(a.vram) - parseInt(b.vram));

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                minVramFilter: `${minVramGb}GB`,
                availableGpus: available,
                count: available.length,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error checking GPUs: ${getErrorMessage(error)}`);
    }
  },

  runpod_create_pod: async (args, ctx) => {
    const {
      name: podName,
      gpu_type,
      gpu_count = 1,
      volume_size = 30,
      image_name,
    } = args as {
      name: string;
      gpu_type: string;
      gpu_count?: number;
      volume_size?: number;
      image_name?: string;
    };

    ctx.logger.info(`Creating pod: ${podName} with ${gpu_type}`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      const pod = (await ctx.runpodClient.createPod({
        name: podName,
        gpuTypeId: gpu_type,
        gpuCount: gpu_count,
        volumeInGb: volume_size,
        imageName: image_name,
      })) as { id: string; name: string; desiredStatus: string; costPerHr: number };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                message: `Pod ${pod.name} created successfully`,
                pod: {
                  id: pod.id,
                  name: pod.name,
                  status: pod.desiredStatus,
                  costPerHr: pod.costPerHr,
                },
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error creating pod: ${getErrorMessage(error)}`);
    }
  },

  runpod_start_pod: async (args, ctx) => {
    const { pod_id } = args as { pod_id: string };
    ctx.logger.info(`Starting pod: ${pod_id}`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      const pod = (await ctx.runpodClient.startPod(pod_id)) as {
        id: string;
        name: string;
        desiredStatus: string;
        costPerHr: number;
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                message: `Pod ${pod.name} starting`,
                pod: {
                  id: pod.id,
                  name: pod.name,
                  status: pod.desiredStatus,
                  costPerHr: pod.costPerHr,
                },
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error starting pod: ${getErrorMessage(error)}`);
    }
  },

  runpod_stop_pod: async (args, ctx) => {
    const { pod_id } = args as { pod_id: string };
    ctx.logger.info(`Stopping pod: ${pod_id}`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      const pod = (await ctx.runpodClient.stopPod(pod_id)) as {
        id: string;
        name: string;
        desiredStatus: string;
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                message: `Pod ${pod.name} stopped. Volume data preserved.`,
                pod: {
                  id: pod.id,
                  name: pod.name,
                  status: pod.desiredStatus,
                },
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error stopping pod: ${getErrorMessage(error)}`);
    }
  },

  runpod_terminate_pod: async (args, ctx) => {
    const { pod_id } = args as { pod_id: string };
    ctx.logger.warn(`Terminating pod: ${pod_id}`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      await ctx.runpodClient.terminatePod(pod_id);

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                message: `Pod ${pod_id} terminated. All data has been deleted.`,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error terminating pod: ${getErrorMessage(error)}`);
    }
  },

  runpod_start_training: async (args, ctx) => {
    const {
      pod_id,
      model_name,
      dataset_path,
      output_dir = '/workspace/output',
      max_steps = 200,
    } = args as {
      pod_id: string;
      model_name: string;
      dataset_path: string;
      output_dir?: string;
      max_steps?: number;
    };

    ctx.logger.info(`Starting training on ${pod_id} with ${model_name}`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      // Check if pod is running
      const pod = (await ctx.runpodClient.getPod(pod_id)) as { runtime?: unknown } | null;
      if (!pod || !pod.runtime) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(
                {
                  success: false,
                  error: `Pod ${pod_id} is not running. Start it first with runpod_start_pod.`,
                },
                null,
                2
              ),
            },
          ],
        };
      }

      const job = (await ctx.runpodClient.startTraining(pod_id, {
        baseModel: model_name,
        datasetPath: dataset_path,
        outputDir: output_dir,
        maxSteps: max_steps,
      })) as { id: string; status: string };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                message: 'Training job started',
                job: {
                  id: job.id,
                  status: job.status,
                  baseModel: model_name,
                  outputDir: output_dir,
                },
                nextSteps: [
                  'Use runpod_get_training_status to monitor progress',
                  'Use runpod_get_training_logs to view training output',
                ],
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error starting training: ${getErrorMessage(error)}`);
    }
  },

  runpod_get_training_status: async (args, ctx) => {
    const { pod_id, job_id } = args as { pod_id: string; job_id?: string };
    ctx.logger.info(`Checking training status for pod: ${pod_id}`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      const status = await ctx.runpodClient.getTrainingStatus(pod_id, job_id || '');

      return {
        content: [
          { type: 'text', text: JSON.stringify({ success: true, training: status }, null, 2) },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error getting training status: ${getErrorMessage(error)}`);
    }
  },

  runpod_get_training_logs: async (args, ctx) => {
    const { pod_id, lines = 100 } = args as { pod_id: string; lines?: number };
    ctx.logger.info(`Getting logs for pod: ${pod_id}`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      const logs = await ctx.runpodClient.getTrainingLogs(pod_id, '', lines);

      return {
        content: [{ type: 'text', text: JSON.stringify({ success: true, logs }, null, 2) }],
      };
    } catch (error: unknown) {
      throw new Error(`Error getting training logs: ${getErrorMessage(error)}`);
    }
  },

  runpod_estimate_cost: async (args, ctx) => {
    const {
      gpu_type,
      training_hours,
      gpu_count = 1,
    } = args as {
      gpu_type: string;
      training_hours: number;
      gpu_count?: number;
    };

    ctx.logger.info(`Estimating cost: ${gpu_type} x ${training_hours}h`);

    if (!ctx.runpodClient) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              { success: false, error: 'RunPod client not configured.' },
              null,
              2
            ),
          },
        ],
        isError: true,
      };
    }

    try {
      const estimate = (await ctx.runpodClient.estimateTrainingCost({
        gpuType: gpu_type,
        trainingHours: training_hours,
        gpuCount: gpu_count,
      })) as { estimatedCost: number; estimatedHours: number };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                success: true,
                estimate: {
                  ...estimate,
                  gpuType: gpu_type,
                  trainingHours: training_hours,
                  gpuCount: gpu_count,
                  summary: `Estimated ${estimate.estimatedHours} hours = $${estimate.estimatedCost}`,
                },
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error: unknown) {
      throw new Error(`Error estimating cost: ${getErrorMessage(error)}`);
    }
  },
};

/**
 * Module export
 */
export const runpodModule: ToolModule = {
  tools: RUNPOD_TOOLS,
  handlers: RUNPOD_HANDLERS,
};

export default runpodModule;
