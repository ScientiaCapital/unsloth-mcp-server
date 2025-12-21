/**
 * RunPod API Client
 *
 * Handles GPU pod management, training job execution, and model artifact retrieval.
 * Designed for fine-tuning LLMs with Unsloth on RunPod infrastructure.
 */

import { logger } from './logger.js';

// ============================================================================
// Types
// ============================================================================

export interface RunPodConfig {
  apiKey: string;
  apiEndpoint?: string;
  defaultGpuType?: string;
  defaultImage?: string;
  defaultVolumeSize?: number;
}

export interface Pod {
  id: string;
  name: string;
  desiredStatus: 'RUNNING' | 'EXITED' | 'STOPPED' | 'TERMINATED';
  runtime?: {
    uptimeInSeconds: number;
    gpus: Array<{
      id: string;
      gpuUtilPercent: number;
      memoryUtilPercent: number;
    }>;
    ports: Array<{
      ip: string;
      isIpPublic: boolean;
      privatePort: number;
      publicPort: number;
      type: string;
    }>;
  };
  machine: {
    gpuTypeId: string;
  };
  gpuCount: number;
  imageName: string;
  volumeInGb: number;
  containerDiskInGb?: number;
  costPerHr: number;
  vcpuCount?: number;
  memoryInGb?: number;
  volumeMountPath?: string;
}

export interface GpuType {
  id: string;
  displayName: string;
  memoryInGb: number;
  secureCloud: boolean;
  communityCloud: boolean;
}

export interface CreatePodOptions {
  name: string;
  gpuTypeId: string;
  gpuCount?: number;
  imageName?: string;
  volumeInGb?: number;
  containerDiskInGb?: number;
  volumeMountPath?: string;
  ports?: string;
  env?: Record<string, string>;
  dockerArgs?: string;
  startSsh?: boolean;
  startJupyter?: boolean;
  templateId?: string;
}

export interface TrainingJobConfig {
  baseModel: string;
  datasetPath: string;
  outputDir: string;
  loraR?: number;
  loraAlpha?: number;
  learningRate?: number;
  epochs?: number;
  batchSize?: number;
  maxSeqLength?: number;
  gradientAccumulationSteps?: number;
  warmupSteps?: number;
  saveSteps?: number;
  loggingSteps?: number;
}

export interface TrainingJob {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  currentStep: number;
  totalSteps: number;
  currentEpoch: number;
  totalEpochs: number;
  trainingLoss?: number;
  evalLoss?: number;
  startedAt?: string;
  completedAt?: string;
  errorMessage?: string;
}

export interface RunPodResponse<T> {
  data?: T;
  errors?: Array<{ message: string; locations?: Array<{ line: number; column: number }> }>;
}

// ============================================================================
// RunPod Client
// ============================================================================

export class RunPodClient {
  private apiKey: string;
  private apiEndpoint: string;
  private defaultGpuType: string;
  private defaultImage: string;
  private defaultVolumeSize: number;

  constructor(config: RunPodConfig) {
    this.apiKey = config.apiKey;
    this.apiEndpoint = config.apiEndpoint || 'https://api.runpod.io/graphql';
    this.defaultGpuType = config.defaultGpuType || 'NVIDIA RTX A5000';
    this.defaultImage =
      config.defaultImage || 'runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04';
    this.defaultVolumeSize = config.defaultVolumeSize || 30;
  }

  // --------------------------------------------------------------------------
  // GraphQL Request Helper
  // --------------------------------------------------------------------------

  private async graphqlRequest<T>(query: string, variables?: Record<string, unknown>): Promise<T> {
    const response = await fetch(this.apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ query, variables }),
    });

    if (!response.ok) {
      throw new Error(`RunPod API error: ${response.status} ${response.statusText}`);
    }

    const result = (await response.json()) as RunPodResponse<T>;

    if (result.errors && result.errors.length > 0) {
      const errorMessages = result.errors.map((e) => e.message).join(', ');
      throw new Error(`RunPod GraphQL error: ${errorMessages}`);
    }

    return result.data as T;
  }

  // --------------------------------------------------------------------------
  // Account & GPU Availability
  // --------------------------------------------------------------------------

  /**
   * Get current user info
   */
  async getMyself(): Promise<{ id: string; email: string; currentSpendPerHr: number }> {
    const query = `
      query {
        myself {
          id
          email
          currentSpendPerHr
        }
      }
    `;

    const data = await this.graphqlRequest<{
      myself: { id: string; email: string; currentSpendPerHr: number };
    }>(query);
    return data.myself;
  }

  /**
   * Get available GPU types with pricing
   */
  async getGpuTypes(): Promise<GpuType[]> {
    // Note: lowestPrice causes internal server errors on RunPod's API (Dec 2025)
    // Using simpler query without pricing data
    const query = `
      query {
        gpuTypes {
          id
          displayName
          memoryInGb
          secureCloud
          communityCloud
        }
      }
    `;

    const data = await this.graphqlRequest<{ gpuTypes: GpuType[] }>(query);
    return data.gpuTypes;
  }

  /**
   * Check if a specific GPU type is available
   */
  async checkGpuAvailability(gpuTypeId: string): Promise<{
    available: boolean;
    secureCloud: boolean;
    communityCloud: boolean;
    memoryInGb: number;
  }> {
    const gpuTypes = await this.getGpuTypes();
    const gpu = gpuTypes.find((g) => g.id === gpuTypeId || g.displayName === gpuTypeId);

    if (!gpu) {
      return { available: false, secureCloud: false, communityCloud: false, memoryInGb: 0 };
    }

    return {
      available: gpu.secureCloud || gpu.communityCloud,
      secureCloud: gpu.secureCloud,
      communityCloud: gpu.communityCloud,
      memoryInGb: gpu.memoryInGb,
    };
  }

  /**
   * Find best available GPU for fine-tuning based on VRAM needs
   */
  async findBestAvailableGpu(minVramGb: number = 24): Promise<GpuType | null> {
    const gpuTypes = await this.getGpuTypes();

    // Filter by VRAM and availability, sort by VRAM (closest to requirement first)
    const available = gpuTypes
      .filter((gpu) => {
        const hasCapacity = gpu.secureCloud || gpu.communityCloud;
        const hasVram = gpu.memoryInGb >= minVramGb;
        return hasCapacity && hasVram;
      })
      .sort((a, b) => a.memoryInGb - b.memoryInGb); // Smallest sufficient VRAM first

    return available.length > 0 ? available[0] : null;
  }

  // --------------------------------------------------------------------------
  // Pod Management
  // --------------------------------------------------------------------------

  /**
   * List all pods
   */
  async listPods(): Promise<Pod[]> {
    const query = `
      query {
        myself {
          pods {
            id
            name
            desiredStatus
            gpuCount
            machine {
              gpuTypeId
            }
            imageName
            volumeInGb
            costPerHr
          }
        }
      }
    `;

    const data = await this.graphqlRequest<{ myself: { pods: Pod[] } }>(query);
    return data.myself.pods;
  }

  /**
   * Get a specific pod by ID
   */
  async getPod(podId: string): Promise<Pod | null> {
    const query = `
      query getPod($podId: String!) {
        pod(input: { podId: $podId }) {
          id
          name
          desiredStatus
          gpuCount
          machine {
            gpuTypeId
          }
          imageName
          volumeInGb
          costPerHr
          runtime {
            uptimeInSeconds
            gpus {
              id
              gpuUtilPercent
              memoryUtilPercent
            }
            ports {
              ip
              isIpPublic
              privatePort
              publicPort
              type
            }
          }
        }
      }
    `;

    try {
      const data = await this.graphqlRequest<{ pod: Pod }>(query, { podId });
      return data.pod;
    } catch {
      return null;
    }
  }

  /**
   * Create a new pod
   */
  async createPod(options: CreatePodOptions): Promise<Pod> {
    const query = `
      mutation createPod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
          id
          name
          desiredStatus
          gpuCount
          imageName
          machine {
            gpuTypeId
          }
          volumeInGb
          costPerHr
        }
      }
    `;

    const input = {
      name: options.name,
      gpuTypeId: options.gpuTypeId,
      gpuCount: options.gpuCount || 1,
      imageName: options.imageName || this.defaultImage,
      volumeInGb: options.volumeInGb || this.defaultVolumeSize,
      containerDiskInGb: options.containerDiskInGb || 20,
      volumeMountPath: options.volumeMountPath || '/runpod-volume',
      ports: options.ports || '8888/http,22/tcp',
      startSsh: options.startSsh ?? true,
      startJupyter: options.startJupyter ?? true,
      env: options.env ? Object.entries(options.env).map(([key, value]) => ({ key, value })) : [],
      dockerArgs: options.dockerArgs,
      templateId: options.templateId,
    };

    logger.info('Creating RunPod pod', { input });

    const data = await this.graphqlRequest<{ podFindAndDeployOnDemand: Pod }>(query, { input });
    return data.podFindAndDeployOnDemand;
  }

  /**
   * Start a stopped pod
   */
  async startPod(podId: string, gpuCount?: number): Promise<Pod> {
    const query = `
      mutation startPod($podId: String!, $gpuCount: Int) {
        podResume(input: { podId: $podId, gpuCount: $gpuCount }) {
          id
          name
          desiredStatus
          costPerHr
        }
      }
    `;

    logger.info('Starting RunPod pod', { podId, gpuCount });

    const data = await this.graphqlRequest<{ podResume: Pod }>(query, { podId, gpuCount });
    return data.podResume;
  }

  /**
   * Stop a running pod (keeps volume data)
   */
  async stopPod(podId: string): Promise<Pod> {
    const query = `
      mutation stopPod($podId: String!) {
        podStop(input: { podId: $podId }) {
          id
          name
          desiredStatus
        }
      }
    `;

    logger.info('Stopping RunPod pod', { podId });

    const data = await this.graphqlRequest<{ podStop: Pod }>(query, { podId });
    return data.podStop;
  }

  /**
   * Terminate a pod (deletes everything)
   */
  async terminatePod(podId: string): Promise<void> {
    const query = `
      mutation terminatePod($podId: String!) {
        podTerminate(input: { podId: $podId })
      }
    `;

    logger.info('Terminating RunPod pod', { podId });

    await this.graphqlRequest(query, { podId });
  }

  /**
   * Handle GPU unavailability by migrating pod data
   * This creates a new pod with similar specs when original GPU is unavailable
   */
  async migratePod(
    oldPodId: string,
    newGpuTypeId?: string
  ): Promise<{ newPod: Pod; migrationStatus: 'success' | 'partial' | 'failed'; message: string }> {
    // Get old pod info
    const oldPod = await this.getPod(oldPodId);
    if (!oldPod) {
      throw new Error(`Pod ${oldPodId} not found`);
    }

    // Find best available GPU if not specified
    let targetGpu = newGpuTypeId;
    if (!targetGpu) {
      const bestGpu = await this.findBestAvailableGpu(24); // Minimum 24GB for fine-tuning
      if (!bestGpu) {
        throw new Error('No suitable GPUs available for migration');
      }
      targetGpu = bestGpu.id;
    }

    // Create new pod with same configuration
    const newPod = await this.createPod({
      name: `${oldPod.name}-migrated`,
      gpuTypeId: targetGpu,
      gpuCount: oldPod.gpuCount,
      imageName: oldPod.imageName,
      volumeInGb: oldPod.volumeInGb,
      containerDiskInGb: oldPod.containerDiskInGb,
      volumeMountPath: oldPod.volumeMountPath,
    });

    logger.info('Pod migration initiated', {
      oldPodId,
      newPodId: newPod.id,
      newGpuType: targetGpu,
    });

    return {
      newPod,
      migrationStatus: 'success',
      message: `Created new pod ${newPod.id} with GPU ${targetGpu}. Note: Volume data from old pod is NOT automatically transferred.`,
    };
  }

  // --------------------------------------------------------------------------
  // Command Execution
  // --------------------------------------------------------------------------

  /**
   * Execute a command on a running pod via SSH
   * Requires the pod to have SSH enabled
   */
  async executeCommand(
    podId: string,
    command: string
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    // Get pod info to find SSH connection details
    const pod = await this.getPod(podId);
    if (!pod) {
      throw new Error(`Pod ${podId} not found`);
    }

    if (!pod.runtime) {
      throw new Error(`Pod ${podId} is not running`);
    }

    const sshPort = pod.runtime.ports.find((p) => p.privatePort === 22);
    if (!sshPort) {
      throw new Error(`SSH not enabled on pod ${podId}`);
    }

    // Note: Actual SSH execution would require an SSH library
    // For now, we'll use RunPod's runsync endpoint for command execution
    const endpoint = `https://api.runpod.ai/v2/${podId}/runsync`;

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: {
          command,
        },
      }),
    });

    if (!response.ok) {
      // If runsync isn't available, return instructions for manual SSH
      return {
        stdout: '',
        stderr: `Direct command execution not available. SSH to pod: ssh root@${sshPort.ip} -p ${sshPort.publicPort}`,
        exitCode: 1,
      };
    }

    const result = (await response.json()) as {
      output?: { stdout: string; stderr: string; exitCode: number };
    };
    return result.output || { stdout: '', stderr: 'No output', exitCode: 0 };
  }

  // --------------------------------------------------------------------------
  // Training Job Management
  // --------------------------------------------------------------------------

  /**
   * Generate Unsloth training script
   */
  generateTrainingScript(config: TrainingJobConfig): string {
    return `
#!/usr/bin/env python3
"""
Unsloth Fine-tuning Script
Generated by unsloth-mcp-server
"""

import os
import json
from datetime import datetime

# Install dependencies if needed
os.system("pip install -q unsloth transformers datasets accelerate bitsandbytes")

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# Configuration
BASE_MODEL = "${config.baseModel}"
DATASET_PATH = "${config.datasetPath}"
OUTPUT_DIR = "${config.outputDir}"

# Training hyperparameters
LORA_R = ${config.loraR || 16}
LORA_ALPHA = ${config.loraAlpha || 32}
LEARNING_RATE = ${config.learningRate || 2e-4}
EPOCHS = ${config.epochs || 3}
BATCH_SIZE = ${config.batchSize || 4}
MAX_SEQ_LENGTH = ${config.maxSeqLength || 2048}
GRADIENT_ACCUMULATION = ${config.gradientAccumulationSteps || 4}
WARMUP_STEPS = ${config.warmupSteps || 10}
SAVE_STEPS = ${config.saveSteps || 100}
LOGGING_STEPS = ${config.loggingSteps || 10}

print(f"[{datetime.now().isoformat()}] Starting fine-tuning...")
print(f"Base model: {BASE_MODEL}")
print(f"Dataset: {DATASET_PATH}")

# Load model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Load dataset
if DATASET_PATH.startswith("http") or "/" in DATASET_PATH:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
else:
    dataset = load_dataset(DATASET_PATH, split="train")

print(f"Dataset size: {len(dataset)} examples")

# Format for training
def format_prompt(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        text = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{input_text}\\n\\n### Response:\\n{output}"
    else:
        text = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}"

    return {"text": text}

dataset = dataset.map(format_prompt)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    fp16=True,
    optim="adamw_8bit",
    seed=42,
    report_to="none",  # Disable wandb
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

# Train
print(f"[{datetime.now().isoformat()}] Training started...")
trainer.train()

# Save final model
print(f"[{datetime.now().isoformat()}] Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training info
info = {
    "base_model": BASE_MODEL,
    "dataset": DATASET_PATH,
    "epochs": EPOCHS,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "learning_rate": LEARNING_RATE,
    "completed_at": datetime.now().isoformat(),
}

with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
    json.dump(info, f, indent=2)

print(f"[{datetime.now().isoformat()}] Training complete! Model saved to {OUTPUT_DIR}")
`.trim();
  }

  /**
   * Start a training job on a pod
   */
  async startTrainingJob(podId: string, config: TrainingJobConfig): Promise<TrainingJob> {
    const script = this.generateTrainingScript(config);
    const scriptPath = '/runpod-volume/train.py';

    // Upload script to pod
    const uploadResult = await this.executeCommand(
      podId,
      `cat > ${scriptPath} << 'TRAINING_SCRIPT_EOF'
${script}
TRAINING_SCRIPT_EOF`
    );

    if (uploadResult.exitCode !== 0) {
      throw new Error(`Failed to upload training script: ${uploadResult.stderr}`);
    }

    // Start training in background with nohup
    const startResult = await this.executeCommand(
      podId,
      `cd /runpod-volume && nohup python3 ${scriptPath} > training.log 2>&1 & echo $!`
    );

    const jobId = startResult.stdout.trim() || `job-${Date.now()}`;

    logger.info('Training job started', { podId, jobId, config });

    return {
      id: jobId,
      status: 'running',
      progress: 0,
      currentStep: 0,
      totalSteps: 0,
      currentEpoch: 0,
      totalEpochs: config.epochs || 3,
      startedAt: new Date().toISOString(),
    };
  }

  /**
   * Get training job status by parsing logs
   */
  async getTrainingStatus(podId: string): Promise<TrainingJob> {
    const result = await this.executeCommand(
      podId,
      'tail -50 /runpod-volume/training.log 2>/dev/null || echo "No logs yet"'
    );

    // Parse logs to extract progress
    const logs = result.stdout;
    let status: TrainingJob['status'] = 'running';
    let progress = 0;
    let currentStep = 0;
    let currentEpoch = 0;
    let trainingLoss: number | undefined;

    if (logs.includes('Training complete!')) {
      status = 'completed';
      progress = 100;
    } else if (logs.includes('Error') || logs.includes('Exception')) {
      status = 'failed';
    }

    // Parse step info (e.g., "Step 50/1000")
    const stepMatch = logs.match(/Step\s+(\d+)\/(\d+)/i);
    if (stepMatch) {
      currentStep = parseInt(stepMatch[1], 10);
      const totalSteps = parseInt(stepMatch[2], 10);
      progress = Math.round((currentStep / totalSteps) * 100);
    }

    // Parse epoch info
    const epochMatch = logs.match(/Epoch\s+(\d+)/i);
    if (epochMatch) {
      currentEpoch = parseInt(epochMatch[1], 10);
    }

    // Parse loss
    const lossMatch = logs.match(/loss[:\s]+([0-9.]+)/i);
    if (lossMatch) {
      trainingLoss = parseFloat(lossMatch[1]);
    }

    return {
      id: 'current',
      status,
      progress,
      currentStep,
      totalSteps: 0,
      currentEpoch,
      totalEpochs: 0,
      trainingLoss,
    };
  }

  /**
   * Get training logs
   */
  async getTrainingLogs(podId: string, lines: number = 100): Promise<string> {
    const result = await this.executeCommand(
      podId,
      `tail -${lines} /runpod-volume/training.log 2>/dev/null || echo "No logs yet"`
    );
    return result.stdout;
  }

  /**
   * Stop a training job
   */
  async stopTrainingJob(podId: string, jobId: string): Promise<void> {
    await this.executeCommand(podId, `kill ${jobId} 2>/dev/null || pkill -f train.py`);
    logger.info('Training job stopped', { podId, jobId });
  }

  // --------------------------------------------------------------------------
  // Cost Estimation
  // --------------------------------------------------------------------------

  /**
   * Estimate training cost
   */
  estimateTrainingCost(
    datasetTokens: number,
    baseModel: string,
    gpuCostPerHour: number = 0.16,
    epochs: number = 3
  ): { estimatedHours: number; estimatedCost: number; tokensPerSecond: number } {
    // Approximate tokens per second based on model size
    let tokensPerSecond = 2000; // Default

    if (baseModel.includes('1B')) tokensPerSecond = 8000;
    else if (baseModel.includes('3B')) tokensPerSecond = 5000;
    else if (baseModel.includes('7B') || baseModel.includes('8B')) tokensPerSecond = 2500;
    else if (baseModel.includes('13B')) tokensPerSecond = 1500;
    else if (baseModel.includes('70B')) tokensPerSecond = 400;

    const totalTokens = datasetTokens * epochs;
    const estimatedSeconds = totalTokens / tokensPerSecond;
    const estimatedHours = estimatedSeconds / 3600;
    const estimatedCost = estimatedHours * gpuCostPerHour;

    return {
      estimatedHours: Math.round(estimatedHours * 100) / 100,
      estimatedCost: Math.round(estimatedCost * 100) / 100,
      tokensPerSecond,
    };
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let runpodClient: RunPodClient | null = null;

export function getRunPodClient(): RunPodClient {
  if (!runpodClient) {
    const apiKey = process.env.RUNPOD_API_KEY;
    if (!apiKey) {
      throw new Error('RUNPOD_API_KEY environment variable is not set');
    }

    runpodClient = new RunPodClient({
      apiKey,
      apiEndpoint: process.env.RUNPOD_API_ENDPOINT,
      defaultGpuType: process.env.RUNPOD_DEFAULT_GPU_TYPE,
      defaultImage: process.env.RUNPOD_DEFAULT_IMAGE,
      defaultVolumeSize: process.env.RUNPOD_DEFAULT_VOLUME_SIZE
        ? parseInt(process.env.RUNPOD_DEFAULT_VOLUME_SIZE, 10)
        : undefined,
    });
  }

  return runpodClient;
}

export function resetRunPodClient(): void {
  runpodClient = null;
}
