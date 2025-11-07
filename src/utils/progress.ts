import logger from './logger.js';

export interface ProgressUpdate {
  stage: string;
  progress: number; // 0-1
  message: string;
  eta?: number; // seconds
  details?: Record<string, any>;
}

export type ProgressCallback = (update: ProgressUpdate) => void;

class ProgressTracker {
  private callbacks: Map<string, ProgressCallback[]>;

  constructor() {
    this.callbacks = new Map();
  }

  subscribe(operationId: string, callback: ProgressCallback): void {
    if (!this.callbacks.has(operationId)) {
      this.callbacks.set(operationId, []);
    }
    this.callbacks.get(operationId)!.push(callback);
  }

  unsubscribe(operationId: string, callback?: ProgressCallback): void {
    if (!callback) {
      this.callbacks.delete(operationId);
    } else {
      const cbs = this.callbacks.get(operationId);
      if (cbs) {
        const index = cbs.indexOf(callback);
        if (index > -1) {
          cbs.splice(index, 1);
        }
      }
    }
  }

  report(operationId: string, update: ProgressUpdate): void {
    const cbs = this.callbacks.get(operationId);
    if (cbs) {
      cbs.forEach(cb => {
        try {
          cb(update);
        } catch (error: any) {
          logger.error('Progress callback error', { error: error.message });
        }
      });
    }

    // Also log progress
    logger.info('Progress update', {
      operationId,
      stage: update.stage,
      progress: `${Math.round(update.progress * 100)}%`,
      message: update.message,
      eta: update.eta ? `${update.eta}s` : undefined,
    });
  }

  complete(operationId: string, message?: string): void {
    this.report(operationId, {
      stage: 'complete',
      progress: 1.0,
      message: message || 'Operation completed',
    });
    this.unsubscribe(operationId);
  }

  error(operationId: string, error: string): void {
    this.report(operationId, {
      stage: 'error',
      progress: 0,
      message: error,
    });
    this.unsubscribe(operationId);
  }
}

// Export both the class and a singleton instance
export { ProgressTracker };
export const progressTracker = new ProgressTracker();

// Helper to create progress-reporting Python scripts
export function wrapWithProgress(
  operationId: string,
  script: string,
  stages: Array<{ name: string; weight: number }>
): string {
  const stageJson = JSON.stringify(stages);

  return `
import json
import sys

OPERATION_ID = "${operationId}"
STAGES = ${stageJson}

def report_progress(stage_name, progress, message, eta=None):
    # Find stage
    stage_index = next((i for i, s in enumerate(STAGES) if s["name"] == stage_name), 0)

    # Calculate overall progress
    completed_weight = sum(s["weight"] for i, s in enumerate(STAGES) if i < stage_index)
    current_weight = STAGES[stage_index]["weight"] if stage_index < len(STAGES) else 0
    total_weight = sum(s["weight"] for s in STAGES)

    overall_progress = (completed_weight + current_weight * progress) / total_weight

    # Print progress (will be captured by stdout)
    progress_data = {
        "type": "progress",
        "operation_id": OPERATION_ID,
        "stage": stage_name,
        "progress": overall_progress,
        "message": message,
        "eta": eta
    }
    print("PROGRESS:" + json.dumps(progress_data), file=sys.stderr, flush=True)

# Original script with progress reporting injected
${script}
`;
}

// Parse progress from Python stdout/stderr
export function parseProgress(output: string): ProgressUpdate | null {
  const match = output.match(/PROGRESS:({.*})/);
  if (match) {
    try {
      const data = JSON.parse(match[1]);
      return {
        stage: data.stage,
        progress: data.progress,
        message: data.message,
        eta: data.eta,
      };
    } catch (error) {
      return null;
    }
  }
  return null;
}

export default progressTracker;
