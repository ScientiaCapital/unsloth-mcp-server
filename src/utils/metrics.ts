import logger from './logger.js';

export interface ToolMetrics {
  toolName: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  success: boolean;
  error?: string;
}

class MetricsCollector {
  private metrics: ToolMetrics[] = [];
  private readonly maxMetrics = 1000;

  startTool(toolName: string): number {
    const startTime = Date.now();
    logger.debug(`Tool started: ${toolName}`, { startTime });
    return startTime;
  }

  endTool(toolName: string, startTime: number, success: boolean, error?: string): void {
    const endTime = Date.now();
    const duration = endTime - startTime;

    const metric: ToolMetrics = {
      toolName,
      startTime,
      endTime,
      duration,
      success,
      error,
    };

    this.metrics.push(metric);

    // Keep only last N metrics
    if (this.metrics.length > this.maxMetrics) {
      this.metrics.shift();
    }

    logger.info(`Tool completed: ${toolName}`, {
      duration: `${duration}ms`,
      success,
      error,
    });
  }

  getMetrics(toolName?: string): ToolMetrics[] {
    if (toolName) {
      return this.metrics.filter(m => m.toolName === toolName);
    }
    return [...this.metrics];
  }

  getStats(toolName?: string): {
    totalCalls: number;
    successfulCalls: number;
    failedCalls: number;
    averageDuration: number;
    minDuration: number;
    maxDuration: number;
  } {
    const metrics = toolName
      ? this.metrics.filter(m => m.toolName === toolName)
      : this.metrics;

    if (metrics.length === 0) {
      return {
        totalCalls: 0,
        successfulCalls: 0,
        failedCalls: 0,
        averageDuration: 0,
        minDuration: 0,
        maxDuration: 0,
      };
    }

    const durations = metrics.map(m => m.duration || 0).filter(d => d > 0);
    const successful = metrics.filter(m => m.success).length;

    return {
      totalCalls: metrics.length,
      successfulCalls: successful,
      failedCalls: metrics.length - successful,
      averageDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
      minDuration: Math.min(...durations),
      maxDuration: Math.max(...durations),
    };
  }

  clear(): void {
    this.metrics = [];
    logger.debug('Metrics cleared');
  }
}

export const metricsCollector = new MetricsCollector();
