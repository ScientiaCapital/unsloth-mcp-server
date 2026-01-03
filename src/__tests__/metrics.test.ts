import { describe, test, expect, beforeEach } from '@jest/globals';
import { metricsCollector } from '../utils/metrics.js';

describe('metricsCollector', () => {
  beforeEach(() => {
    metricsCollector.clear();
  });

  test('should track tool start and end', () => {
    const startTime = metricsCollector.startTool('test_tool');
    expect(startTime).toBeGreaterThan(0);

    metricsCollector.endTool('test_tool', startTime, true);

    const metrics = metricsCollector.getMetrics('test_tool');
    expect(metrics).toHaveLength(1);
    expect(metrics[0].toolName).toBe('test_tool');
    expect(metrics[0].success).toBe(true);
    expect(metrics[0].duration).toBeGreaterThanOrEqual(0);
  });

  test('should track failed executions', () => {
    const startTime = metricsCollector.startTool('failing_tool');
    metricsCollector.endTool('failing_tool', startTime, false, 'Test error');

    const metrics = metricsCollector.getMetrics('failing_tool');
    expect(metrics).toHaveLength(1);
    expect(metrics[0].success).toBe(false);
    expect(metrics[0].error).toBe('Test error');
  });

  test('should calculate stats correctly', async () => {
    // Add multiple metrics
    const tools = ['tool1', 'tool1', 'tool1', 'tool2'];
    const successes = [true, false, true, true];

    for (let i = 0; i < tools.length; i++) {
      const start = Date.now();
      // Add small delay to ensure duration > 0
      await new Promise((resolve) => setTimeout(resolve, 1));
      metricsCollector.endTool(tools[i], start, successes[i]);
    }

    const stats = metricsCollector.getStats('tool1');
    expect(stats.totalCalls).toBe(3);
    expect(stats.successfulCalls).toBe(2);
    expect(stats.failedCalls).toBe(1);
    expect(stats.averageDuration).toBeGreaterThanOrEqual(0);
    expect(isNaN(stats.averageDuration)).toBe(false);
  });

  test('should return all metrics when no filter provided', () => {
    metricsCollector.endTool('tool1', Date.now(), true);
    metricsCollector.endTool('tool2', Date.now(), true);

    const metrics = metricsCollector.getMetrics();
    expect(metrics).toHaveLength(2);
  });

  test('should filter metrics by tool name', () => {
    metricsCollector.endTool('tool1', Date.now(), true);
    metricsCollector.endTool('tool2', Date.now(), true);
    metricsCollector.endTool('tool1', Date.now(), false);

    const tool1Metrics = metricsCollector.getMetrics('tool1');
    expect(tool1Metrics).toHaveLength(2);
    expect(tool1Metrics.every((m) => m.toolName === 'tool1')).toBe(true);
  });

  test('should handle empty metrics gracefully', () => {
    const stats = metricsCollector.getStats('nonexistent_tool');
    expect(stats.totalCalls).toBe(0);
    expect(stats.successfulCalls).toBe(0);
    expect(stats.failedCalls).toBe(0);
    expect(stats.averageDuration).toBe(0);
  });

  test('should clear all metrics', () => {
    metricsCollector.endTool('tool1', Date.now(), true);
    metricsCollector.endTool('tool2', Date.now(), true);

    expect(metricsCollector.getMetrics()).toHaveLength(2);

    metricsCollector.clear();

    expect(metricsCollector.getMetrics()).toHaveLength(0);
  });

  test('should limit metrics to max size', () => {
    // Add more than max metrics (1000)
    for (let i = 0; i < 1100; i++) {
      metricsCollector.endTool(`tool_${i}`, Date.now(), true);
    }

    const metrics = metricsCollector.getMetrics();
    expect(metrics.length).toBeLessThanOrEqual(1000);
  });
});
