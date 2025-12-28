/**
 * Cost Tracker Tests
 */

import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import * as fs from 'fs';
import { CostTracker } from '../utils/cost-tracker.js';

describe('CostTracker', () => {
  const testDir = '/tmp/test_costs_' + Date.now();
  let tracker: CostTracker;

  beforeEach(() => {
    tracker = new CostTracker({
      data_dir: testDir,
      budget: {
        daily_limit: 10,
        weekly_limit: 50,
        monthly_limit: 200,
        per_job_limit: 5,
        alert_thresholds: [0.5, 0.75, 0.9, 1.0],
      },
    });
  });

  afterEach(() => {
    // Cleanup test directory
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  });

  describe('Session Tracking', () => {
    test('should start a session', () => {
      tracker.startSession('session1', 'pod123', 'RTX A5000', 0.16, 'job123', 'project1');

      const sessions = tracker.getActiveSessions();
      expect(sessions.length).toBe(1);
      expect(sessions[0].session_id).toBe('session1');
      expect(sessions[0].pod_id).toBe('pod123');
    });

    test('should track multiple active sessions', () => {
      tracker.startSession('session1', 'pod123', 'RTX A5000', 0.16);
      tracker.startSession('session2', 'pod456', 'RTX 4090', 0.44);

      const sessions = tracker.getActiveSessions();
      expect(sessions.length).toBe(2);
    });

    test('should get current session cost', () => {
      tracker.startSession('session1', 'pod123', 'RTX A5000', 0.16);

      // Wait a tiny bit to accumulate some cost
      const cost = tracker.getCurrentSessionCost('session1');

      expect(cost).not.toBeNull();
      expect(cost!.hours).toBeGreaterThanOrEqual(0);
      expect(cost!.cost).toBeGreaterThanOrEqual(0);
    });

    test('should return null for non-existent session', () => {
      const cost = tracker.getCurrentSessionCost('nonexistent');
      expect(cost).toBeNull();
    });

    test('should end a session and record entry', () => {
      tracker.startSession('session1', 'pod123', 'RTX A5000', 0.16, 'job123');

      const entry = tracker.endSession('session1', 'completed', 75, 50);

      expect(entry).not.toBeNull();
      expect(entry!.pod_id).toBe('pod123');
      expect(entry!.status).toBe('completed');
      expect(entry!.gpu_utilization_avg).toBe(75);
    });

    test('should remove session after ending', () => {
      tracker.startSession('session1', 'pod123', 'RTX A5000', 0.16);
      tracker.endSession('session1');

      const sessions = tracker.getActiveSessions();
      expect(sessions.length).toBe(0);
    });

    test('should return null when ending non-existent session', () => {
      const entry = tracker.endSession('nonexistent');
      expect(entry).toBeNull();
    });
  });

  describe('Cost Analysis', () => {
    beforeEach(() => {
      // Add some test entries by starting and ending sessions
      tracker.startSession('s1', 'pod1', 'RTX A5000', 0.16, 'job1', 'project1');
      tracker.endSession('s1', 'completed', 80);

      tracker.startSession('s2', 'pod2', 'RTX 4090', 0.44, 'job2', 'project2');
      tracker.endSession('s2', 'completed', 90);
    });

    test('should get cost summary', () => {
      const summary = tracker.getSummary();

      expect(summary.entries_count).toBe(2);
      expect(summary.total_cost).toBeGreaterThanOrEqual(0);
      expect(Object.keys(summary.by_pod).length).toBe(2);
      expect(Object.keys(summary.by_gpu_type).length).toBe(2);
    });

    test('should filter summary by project', () => {
      const summary = tracker.getSummary(undefined, undefined, 'project1');

      expect(summary.entries_count).toBe(1);
    });

    test('should get today cost', () => {
      const todayCost = tracker.getTodayCost();
      expect(todayCost).toBeGreaterThanOrEqual(0);
    });

    test('should get week cost', () => {
      const weekCost = tracker.getWeekCost();
      expect(weekCost).toBeGreaterThanOrEqual(0);
    });

    test('should get month cost', () => {
      const monthCost = tracker.getMonthCost();
      expect(monthCost).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Alerts', () => {
    test('should create an alert', () => {
      const alert = tracker.createAlert(
        'high_cost_job',
        'warning',
        'Job xyz exceeded expected cost',
        { job_id: 'xyz', cost: 10 }
      );

      expect(alert.id).toBeDefined();
      expect(alert.type).toBe('high_cost_job');
      expect(alert.severity).toBe('warning');
      expect(alert.acknowledged).toBe(false);
    });

    test('should get all alerts', () => {
      tracker.createAlert('idle_pod', 'info', 'Pod has been idle');
      tracker.createAlert('cost_spike', 'warning', 'Unusual cost increase');

      const alerts = tracker.getAlerts();
      expect(alerts.length).toBe(2);
    });

    test('should get only unacknowledged alerts', () => {
      const alert1 = tracker.createAlert('idle_pod', 'info', 'Idle');
      tracker.createAlert('cost_spike', 'warning', 'Spike');

      tracker.acknowledgeAlert(alert1.id);

      const unacknowledged = tracker.getAlerts(true);
      expect(unacknowledged.length).toBe(1);
    });

    test('should acknowledge an alert', () => {
      const alert = tracker.createAlert('idle_pod', 'info', 'Idle');

      const result = tracker.acknowledgeAlert(alert.id);
      expect(result).toBe(true);

      const alerts = tracker.getAlerts();
      const acknowledged = alerts.find((a) => a.id === alert.id);
      expect(acknowledged?.acknowledged).toBe(true);
    });

    test('should return false when acknowledging non-existent alert', () => {
      const result = tracker.acknowledgeAlert('nonexistent');
      expect(result).toBe(false);
    });
  });

  describe('Dashboard', () => {
    test('should get dashboard data', () => {
      tracker.startSession('s1', 'pod1', 'RTX A5000', 0.16);

      const dashboard = tracker.getDashboard();

      expect(dashboard.current_sessions.length).toBe(1);
      expect(dashboard.today_cost).toBeDefined();
      expect(dashboard.budget_status.daily).toBeDefined();
      expect(dashboard.summary).toBeDefined();
    });

    test('should include budget percentages', () => {
      const dashboard = tracker.getDashboard();

      expect(dashboard.budget_status.daily?.percent).toBeDefined();
      expect(dashboard.budget_status.weekly?.percent).toBeDefined();
      expect(dashboard.budget_status.monthly?.percent).toBeDefined();
    });
  });

  describe('Export', () => {
    test('should export to CSV', () => {
      tracker.startSession('s1', 'pod1', 'RTX A5000', 0.16, 'job1');
      tracker.endSession('s1', 'completed');

      const csv = tracker.exportToCSV();

      expect(csv).toContain('timestamp');
      expect(csv).toContain('pod_id');
      expect(csv).toContain('pod1');
      expect(csv).toContain('job1');
    });
  });

  describe('Budget Management', () => {
    test('should update budget', () => {
      tracker.updateBudget({
        daily_limit: 20,
        monthly_limit: 400,
      });

      const dashboard = tracker.getDashboard();
      expect(dashboard.budget_status.daily?.limit).toBe(20);
      expect(dashboard.budget_status.monthly?.limit).toBe(400);
    });
  });

  describe('Data Persistence', () => {
    test('should persist entries to disk', () => {
      tracker.startSession('s1', 'pod1', 'RTX A5000', 0.16);
      tracker.endSession('s1', 'completed');

      // Check file exists
      expect(fs.existsSync(`${testDir}/cost_entries.json`)).toBe(true);
    });

    test('should load entries on initialization', () => {
      tracker.startSession('s1', 'pod1', 'RTX A5000', 0.16);
      tracker.endSession('s1', 'completed');

      // Create new tracker instance (simulating restart)
      const newTracker = new CostTracker({ data_dir: testDir });

      const summary = newTracker.getSummary();
      expect(summary.entries_count).toBe(1);
    });
  });
});
