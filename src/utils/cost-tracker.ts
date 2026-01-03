/**
 * Cost Tracking Dashboard
 *
 * Real-time GPU cost monitoring with alerts and analytics.
 * Tracks spending across training jobs, pods, and projects.
 */

import * as fs from 'fs';
import * as path from 'path';
import { logger } from './logger.js';

// ============================================================================
// Types
// ============================================================================

export interface CostEntry {
  id: string;
  timestamp: string;
  pod_id: string;
  job_id?: string;
  project?: string;

  // Duration
  duration_seconds: number;
  duration_hours: number;

  // Costs
  gpu_type: string;
  cost_per_hour: number;
  total_cost: number;

  // Usage
  gpu_utilization_avg?: number;
  memory_utilization_avg?: number;

  // Status
  status: 'running' | 'completed' | 'failed' | 'cancelled';
}

export interface CostSummary {
  total_cost: number;
  total_hours: number;
  entries_count: number;

  // Breakdown
  by_pod: Record<string, number>;
  by_job: Record<string, number>;
  by_project: Record<string, number>;
  by_gpu_type: Record<string, number>;
  by_status: Record<string, number>;

  // Time-based
  daily_costs: Array<{ date: string; cost: number; hours: number }>;
  hourly_costs: Array<{ hour: string; cost: number }>;

  // Efficiency
  avg_gpu_utilization: number;
  avg_cost_per_hour: number;
  estimated_monthly_cost: number;
}

export interface CostAlert {
  id: string;
  type: 'budget_threshold' | 'high_cost_job' | 'idle_pod' | 'cost_spike';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: string;
  acknowledged: boolean;
  metadata?: Record<string, unknown>;
}

export interface CostBudget {
  daily_limit?: number;
  weekly_limit?: number;
  monthly_limit?: number;
  per_job_limit?: number;
  alert_thresholds: number[]; // e.g., [0.5, 0.75, 0.9, 1.0] for 50%, 75%, 90%, 100%
}

export interface CostTrackerConfig {
  data_dir: string;
  budget: CostBudget;
  auto_stop_on_budget: boolean;
  retention_days: number;
  alert_webhook?: string;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: CostTrackerConfig = {
  data_dir: './data/costs',
  budget: {
    daily_limit: 10,
    weekly_limit: 50,
    monthly_limit: 200,
    per_job_limit: 5,
    alert_thresholds: [0.5, 0.75, 0.9, 1.0],
  },
  auto_stop_on_budget: false,
  retention_days: 90,
};

// ============================================================================
// Cost Tracker
// ============================================================================

export class CostTracker {
  private config: CostTrackerConfig;
  private entries: CostEntry[] = [];
  private alerts: CostAlert[] = [];
  private activeSessions: Map<
    string,
    {
      start_time: Date;
      pod_id: string;
      cost_per_hour: number;
      job_id?: string;
      project?: string;
      gpu_type: string;
    }
  > = new Map();

  constructor(config: Partial<CostTrackerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.ensureDataDir();
    this.loadData();
  }

  /**
   * Ensure data directory exists
   */
  private ensureDataDir(): void {
    if (!fs.existsSync(this.config.data_dir)) {
      fs.mkdirSync(this.config.data_dir, { recursive: true });
      logger.info('Created cost tracking directory', { path: this.config.data_dir });
    }
  }

  /**
   * Load existing data from disk
   */
  private loadData(): void {
    const entriesPath = path.join(this.config.data_dir, 'cost_entries.json');
    const alertsPath = path.join(this.config.data_dir, 'alerts.json');

    if (fs.existsSync(entriesPath)) {
      try {
        this.entries = JSON.parse(fs.readFileSync(entriesPath, 'utf-8'));
      } catch (error) {
        logger.warn('Failed to load cost entries', { error });
        this.entries = [];
      }
    }

    if (fs.existsSync(alertsPath)) {
      try {
        this.alerts = JSON.parse(fs.readFileSync(alertsPath, 'utf-8'));
      } catch (error) {
        logger.warn('Failed to load alerts', { error });
        this.alerts = [];
      }
    }
  }

  /**
   * Save data to disk
   */
  private saveData(): void {
    const entriesPath = path.join(this.config.data_dir, 'cost_entries.json');
    const alertsPath = path.join(this.config.data_dir, 'alerts.json');

    fs.writeFileSync(entriesPath, JSON.stringify(this.entries, null, 2));
    fs.writeFileSync(alertsPath, JSON.stringify(this.alerts, null, 2));
  }

  /**
   * Generate entry ID
   */
  private generateEntryId(): string {
    return `cost_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
  }

  /**
   * Generate alert ID
   */
  private generateAlertId(): string {
    return `alert_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
  }

  // -------------------------------------------------------------------------
  // Session Tracking
  // -------------------------------------------------------------------------

  /**
   * Start tracking a pod/job session
   */
  startSession(
    sessionId: string,
    podId: string,
    gpuType: string,
    costPerHour: number,
    jobId?: string,
    project?: string
  ): void {
    this.activeSessions.set(sessionId, {
      start_time: new Date(),
      pod_id: podId,
      cost_per_hour: costPerHour,
      job_id: jobId,
      project: project,
      gpu_type: gpuType,
    });

    logger.info('Cost tracking session started', {
      sessionId,
      podId,
      gpuType,
      costPerHour,
    });
  }

  /**
   * End a tracking session and record costs
   */
  endSession(
    sessionId: string,
    status: CostEntry['status'] = 'completed',
    gpuUtilization?: number,
    memoryUtilization?: number
  ): CostEntry | null {
    const session = this.activeSessions.get(sessionId);

    if (!session) {
      logger.warn('No active session found', { sessionId });
      return null;
    }

    const endTime = new Date();
    const durationSeconds = (endTime.getTime() - session.start_time.getTime()) / 1000;
    const durationHours = durationSeconds / 3600;
    const totalCost = durationHours * session.cost_per_hour;

    const entry: CostEntry = {
      id: this.generateEntryId(),
      timestamp: endTime.toISOString(),
      pod_id: session.pod_id,
      job_id: session.job_id,
      project: session.project,
      duration_seconds: Math.round(durationSeconds),
      duration_hours: Math.round(durationHours * 100) / 100,
      gpu_type: session.gpu_type,
      cost_per_hour: session.cost_per_hour,
      total_cost: Math.round(totalCost * 100) / 100,
      gpu_utilization_avg: gpuUtilization,
      memory_utilization_avg: memoryUtilization,
      status,
    };

    this.entries.push(entry);
    this.activeSessions.delete(sessionId);
    this.saveData();

    // Check for alerts
    this.checkBudgetAlerts();

    logger.info('Cost tracking session ended', {
      sessionId,
      duration: durationHours.toFixed(2),
      cost: totalCost.toFixed(2),
    });

    return entry;
  }

  /**
   * Get current running cost for a session
   */
  getCurrentSessionCost(sessionId: string): { hours: number; cost: number } | null {
    const session = this.activeSessions.get(sessionId);

    if (!session) {
      return null;
    }

    const now = new Date();
    const durationHours = (now.getTime() - session.start_time.getTime()) / 3600000;
    const cost = durationHours * session.cost_per_hour;

    return {
      hours: Math.round(durationHours * 100) / 100,
      cost: Math.round(cost * 100) / 100,
    };
  }

  /**
   * Get all active sessions
   */
  getActiveSessions(): Array<{
    session_id: string;
    pod_id: string;
    job_id?: string;
    gpu_type: string;
    running_hours: number;
    current_cost: number;
  }> {
    const sessions: Array<{
      session_id: string;
      pod_id: string;
      job_id?: string;
      gpu_type: string;
      running_hours: number;
      current_cost: number;
    }> = [];

    const now = new Date();

    for (const [sessionId, session] of this.activeSessions) {
      const durationHours = (now.getTime() - session.start_time.getTime()) / 3600000;
      const cost = durationHours * session.cost_per_hour;

      sessions.push({
        session_id: sessionId,
        pod_id: session.pod_id,
        job_id: session.job_id,
        gpu_type: session.gpu_type,
        running_hours: Math.round(durationHours * 100) / 100,
        current_cost: Math.round(cost * 100) / 100,
      });
    }

    return sessions;
  }

  // -------------------------------------------------------------------------
  // Cost Analysis
  // -------------------------------------------------------------------------

  /**
   * Get cost summary for a time period
   */
  getSummary(startDate?: Date, endDate?: Date, project?: string): CostSummary {
    let filteredEntries = this.entries;

    // Filter by date range
    if (startDate) {
      filteredEntries = filteredEntries.filter((e) => new Date(e.timestamp) >= startDate);
    }
    if (endDate) {
      filteredEntries = filteredEntries.filter((e) => new Date(e.timestamp) <= endDate);
    }

    // Filter by project
    if (project) {
      filteredEntries = filteredEntries.filter((e) => e.project === project);
    }

    const summary: CostSummary = {
      total_cost: 0,
      total_hours: 0,
      entries_count: filteredEntries.length,
      by_pod: {},
      by_job: {},
      by_project: {},
      by_gpu_type: {},
      by_status: {},
      daily_costs: [],
      hourly_costs: [],
      avg_gpu_utilization: 0,
      avg_cost_per_hour: 0,
      estimated_monthly_cost: 0,
    };

    const dailyMap = new Map<string, { cost: number; hours: number }>();
    const hourlyMap = new Map<string, number>();
    let totalUtilization = 0;
    let utilizationCount = 0;

    for (const entry of filteredEntries) {
      summary.total_cost += entry.total_cost;
      summary.total_hours += entry.duration_hours;

      // By pod
      summary.by_pod[entry.pod_id] = (summary.by_pod[entry.pod_id] || 0) + entry.total_cost;

      // By job
      if (entry.job_id) {
        summary.by_job[entry.job_id] = (summary.by_job[entry.job_id] || 0) + entry.total_cost;
      }

      // By project
      if (entry.project) {
        summary.by_project[entry.project] =
          (summary.by_project[entry.project] || 0) + entry.total_cost;
      }

      // By GPU type
      summary.by_gpu_type[entry.gpu_type] =
        (summary.by_gpu_type[entry.gpu_type] || 0) + entry.total_cost;

      // By status
      summary.by_status[entry.status] = (summary.by_status[entry.status] || 0) + entry.total_cost;

      // Daily
      const date = new Date(entry.timestamp).toISOString().split('T')[0];
      const daily = dailyMap.get(date) || { cost: 0, hours: 0 };
      daily.cost += entry.total_cost;
      daily.hours += entry.duration_hours;
      dailyMap.set(date, daily);

      // Hourly
      const hour = new Date(entry.timestamp).toISOString().substring(0, 13);
      hourlyMap.set(hour, (hourlyMap.get(hour) || 0) + entry.total_cost);

      // Utilization
      if (entry.gpu_utilization_avg !== undefined) {
        totalUtilization += entry.gpu_utilization_avg;
        utilizationCount++;
      }
    }

    // Convert maps to arrays
    summary.daily_costs = Array.from(dailyMap.entries())
      .map(([date, data]) => ({
        date,
        cost: Math.round(data.cost * 100) / 100,
        hours: Math.round(data.hours * 100) / 100,
      }))
      .sort((a, b) => a.date.localeCompare(b.date));

    summary.hourly_costs = Array.from(hourlyMap.entries())
      .map(([hour, cost]) => ({
        hour,
        cost: Math.round(cost * 100) / 100,
      }))
      .sort((a, b) => a.hour.localeCompare(b.hour));

    // Calculate averages
    summary.avg_gpu_utilization =
      utilizationCount > 0 ? Math.round(totalUtilization / utilizationCount) : 0;

    summary.avg_cost_per_hour =
      summary.total_hours > 0
        ? Math.round((summary.total_cost / summary.total_hours) * 100) / 100
        : 0;

    // Round totals
    summary.total_cost = Math.round(summary.total_cost * 100) / 100;
    summary.total_hours = Math.round(summary.total_hours * 100) / 100;

    // Estimate monthly cost based on recent usage
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
    const recentEntries = filteredEntries.filter((e) => new Date(e.timestamp) >= thirtyDaysAgo);
    const recentCost = recentEntries.reduce((sum, e) => sum + e.total_cost, 0);
    summary.estimated_monthly_cost = Math.round(recentCost * 100) / 100;

    return summary;
  }

  /**
   * Get today's costs
   */
  getTodayCost(): number {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const todayEntries = this.entries.filter((e) => new Date(e.timestamp) >= today);

    // Include running sessions
    let runningCost = 0;
    for (const session of this.activeSessions.values()) {
      const hours = (Date.now() - session.start_time.getTime()) / 3600000;
      runningCost += hours * session.cost_per_hour;
    }

    const completedCost = todayEntries.reduce((sum, e) => sum + e.total_cost, 0);
    return Math.round((completedCost + runningCost) * 100) / 100;
  }

  /**
   * Get this week's costs
   */
  getWeekCost(): number {
    const weekStart = new Date();
    weekStart.setDate(weekStart.getDate() - weekStart.getDay());
    weekStart.setHours(0, 0, 0, 0);

    const weekEntries = this.entries.filter((e) => new Date(e.timestamp) >= weekStart);

    return Math.round(weekEntries.reduce((sum, e) => sum + e.total_cost, 0) * 100) / 100;
  }

  /**
   * Get this month's costs
   */
  getMonthCost(): number {
    const monthStart = new Date();
    monthStart.setDate(1);
    monthStart.setHours(0, 0, 0, 0);

    const monthEntries = this.entries.filter((e) => new Date(e.timestamp) >= monthStart);

    return Math.round(monthEntries.reduce((sum, e) => sum + e.total_cost, 0) * 100) / 100;
  }

  // -------------------------------------------------------------------------
  // Alerts
  // -------------------------------------------------------------------------

  /**
   * Check budget thresholds and create alerts
   */
  private checkBudgetAlerts(): void {
    const { budget } = this.config;

    // Daily budget
    if (budget.daily_limit) {
      const todayCost = this.getTodayCost();
      this.checkThreshold(todayCost, budget.daily_limit, 'daily', budget.alert_thresholds);
    }

    // Weekly budget
    if (budget.weekly_limit) {
      const weekCost = this.getWeekCost();
      this.checkThreshold(weekCost, budget.weekly_limit, 'weekly', budget.alert_thresholds);
    }

    // Monthly budget
    if (budget.monthly_limit) {
      const monthCost = this.getMonthCost();
      this.checkThreshold(monthCost, budget.monthly_limit, 'monthly', budget.alert_thresholds);
    }
  }

  /**
   * Check threshold and create alert if needed
   */
  private checkThreshold(
    currentCost: number,
    limit: number,
    period: string,
    thresholds: number[]
  ): void {
    const percentage = currentCost / limit;

    for (const threshold of thresholds.sort((a, b) => b - a)) {
      if (percentage >= threshold) {
        const alertId = `${period}_${threshold}_${new Date().toISOString().split('T')[0]}`;

        // Check if alert already exists for today
        const existingAlert = this.alerts.find(
          (a) =>
            a.id.startsWith(alertId.split('_').slice(0, 2).join('_')) &&
            a.timestamp.startsWith(new Date().toISOString().split('T')[0])
        );

        if (!existingAlert) {
          const severity: CostAlert['severity'] =
            threshold >= 1.0 ? 'critical' : threshold >= 0.9 ? 'warning' : 'info';

          this.createAlert(
            'budget_threshold',
            severity,
            `${period.charAt(0).toUpperCase() + period.slice(1)} budget at ${Math.round(percentage * 100)}% ($${currentCost.toFixed(2)} / $${limit.toFixed(2)})`,
            { period, threshold, current_cost: currentCost, limit }
          );
        }
        break; // Only create one alert for highest threshold crossed
      }
    }
  }

  /**
   * Create a new alert
   */
  createAlert(
    type: CostAlert['type'],
    severity: CostAlert['severity'],
    message: string,
    metadata?: Record<string, unknown>
  ): CostAlert {
    const alert: CostAlert = {
      id: this.generateAlertId(),
      type,
      severity,
      message,
      timestamp: new Date().toISOString(),
      acknowledged: false,
      metadata,
    };

    this.alerts.push(alert);
    this.saveData();

    logger.warn('Cost alert created', { alert });

    return alert;
  }

  /**
   * Get active alerts
   */
  getAlerts(unacknowledgedOnly: boolean = false): CostAlert[] {
    if (unacknowledgedOnly) {
      return this.alerts.filter((a) => !a.acknowledged);
    }
    return this.alerts;
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string): boolean {
    const alert = this.alerts.find((a) => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      this.saveData();
      return true;
    }
    return false;
  }

  /**
   * Clear old alerts
   */
  clearOldAlerts(daysToKeep: number = 30): number {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - daysToKeep);

    const originalCount = this.alerts.length;
    this.alerts = this.alerts.filter((a) => new Date(a.timestamp) >= cutoff);
    this.saveData();

    return originalCount - this.alerts.length;
  }

  // -------------------------------------------------------------------------
  // Dashboard Data
  // -------------------------------------------------------------------------

  /**
   * Get dashboard data for display
   */
  getDashboard(): {
    current_sessions: Array<{
      session_id: string;
      pod_id: string;
      job_id?: string;
      gpu_type: string;
      running_hours: number;
      current_cost: number;
    }>;
    today_cost: number;
    week_cost: number;
    month_cost: number;
    budget_status: {
      daily: { used: number; limit: number; percent: number } | null;
      weekly: { used: number; limit: number; percent: number } | null;
      monthly: { used: number; limit: number; percent: number } | null;
    };
    active_alerts: CostAlert[];
    recent_entries: CostEntry[];
    summary: CostSummary;
  } {
    const todayCost = this.getTodayCost();
    const weekCost = this.getWeekCost();
    const monthCost = this.getMonthCost();
    const { budget } = this.config;

    return {
      current_sessions: this.getActiveSessions(),
      today_cost: todayCost,
      week_cost: weekCost,
      month_cost: monthCost,
      budget_status: {
        daily: budget.daily_limit
          ? {
              used: todayCost,
              limit: budget.daily_limit,
              percent: Math.round((todayCost / budget.daily_limit) * 100),
            }
          : null,
        weekly: budget.weekly_limit
          ? {
              used: weekCost,
              limit: budget.weekly_limit,
              percent: Math.round((weekCost / budget.weekly_limit) * 100),
            }
          : null,
        monthly: budget.monthly_limit
          ? {
              used: monthCost,
              limit: budget.monthly_limit,
              percent: Math.round((monthCost / budget.monthly_limit) * 100),
            }
          : null,
      },
      active_alerts: this.getAlerts(true),
      recent_entries: this.entries.slice(-10).reverse(),
      summary: this.getSummary(),
    };
  }

  /**
   * Export cost data to CSV
   */
  exportToCSV(startDate?: Date, endDate?: Date): string {
    let entries = this.entries;

    if (startDate) {
      entries = entries.filter((e) => new Date(e.timestamp) >= startDate);
    }
    if (endDate) {
      entries = entries.filter((e) => new Date(e.timestamp) <= endDate);
    }

    const headers = [
      'timestamp',
      'pod_id',
      'job_id',
      'project',
      'gpu_type',
      'duration_hours',
      'cost_per_hour',
      'total_cost',
      'gpu_utilization',
      'status',
    ];

    const rows = entries.map((e) => [
      e.timestamp,
      e.pod_id,
      e.job_id || '',
      e.project || '',
      e.gpu_type,
      e.duration_hours.toString(),
      e.cost_per_hour.toString(),
      e.total_cost.toString(),
      e.gpu_utilization_avg?.toString() || '',
      e.status,
    ]);

    return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
  }

  /**
   * Update budget configuration
   */
  updateBudget(budget: Partial<CostBudget>): void {
    this.config.budget = { ...this.config.budget, ...budget };
    logger.info('Budget updated', { budget: this.config.budget });
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let costTracker: CostTracker | null = null;

export function getCostTracker(config?: Partial<CostTrackerConfig>): CostTracker {
  if (!costTracker) {
    costTracker = new CostTracker(config);
  }
  return costTracker;
}

export function resetCostTracker(): void {
  costTracker = null;
}
