'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Activity, 
  Cpu, 
  HardDrive, 
  MemoryStick, 
  TrendingUp, 
  AlertCircle,
  CheckCircle,
  Database,
  Clock,
  Zap,
  RefreshCw,
  Loader2
} from 'lucide-react';
import { toast } from 'sonner';

interface SystemHealth {
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  disk_percent: number;
  disk_used_gb: number;
  disk_total_gb: number;
  uptime_seconds: number;
  timestamp: string;
}

interface FactorQuality {
  total_factors: number;
  success_rate: number;
  avg_calculation_time_ms: number;
  failed_factors: string[];
  recent_runs: Array<{
    timestamp: string;
    success_rate: number;
    total_factors: number;
  }>;
}

interface PipelineMetrics {
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  avg_tickers_per_run: number;
  avg_signals_per_run: number;
  avg_runtime_minutes: number;
  last_run_time: string | null;
  runs_last_24h: number;
}

interface StorageMetrics {
  total_pipeline_runs: number;
  total_signals: number;
  total_analytics: number;
  database_size_mb: number;
  table_sizes: { [table: string]: number };
}

interface DashboardData {
  system_health: SystemHealth;
  factor_quality: FactorQuality;
  pipeline_metrics: PipelineMetrics;
  storage_metrics: StorageMetrics;
}

export default function AdminDashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchDashboardData = async () => {
    try {
      const token = localStorage.getItem('admin_token');
      const response = await fetch('http://127.0.0.1:8000/api/monitoring/dashboard', {
        headers: token ? { 'Authorization': `Bearer ${token}` } : {}
      });
      
      if (response.status === 401) {
        toast.error('Session expired. Please login again.');
        window.location.href = '/admin/login';
        return;
      }
      
      if (!response.ok) {
        throw new Error('Failed to fetch dashboard data');
      }
      
      const dashboardData = await response.json();
      setData(dashboardData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching dashboard:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchDashboardData, 30000); // 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const formatRelativeTime = (timestamp: string | null) => {
    if (!timestamp) return 'Never';
    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diff = now.getTime() - date.getTime();
      const minutes = Math.floor(diff / 60000);
      const hours = Math.floor(diff / 3600000);
      const days = Math.floor(diff / 86400000);
      
      if (minutes < 1) return 'Just now';
      if (minutes < 60) return `${minutes}m ago`;
      if (hours < 24) return `${hours}h ago`;
      return `${days}d ago`;
    } catch {
      return timestamp;
    }
  };

  const getHealthColor = (percent: number) => {
    if (percent < 60) return 'text-green-500';
    if (percent < 80) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getHealthStatus = (percent: number) => {
    if (percent < 60) return { status: 'Healthy', icon: CheckCircle, color: 'text-green-500' };
    if (percent < 80) return { status: 'Warning', icon: AlertCircle, color: 'text-yellow-500' };
    return { status: 'Critical', icon: AlertCircle, color: 'text-red-500' };
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin" />
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardContent className="pt-6 pb-6 text-center text-muted-foreground">
            <AlertCircle className="w-12 h-12 mx-auto mb-4" />
            <p>Failed to load dashboard data</p>
            <Button onClick={fetchDashboardData} className="mt-4">
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const cpuStatus = getHealthStatus(data.system_health.cpu_percent);
  const memStatus = getHealthStatus(data.system_health.memory_percent);
  const diskStatus = getHealthStatus(data.system_health.disk_percent);

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">System Dashboard</h1>
            <p className="text-muted-foreground mt-2">
              Monitor system health, pipeline performance, and factor quality
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-sm text-muted-foreground">
              Last updated: {formatRelativeTime(lastUpdate.toISOString())}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setLoading(true);
                fetchDashboardData();
              }}
              disabled={loading}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button
              variant={autoRefresh ? 'default' : 'outline'}
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              <Activity className="w-4 h-4 mr-2" />
              Auto-refresh {autoRefresh ? 'On' : 'Off'}
            </Button>
          </div>
        </div>
      </div>

      {/* System Health Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          System Health
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4" />
                  CPU Usage
                </div>
                <cpuStatus.icon className={`w-4 h-4 ${cpuStatus.color}`} />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-3xl font-bold ${getHealthColor(data.system_health.cpu_percent)}`}>
                {data.system_health.cpu_percent.toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground mt-2">{cpuStatus.status}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MemoryStick className="w-4 h-4" />
                  Memory Usage
                </div>
                <memStatus.icon className={`w-4 h-4 ${memStatus.color}`} />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-3xl font-bold ${getHealthColor(data.system_health.memory_percent)}`}>
                {data.system_health.memory_percent.toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {data.system_health.memory_used_gb.toFixed(1)} / {data.system_health.memory_total_gb.toFixed(1)} GB
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <HardDrive className="w-4 h-4" />
                  Disk Usage
                </div>
                <diskStatus.icon className={`w-4 h-4 ${diskStatus.color}`} />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-3xl font-bold ${getHealthColor(data.system_health.disk_percent)}`}>
                {data.system_health.disk_percent.toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {data.system_health.disk_used_gb.toFixed(0)} / {data.system_health.disk_total_gb.toFixed(0)} GB
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Pipeline Metrics Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Pipeline Performance
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Database className="w-4 h-4" />
                Total Runs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{data.pipeline_metrics.total_runs}</div>
              <p className="text-xs text-muted-foreground mt-2">
                {data.pipeline_metrics.runs_last_24h} in last 24h
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                Success Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-500">
                {((data.pipeline_metrics.successful_runs / data.pipeline_metrics.total_runs) * 100).toFixed(0)}%
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {data.pipeline_metrics.successful_runs} / {data.pipeline_metrics.total_runs} runs
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Avg Signals
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{data.pipeline_metrics.avg_signals_per_run.toFixed(0)}</div>
              <p className="text-xs text-muted-foreground mt-2">
                per run
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Clock className="w-4 h-4" />
                Last Run
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-lg font-bold">
                {formatRelativeTime(data.pipeline_metrics.last_run_time)}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Avg: {data.pipeline_metrics.avg_tickers_per_run.toFixed(0)} tickers
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Factor Quality Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <CheckCircle className="w-5 h-5" />
          Factor Quality
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="text-base">Success Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-bold text-green-500 mb-4">
                {(data.factor_quality.success_rate * 100).toFixed(1)}%
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Factors:</span>
                  <span className="font-medium">{data.factor_quality.total_factors}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Avg Time:</span>
                  <span className="font-medium">{data.factor_quality.avg_calculation_time_ms.toFixed(1)}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Failed:</span>
                  <span className="font-medium text-red-500">
                    {data.factor_quality.failed_factors.length}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-base">Recent Runs</CardTitle>
              <CardDescription>Factor success rate over last 5 runs</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {data.factor_quality.recent_runs.map((run, idx) => {
                  const successPercent = (run.success_rate * 100).toFixed(1);
                  const getWidthClass = (rate: number) => {
                    const percent = rate * 100;
                    if (percent >= 95) return 'w-full';
                    if (percent >= 90) return 'w-11/12';
                    if (percent >= 80) return 'w-5/6';
                    if (percent >= 70) return 'w-3/4';
                    if (percent >= 60) return 'w-2/3';
                    if (percent >= 50) return 'w-1/2';
                    return 'w-1/3';
                  };
                  
                  return (
                    <div key={idx} className="flex items-center gap-3">
                      <div className="text-xs text-muted-foreground w-32">
                        {formatRelativeTime(run.timestamp)}
                      </div>
                      <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden relative">
                        <div className={`h-full bg-green-500 transition-all ${getWidthClass(run.success_rate)}`} />
                      </div>
                      <div className="text-sm font-medium w-16 text-right">
                        {successPercent}%
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Storage Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Database className="w-5 h-5" />
          Storage Metrics
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Pipeline Runs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{data.storage_metrics.total_pipeline_runs}</div>
              <p className="text-xs text-muted-foreground mt-2">Total runs stored</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{data.storage_metrics.total_signals.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-2">Total signals</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Analytics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{data.storage_metrics.total_analytics.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-2">Analytics records</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Database Size</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{data.storage_metrics.database_size_mb.toFixed(1)}</div>
              <p className="text-xs text-muted-foreground mt-2">MB (estimated)</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
