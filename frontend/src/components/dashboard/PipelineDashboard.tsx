// VP Investments - Enhanced Pipeline Dashboard
// Modern dashboard component showing real-time trading data from all sources

'use client';

import { useState } from 'react';
import { 
  RefreshCw, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  AlertCircle, 
  Clock,
  Database,
  Activity,
  DollarSign,
  BarChart3,
  Zap
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Input } from '@/components/ui/input';
import { useDashboardData, useDataSources, useFetchPipelineData } from '@/hooks/usePipelineData';
import { cn } from '@/lib/utils';

export function PipelineDashboard() {
  const [tickerInput, setTickerInput] = useState('AAPL,TSLA,GOOGL,MSFT,AMZN');
  
  // React Query hooks
  const { data: dashboardData, isLoading, error, refetch } = useDashboardData(tickerInput);
  const { data: dataSourcesInfo } = useDataSources();
  const fetchPipelineData = useFetchPipelineData();

  // Handle manual data refresh
  const handleRefresh = async () => {
    const tickers = tickerInput.split(',').map(t => t.trim().toUpperCase()).filter(Boolean);
    try {
      await fetchPipelineData.mutateAsync(tickers);
    } catch (error) {
      console.error('Failed to fetch pipeline data:', error);
      // Fallback to regular refetch
      refetch();
    }
  };

  const handleTickerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTickerInput(e.target.value);
  };

  const handleTickerSubmit = () => {
    refetch();
  };

  const getRecommendationIcon = (recommendation?: string) => {
    switch (recommendation) {
      case 'BUY':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'SELL':
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      case 'HOLD':
        return <Minus className="h-4 w-4 text-yellow-600" />;
      default:
        return <Minus className="h-4 w-4 text-gray-400" />;
    }
  };

  const getRecommendationBadgeVariant = (recommendation?: string) => {
    switch (recommendation) {
      case 'BUY':
        return 'default'; // Green
      case 'SELL':
        return 'destructive'; // Red  
      case 'HOLD':
        return 'secondary'; // Gray
      default:
        return 'outline';
    }
  };

  const getConfidenceColor = (confidence?: number) => {
    if (!confidence) return 'text-gray-400';
    if (confidence >= 0.8) return 'text-green-600 font-semibold';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-500';
  };

  const formatPrice = (price?: number) => {
    if (price === undefined || price === null) return 'N/A';
    return `$${price.toFixed(2)}`;
  };

  const formatPercent = (percent?: number) => {
    if (percent === undefined || percent === null) return 'N/A';
    const sign = percent >= 0 ? '+' : '';
    const color = percent >= 0 ? 'text-green-600' : 'text-red-600';
    return <span className={color}>{sign}{percent.toFixed(2)}%</span>;
  };

  if (error) {
    return (
      <Card className="max-w-2xl mx-auto mt-8">
        <CardContent className="pt-6">
          <div className="flex items-center space-x-2 text-red-600">
            <AlertCircle className="h-5 w-5" />
            <span>Failed to load dashboard data. Please check if the API server is running on port 8000.</span>
          </div>
          <Button 
            onClick={() => refetch()} 
            className="mt-4"
            variant="outline"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">VP Investments</h1>
          <p className="text-gray-500">Real-Time Trading Dashboard</p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="flex items-center space-x-1">
            <Activity className="h-3 w-3" />
            <span>{dashboardData?.system_status.pipeline_online ? 'Online' : 'Offline'}</span>
          </Badge>
          <Button
            onClick={handleRefresh}
            disabled={fetchPipelineData.isPending || isLoading}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={cn(
              "h-4 w-4", 
              (fetchPipelineData.isPending || isLoading) && "animate-spin"
            )} />
            <span>Refresh Data</span>
          </Button>
        </div>
      </div>

      {/* Ticker Input */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5" />
            <span>Ticker Selection</span>
          </CardTitle>
          <CardDescription>
            Enter comma-separated ticker symbols to analyze
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-2">
            <Input
              placeholder="AAPL,TSLA,GOOGL,MSFT,AMZN"
              value={tickerInput}
              onChange={handleTickerChange}
              onKeyPress={(e) => e.key === 'Enter' && handleTickerSubmit()}
              className="flex-1"
            />
            <Button onClick={handleTickerSubmit} variant="outline">
              Update
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Summary Cards */}
      {dashboardData && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center space-x-2">
                <Database className="h-4 w-4" />
                <span>Total Tickers</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboardData.summary.total_tickers}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-green-600" />
                <span>Buy Signals</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {dashboardData.summary.buy_recommendations}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center space-x-2">
                <TrendingDown className="h-4 w-4 text-red-600" />
                <span>Sell Signals</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">
                {dashboardData.summary.sell_recommendations}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center space-x-2">
                <Minus className="h-4 w-4 text-yellow-600" />
                <span>Hold Signals</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-600">
                {dashboardData.summary.hold_recommendations}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Data Sources Status */}
      {dataSourcesInfo && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Data Sources</span>
            </CardTitle>
            <CardDescription>
              Real-time status of all data collection sources
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {dataSourcesInfo.sources.map((source) => (
                <div key={source.name} className="flex items-center space-x-2">
                  <div className={cn(
                    "h-3 w-3 rounded-full",
                    source.enabled ? "bg-green-500" : "bg-gray-300"
                  )} />
                  <span className="text-sm capitalize">{source.name}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Ticker Data Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <DollarSign className="h-5 w-5" />
            <span>Trading Recommendations</span>
          </CardTitle>
          <CardDescription>
            {isLoading ? 'Loading ticker data...' : 
             `${dashboardData?.tickers.length || 0} tickers analyzed`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-6 w-6 animate-spin" />
              <span className="ml-2">Loading data from all sources...</span>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Ticker</TableHead>
                  <TableHead>Company</TableHead>
                  <TableHead>Price</TableHead>
                  <TableHead>Change</TableHead>
                  <TableHead>Volume</TableHead>
                  <TableHead>Recommendation</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Updated</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dashboardData?.tickers.map((ticker, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{ticker.ticker}</TableCell>
                    <TableCell>{ticker.company_name || 'N/A'}</TableCell>
                    <TableCell>{formatPrice(ticker.current_price)}</TableCell>
                    <TableCell>{formatPercent(ticker.price_change_percent)}</TableCell>
                    <TableCell>
                      {ticker.volume ? ticker.volume.toLocaleString() : 'N/A'}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        {getRecommendationIcon(ticker.recommendation)}
                        <Badge variant={getRecommendationBadgeVariant(ticker.recommendation)}>
                          {ticker.recommendation || 'N/A'}
                        </Badge>
                      </div>
                    </TableCell>
                    <TableCell className={getConfidenceColor(ticker.confidence)}>
                      {ticker.confidence ? `${(ticker.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-1 text-sm text-gray-500">
                        <Clock className="h-3 w-3" />
                        <span>{new Date(ticker.last_updated).toLocaleTimeString()}</span>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
                {(!dashboardData?.tickers.length && !isLoading) && (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center text-gray-500 py-8">
                      No ticker data available. Click &ldquo;Refresh Data&rdquo; to fetch latest information.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* System Status Footer */}
      {dashboardData && (
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between text-sm text-gray-500">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1">
                  <Activity className="h-3 w-3" />
                  <span>API: {dashboardData.system_status.api_online ? 'Online' : 'Offline'}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Database className="h-3 w-3" />
                  <span>Pipeline: {dashboardData.system_status.pipeline_online ? 'Online' : 'Offline'}</span>
                </div>
              </div>
              <div className="flex items-center space-x-1">
                <Clock className="h-3 w-3" />
                <span>Last updated: {new Date(dashboardData.system_status.last_updated).toLocaleString()}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}