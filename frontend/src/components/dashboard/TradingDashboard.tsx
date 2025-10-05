// VP Investments - Trading Signals Dashboard Component

'use client';

import { useState } from 'react';
import { RefreshCw, TrendingUp, TrendingDown, Minus, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { useSignals, useRefreshSignals } from '@/hooks/useSignals';
import { useSignalsStats } from '@/hooks/usePipelineData';
import { DashboardFilters } from '@/types/api';
import { cn } from '@/lib/utils';

export function TradingDashboard() {
  const [filters, setFilters] = useState<DashboardFilters>({
    signal_type: 'ALL',
  });
  
  const [tickerInput, setTickerInput] = useState('');
  
  // React Query hooks
  const { data: signals = [], isLoading, error, refetch } = useSignals(filters);
  const { data: stats } = useSignalsStats();
  const refreshMutation = useRefreshSignals();

  // Handle filter changes
  const handleFilterChange = (key: keyof DashboardFilters, value: string | number | string[] | undefined) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const handleTickerFilter = () => {
    if (tickerInput.trim()) {
      const tickers = tickerInput.split(',').map(t => t.trim().toUpperCase());
      handleFilterChange('tickers', tickers);
    } else {
      handleFilterChange('tickers', undefined);
    }
  };

  const getSignalIcon = (signalType: string) => {
    switch (signalType) {
      case 'BUY':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'SELL':
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      case 'HOLD':
        return <Minus className="h-4 w-4 text-yellow-600" />;
      default:
        return <Minus className="h-4 w-4" />;
    }
  };

  const getSignalBadgeVariant = (signalType: string) => {
    switch (signalType) {
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

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 font-semibold';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-500';
  };

  if (error) {
    return (
      <Card className="max-w-2xl mx-auto mt-8">
        <CardContent className="pt-6">
          <div className="flex items-center space-x-2 text-red-600">
            <AlertCircle className="h-5 w-5" />
            <span>Failed to load trading signals. Please check if the backend is running.</span>
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
          <p className="text-gray-500">Trading Signals Dashboard</p>
        </div>
        <Button
          onClick={() => refreshMutation.mutate()}
          disabled={refreshMutation.isPending}
          className="flex items-center space-x-2"
        >
          <RefreshCw className={cn("h-4 w-4", refreshMutation.isPending && "animate-spin")} />
          <span>Refresh Signals</span>
        </Button>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Total Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_signals}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Buy Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{stats.buy_signals}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Sell Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{stats.sell_signals}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(stats.avg_confidence * 100).toFixed(1)}%</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
          <CardDescription>Filter and search trading signals</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Signal Type</label>
              <Select
                value={filters.signal_type || 'ALL'}
                onValueChange={(value) => handleFilterChange('signal_type', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ALL">All Signals</SelectItem>
                  <SelectItem value="BUY">Buy Only</SelectItem>
                  <SelectItem value="SELL">Sell Only</SelectItem>
                  <SelectItem value="HOLD">Hold Only</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Min Confidence</label>
              <Select
                value={filters.min_confidence?.toString() || '0'}
                onValueChange={(value) => handleFilterChange('min_confidence', parseFloat(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">Any Confidence</SelectItem>
                  <SelectItem value="0.5">50%+</SelectItem>
                  <SelectItem value="0.7">70%+</SelectItem>
                  <SelectItem value="0.8">80%+</SelectItem>
                  <SelectItem value="0.9">90%+</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Tickers</label>
              <div className="flex space-x-2">
                <Input
                  placeholder="AAPL,MSFT,GOOGL"
                  value={tickerInput}
                  onChange={(e) => setTickerInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleTickerFilter()}
                />
                <Button onClick={handleTickerFilter} variant="outline">
                  Filter
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Signals Table */}
      <Card>
        <CardHeader>
          <CardTitle>Trading Signals</CardTitle>
          <CardDescription>
            {isLoading ? 'Loading signals...' : `${signals.length} signals found`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-6 w-6 animate-spin" />
              <span className="ml-2">Loading signals...</span>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Ticker</TableHead>
                  <TableHead>Signal</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Technical</TableHead>
                  <TableHead>Sentiment</TableHead>
                  <TableHead>Combined</TableHead>
                  <TableHead>Date</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {signals.map((signal, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{signal.ticker}</TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        {getSignalIcon(signal.signal_type)}
                        <Badge variant={getSignalBadgeVariant(signal.signal_type)}>
                          {signal.signal_type}
                        </Badge>
                      </div>
                    </TableCell>
                    <TableCell className={getConfidenceColor(signal.confidence)}>
                      {(signal.confidence * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell>{signal.technical_score.toFixed(2)}</TableCell>
                    <TableCell>{signal.sentiment_score.toFixed(2)}</TableCell>
                    <TableCell>{signal.combined_score.toFixed(2)}</TableCell>
                    <TableCell>
                      {new Date(signal.created_at).toLocaleDateString()}
                    </TableCell>
                  </TableRow>
                ))}
                {signals.length === 0 && !isLoading && (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center text-gray-500 py-8">
                      No signals found. Try adjusting your filters or refresh the data.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}