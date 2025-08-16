'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ScoreDistributionChart, TradeTypeChart } from '@/components/Charts';
import { StocksTable } from '@/components/StocksTable';
import { StockData, DashboardStats } from '@/types/stock';
import { TrendingUp, Users, DollarSign, Target, RefreshCw } from 'lucide-react';
import Image from 'next/image';

export default function Dashboard() {
  const [stocks, setStocks] = useState<StockData[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  // Load from Next.js API which proxies Python Flask backend
  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const resp = await fetch('/api/stocks', { cache: 'no-store' });
        const json = await resp.json();
        const data: StockData[] = json.data || [];
        setStocks(data);
        setLastUpdated(new Date().toLocaleString());

        const totalStocks = data.length || 1;
        const avgWeightedScore = data.reduce((sum, s) => sum + (s.weighted_score || 0), 0) / totalStocks;
        const topScore = data.reduce((max, s) => Math.max(max, s.weighted_score || 0), 0);
        const avgSentiment = data.reduce((sum, s) => sum + (s.sentiment || 0), 0) / totalStocks;
        const totalMentions = data.reduce((sum, s) => sum + (s.mentions || 0), 0);
        const tradeTypeBreakdown: Record<string, number> = {};
        data.forEach(s => { tradeTypeBreakdown[s.trade_type] = (tradeTypeBreakdown[s.trade_type] || 0) + 1; });
        setStats({ totalStocks: data.length, avgWeightedScore, topScore, avgSentiment, totalMentions, tradeTypeBreakdown });
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const refresh = () => {
    // Re-run the fetch
    (async () => {
      setLoading(true);
      try {
        const resp = await fetch('/api/stocks', { cache: 'no-store' });
        const json = await resp.json();
        const data: StockData[] = json.data || [];
        setStocks(data);
        setLastUpdated(new Date().toLocaleString());
      } finally {
        setLoading(false);
      }
    })();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading VP Investments analysis...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">VP Investments</h1>
              <p className="text-gray-600 mt-1">AI-Powered Stock Analysis Dashboard</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-500">
                Last updated: {lastUpdated}
              </div>
              <Button onClick={refresh} disabled={loading}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats Overview */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Stocks</CardTitle>
                <Target className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.totalStocks}</div>
                <p className="text-xs text-muted-foreground">Analyzed today</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avg Score</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.avgWeightedScore.toFixed(1)}</div>
                <p className="text-xs text-muted-foreground">Weighted average</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Top Score</CardTitle>
                <DollarSign className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.topScore.toFixed(1)}</div>
                <p className="text-xs text-muted-foreground">Best opportunity</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Mentions</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.totalMentions}</div>
                <p className="text-xs text-muted-foreground">Reddit discussions</p>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Charts */}
        {stats && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <Card>
              <CardHeader>
                <CardTitle>Score Distribution</CardTitle>
                <CardDescription>Distribution of weighted scores across all stocks</CardDescription>
              </CardHeader>
              <CardContent>
                <ScoreDistributionChart stocks={stocks} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Trade Types</CardTitle>
                <CardDescription>Breakdown by trading strategy</CardDescription>
              </CardHeader>
              <CardContent>
                <TradeTypeChart stats={stats} />
              </CardContent>
            </Card>
          </div>
        )}

        {/* Stocks Table */}
        <StocksTable stocks={stocks} />

        {/* Footer */}

        <div className="mt-12 text-center text-gray-500 text-sm">
          <p>VP Investments Dashboard - Powered by Reddit sentiment, technical analysis, and fundamental data</p>
        </div>
        <footer className="row-start-3 flex gap-[24px] flex-wrap items-center justify-center">
          <a
            className="flex items-center gap-2 hover:underline hover:underline-offset-4"
            href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              aria-hidden
              src="/file.svg"
              alt="File icon"
              width={16}
              height={16}
            />
            Learn
          </a>
          <a
            className="flex items-center gap-2 hover:underline hover:underline-offset-4"
            href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              aria-hidden
              src="/window.svg"
              alt="Window icon"
              width={16}
              height={16}
            />
            Examples
          </a>
          <a
            className="flex items-center gap-2 hover:underline hover:underline-offset-4"
            href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              aria-hidden
              src="/globe.svg"
              alt="Globe icon"
              width={16}
              height={16}
            />
            Go to nextjs.org →
          </a>
        </footer>
      </div>
    </div>
  );
}
