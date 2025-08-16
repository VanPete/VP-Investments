'use client';

import React, { useState } from 'react';
import type { StockData } from '@/types/stock';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from './ui/badge';
import { Brain, MessageSquare, TrendingUp, AlertTriangle, Loader2 } from 'lucide-react';

interface AIInsightsProps {
  ticker: string;
  score: number;
  metrics: {
  redditSentiment: number;
  mentions: number;
  price7d: number;
  volumeSpike: number;
  marketCap?: string;
  peRatio?: number;
  rsi: number;
  };
}

export function AIInsights({ ticker, score, metrics }: AIInsightsProps) {
  const [commentary, setCommentary] = useState<string>('');
  const [scoreExplanation, setScoreExplanation] = useState<string>('');
  const [riskAssessment, setRiskAssessment] = useState<string>('');
  const [loading, setLoading] = useState<{[key: string]: boolean}>({});

  const generateCommentary = async () => {
    setLoading(prev => ({ ...prev, commentary: true }));
    
    try {
      // For demo purposes, we'll use mock responses
      // In production, this would call the ChatGPT API
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate API delay
      
  const mockCommentary = `${ticker} shows strong momentum with ${(metrics.redditSentiment ?? 0).toFixed(2)} sentiment and ${metrics.mentions} mentions. The ${metrics.price7d > 0 ? 'positive' : 'negative'} 7-day price movement (${(metrics.price7d ?? 0).toFixed(1)}%) combined with ${(metrics.volumeSpike ?? 0).toFixed(1)}x volume spike suggests ${score > 70 ? 'bullish' : 'mixed'} sentiment. Monitor for ${metrics.rsi > 70 ? 'overbought conditions' : 'technical breakout opportunities'}.`;
      
      setCommentary(mockCommentary);
    } catch (error) {
      console.error('AI commentary error', error);
      setCommentary('Error generating AI commentary. Please try again.');
    } finally {
      setLoading(prev => ({ ...prev, commentary: false }));
    }
  };

  const explainScore = async () => {
    setLoading(prev => ({ ...prev, explanation: true }));
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      
  const mockExplanation = `${ticker} scored ${score}/100 primarily due to ${score > 70 ? 'strong' : score > 50 ? 'moderate' : 'weak'} Reddit sentiment ${(metrics.redditSentiment ?? 0).toFixed(2)} and ${metrics.price7d > 0 ? 'positive' : 'negative'} price momentum. Key factors include ${metrics.mentions} social mentions and ${(metrics.volumeSpike ?? 0).toFixed(1)}x volume activity.`;
      
      setScoreExplanation(mockExplanation);
    } catch (error) {
      console.error('Explain score error', error);
      setScoreExplanation('Error explaining score. Please try again.');
    } finally {
      setLoading(prev => ({ ...prev, explanation: false }));
    }
  };

  const assessRisk = async () => {
    setLoading(prev => ({ ...prev, risk: true }));
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1800));
      
  const pe = typeof metrics.peRatio === 'number' ? metrics.peRatio : 0;
  const riskLevel = metrics.rsi > 70 || metrics.volumeSpike > 3 ? 'High' : 
           pe > 30 || metrics.price7d > 10 ? 'Medium' : 'Low';
      
      const mockRisk = `${riskLevel} Risk - ${riskLevel === 'High' ? 'Monitor for volatility and consider smaller position size' : riskLevel === 'Medium' ? 'Standard risk management applies' : 'Relatively stable metrics suggest lower risk'}. Key factors: RSI at ${metrics.rsi}, volume spike ${metrics.volumeSpike.toFixed(1)}x.`;
      
      setRiskAssessment(mockRisk);
    } catch (error) {
      console.error('Risk assessment error', error);
      setRiskAssessment('Error assessing risk. Please try again.');
    } finally {
      setLoading(prev => ({ ...prev, risk: false }));
    }
  };

  const getRiskColor = (assessment: string) => {
    if (assessment.includes('High Risk')) return 'destructive';
    if (assessment.includes('Medium Risk')) return 'default';
    return 'secondary';
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-blue-500" />
            AI Analysis for {ticker}
          </CardTitle>
          <CardDescription>
            ChatGPT-powered insights and explanations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* AI Commentary */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-medium flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                AI Commentary
              </h4>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={generateCommentary}
                disabled={loading.commentary}
              >
                {loading.commentary ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Generate'
                )}
              </Button>
            </div>
            {commentary && (
              <div className="p-3 bg-blue-50 rounded-lg text-sm">
                {commentary}
              </div>
            )}
          </div>

          {/* Score Explanation */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-medium flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Score Explanation
              </h4>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={explainScore}
                disabled={loading.explanation}
              >
                {loading.explanation ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Explain'
                )}
              </Button>
            </div>
            {scoreExplanation && (
              <div className="p-3 bg-green-50 rounded-lg text-sm">
                {scoreExplanation}
              </div>
            )}
          </div>

          {/* Risk Assessment */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-medium flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                Risk Assessment
              </h4>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={assessRisk}
                disabled={loading.risk}
              >
                {loading.risk ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Assess'
                )}
              </Button>
            </div>
            {riskAssessment && (
              <div className="space-y-2">
                <Badge variant={getRiskColor(riskAssessment)}>
                  {riskAssessment.includes('High') ? 'High Risk' : 
                   riskAssessment.includes('Medium') ? 'Medium Risk' : 'Low Risk'}
                </Badge>
                <div className="p-3 bg-yellow-50 rounded-lg text-sm">
                  {riskAssessment}
                </div>
              </div>
            )}
          </div>

          {/* Quick Metrics Display */}
          <div className="grid grid-cols-2 gap-4 pt-4 border-t">
            <div className="space-y-1">
              <p className="text-xs text-gray-500">Reddit Sentiment</p>
              <p className="font-medium">{metrics.redditSentiment.toFixed(2)}</p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-gray-500">Mentions</p>
              <p className="font-medium">{metrics.mentions}</p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-gray-500">7D Price %</p>
              <p className={`font-medium ${metrics.price7d >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {metrics.price7d.toFixed(1)}%
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-gray-500">Volume Spike</p>
              <p className="font-medium">{metrics.volumeSpike.toFixed(1)}x</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Portfolio-level AI insights component
export function PortfolioAIInsights({ signals }: { signals: StockData[] }) {
  const [insights, setInsights] = useState<string>('');
  const [marketOutlook, setMarketOutlook] = useState<string>('');
  const [loading, setLoading] = useState<{[key: string]: boolean}>({});

  const generatePortfolioInsights = async () => {
    setLoading(prev => ({ ...prev, portfolio: true }));
    
    try {
      await new Promise(resolve => setTimeout(resolve, 2500));
      
  const avgScore = signals.reduce((acc, s) => acc + (s.weighted_score || 0), 0) / (signals.length || 1);
  const highQualityCount = signals.filter(s => (s.weighted_score || 0) > 70).length;
      
      const mockInsights = `Portfolio Analysis: ${signals.length} signals detected with average quality of ${avgScore.toFixed(1)}/100. ${highQualityCount} high-confidence opportunities identified. Technology and healthcare sectors showing strongest momentum. Recommend diversified approach with emphasis on top-tier signals. Monitor for sector rotation and adjust position sizing based on individual risk profiles.`;
      
      setInsights(mockInsights);
    } catch (error) {
      console.error('Portfolio insights error', error);
      setInsights('Error generating portfolio insights.');
    } finally {
      setLoading(prev => ({ ...prev, portfolio: false }));
    }
  };

  const generateMarketOutlook = async () => {
    setLoading(prev => ({ ...prev, market: true }));
    
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockOutlook = `Market Sentiment: Alternative data signals indicate cautiously optimistic retail sentiment. Reddit activity suggests growing interest in growth stocks and technology plays. Quality of current signals is above historical average, indicating potential opportunities. Recommend selective positioning with emphasis on risk management given current market volatility.`;
      
      setMarketOutlook(mockOutlook);
    } catch (error) {
      console.error('Market outlook error', error);
      setMarketOutlook('Error generating market outlook.');
    } finally {
      setLoading(prev => ({ ...prev, market: false }));
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-500" />
            Portfolio AI Insights
          </CardTitle>
          <CardDescription>
            ChatGPT analysis of your entire signal portfolio
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-medium">Portfolio Strategy</h4>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={generatePortfolioInsights}
                disabled={loading.portfolio}
              >
                {loading.portfolio ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Analyze Portfolio'
                )}
              </Button>
            </div>
            {insights && (
              <div className="p-4 bg-purple-50 rounded-lg text-sm">
                {insights}
              </div>
            )}
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-medium">Market Outlook</h4>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={generateMarketOutlook}
                disabled={loading.market}
              >
                {loading.market ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Generate Outlook'
                )}
              </Button>
            </div>
            {marketOutlook && (
              <div className="p-4 bg-indigo-50 rounded-lg text-sm">
                {marketOutlook}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
