'use client';

import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import { StockData, DashboardStats } from '@/types/stock';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface ScoreDistributionChartProps {
  stocks: StockData[];
}

export function ScoreDistributionChart({ stocks }: ScoreDistributionChartProps) {
  const scoreRanges = [
    { label: '0-2', min: 0, max: 2 },
    { label: '2-4', min: 2, max: 4 },
    { label: '4-6', min: 4, max: 6 },
    { label: '6-8', min: 6, max: 8 },
    { label: '8+', min: 8, max: 100 },
  ];

  const data = {
    labels: scoreRanges.map(range => range.label),
    datasets: [
      {
        label: 'Number of Stocks',
        data: scoreRanges.map(range => 
          stocks.filter(stock => 
            stock.weighted_score >= range.min && stock.weighted_score < range.max
          ).length
        ),
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(255, 159, 64, 0.5)',
          'rgba(255, 205, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(54, 162, 235, 0.5)',
        ],
        borderColor: [
          'rgb(255, 99, 132)',
          'rgb(255, 159, 64)',
          'rgb(255, 205, 86)',
          'rgb(75, 192, 192)',
          'rgb(54, 162, 235)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Score Distribution',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return <Bar data={data} options={options} />;
}

interface TradeTypeChartProps {
  stats: DashboardStats;
}

export function TradeTypeChart({ stats }: TradeTypeChartProps) {
  const data = {
    labels: Object.keys(stats.tradeTypeBreakdown),
    datasets: [
      {
        data: Object.values(stats.tradeTypeBreakdown),
        backgroundColor: [
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 99, 132, 0.5)',
          'rgba(255, 205, 86, 0.5)',
        ],
        borderColor: [
          'rgb(54, 162, 235)',
          'rgb(255, 99, 132)',
          'rgb(255, 205, 86)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Trade Type Distribution',
      },
    },
  };

  return <Doughnut data={data} options={options} />;
}
