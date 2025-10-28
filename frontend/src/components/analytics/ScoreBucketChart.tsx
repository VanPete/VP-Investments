import React from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface BucketMetrics {
  threshold: string;
  count: number;
  [interval: string]: any;
}

interface ScoreBucketData {
  strong_buy: BucketMetrics;
  buy: BucketMetrics;
  hold: BucketMetrics;
  sell: BucketMetrics;
  strong_sell: BucketMetrics;
}

interface ScoreBucketChartProps {
  data: ScoreBucketData | null;
  interval?: string;
}

const BUCKET_LABELS = {
  strong_buy: 'Strong Buy (>0.75)',
  buy: 'Buy (0.50-0.75)',
  hold: 'Hold (-0.25-0.50)',
  sell: 'Sell (-0.50 to -0.25)',
  strong_sell: 'Strong Sell (<-0.50)'
};

const BUCKET_COLORS = {
  strong_buy: '#10b981',  // green
  buy: '#22c55e',
  hold: '#fbbf24',        // yellow
  sell: '#f87171',        // red
  strong_sell: '#dc2626'
};

export const ScoreBucketChart: React.FC<ScoreBucketChartProps> = ({ 
  data, 
  interval = '7d' 
}) => {
  if (!data) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-400">
        No score bucket data available
      </div>
    );
  }

  const buckets = Object.keys(BUCKET_LABELS) as Array<keyof typeof BUCKET_LABELS>;
  
  // Extract data for the selected interval
  const avgReturns: number[] = [];
  const winRates: number[] = [];
  const counts: number[] = [];
  const labels: string[] = [];
  const colors: string[] = [];

  buckets.forEach((bucket) => {
    const bucketData = data[bucket];
    const intervalData = bucketData[interval];
    
    if (intervalData && bucketData.count > 0) {
      avgReturns.push((intervalData.avg_return || 0) * 100); // Convert to percentage
      winRates.push((intervalData.win_rate || 0) * 100);
      counts.push(bucketData.count);
      labels.push(BUCKET_LABELS[bucket]);
      colors.push(BUCKET_COLORS[bucket]);
    }
  });

  if (avgReturns.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-400">
        No data for {interval} interval
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {/* Average Return Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-2">
            Average Return by Score Bucket ({interval})
          </h3>
          <Plot
            data={[
              {
                x: labels,
                y: avgReturns,
                type: 'bar',
                marker: {
                  color: colors,
                },
                text: avgReturns.map(r => `${r.toFixed(2)}%`),
                textposition: 'outside',
                hovertemplate: '<b>%{x}</b><br>Avg Return: %{y:.2f}%<extra></extra>',
              },
            ]}
            layout={{
              height: 400,
              xaxis: {
                title: 'Score Bucket',
                tickangle: -45,
              },
              yaxis: {
                title: 'Average Return (%)',
                zeroline: true,
              },
              showlegend: false,
              margin: { t: 20, r: 20, b: 100, l: 60 },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Win Rate Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-2">
            Win Rate by Score Bucket ({interval})
          </h3>
          <Plot
            data={[
              {
                x: labels,
                y: winRates,
                type: 'bar',
                marker: {
                  color: colors,
                },
                text: winRates.map(r => `${r.toFixed(1)}%`),
                textposition: 'outside',
                hovertemplate: '<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>',
              },
            ]}
            layout={{
              height: 400,
              xaxis: {
                title: 'Score Bucket',
                tickangle: -45,
              },
              yaxis: {
                title: 'Win Rate (%)',
                range: [0, 100],
              },
              showlegend: false,
              margin: { t: 20, r: 20, b: 100, l: 60 },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Summary Stats */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-3">
          Signal Distribution
        </h3>
        <div className="grid grid-cols-5 gap-2">
          {buckets.map((bucket, idx) => {
            const bucketData = data[bucket];
            if (bucketData.count === 0) return null;
            
            return (
              <div
                key={bucket}
                className="text-center p-3 rounded-lg"
                style={{ backgroundColor: `${BUCKET_COLORS[bucket]}20` }}
              >
                <div className="text-2xl font-bold" style={{ color: BUCKET_COLORS[bucket] }}>
                  {bucketData.count}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  {BUCKET_LABELS[bucket].split('(')[0].trim()}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
