import React from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface BacktestData {
  start_date: string;
  end_date: string;
  daily_returns: Array<{
    date: string;
    vp_strategy: number;
    spy: number;
    qqq: number;
  }>;
  summary: {
    vp_total_return: number;
    spy_total_return: number;
    qqq_total_return: number;
    vp_sharpe: number;
    vp_max_drawdown: number;
    vp_win_rate: number;
  };
}

interface BacktestChartProps {
  data: BacktestData | null;
}

export const BacktestChart: React.FC<BacktestChartProps> = ({ data }) => {
  if (!data || !data.daily_returns || data.daily_returns.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-400">
        No backtest data available
      </div>
    );
  }

  const dates = data.daily_returns.map(d => d.date);
  const vpReturns = data.daily_returns.map(d => (d.vp_strategy - 1) * 100);
  const spyReturns = data.daily_returns.map(d => (d.spy - 1) * 100);
  const qqqReturns = data.daily_returns.map(d => (d.qqq - 1) * 100);

  return (
    <div className="space-y-6">
      {/* Cumulative Returns Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-3">
          Cumulative Returns: VP Strategy vs Benchmarks
        </h3>
        <Plot
          data={[
            {
              x: dates,
              y: vpReturns,
              type: 'scatter',
              mode: 'lines',
              name: 'VP Strategy',
              line: {
                color: '#10b981',
                width: 3
              },
              hovertemplate: '<b>VP Strategy</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>',
            },
            {
              x: dates,
              y: spyReturns,
              type: 'scatter',
              mode: 'lines',
              name: 'SPY (S&P 500)',
              line: {
                color: '#3b82f6',
                width: 2,
                dash: 'dash'
              },
              hovertemplate: '<b>SPY</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>',
            },
            {
              x: dates,
              y: qqqReturns,
              type: 'scatter',
              mode: 'lines',
              name: 'QQQ (Nasdaq)',
              line: {
                color: '#f59e0b',
                width: 2,
                dash: 'dot'
              },
              hovertemplate: '<b>QQQ</b><br>%{x}<br>Return: %{y:.2f}%<extra></extra>',
            },
          ] as any}
          layout={{
            height: 500,
            xaxis: {
              title: 'Date',
              type: 'date',
            },
            yaxis: {
              title: 'Cumulative Return (%)',
              zeroline: true,
            },
            hovermode: 'x unified',
            legend: {
              x: 0.02,
              y: 0.98,
              bgcolor: 'rgba(255,255,255,0.8)',
              bordercolor: '#ddd',
              borderwidth: 1
            },
            margin: { t: 40, r: 40, b: 60, l: 60 },
          }}
          config={{ displayModeBar: true, responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      {/* Performance Summary */}
      {data.summary && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">
            Performance Summary
          </h3>
          <div className="grid grid-cols-3 gap-6">
            {/* VP Strategy */}
            <div className="border-l-4 border-green-500 pl-4">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                VP Strategy
              </div>
              <div className="text-3xl font-bold text-green-600 mb-2">
                {(data.summary.vp_total_return * 100).toFixed(2)}%
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Sharpe Ratio:</span>
                  <span className="font-mono font-semibold">{data.summary.vp_sharpe.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Max Drawdown:</span>
                  <span className="font-mono font-semibold text-red-600">
                    {data.summary.vp_max_drawdown.toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Win Rate:</span>
                  <span className="font-mono font-semibold">
                    {(data.summary.vp_win_rate * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* SPY Benchmark */}
            <div className="border-l-4 border-blue-500 pl-4">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                SPY (S&P 500)
              </div>
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {(data.summary.spy_total_return * 100).toFixed(2)}%
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Outperformance:</span>
                  <span className={`font-mono font-semibold ${
                    data.summary.vp_total_return > data.summary.spy_total_return 
                      ? 'text-green-600' 
                      : 'text-red-600'
                  }`}>
                    {((data.summary.vp_total_return - data.summary.spy_total_return) * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>

            {/* QQQ Benchmark */}
            <div className="border-l-4 border-orange-500 pl-4">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                QQQ (Nasdaq)
              </div>
              <div className="text-3xl font-bold text-orange-600 mb-2">
                {(data.summary.qqq_total_return * 100).toFixed(2)}%
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Outperformance:</span>
                  <span className={`font-mono font-semibold ${
                    data.summary.vp_total_return > data.summary.qqq_total_return 
                      ? 'text-green-600' 
                      : 'text-red-600'
                  }`}>
                    {((data.summary.vp_total_return - data.summary.qqq_total_return) * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Period Info */}
          <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex justify-between">
              <span>Period: {data.start_date} to {data.end_date}</span>
              <span>{data.daily_returns.length} trading days</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
