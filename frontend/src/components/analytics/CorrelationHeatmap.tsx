import React from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface CorrelationData {
  group_correlations: {
    matrix: number[][];
    labels: string[];
  };
  top_positive_pairs?: Array<{
    factor1: string;
    factor2: string;
    correlation: number;
  }>;
  top_negative_pairs?: Array<{
    factor1: string;
    factor2: string;
    correlation: number;
  }>;
}

interface CorrelationHeatmapProps {
  data: CorrelationData | null;
}

const GROUP_DISPLAY_NAMES: Record<string, string> = {
  'technical': 'Technical',
  'fundamental': 'Fundamental',
  'news_macro': 'News/Macro',
  'social_alternative': 'Social/Alt',
  'risk_stability': 'Risk/Stability',
  'institutional_smart_money': 'Institutional'
};

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ data }) => {
  if (!data || !data.group_correlations) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-400">
        No correlation data available
      </div>
    );
  }

  const { matrix, labels } = data.group_correlations;
  
  // Convert labels to display names
  const displayLabels = labels.map(label => GROUP_DISPLAY_NAMES[label] || label);

  return (
    <div className="space-y-6">
      {/* Heatmap */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-3">
          Factor Group Correlation Matrix
        </h3>
        <div className="flex justify-center">
          {/* @ts-expect-error - Plotly.js has complex type definitions */}
          <Plot
            data={[
              {
                z: matrix,
                x: displayLabels,
                y: displayLabels,
                type: 'heatmap',
                colorscale: 'RdBu',
                reversescale: true,
                zmid: 0,
                zmin: -1,
                zmax: 1,
                text: matrix.map(row => 
                  row.map(val => val.toFixed(2))
                ),
                texttemplate: '%{text}',
                textfont: {
                  size: 12
                },
                hovertemplate: '<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>',
                showscale: true,
                colorbar: {
                  title: 'Correlation',
                  titleside: 'right',
                  tickvals: [-1, -0.5, 0, 0.5, 1],
                  ticktext: ['-1.0', '-0.5', '0.0', '0.5', '1.0']
                }
              }
            ] as Plotly.Data[]}
            layout={{
              height: 500,
              xaxis: {
                title: '',
                side: 'bottom',
                tickangle: -45
              },
              yaxis: {
                title: '',
                autorange: 'reversed'
              },
              margin: { t: 40, r: 100, b: 120, l: 120 },
            }}
            config={{ displayModeBar: true, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-2 text-center">
          Positive correlations (blue) indicate factors move together. 
          Negative correlations (red) indicate factors move opposite.
        </p>
      </div>

      {/* Top Correlations */}
      {(data.top_positive_pairs || data.top_negative_pairs) && (
        <div className="grid grid-cols-2 gap-4">
          {/* Positive Correlations */}
          {data.top_positive_pairs && data.top_positive_pairs.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 text-blue-600">
                Top Positive Correlations
              </h3>
              <div className="space-y-2">
                {data.top_positive_pairs.slice(0, 5).map((pair, idx) => (
                  <div key={idx} className="flex justify-between items-center p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                    <span className="text-sm">
                      {GROUP_DISPLAY_NAMES[pair.factor1] || pair.factor1} ↔{' '}
                      {GROUP_DISPLAY_NAMES[pair.factor2] || pair.factor2}
                    </span>
                    <span className="font-mono font-bold text-blue-600">
                      {pair.correlation.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Negative Correlations */}
          {data.top_negative_pairs && data.top_negative_pairs.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 text-red-600">
                Top Negative Correlations
              </h3>
              <div className="space-y-2">
                {data.top_negative_pairs.slice(0, 5).map((pair, idx) => (
                  <div key={idx} className="flex justify-between items-center p-2 bg-red-50 dark:bg-red-900/20 rounded">
                    <span className="text-sm">
                      {GROUP_DISPLAY_NAMES[pair.factor1] || pair.factor1} ↔{' '}
                      {GROUP_DISPLAY_NAMES[pair.factor2] || pair.factor2}
                    </span>
                    <span className="font-mono font-bold text-red-600">
                      {pair.correlation.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
