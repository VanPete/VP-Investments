'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { MethodologyConfig } from '@/types/pipeline';

interface ScoringExplainerProps {
  methodologyConfig: MethodologyConfig;
}

export function ScoringExplainer({ methodologyConfig }: ScoringExplainerProps) {
  return (
    <div className="space-y-6 mb-8">
      {/* Scoring Process */}
      <Card>
        <CardHeader>
          <CardTitle>Scoring Process</CardTitle>
          <CardDescription>
            How raw data is transformed into actionable signals
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Normalization */}
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
              1. Normalization ({methodologyConfig.scoring.normalization.method})
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              {methodologyConfig.scoring.normalization.description}
            </p>
            <div className="mt-3">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Advantages:</h4>
              <div className="flex flex-wrap gap-2">
                {methodologyConfig.scoring.normalization.advantages.map((advantage, idx) => (
                  <Badge key={idx} variant="outline" className="text-xs">
                    {advantage}
                  </Badge>
                ))}
              </div>
            </div>
          </div>

          {/* Factor Weighting */}
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
              2. Factor Weighting
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              {methodologyConfig.scoring.factor_weighting.description}
            </p>
          </div>

          {/* Group Weighting */}
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
              3. Group Weighting
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              {methodologyConfig.scoring.group_weighting.description}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Score Interpretation */}
      <Card>
        <CardHeader>
          <CardTitle>Score Interpretation</CardTitle>
          <CardDescription>
            Understanding what the scores mean
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Overall Score */}
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Overall Score</h3>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                {methodologyConfig.interpretation.overall_score.description}
              </p>
              <div className="space-y-2">
                {Object.entries(methodologyConfig.interpretation.overall_score.ranges).map(
                  ([key, range]) => (
                    <div
                      key={key}
                      className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg"
                    >
                      <span className="font-medium capitalize">{key.replace('_', ' ')}</span>
                      <Badge variant="outline" className="text-xs">
                        {range}
                      </Badge>
                    </div>
                  )
                )}
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-3">
                {methodologyConfig.interpretation.overall_score.notes}
              </p>
            </div>

            {/* Coverage */}
            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Coverage Quality</h3>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                {methodologyConfig.interpretation.coverage.description}
              </p>
              <div className="space-y-2">
                {Object.entries(methodologyConfig.interpretation.coverage.guidance).map(
                  ([key, guidanceText]) => (
                    <div key={key} className="flex items-center gap-3 p-2 bg-gray-50 dark:bg-gray-800/50 rounded">
                      <span className="text-sm font-medium capitalize min-w-[100px]">
                        {key.replace('_', ' ')}:
                      </span>
                      <span className="text-sm text-gray-700 dark:text-gray-300">{guidanceText}</span>
                    </div>
                  )
                )}
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-3">
                {methodologyConfig.interpretation.coverage.notes}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Limitations */}
      <Card className="border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-900/20">
        <CardHeader>
          <CardTitle className="text-orange-900 dark:text-orange-400">Important Limitations</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {methodologyConfig.limitations.map((limitation, idx) => (
              <li key={idx} className="flex items-start text-sm text-orange-900 dark:text-orange-300">
                <span className="mr-2">•</span>
                <span>{limitation}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
