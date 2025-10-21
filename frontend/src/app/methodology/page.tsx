import { readMethodologyConfig, readWeightsConfig, readFactorToGroup, getFactorCounts } from '@/lib/pipeline-data';
import { WeightsOverview } from '@/components/methodology/WeightsOverview';
import { ScoringExplainer } from '@/components/methodology/ScoringExplainer';
import { FactorLibrary } from '@/components/methodology/FactorLibrary';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

export default function MethodologyPage() {
  const methodologyConfig = readMethodologyConfig();
  const weightsConfig = readWeightsConfig();
  const factorToGroup = readFactorToGroup();
  const factorCounts = factorToGroup ? getFactorCounts(factorToGroup) : {};

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Methodology
          </h1>
          <p className="text-lg text-gray-600">
            Understanding the VP Investments signal ranking system
          </p>
        </div>

        {/* Overview Section */}
        {methodologyConfig && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle>{methodologyConfig.overview.title}</CardTitle>
              <CardDescription>
                {methodologyConfig.overview.description}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <h3 className="font-semibold text-gray-900 mb-3">Key Principles</h3>
              <ul className="space-y-2">
                {methodologyConfig.overview.key_principles.map((principle, idx) => (
                  <li key={idx} className="flex items-start">
                    <span className="text-gray-600 mr-2">•</span>
                    <span className="text-gray-700">{principle}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {/* Weights Overview */}
        {weightsConfig && (
          <WeightsOverview
            weightsConfig={weightsConfig}
            factorCounts={factorCounts}
          />
        )}

        {/* Scoring Explanation */}
        {methodologyConfig && (
          <ScoringExplainer methodologyConfig={methodologyConfig} />
        )}

        {/* Factor Library */}
        {factorToGroup && weightsConfig && (
          <FactorLibrary
            factorToGroup={factorToGroup}
            weightsConfig={weightsConfig}
          />
        )}

        {/* Version Info */}
        {methodologyConfig && (
          <Card className="mt-8">
            <CardContent className="pt-6">
              <p className="text-sm text-gray-500 text-center">
                Methodology Version {methodologyConfig.version.current} • 
                Last Updated: {methodologyConfig.version.last_updated}
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
