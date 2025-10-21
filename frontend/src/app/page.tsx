import { SignalsDashboard } from '@/components/dashboard/SignalsDashboard';
import {
  getLatestResults,
  getAvailableResults,
  readWeightsConfig,
  readFactorToGroup,
} from '@/lib/pipeline-data';

export default function Home() {
  // Read data at build time
  const latestResults = getLatestResults();
  const availableFiles = getAvailableResults();
  const weightsConfig = readWeightsConfig();
  const factorToGroup = readFactorToGroup();

  return (
    <SignalsDashboard
      initialResults={latestResults}
      availableFiles={availableFiles}
      weightsConfig={weightsConfig}
      factorToGroup={factorToGroup}
    />
  );
}
