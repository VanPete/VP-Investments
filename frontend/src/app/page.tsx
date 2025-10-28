import { SignalsDashboard } from '@/components/dashboard/SignalsDashboard';
import {
  readWeightsConfig,
  readFactorToGroup,
  readMethodologyConfig,
} from '@/lib/pipeline-data';

export default function Home() {
  // Read config data at build time (static files)
  const weightsConfig = readWeightsConfig();
  const factorToGroup = readFactorToGroup();
  const methodologyConfig = readMethodologyConfig();

  // Signals data now fetched from Supabase on client-side
  return (
    <SignalsDashboard
      weightsConfig={weightsConfig}
      factorToGroup={factorToGroup}
      methodologyConfig={methodologyConfig}
    />
  );
}
