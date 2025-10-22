import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { HelpCircle } from 'lucide-react';

interface MetricTooltipProps {
  title: string;
  description: string;
  children?: React.ReactNode;
}

export function MetricTooltip({ title, description, children }: MetricTooltipProps) {
  return (
    <TooltipProvider>
      <Tooltip delayDuration={200}>
        <TooltipTrigger asChild>
          {children || (
            <HelpCircle className="h-3.5 w-3.5 text-gray-400 hover:text-[#00AEEF] transition-colors cursor-help inline-block ml-1" />
          )}
        </TooltipTrigger>
        <TooltipContent 
          side="top" 
          className="max-w-xs bg-white dark:bg-gray-900 border-[#00AEEF]/30 shadow-xl"
        >
          <div className="space-y-1">
            <p className="font-semibold text-sm text-gray-900 dark:text-gray-100">
              {title}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              {description}
            </p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Predefined tooltip content for common metrics
export const METRIC_TOOLTIPS = {
  overallScore: {
    title: "Overall Score",
    description: "Weighted combination of all signal groups. Ranges from -5 (very bearish) to +5 (very bullish). Higher scores indicate stronger positive signals."
  },
  coverage: {
    title: "Data Coverage",
    description: "Percentage of available data points that were successfully collected. Higher coverage means more reliable scoring. 90%+ is excellent."
  },
  technical: {
    title: "Technical Signals",
    description: "Price momentum, trend strength, moving averages, and volume analysis. Captures market sentiment from chart patterns."
  },
  fundamental: {
    title: "Fundamental Signals",
    description: "Financial health metrics like revenue growth, profit margins, P/E ratios, and balance sheet strength. Measures business quality."
  },
  newsMacro: {
    title: "News & Macro Signals",
    description: "Sentiment from news articles, economic indicators, and market-wide trends. Captures external factors affecting the stock."
  },
  social: {
    title: "Social Sentiment",
    description: "Analysis of social media discussions, Reddit mentions, and community sentiment. Reflects retail investor interest."
  },
  risk: {
    title: "Risk Assessment",
    description: "Volatility, drawdown risk, beta, and other risk metrics. Lower risk scores are better. Negative scores indicate higher risk."
  },
  institutional: {
    title: "Institutional Activity",
    description: "Hedge fund holdings, insider trading, and institutional ownership changes. Shows smart money positioning."
  },
  rank: {
    title: "Overall Rank",
    description: "Position in the ranked list based on Overall Score. Rank 1 is the highest-scoring ticker in the current universe."
  }
};
