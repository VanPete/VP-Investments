/**
 * Utility function to fetch top 5 factors for a specific group from the detail tables
 * Used for expandable rows in the signals table
 */

import { supabase } from './supabase';

export interface TopFactor {
  name: string;
  raw: number;
  normalized: number;
  percentile: number;
  absNormalized: number; // For sorting by impact
}

type GroupName = 
  | 'technical' 
  | 'fundamental' 
  | 'news_macro' 
  | 'social_alternative' 
  | 'risk_stability' 
  | 'institutional_smart_money';

// Map group names to detail table names
const GROUP_TABLE_MAP: Record<GroupName, string> = {
  technical: 'signals_technical',
  fundamental: 'signals_fundamental',
  news_macro: 'signals_news_macro',
  social_alternative: 'signals_social_alternative',
  risk_stability: 'signals_risk_stability',
  institutional_smart_money: 'signals_institutional_smart_money',
};

/**
 * Fetch and parse top 5 factors by absolute normalized score for a given signal and group
 * 
 * @param signalId - The signal ID to fetch factors for
 * @param groupName - The group name (e.g., 'technical', 'fundamental')
 * @returns Array of top 5 factors sorted by absolute normalized value (descending)
 */
export async function getTopFactorsForGroup(
  signalId: string,
  groupName: GroupName
): Promise<TopFactor[]> {
  // Get the appropriate table name
  const tableName = GROUP_TABLE_MAP[groupName];
  
  if (!tableName) {
    console.error(`Unknown group name: ${groupName}`);
    return [];
  }

  try {
    // Query the detail table for this signal
    const { data, error } = await supabase
      .from(tableName)
      .select('factors')
      .eq('signal_id', signalId)
      .single();

    if (error) {
      console.error(`Error fetching factors for ${groupName}:`, error);
      return [];
    }

    if (!data || !data.factors) {
      console.warn(`No factors found for signal ${signalId} in ${groupName}`);
      return [];
    }

    // Parse JSONB factors structure: { "factor_name": { "raw": X, "normalized": Y, "percentile": Z } }
    const factors = data.factors as Record<string, { raw: number; normalized: number; percentile: number }>;
    
    // Convert to array and add absolute normalized value for sorting
    const factorArray: TopFactor[] = Object.entries(factors).map(([name, values]) => ({
      name,
      raw: values.raw,
      normalized: values.normalized,
      percentile: values.percentile,
      absNormalized: Math.abs(values.normalized),
    }));

    // Sort by absolute normalized value (descending) and take top 5
    const top5 = factorArray
      .sort((a, b) => b.absNormalized - a.absNormalized)
      .slice(0, 5);

    return top5;
  } catch (err) {
    console.error(`Exception fetching top factors for ${groupName}:`, err);
    return [];
  }
}

/**
 * Batch fetch top 5 factors for all groups for a single signal
 * Useful for loading all group factors at once when a row is expanded
 * 
 * @param signalId - The signal ID to fetch factors for
 * @returns Object with top 5 factors for each group
 */
export async function getTopFactorsForAllGroups(signalId: string): Promise<Record<GroupName, TopFactor[]>> {
  const groups: GroupName[] = [
    'technical',
    'fundamental',
    'news_macro',
    'social_alternative',
    'risk_stability',
    'institutional_smart_money',
  ];

  // Fetch all groups in parallel
  const results = await Promise.all(
    groups.map(group => getTopFactorsForGroup(signalId, group))
  );

  // Build result object
  const topFactors: Record<GroupName, TopFactor[]> = {
    technical: results[0],
    fundamental: results[1],
    news_macro: results[2],
    social_alternative: results[3],
    risk_stability: results[4],
    institutional_smart_money: results[5],
  };

  return topFactors;
}
