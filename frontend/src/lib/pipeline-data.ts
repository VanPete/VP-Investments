// VP Investments - Pipeline Data Utilities
// Utilities for reading pipeline results and config files

import fs from 'fs';
import path from 'path';
import yaml from 'yaml';
import type {
  PipelineResults,
  WeightsConfig,
  FactorToGroup,
  MethodologyConfig,
  FileOption,
} from '@/types/pipeline';

/**
 * Get the root directory of the project (parent of frontend)
 */
export function getProjectRoot(): string {
  // From frontend/src/lib -> go up 3 levels to project root
  return path.join(process.cwd(), '..');
}

/**
 * Get all pipeline result files sorted by timestamp (newest first)
 */
export function getAvailableResults(): FileOption[] {
  // Look in frontend/public/results where backend saves files
  const resultsDir = path.join(process.cwd(), 'public', 'results');
  
  if (!fs.existsSync(resultsDir)) {
    return [];
  }

  const files = fs.readdirSync(resultsDir)
    .filter(file => file.startsWith('pipeline_results_') && file.endsWith('.json'))
    .map(filename => {
      // Extract timestamp from filename: pipeline_results_YYYYMMDD_HHMMSS.json
      const match = filename.match(/pipeline_results_(\d{8})_(\d{6})\.json/);
      if (!match) return null;

      const dateStr = match[1]; // YYYYMMDD
      const timeStr = match[2]; // HHMMSS

      const year = dateStr.substring(0, 4);
      const month = dateStr.substring(4, 6);
      const day = dateStr.substring(6, 8);
      const hour = timeStr.substring(0, 2);
      const minute = timeStr.substring(2, 4);
      const second = timeStr.substring(4, 6);

      const timestamp = `${year}-${month}-${day}T${hour}:${minute}:${second}`;
      const date = new Date(timestamp);

      return {
        filename,
        timestamp,
        label: date.toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        }),
      };
    })
    .filter((item): item is FileOption => item !== null)
    .sort((a, b) => b.timestamp.localeCompare(a.timestamp)); // Newest first

  return files;
}

/**
 * Read a specific pipeline results file
 */
export function readPipelineResults(filename: string): PipelineResults | null {
  try {
    const filePath = path.join(process.cwd(), 'public', 'results', filename);
    
    if (!fs.existsSync(filePath)) {
      console.error(`Pipeline results file not found: ${filename}`);
      return null;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(content) as PipelineResults;
  } catch (error) {
    console.error(`Error reading pipeline results: ${filename}`, error);
    return null;
  }
}

/**
 * Get the latest pipeline results
 */
export function getLatestResults(): PipelineResults | null {
  const availableFiles = getAvailableResults();
  
  if (availableFiles.length === 0) {
    console.error('No pipeline results found');
    return null;
  }

  return readPipelineResults(availableFiles[0].filename);
}

/**
 * Read weights configuration from weights.yaml
 */
export function readWeightsConfig(): WeightsConfig | null {
  try {
    const filePath = path.join(process.cwd(), 'public', 'config', 'weights.yaml');
    
    if (!fs.existsSync(filePath)) {
      console.error('weights.yaml not found');
      return null;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    return yaml.parse(content) as WeightsConfig;
  } catch (error) {
    console.error('Error reading weights config:', error);
    return null;
  }
}

/**
 * Read factor to group mapping from factor_to_group.yaml
 */
export function readFactorToGroup(): FactorToGroup | null {
  try {
    const filePath = path.join(process.cwd(), 'public', 'config', 'factor_to_group.yaml');
    
    if (!fs.existsSync(filePath)) {
      console.error('factor_to_group.yaml not found');
      return null;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const parsed = yaml.parse(content);

    // Extract only the group sections (skip validation)
    return {
      technical: parsed.technical || {},
      fundamental: parsed.fundamental || {},
      news_macro: parsed.news_macro || {},
      social_alternative: parsed.social_alternative || {},
      risk_stability: parsed.risk_stability || {},
      institutional_smart_money: parsed.institutional_smart_money || {},
    } as FactorToGroup;
  } catch (error) {
    console.error('Error reading factor_to_group config:', error);
    return null;
  }
}

/**
 * Read methodology configuration from methodology.yaml
 */
export function readMethodologyConfig(): MethodologyConfig | null {
  try {
    const filePath = path.join(process.cwd(), 'public', 'config', 'methodology.yaml');
    
    if (!fs.existsSync(filePath)) {
      console.error('methodology.yaml not found');
      return null;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    return yaml.parse(content) as MethodologyConfig;
  } catch (error) {
    console.error('Error reading methodology config:', error);
    return null;
  }
}

/**
 * Get factor count per group
 */
export function getFactorCounts(factorToGroup: FactorToGroup): Record<string, number> {
  return {
    technical: Object.keys(factorToGroup.technical).length,
    fundamental: Object.keys(factorToGroup.fundamental).length,
    news_macro: Object.keys(factorToGroup.news_macro).length,
    social_alternative: Object.keys(factorToGroup.social_alternative).length,
    risk_stability: Object.keys(factorToGroup.risk_stability).length,
    institutional_smart_money: Object.keys(factorToGroup.institutional_smart_money).length,
  };
}

/**
 * Get total factor count
 */
export function getTotalFactorCount(factorToGroup: FactorToGroup): number {
  const counts = getFactorCounts(factorToGroup);
  return Object.values(counts).reduce((sum, count) => sum + count, 0);
}
