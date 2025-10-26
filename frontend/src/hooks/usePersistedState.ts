'use client';

import { useState, useEffect } from 'react';
import type { FilterState } from '@/types/pipeline';

const STORAGE_KEYS = {
  FILTERS: 'vanpiq_filters',
  SORT: 'vanpiq_sort',
  EXPANDED_ROW: 'vanpiq_expanded_row',
  THEME: 'vanpiq_theme',
} as const;

// Generic localStorage hook
function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T | ((prev: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(initialValue);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    try {
      const item = window.localStorage.getItem(key);
      if (item) {
        setStoredValue(JSON.parse(item));
      }
    } catch (error) {
      console.warn(`Error loading ${key} from localStorage:`, error);
    } finally {
      setIsInitialized(true);
    }
  }, [key]);

  const setValue = (value: T | ((prev: T) => T)) => {
    try {
      const newValue = value instanceof Function ? value(storedValue) : value;
      setStoredValue(newValue);
      if (isInitialized) {
        window.localStorage.setItem(key, JSON.stringify(newValue));
      }
    } catch (error) {
      console.warn(`Error saving ${key} to localStorage:`, error);
    }
  };

  return [storedValue, setValue];
}

// Hook for persisting filter state
export function usePersistedFilters(initialFilters: FilterState) {
  return useLocalStorage<FilterState>(STORAGE_KEYS.FILTERS, initialFilters);
}

// Hook for persisting sort state
export function usePersistedSort(initialSort: SortState) {
  return useLocalStorage<SortState>(STORAGE_KEYS.SORT, initialSort);
}

// Hook for persisting expanded row
export function usePersistedExpandedRow(initialRow: string | null) {
  return useLocalStorage<string | null>(STORAGE_KEYS.EXPANDED_ROW, initialRow);
}

// Hook for persisting column visibility
export function usePersistedColumnVisibility(initialVisibility: ColumnVisibility) {
  return useLocalStorage<ColumnVisibility>('vanpiq_column_visibility', initialVisibility);
}

// Clear all persisted data
export function clearAllPersistedData() {
  Object.values(STORAGE_KEYS).forEach(key => {
    try {
      window.localStorage.removeItem(key);
    } catch (error) {
      console.warn(`Error clearing ${key}:`, error);
    }
  });
  // Also clear column visibility
  try {
    window.localStorage.removeItem('vanpiq_column_visibility');
  } catch (error) {
    console.warn('Error clearing column visibility:', error);
  }
}

// Sort state type
export interface SortState {
  column: string;
  direction: 'asc' | 'desc';
}

// Column visibility type
export interface ColumnVisibility {
  rank: boolean;
  ticker: boolean;
  overallScore: boolean;
  coverage: boolean;
  technical: boolean;
  fundamental: boolean;
  newsMacro: boolean;
  social: boolean;
  risk: boolean;
  institutional: boolean;
  // Backtest columns (Phase 6)
  baseline?: boolean;
  return1d?: boolean;
  return7d?: boolean;
  return30d?: boolean;
  return90d?: boolean;
  vsSpy?: boolean;
}
