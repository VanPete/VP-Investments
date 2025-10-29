'use client';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Columns3 } from 'lucide-react';

export interface ColumnVisibility {
  rank: boolean;
  ticker: boolean;
  companyName: boolean;
  sector: boolean;  // v3.3: Sector column (required, always true)
  currentPrice: boolean;
  overallScore: boolean;
  coverage: boolean;
  technical: boolean;
  fundamental: boolean;
  newsMacro: boolean;
  social: boolean;
  risk: boolean;
  institutional: boolean;
  // Performance columns
  return1d?: boolean;
  return7d?: boolean;
  return30d?: boolean;
  return90d?: boolean;
  vsSpy?: boolean;
}

interface ColumnVisibilityToggleProps {
  visibility: ColumnVisibility;
  onVisibilityChange: (visibility: ColumnVisibility) => void;
}

export function ColumnVisibilityToggle({
  visibility,
  onVisibilityChange,
}: ColumnVisibilityToggleProps) {
  const columns = [
    { key: 'rank' as keyof ColumnVisibility, label: 'Rank', required: false },
    { key: 'ticker' as keyof ColumnVisibility, label: 'Ticker', required: true },
    { key: 'companyName' as keyof ColumnVisibility, label: 'Company Name', required: false },
    { key: 'sector' as keyof ColumnVisibility, label: 'Sector', required: false },  // v3.3: Sector column
    { key: 'currentPrice' as keyof ColumnVisibility, label: 'Current Price', required: false },
    { key: 'overallScore' as keyof ColumnVisibility, label: 'Overall Score', required: true },
    { key: 'coverage' as keyof ColumnVisibility, label: 'Coverage', required: false },
    { key: 'technical' as keyof ColumnVisibility, label: 'Technical', required: false },
    { key: 'fundamental' as keyof ColumnVisibility, label: 'Fundamental', required: false },
    { key: 'newsMacro' as keyof ColumnVisibility, label: 'News/Macro', required: false },
    { key: 'social' as keyof ColumnVisibility, label: 'Social', required: false },
    { key: 'risk' as keyof ColumnVisibility, label: 'Risk', required: false },
    { key: 'institutional' as keyof ColumnVisibility, label: 'Institutional', required: false },
    // Performance columns
    { key: 'return1d' as keyof ColumnVisibility, label: '1D Return', required: false },
    { key: 'return7d' as keyof ColumnVisibility, label: '7D Return', required: false },
    { key: 'return30d' as keyof ColumnVisibility, label: '30D Return', required: false },
    { key: 'return90d' as keyof ColumnVisibility, label: '90D Return', required: false },
    { key: 'vsSpy' as keyof ColumnVisibility, label: 'vs SPY', required: false },
  ];

  const visibleCount = Object.values(visibility).filter(Boolean).length;

  const handleShowAll = () => {
    const allVisible = columns.reduce((acc, col) => {
      acc[col.key] = true;
      return acc;
    }, {} as ColumnVisibility);
    onVisibilityChange(allVisible);
  };

  const handleHideOptional = () => {
    const essentialOnly = columns.reduce((acc, col) => {
      acc[col.key] = col.required;
      return acc;
    }, {} as ColumnVisibility);
    onVisibilityChange(essentialOnly);
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="h-9 gap-2 bg-white dark:bg-gray-900 hover:bg-gradient-to-r hover:from-[#001F3F]/5 hover:to-[#00AEEF]/5"
        >
          <Columns3 className="h-4 w-4" />
          <span className="text-sm">Columns ({visibleCount}/{columns.length})</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>Toggle Columns</DropdownMenuLabel>
        <DropdownMenuSeparator />
        
        {columns.map((column) => (
          <DropdownMenuCheckboxItem
            key={column.key}
            checked={visibility[column.key]}
            onCheckedChange={(checked: boolean) => {
              onVisibilityChange({
                ...visibility,
                [column.key]: checked,
              });
            }}
            disabled={column.required}
            className="cursor-pointer"
          >
            {column.label}
            {column.required && (
              <span className="ml-2 text-xs text-gray-400">(required)</span>
            )}
          </DropdownMenuCheckboxItem>
        ))}
        
        <DropdownMenuSeparator />
        
        <div className="flex gap-2 p-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleShowAll}
            className="flex-1 h-7 text-xs"
          >
            Show All
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleHideOptional}
            className="flex-1 h-7 text-xs"
          >
            Essential Only
          </Button>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
