'use client';

import type { PipelineMetadata, FileOption } from '@/types/pipeline';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { RefreshCw } from 'lucide-react';
import { formatTimestamp } from '@/lib/utils';

interface DashboardHeaderProps {
  metadata: PipelineMetadata;
  availableFiles: FileOption[];
  selectedFile: string;
  onFileChange: (filename: string) => void;
  onRefresh: () => void;
  totalCount: number;
  displayedCount: number;
}

export function DashboardHeader({
  metadata,
  availableFiles,
  selectedFile,
  onFileChange,
  onRefresh,
  totalCount,
  displayedCount,
}: DashboardHeaderProps) {
  return (
    <div className="bg-white border-b border-gray-200 shadow-sm">
      <div className="container mx-auto px-4 py-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          {/* Left: Title and Stats */}
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">
              VP INVESTMENTS SIGNALS
            </h1>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-gray-600">
              <span>
                Latest Analysis: {formatTimestamp(metadata.timestamp)}
              </span>
              <span className="hidden md:inline">•</span>
              <span>
                Total Tickers: {totalCount} | Showing: {displayedCount}
              </span>
            </div>
          </div>

          {/* Right: File Selector and Refresh */}
          <div className="flex items-center gap-3">
            <Select value={selectedFile} onValueChange={onFileChange}>
              <SelectTrigger className="w-[280px]">
                <SelectValue placeholder="Select dataset" />
              </SelectTrigger>
              <SelectContent>
                {availableFiles.map((file) => (
                  <SelectItem key={file.filename} value={file.filename}>
                    {file.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Button
              variant="outline"
              size="icon"
              onClick={onRefresh}
              title="Refresh data"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Discovery Stats (if available) */}
        {metadata.discovery && (
          <Card className="mt-4 bg-gray-50">
            <CardContent className="py-3">
              <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-gray-700">
                <span>
                  Reddit Tickers: {metadata.discovery.reddit_tickers}
                </span>
                <span>
                  News Tickers: {metadata.discovery.news_tickers}
                </span>
                <span>
                  Total Universe: {metadata.discovery.total_universe}
                </span>
                <span>
                  Source: {metadata.source}
                </span>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
