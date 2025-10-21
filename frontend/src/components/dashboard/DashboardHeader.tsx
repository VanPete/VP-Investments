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
}: DashboardHeaderProps) {
  return (
    <div className="bg-white dark:bg-gray-900/50 border-b border-gray-200 dark:border-gray-800 shadow-lg rounded-2xl mx-4 mt-4 backdrop-blur-sm [border-image:linear-gradient(to_right,#001F3F,#00AEEF)_1] [border-top:2px_solid]">
      <div className="container mx-auto px-6 py-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          {/* Left: Title and Inline Stats */}
          <div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
              VanPiQ Signals Dashboard
            </h1>
            <div className="flex flex-wrap gap-x-2 text-sm text-gray-600 dark:text-gray-400 font-medium">
              <span>Latest: {formatTimestamp(metadata.timestamp)}</span>
              <span>•</span>
              <span>Tickers: {totalCount}</span>
              <span>•</span>
              <span>Source: {metadata.source}</span>
            </div>
          </div>

          {/* Right: File Selector and Refresh */}
          <div className="flex items-center gap-3">
            <Select value={selectedFile} onValueChange={onFileChange}>
              <SelectTrigger className="w-[280px] bg-gradient-to-r from-[#001F3F] to-[#00AEEF] text-white border-none hover:opacity-90 transition-opacity font-medium">
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
              className="bg-gradient-to-r from-[#001F3F] to-[#00AEEF] text-white border-none hover:opacity-90 transition-opacity"
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
