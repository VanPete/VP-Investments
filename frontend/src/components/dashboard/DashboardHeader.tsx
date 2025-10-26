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
import Image from 'next/image';
import { Badge } from '@/components/ui/badge';
import { ThemeToggle } from '@/components/theme-toggle';

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
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          {/* Left: Logo + Title and Inline Stats */}
          <div className="flex flex-col gap-3">
            {/* VanPIQ Logo */}
            <div className="flex items-center">
              <Image 
                src="/vanpiq-logo.svg" 
                alt="VanPIQ" 
                width={120} 
                height={40}
                className="h-[40px] w-auto transition-all duration-300 hover:drop-shadow-[0_0_8px_rgba(0,174,239,0.5)]"
              />
            </div>
            
            {/* Title with Discovery Badge Tooltip */}
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                VanPiQ Signals Dashboard
              </h1>
              
              {/* Discovery Stats Tooltip on Hover */}
              {metadata.discovery && (
                <div className="relative group">
                  <Badge 
                    variant="outline" 
                    className="cursor-help bg-gradient-to-r from-[#001F3F]/10 to-[#00AEEF]/10 border-[#00AEEF]/30 text-gray-900 dark:text-gray-100"
                  >
                    {totalCount} tickers
                  </Badge>
                  
                  {/* Hover Tooltip */}
                  <div className="absolute left-0 top-full mt-2 hidden group-hover:block z-50 w-64">
                    <Card className="shadow-xl border-[#00AEEF]/50">
                      <CardContent className="p-3 space-y-2">
                        <div className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
                          Discovery Breakdown
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <Badge variant="outline" className="bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950 dark:text-orange-300">
                            Reddit
                          </Badge>
                          <span className="font-semibold text-gray-900 dark:text-gray-100">
                            {metadata.discovery.reddit_tickers}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300">
                            News
                          </Badge>
                          <span className="font-semibold text-gray-900 dark:text-gray-100">
                            {metadata.discovery.news_tickers}
                          </span>
                        </div>
                        <div className="pt-2 mt-2 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-600 dark:text-gray-400">
                          Total discovered: {metadata.discovery.total_universe}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
            </div>
            
            {/* Inline Stats */}
            <div className="flex flex-wrap gap-x-2 text-sm text-gray-600 dark:text-gray-400 font-medium">
              <span>Latest: {formatTimestamp(metadata.timestamp)}</span>
              <span>•</span>
              <span>Source: {metadata.source}</span>
              <span>•</span>
              <span>Last updated: {new Date().toLocaleTimeString()}</span>
            </div>
          </div>

          {/* Right: File Selector and Refresh */}
          <div className="flex items-center gap-3 md:mt-12">
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

            <ThemeToggle />
          </div>
        </div>
      </div>
    </div>
  );
}
