import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import type { FileOption } from '@/types/pipeline';

export async function GET() {
  try {
    const resultsDir = path.join(process.cwd(), 'public', 'results');
    
    if (!fs.existsSync(resultsDir)) {
      return NextResponse.json([]);
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
        } as FileOption;
      })
      .filter((item): item is FileOption => item !== null)
      .sort((a, b) => b.timestamp.localeCompare(a.timestamp)); // Newest first

    return NextResponse.json(files);
  } catch (error) {
    console.error('Error reading results directory:', error);
    return NextResponse.json([]);
  }
}
