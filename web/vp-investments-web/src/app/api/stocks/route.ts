// @ts-nocheck
// API endpoint to fetch stock analysis data
// This is currently returning mock data, but you can modify it to:
// 1. Read your CSV files from the Python output
// 2. Connect to your Python backend API
// 3. Read from a database

import { NextResponse } from 'next/server';
import { StockData } from '@/types/stock';

// Minimal shape for the Python Final Analysis row
type FinalRow = Record<string, string | number | null | undefined>;

export async function GET() {
  try {
    // Prefer live data from Python Flask backend; fallback to mock on failure
    const backend = process.env.PY_BACKEND_URL || 'http://localhost:5001';
    const resp = await fetch(`${backend}/api/outputs/final`, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`Backend responded ${resp.status}`);
    const payload = await resp.json();
    const rows: FinalRow[] = (payload.rows || []) as FinalRow[];

    const data: StockData[] = rows.map((r: FinalRow): StockData => ({
      rank: Number(r['Rank'] ?? 0),
      ticker: String((r['Ticker'] || '')).replace('$',''),
      company: String(r['Company'] || ''),
      trade_type: (['Swing','Long-Term','Balanced'].includes(String(r['Trade Type']))
        ? (r['Trade Type'] as StockData['trade_type'])
        : 'Balanced'),
      sector: (r['Sector'] as string) || undefined,
      run_datetime: String(r['Run Datetime'] || new Date().toISOString()),
      source: String(r['Source'] || 'Unknown'),
  // raw_score removed in favor of weighted_score only
      weighted_score: Number(r['Weighted Score'] ?? 0),
      active_signals: String(r['Top Factors'] || ''),
      mentions: Number(r['Mentions'] ?? 0) || 0,
      upvotes: Number(r['Upvotes'] ?? 0) || 0,
      sentiment: Number(r['Reddit Sentiment'] ?? 0) || 0,
      current_price: Number(r['Current Price'] ?? 0) || 0,
      price_1d: Number(r['Price 1D %'] ?? 0) || 0,
      price_7d: Number(r['Price 7D %'] ?? 0) || 0,
      volume: Number(r['Volume'] ?? 0) || 0,
      market_cap: r['Market Cap'] !== '' && r['Market Cap'] != null ? Number(r['Market Cap'] as number) : undefined,
      pe_ratio: r['P/E Ratio'] !== '' && r['P/E Ratio'] != null ? Number(r['P/E Ratio'] as number) : undefined,
      rsi: r['RSI'] !== '' && r['RSI'] != null ? Number(r['RSI'] as number) : undefined,
    }));

    return NextResponse.json({
      success: true,
      data,
      timestamp: new Date().toISOString(),
      total: data.length,
      sourcePath: payload.path || null,
    });
  } catch (error) {
    console.error('Error fetching stock data:', error);
    // Fallback mock minimal structure
    const mock: StockData[] = [];
    return NextResponse.json(
      { success: true, data: mock, timestamp: new Date().toISOString(), total: 0 },
      { status: 200 }
    );
  }
}

// Example of how to integrate with your Python backend:
/*
async function fetchFromPythonBackend() {
  // Option 1: Read CSV file directly
  const fs = require('fs');
  const csv = require('csvtojson');
  const csvFilePath = '../../../outputs/Final Analysis.csv';
  const jsonArray = await csv().fromFile(csvFilePath);
  return jsonArray;

  // Option 2: Call Python API
  const response = await fetch('http://localhost:8000/api/stocks');
  const data = await response.json();
  return data;
}
*/
