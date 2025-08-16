'use client';

import React, { useState } from 'react';
import { StockData } from '@/types/stock';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface StocksTableProps {
  stocks: StockData[];
}

export function StocksTable({ stocks }: StocksTableProps) {
  const [sortBy, setSortBy] = useState<keyof StockData>('weighted_score');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const sortedStocks = [...stocks].sort((a, b) => {
    const aVal = a[sortBy];
    const bVal = b[sortBy];
    
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortDirection === 'asc' 
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal);
    }
    
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    }
    
    return 0;
  });

  const paginatedStocks = sortedStocks.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const totalPages = Math.ceil(stocks.length / itemsPerPage);

  const handleSort = (column: keyof StockData) => {
    if (sortBy === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortDirection('desc');
    }
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(price);
  };

  const formatPercent = (value: number) => {
    const v = Number.isFinite(value) ? value : 0;
    return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
  };

  const getTradeTypeColor = (tradeType: string) => {
    switch (tradeType) {
      case 'Swing':
        return 'bg-blue-100 text-blue-800';
      case 'Long-Term':
        return 'bg-green-100 text-green-800';
      case 'Balanced':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Top Stock Picks</CardTitle>
        <CardDescription>
          Showing {paginatedStocks.length} of {stocks.length} stocks
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b">
                <th className="text-left p-2">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('rank')}
                    className="h-auto p-0 font-semibold"
                  >
                    Rank
                  </Button>
                </th>
                <th className="text-left p-2">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('ticker')}
                    className="h-auto p-0 font-semibold"
                  >
                    Ticker
                  </Button>
                </th>
                <th className="text-left p-2">Company</th>
                <th className="text-left p-2">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('weighted_score')}
                    className="h-auto p-0 font-semibold"
                  >
                    Score
                  </Button>
                </th>
                <th className="text-left p-2">Trade Type</th>
                <th className="text-left p-2">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('current_price')}
                    className="h-auto p-0 font-semibold"
                  >
                    Price
                  </Button>
                </th>
                <th className="text-left p-2">1D %</th>
                <th className="text-left p-2">7D %</th>
                <th className="text-left p-2">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort('sentiment')}
                    className="h-auto p-0 font-semibold"
                  >
                    Sentiment
                  </Button>
                </th>
              </tr>
            </thead>
            <tbody>
              {paginatedStocks.map((stock) => (
                <tr key={stock.ticker} className="border-b hover:bg-gray-50">
                  <td className="p-2 font-mono">{stock.rank}</td>
                  <td className="p-2 font-bold text-blue-600">{stock.ticker}</td>
                  <td className="p-2 max-w-48 truncate">{stock.company}</td>
                  <td className="p-2">
                    <span className="font-semibold">
                      {stock.weighted_score.toFixed(2)}
                    </span>
                  </td>
                  <td className="p-2">
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium ${getTradeTypeColor(
                        stock.trade_type
                      )}`}
                    >
                      {stock.trade_type}
                    </span>
                  </td>
                  <td className="p-2 font-mono">{formatPrice(stock.current_price)}</td>
                  <td className="p-2">
                    <div className="flex items-center">
                      {stock.price_1d >= 0 ? (
                        <TrendingUp className="w-4 h-4 text-green-600 mr-1" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-600 mr-1" />
                      )}
                      <span
                        className={
                          stock.price_1d >= 0 ? 'text-green-600' : 'text-red-600'
                        }
                      >
                        {formatPercent(stock.price_1d)}
                      </span>
                    </div>
                  </td>
                  <td className="p-2">
                    <div className="flex items-center">
                      {stock.price_7d >= 0 ? (
                        <TrendingUp className="w-4 h-4 text-green-600 mr-1" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-600 mr-1" />
                      )}
                      <span
                        className={
                          stock.price_7d >= 0 ? 'text-green-600' : 'text-red-600'
                        }
                      >
                        {formatPercent(stock.price_7d)}
                      </span>
                    </div>
                  </td>
                  <td className="p-2">
                    <div className="flex items-center">
                      <div
                        className={`w-3 h-3 rounded-full mr-2 ${
                          stock.sentiment >= 0.5
                            ? 'bg-green-500'
                            : stock.sentiment >= 0
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                        }`}
                      />
                      {stock.sentiment.toFixed(2)}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex justify-between items-center mt-4">
          <div className="text-sm text-gray-600">
            Page {currentPage} of {totalPages}
          </div>
          <div className="flex space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
              disabled={currentPage === totalPages}
            >
              Next
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
