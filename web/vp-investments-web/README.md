# VP Investments Web Dashboard

A modern, responsive web dashboard for visualizing VP Investments stock analysis results. Built with Next.js 14, TypeScript, Tailwind CSS, and Chart.js.

## Features

- 📊 **Interactive Charts**: Score distribution and trade type breakdown
- 📈 **Real-time Data**: Live stock analysis with sentiment scoring
- 🎯 **Smart Filtering**: Sort and filter stocks by various metrics
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🔄 **Auto-refresh**: Real-time updates from your Python analysis

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS + shadcn/ui components
- **Charts**: Chart.js with react-chartjs-2
- **Icons**: Lucide React
- **Deployment**: Ready for Vercel

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

1. **Install dependencies:**

   ```bash
   npm install
   ```

2. **Start development server:**

   ```bash
   npm run dev
   ```

3. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Connecting to Your Python Backend

The dashboard currently uses mock data. To connect it to your VP Investments Python system:

### Option 1: CSV File Integration

```typescript
// In src/app/api/stocks/route.ts
import csvtojson from 'csvtojson';

const csvFilePath = '../../../outputs/Final Analysis.csv';
const stocks = await csvtojson().fromFile(csvFilePath);
```

### Option 2: Python API Integration

1. **Create a FastAPI server** in your Python project:

   ```python
   from fastapi import FastAPI
   from fastapi.middleware.cors import CORSMiddleware
   import pandas as pd
   
   app = FastAPI()
   app.add_middleware(CORSMiddleware, allow_origins=["*"])
   
   @app.get("/api/stocks")
   def get_stocks():
       df = pd.read_csv("outputs/Final Analysis.csv")
       return df.to_dict("records")
   ```

2. **Update the frontend** to call your API:

   ```typescript
   const response = await fetch('http://localhost:8000/api/stocks');
   const data = await response.json();
   ```

### Option 3: Database Integration

Connect to SQLite or PostgreSQL where your Python script stores results.

## Deployment

### Deploy to Vercel (Recommended)

1. **Push to GitHub:**

   ```bash
   git add .
   git commit -m "Initial dashboard commit"
   git push origin main
   ```

2. **Connect to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Deploy automatically

---

**VP Investments Dashboard** - Turning Reddit sentiment and technical analysis into actionable investment insights 🚀
