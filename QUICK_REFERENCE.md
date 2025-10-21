# VP INVESTMENTS - QUICK REFERENCE

## 📋 Daily Commands

### Update Pipeline Results & Deploy
```bash
# Run pipeline (saves directly to frontend/public/results/)
python run_full_pipeline.py

# Deploy
git add frontend/public/results/*.json
git commit -m "Update results"
git push origin main
```

### Local Development
```bash
cd frontend
npm run dev              # Start dev server at http://localhost:3000
npm run build            # Test production build
```

---

## 🗂️ File Structure

```
VP Investments/
├── config/
│   ├── weights.yaml              # Group & factor weights
│   ├── factor_to_group.yaml      # 158 factors with descriptions
│   └── methodology.yaml          # Methodology documentation
└── frontend/
    ├── public/
    │   └── results/              # Pipeline results (HTTP accessible)
    │       └── pipeline_results_*.json  # Timestamped results
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx          # Dashboard route
    │   │   └── methodology/      # Methodology page
    │   ├── components/
    │   │   ├── Navigation.tsx    # Header nav
    │   │   ├── dashboard/        # Dashboard components (4)
    │   │   └── methodology/      # Methodology components (3)
    │   ├── types/
    │   │   └── pipeline.ts       # TypeScript types
    │   └── lib/
    │       ├── pipeline-data.ts  # File reading utilities
    │       └── utils.ts          # Formatting utilities
    └── package.json
```

---

## 🎯 Key Features

### Dashboard
- **Top 10 View**: Default shows top 10 ranked tickers (expandable)
- **File Selector**: Dropdown to view historical pipeline runs
- **Filters**: Search, group, factor, coverage, score range
- **Expandable Rows**: Click ticker to see group score breakdown
- **Color Coding**: Green (high), Red (low), Gray (neutral)

### Methodology Page
- **Group Weights**: Visual bar charts for 6 signal groups
- **Scoring Explanation**: Step-by-step scoring process
- **Factor Library**: Searchable catalog of 158 factors

### Navigation
- **Dashboard**: Main signals ranking table
- **Methodology**: Weights, scoring, factor library

---

## ⚙️ Configuration Files

### weights.yaml
Controls signal group and factor weights. Edit to adjust scoring.

```yaml
group_weights:
  technical: 0.20              # 20% weight
  fundamental: 0.25            # 25% weight
  news_macro: 0.15             # etc.
  # ...

factor_weights_technical:
  price_30d_pct: 0.06          # 6% of technical group
  rsi_14: 0.05                 # 5% of technical group
  # ...
```

### factor_to_group.yaml
Factor descriptions displayed in UI. Edit to update descriptions.

```yaml
technical:
  price_1d_pct: "1-day percentage price change"
  price_7d_pct: "7-day percentage price change"
  # ... 158 total factors
```

### methodology.yaml
Methodology page content. Edit to update explanations.

```yaml
overview:
  title: "..."
  description: "..."
  key_principles: [...]

scoring:
  normalization: {...}
  # ...
```

---

## 🔧 npm Scripts

```bash
npm run dev              # Start development server
npm run build            # Build production version
npm run start            # Start production server
npm run lint             # Run ESLint
```

---

## 📊 Data Flow

### Build Time (Static Generation)
```
1. npm run build
   ↓
2. Read latest pipeline_results_*.json from public/results/
   ↓
3. Read weights.yaml, factor_to_group.yaml, methodology.yaml
   ↓
4. Generate static HTML pages
   ↓
5. Deploy to Vercel CDN
```

### Runtime (Client-Side)
```
1. User visits site (loads static HTML)
   ↓
2. User clicks file selector dropdown
   ↓
3. Fetch /results/pipeline_results_*.json
   ↓
4. Update dashboard with new data
   ↓
5. Apply filters/sorts client-side
```

---

## 🚨 Troubleshooting

### File selector not working
```bash
# Run pipeline to generate fresh results
python run_full_pipeline.py

# Rebuild frontend
cd frontend
npm run build
```

### Old data showing
```bash
# Run pipeline to generate new results
python run_full_pipeline.py

# Commit and push
git add frontend/public/results/*.json
git commit -m "Update results"
git push origin main
```

### Build fails
```bash
# Test locally first
cd frontend
npm run build

# Check for TypeScript errors
# Fix any errors, then redeploy
```

---

## 📱 Theme

- **Primary**: Navy Blue (#1e3a8a)
- **Accent**: Silver/Gray (#94a3b8)
- **Background**: Light Gray (#f9fafb)
- **Design**: Minimalist, no emojis

---

## ✅ Pre-Deployment Checklist

- [ ] Run `python run_full_pipeline.py` (saves to frontend/public/results/)
- [ ] Test `npm run build` (should pass)
- [ ] Test file selector in local dev
- [ ] Commit all changes
- [ ] Push to main

---

## 🔗 URLs

- **Local Dev**: <http://localhost:3000>
- **Production**: vanpiq.com (after deployment)
- **Vercel Dashboard**: Check deployment status

---

## 📞 Support

For issues, check:
1. `FRONTEND_V1_COMPLETE.md` - Full implementation details
2. `FILE_SELECTOR_FIX.md` - File selector implementation
3. `DEPLOYMENT_GUIDE.md` - Detailed deployment instructions

---

**Version**: 1.0  
**Status**: Production Ready ✅  
**Last Updated**: January 20, 2025
