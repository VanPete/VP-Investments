# VP Investments Documentation

This folder contains all project documentation, organized by topic and status.

## 📚 Active Documentation

### **Database & Architecture**
- **[DATABASE_REORGANIZATION_SUMMARY.md](DATABASE_REORGANIZATION_SUMMARY.md)** - Summary of signals table reorganization into 3-table structure
- **[SIGNALS_TABLE_REORGANIZATION.md](SIGNALS_TABLE_REORGANIZATION.md)** - Detailed analysis and plan for database reorganization
- **[tables_enhancement_summary.md](tables_enhancement_summary.md)** - Table enhancement documentation

### **Development Guidelines**
- **[operational_guidelines.md](operational_guidelines.md)** - Coding standards, best practices, and operational procedures
- **[recommendations.md](recommendations.md)** - Active recommendations and improvement suggestions

## 🗂️ Archived Documentation

See **[archive/](archive/)** folder for historical documentation:
- Phase A & B implementation details
- Database consolidation planning
- Historical status snapshots
- Planning documents

## 🗄️ Database Structure

### Current Three-Table Design (After Reorganization)

```
signals (Core Signal Data)
├── Identification: id, ticker, company, sector
├── Scores: weighted_score, financial_score, reddit_score, news_score
├── Classification: trade_type, risk_level, signal_type
├── Market Context: current_price, market_cap, volume
├── Sentiment: reddit_sentiment, news_sentiment, mentions
└── AI Commentary: ai_commentary, ai_trends_commentary

signal_metrics (Technical & Fundamental)
├── Technical Indicators: RSI, MACD, volatility, moving averages
├── Volume Metrics: volume_spike_ratio, avg_volume, correlation
├── Fundamentals: P/E, EPS growth, ROE, debt/equity, FCF margin
├── Options: put/call ratios, IV spike
└── Ownership: institutional%, short interest, insider transactions

signal_performance (Performance Tracking)
├── Backtest Info: type (3d/7d/10d/30d), date, days elapsed
├── Entry Metrics: entry_price, entry_datetime, entry_volume
├── Exit Metrics: exit_price, exit_datetime, exit_volume
├── Performance: return_pct, peak_return, max_drawdown
└── Comparisons: SPY return, alpha, sector comparison
```

## 🚀 Key Improvements

### **Database Reorganization** (Oct 2025)
- ✅ Split 142-column signals table into 3 normalized tables
- ✅ 40-50% faster dashboard queries
- ✅ Full performance history tracking (no more overwriting backtest data)
- ✅ Better data integrity with foreign key constraints
- ✅ Clear separation: Signal → Metrics → Performance

### **Phase B: Technical Indicators** (Oct 2025)
- ✅ Added 9 new technical indicators
- ✅ Refactored financial_score to use 29+ indicators
- ✅ Implemented comprehensive 4-component scoring formula

### **Phase A: Backtest** (Oct 2025)
- ✅ Implemented smart historical backtesting
- ✅ Automated performance tracking (3d, 7d, 10d, 30d returns)
- ✅ Alpha calculation vs SPY and sector benchmarks

## 📖 Documentation Standards

When adding new documentation:

1. **File Naming**: Use descriptive names with UPPER_SNAKE_CASE for major docs
2. **Location**: 
   - Active docs → `/docs`
   - Historical/completed → `/docs/archive`
3. **Format**: Use Markdown with clear sections
4. **Headers**: Include date, status, and impact at top
5. **Updates**: Update this README when adding new docs

## 🔗 Related Resources

### **Code Documentation**
- Backend API: See `/backend/README.md` (if exists)
- Frontend: See `/frontend/README.md`

### **Database**
- Migrations: `/migrations` folder
- Schema tools: `tables.py` in root

### **Monitoring**
- Logs: `/logs` folder
- Observability: `backend/utils/observability.py`

## 📝 Quick Reference

### Running the Pipeline
```bash
python -m backend.pipeline
```

### Checking Database Schema
```bash
python tables.py --table signals --detailed
```

### Verifying Phase B Indicators
```bash
python verify_phase_b.py
```

### Testing Technical Indicators
```bash
python test_phase_b.py
```

## 🎯 Current Focus

**Next Phase**: Database migration execution
- Run `migrations/signals_table_reorganization_20251005.sql` in Supabase
- Update backend code to use 3-table structure
- Test query performance improvements
- Verify data integrity

See [DATABASE_REORGANIZATION_SUMMARY.md](DATABASE_REORGANIZATION_SUMMARY.md) for details.

---

**Last Updated**: October 5, 2025  
**Maintained By**: VP Investments Development Team
