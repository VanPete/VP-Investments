# Future Signals - To Be Implemented

## Macro Economic Indicators (news_macro group)

These signals require external data sources (FRED API, economic databases) and are currently stubbed out with NaN values.

### 1. unemployment_rate
- **Source**: FRED (Federal Reserve Economic Data) - Series `UNRATE`
- **Description**: Current U.S. unemployment rate (%)
- **Update Frequency**: Monthly
- **API**: https://fred.stlouisfed.org/series/UNRATE
- **Weight**: 0.03 (3%)
- **Implementation Notes**: 
  - Requires FRED API key
  - Use most recent monthly value
  - Normalize as z-score across historical distribution

### 2. gdp_growth_rate
- **Source**: FRED - Series `A191RL1Q225SBEA`
- **Description**: Real GDP growth rate (quarterly, annualized %)
- **Update Frequency**: Quarterly
- **API**: https://fred.stlouisfed.org/series/A191RL1Q225SBEA
- **Weight**: 0.03 (3%)
- **Implementation Notes**:
  - Quarterly data, use most recent
  - Lag time: ~1 month after quarter end
  - Consider forward-looking indicators (Atlanta Fed GDPNow)

### 3. inflation_rate
- **Source**: FRED - Series `CPIAUCSL` (CPI All Urban Consumers)
- **Description**: Year-over-year CPI inflation rate (%)
- **Update Frequency**: Monthly
- **API**: https://fred.stlouisfed.org/series/CPIAUCSL
- **Weight**: 0.02 (2%)
- **Implementation Notes**:
  - Calculate YoY % change
  - Consider core CPI (`CPILFESL`) for less volatility
  - PCE Price Index (`PCEPI`) is Fed's preferred metric

---

## Implementation Approach

### Option A: FRED API Integration
```python
# pip install fredapi
from fredapi import Fred

fred = Fred(api_key='YOUR_API_KEY')

# Fetch latest values
unemployment = fred.get_series('UNRATE').iloc[-1]
gdp_growth = fred.get_series('A191RL1Q225SBEA').iloc[-1]
cpi = fred.get_series('CPIAUCSL')
inflation = ((cpi.iloc[-1] / cpi.iloc[-13]) - 1) * 100  # YoY
```

### Option B: Alternative Sources
- **Yahoo Finance**: Limited macro data
- **Alpha Vantage**: Economic indicators API
- **Quandl/Nasdaq Data Link**: Comprehensive economic data
- **World Bank API**: Global economic indicators
- **BEA API**: Direct Bureau of Economic Analysis data

### Option C: Web Scraping
- BLS.gov (Bureau of Labor Statistics)
- BEA.gov (Bureau of Economic Analysis)
- Less reliable, requires maintenance

---

## Timeline & Priority

### Phase 1 (High Priority)
- Set up FRED API integration
- Implement basic fetching for all 3 indicators
- Add caching (daily update sufficient)

### Phase 2 (Enhancement)
- Add historical backfill for baseline calculations
- Implement percentile/z-score normalization
- Add forecast indicators (GDPNow, inflation expectations)

### Phase 3 (Advanced)
- Regional economic indicators
- Leading economic indicators (LEI)
- Sector-specific economic data
- International economic data for global stocks

---

## Current Status

**Status**: ⏸️ Paused - Awaiting API setup

**Reason**: Focus on yfinance-based signals first (no external dependencies)

**Expected Impact**: 
- news_macro group: +17% coverage (3/17 factors currently at 0%)
- Overall pipeline: +1-2% improvement

**Weight**: Low priority - only 8% of news_macro group (1.2% of overall)

---

## Notes

- These 3 factors currently return NaN in all calculations
- news_macro group at 67.9% without them (64.9% with them included)
- Can be safely ignored for now without major impact on overall scoring
- Macro data changes slowly (monthly/quarterly) vs daily stock data
- Consider implementing after all yfinance-based signals are optimized

---

**Last Updated**: January 2025 (Updated after factor removal)  
**Status**: Macro factors officially removed from active pipeline

---

## REMOVED FACTORS - Insider Trading Analysis (3 factors)

The following insider trading factors were removed because they require detailed SEC Form 4 filing parsing:

### 4. insider_buy_ratio
- **Source**: SEC Form 4 filings via Edgar
- **Description**: Proportion of insider transactions that are buys (0-1)
- **Calculation**: `buy_count / (buy_count + sell_count)`
- **Time Window**: Last 6 months
- **Weight**: 0.07 (7%) - **REMOVED**
- **Group**: insider_activity
- **Reason for Removal**: yfinance only provides generic "Sale/Purchase" without transaction codes
- **Required Data**: Transaction codes (P=Purchase, S=Sale, A=Award, F=Tax, M=Exercise, G=Gift, L=Small)

### 5. insider_sell_ratio  
- **Source**: SEC Form 4 filings via Edgar
- **Description**: Proportion of insider transactions that are sells (0-1)
- **Calculation**: `sell_count / (buy_count + sell_count)`
- **Time Window**: Last 6 months
- **Weight**: 0.04 (4%) - **REMOVED**
- **Group**: insider_activity
- **Reason for Removal**: Same as insider_buy_ratio

### 6. insider_buy_score
- **Source**: SEC Form 4 filings via Edgar
- **Description**: Net insider sentiment score (-1 to +1)
- **Calculation**: `insider_buy_ratio - insider_sell_ratio`
- **Time Window**: Last 6 months
- **Weight**: 0.05 (5%) - **REMOVED**
- **Group**: insider_activity
- **Reason for Removal**: Composite of above two factors

**Total Insider Weight Removed**: 16% from insider_activity group

**Implementation Requirements**:
1. SEC Edgar Form 4 XML parser
2. Transaction code classification system
3. Exclude non-meaningful transactions (gifts, option exercises)
4. Focus on open market purchases (Code P) vs sales (Code S)
5. Consider alternative: Commercial APIs (InsiderScreener, OpenInsider)

---

## REMOVED FACTORS - Institutional Ownership Deltas (3 factors)

The following institutional factors were removed because they require historical snapshot caching:

### 7. inst_ownership_delta_3m
- **Source**: 13F filings via yfinance institutional_holders (requires caching)
- **Description**: Change in total institutional ownership % over 3 months
- **Calculation**: `current_ownership_pct - ownership_pct_3m_ago`
- **Weight**: 0.07 (7%) - **REMOVED**
- **Group**: institutional_smart_money
- **Reason for Removal**: yfinance only provides current snapshot, no historical data

### 8. inst_holder_count_delta_3m
- **Source**: 13F filings via yfinance institutional_holders (requires caching)
- **Description**: Change in number of institutional holders over 3 months
- **Calculation**: `current_holder_count - holder_count_3m_ago`
- **Weight**: 0.03 (3%) - **REMOVED**
- **Group**: institutional_smart_money
- **Reason for Removal**: Same as above

### 9. institutional_turnover_qoq
- **Source**: 13F filings comparison between quarters
- **Description**: Quarterly turnover rate of institutional holdings
- **Calculation**: `sum(abs(position_changes)) / total_shares_held`
- **Weight**: 0.03 (3%) - **REMOVED**
- **Group**: institutional_smart_money
- **Reason for Removal**: Requires comparing holdings position-by-position across quarters

**Total Institutional Weight Removed**: 13% from institutional_smart_money group

**Implementation Requirements**:
1. SQLite database for institutional holding snapshots
2. Quarterly snapshot capture script (run 50 days after quarter end)
3. Position-level change tracking (new positions, closed positions, size changes)
4. Scheduled automation (cron job for Q1, Q2, Q3, Q4 snapshots)
5. Delta calculation engine comparing current vs historical snapshots

**Snapshot Schedule**:
- Q1 (Jan-Mar): Capture May 15
- Q2 (Apr-Jun): Capture Aug 15
- Q3 (Jul-Sep): Capture Nov 15  
- Q4 (Oct-Dec): Capture Feb 15

---

## Summary of Removed Factors

**Total Removed**: 9 factors
- **Macro Economic**: 3 factors (unemployment, GDP, inflation) - 6% weight
- **Insider Trading**: 3 factors (buy/sell ratios, buy score) - 16% weight
- **Institutional Deltas**: 3 factors (ownership/holder deltas, turnover) - 13% weight
- **Total Weight Impact**: 35% of factor weights removed

**Group Weight Adjustments**:
- `macro_indicators`: 18% → 12% (reduced by 6%)
- `insider_activity`: 23% → 7% (reduced by 16%)
- `institutional_smart_money`: 23% → 12% (reduced by 11%, added 2% for inst_concentration_top10)

**Config Files Updated**:
- `config/factor_to_group.yaml`: Factors commented out with removal reason
- `config/weights.yaml`: Weights removed, group totals adjusted
- `backend/phases/phase2_calculate.py`: Calculation code replaced with comments

**Factor Count**: 167 → 158 total factors

---

## Implementation Priority

**High Priority** (Institutional Deltas - 13% weight):
- Effort: Medium (2 weeks development + 1 quarter data collection)
- Impact: High (institutional flow is strong signal)
- ROI: Best return on investment

**Medium Priority** (Macro Indicators - 6% weight):
- Effort: Low (1-2 days, simple FRED API)
- Impact: Medium (broad market context)
- ROI: Quick win

**Low Priority** (Insider Trading - 16% weight):
- Effort: High (2-3 weeks, complex parsing)
- Impact: Medium (insider data is noisy)
- ROI: Consider commercial API instead

---

**Related Files**:
- `config/factor_to_group.yaml`
- `config/weights.yaml`
- `backend/phases/phase2_calculate.py`
- Implementation templates in comments above
