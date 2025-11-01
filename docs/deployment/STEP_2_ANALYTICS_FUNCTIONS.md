# Step 2: Add Analytics Computation Functions

**Status**: 🔄 IN PROGRESS  
**Goal**: Implement 4 new functions in Phase 7 to compute spec-required metrics

---

## 📋 Overview

The VanPiQ spec requires 20 new analytics metrics. We need to add 4 computation functions to Phase 7:

1. **`compute_ic_series()`** - Predictive Strength metrics
2. **`compute_signal_correlations()`** - 158×158 factor correlation matrix
3. **`compute_predictive_metrics()`** - Hit rate, profit factor, win/loss ratio
4. **`compute_global_performance()`** - CAGR, Sharpe, Sortino, Calmar, Alpha/Beta

---

## 🔧 Function 1: compute_ic_series()

### Purpose
Calculate Rolling Rank Information Coefficient (RankIC) to measure predictive power of signals over time.

### Inputs
- Historical signals from multiple runs
- Corresponding forward returns from performance table

### Outputs
- `ic_series`: `[{"date": "YYYY-MM-DD", "ic": 0.42}, ...]`
- `ic_mean`: Average IC across all periods
- `ic_std`: Standard deviation of IC

### Algorithm
```python
async def _compute_ic_series(self, performance_data: List[Dict]) -> Dict[str, Any]:
    """
    Compute Rolling RankIC series + statistics.
    
    RankIC = Spearman rank correlation between signal scores and forward returns.
    """
    from scipy.stats import spearmanr
    
    # Group by date (baseline_date)
    by_date = defaultdict(list)
    for p in performance_data:
        date = p['baseline_date'][:10]  # YYYY-MM-DD
        by_date[date].append({
            'score': p['signals']['overall_score'],
            'return_30d': p.get('return_30d')  # Use 30d returns
        })
    
    # Calculate RankIC for each date
    ic_series = []
    for date in sorted(by_date.keys()):
        records = by_date[date]
        # Filter records with valid returns
        valid = [(r['score'], r['return_30d']) for r in records if r['return_30d'] is not None]
        
        if len(valid) >= 10:  # Need minimum sample size
            scores, returns = zip(*valid)
            ic, p_value = spearmanr(scores, returns)
            
            if not np.isnan(ic):
                ic_series.append({'date': date, 'ic': round(float(ic), 4)})
    
    # Calculate statistics
    if ic_series:
        ic_values = [x['ic'] for x in ic_series]
        return {
            'ic_series': ic_series,
            'ic_mean': round(float(np.mean(ic_values)), 4),
            'ic_std': round(float(np.std(ic_values)), 4)
        }
    
    return {
        'ic_series': [],
        'ic_mean': None,
        'ic_std': None
    }
```

### Requirements
- Historical data across multiple runs (need to query beyond current run)
- Sufficient sample size per date (≥10 signals)
- Use 30d returns as proxy for "forward returns"

---

## 🔧 Function 2: compute_signal_correlations()

### Purpose
Compute 158×158 pairwise correlations between all signal factors across all stocks.

### Inputs
- All factor scores from `signals_*` tables (technical, fundamental, etc.)
- Current run's signals

### Outputs
- `signal_correlations`: `[{"i": "RSI_14", "j": "MACD", "r": 0.42, "n": 1284}, ...]`
- `top_positive_pairs`: Top 10 positive correlations
- `top_negative_pairs`: Top 10 negative correlations

### Algorithm
```python
async def _compute_signal_correlations(self, run_id: str) -> Dict[str, Any]:
    """
    Compute 158×158 factor correlation matrix.
    
    Reads all factor scores for this run, computes pairwise correlations.
    """
    # 1. Fetch all factor scores for this run
    factors_by_ticker = await self._fetch_all_factors(run_id)
    
    # 2. Build factor matrix (rows=tickers, cols=158 factors)
    factor_names = self._get_all_factor_names()  # From factor_to_group.yaml
    matrix = []
    
    for ticker, factors in factors_by_ticker.items():
        row = [factors.get(fname) for fname in factor_names]
        matrix.append(row)
    
    df = pd.DataFrame(matrix, columns=factor_names)
    
    # 3. Compute correlation matrix
    corr_matrix = df.corr(method='pearson')
    
    # 4. Extract upper triangle (avoid duplicates)
    correlations = []
    for i in range(len(factor_names)):
        for j in range(i+1, len(factor_names)):
            r = corr_matrix.iloc[i, j]
            if not np.isnan(r):
                correlations.append({
                    'i': factor_names[i],
                    'j': factor_names[j],
                    'r': round(float(r), 4),
                    'n': len(df)
                })
    
    # 5. Sort and extract top pairs
    correlations.sort(key=lambda x: abs(x['r']), reverse=True)
    
    positive_pairs = sorted([c for c in correlations if c['r'] > 0], 
                           key=lambda x: x['r'], reverse=True)[:10]
    negative_pairs = sorted([c for c in correlations if c['r'] < 0], 
                           key=lambda x: x['r'])[:10]
    
    return {
        'signal_correlations': correlations,
        'top_positive_pairs': positive_pairs,
        'top_negative_pairs': negative_pairs
    }
```

### Requirements
- Access to `signals_technical`, `signals_fundamental`, etc. tables
- Join on `signal_id` for current run
- Handle missing factors gracefully (NULL → exclude from correlation)

---

## 🔧 Function 3: compute_predictive_metrics()

### Purpose
Calculate binary classification metrics for top-decile signals.

### Inputs
- Performance data with returns and signal scores

### Outputs
- `hit_rate_top_decile`: Fraction of top 10% signals with positive 30d returns
- `profit_factor`: Total gains / Total losses (30d)
- `win_loss_ratio`: Avg win / Avg loss (30d)

### Algorithm
```python
def _compute_predictive_metrics(self, performance_data: List[Dict]) -> Dict[str, Any]:
    """
    Compute predictive metrics for signal quality assessment.
    """
    # Filter to signals with completed 30d returns
    valid = [p for p in performance_data if p.get('return_30d') is not None]
    
    if len(valid) < 10:
        return {
            'hit_rate_top_decile': None,
            'profit_factor': None,
            'win_loss_ratio': None
        }
    
    # Sort by overall_score descending
    valid.sort(key=lambda x: x['signals']['overall_score'], reverse=True)
    
    # Top decile (10%)
    top_n = max(1, len(valid) // 10)
    top_decile = valid[:top_n]
    
    # Hit Rate: fraction with positive returns
    positive_returns = sum(1 for p in top_decile if p['return_30d'] > 0)
    hit_rate = positive_returns / len(top_decile)
    
    # Profit Factor: total gains / total losses
    all_returns = [p['return_30d'] for p in valid]
    gains = sum(r for r in all_returns if r > 0)
    losses = abs(sum(r for r in all_returns if r < 0))
    profit_factor = gains / losses if losses > 0 else None
    
    # Win/Loss Ratio: avg win / avg loss
    wins = [r for r in all_returns if r > 0]
    losses_list = [abs(r) for r in all_returns if r < 0]
    
    if wins and losses_list:
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses_list)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else None
    else:
        win_loss_ratio = None
    
    return {
        'hit_rate_top_decile': round(hit_rate, 4),
        'profit_factor': round(profit_factor, 4) if profit_factor else None,
        'win_loss_ratio': round(win_loss_ratio, 4) if win_loss_ratio else None
    }
```

---

## 🔧 Function 4: compute_global_performance()

### Purpose
Calculate portfolio-level risk/return metrics benchmarked against SPY/QQQ.

### Inputs
- Historical performance records across all runs
- SPY/QQQ returns from performance table

### Outputs
- `cagr`: Compound Annual Growth Rate
- `volatility`: Annualized standard deviation
- `sortino_ratio`: Risk-adjusted return (downside deviation)
- `calmar_ratio`: CAGR / Max Drawdown
- `alpha_vs_spy`, `beta_vs_spy`: Regression coefficients
- `alpha_vs_qqq`, `beta_vs_qqq`: Regression coefficients
- `rolling_sharpe_30d`: 30-day rolling Sharpe ratio series
- `benchmark_correlations`: Correlation with SPY/QQQ

### Algorithm
```python
async def _compute_global_performance(self, performance_data: List[Dict]) -> Dict[str, Any]:
    """
    Compute global portfolio performance metrics.
    """
    # Extract time series of returns
    returns = [p['return_30d'] for p in performance_data if p.get('return_30d')]
    spy_returns = [p['spy_return_30d'] for p in performance_data if p.get('spy_return_30d')]
    qqq_returns = [p['qqq_return_30d'] for p in performance_data if p.get('qqq_return_30d')]
    
    if len(returns) < 30:  # Need minimum data
        return self._null_global_metrics()
    
    returns_arr = np.array(returns) / 100  # Convert % to decimal
    spy_arr = np.array(spy_returns) / 100
    qqq_arr = np.array(qqq_returns) / 100
    
    # CAGR (assume 30d intervals, so 12 periods per year)
    cumulative_return = np.prod(1 + returns_arr) - 1
    years = len(returns_arr) / 12  # 30d = 1 month, 12 months = 1 year
    cagr = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else None
    
    # Volatility (annualized)
    volatility = np.std(returns_arr) * np.sqrt(12)  # Annualize monthly vol
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns_arr[returns_arr < 0]
    downside_dev = np.std(downside_returns) * np.sqrt(12) if len(downside_returns) > 0 else volatility
    sortino = (cagr * 100) / (downside_dev * 100) if downside_dev > 0 else None
    
    # Max Drawdown
    cumulative = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = abs(np.min(drawdowns))
    
    # Calmar Ratio
    calmar = (cagr * 100) / (max_drawdown * 100) if max_drawdown > 0 else None
    
    # Alpha/Beta vs SPY
    if len(spy_arr) == len(returns_arr):
        from sklearn.linear_model import LinearRegression
        X = spy_arr.reshape(-1, 1)
        y = returns_arr
        reg_spy = LinearRegression().fit(X, y)
        beta_spy = reg_spy.coef_[0]
        alpha_spy = reg_spy.intercept_
    else:
        beta_spy, alpha_spy = None, None
    
    # Alpha/Beta vs QQQ
    if len(qqq_arr) == len(returns_arr):
        X = qqq_arr.reshape(-1, 1)
        y = returns_arr
        reg_qqq = LinearRegression().fit(X, y)
        beta_qqq = reg_qqq.coef_[0]
        alpha_qqq = reg_qqq.intercept_
    else:
        beta_qqq, alpha_qqq = None, None
    
    # Rolling Sharpe (30-period window)
    rolling_sharpe = []
    for i in range(30, len(returns_arr)):
        window = returns_arr[i-30:i]
        sharpe = np.mean(window) / np.std(window) * np.sqrt(12) if np.std(window) > 0 else 0
        rolling_sharpe.append({
            'date': performance_data[i]['baseline_date'][:10],
            'sharpe': round(float(sharpe), 4)
        })
    
    # Benchmark Correlations
    corr_spy = np.corrcoef(returns_arr, spy_arr)[0, 1] if len(spy_arr) == len(returns_arr) else None
    corr_qqq = np.corrcoef(returns_arr, qqq_arr)[0, 1] if len(qqq_arr) == len(returns_arr) else None
    
    return {
        'cagr': round(float(cagr), 4) if cagr else None,
        'volatility': round(float(volatility), 4),
        'sortino_ratio': round(float(sortino), 4) if sortino else None,
        'calmar_ratio': round(float(calmar), 4) if calmar else None,
        'max_drawdown': round(float(max_drawdown), 4),
        'total_return': round(float(cumulative_return), 4),
        'alpha_vs_spy': round(float(alpha_spy), 6) if alpha_spy else None,
        'beta_vs_spy': round(float(beta_spy), 4) if beta_spy else None,
        'alpha_vs_qqq': round(float(alpha_qqq), 6) if alpha_qqq else None,
        'beta_vs_qqq': round(float(beta_qqq), 4) if beta_qqq else None,
        'rolling_sharpe_30d': rolling_sharpe,
        'benchmark_correlations': {
            'SPY': round(float(corr_spy), 4) if corr_spy else None,
            'QQQ': round(float(corr_qqq), 4) if corr_qqq else None
        }
    }
```

---

## 🔄 Integration Plan

### 1. Update `_calculate_all_metrics()` in phase7_analytics.py

```python
async def _calculate_all_metrics(self, performance_data: List[Dict]) -> Dict[str, Any]:
    # ... existing code ...
    
    # NEW: Compute IC Series
    self.logger.info("Computing IC series...")
    ic_metrics = await self._compute_ic_series(performance_data)
    metrics.update(ic_metrics)
    
    # NEW: Compute Signal Correlations (158×158)
    self.logger.info("Computing signal correlations...")
    corr_metrics = await self._compute_signal_correlations(metrics['run_id'])
    metrics.update(corr_metrics)
    
    # NEW: Compute Predictive Metrics
    self.logger.info("Computing predictive metrics...")
    pred_metrics = self._compute_predictive_metrics(performance_data)
    metrics.update(pred_metrics)
    
    # NEW: Compute Global Performance
    self.logger.info("Computing global performance...")
    perf_metrics = await self._compute_global_performance(performance_data)
    metrics.update(perf_metrics)
    
    return metrics
```

### 2. Update `_persist_analytics()` to include new columns

Already done in v3.4 - just need to add the new metrics to the INSERT statement.

---

## ✅ Testing Checklist

- [ ] IC series computes correctly with ≥10 signals per date
- [ ] Signal correlations matrix is 158×158 (or N×N for available factors)
- [ ] Top positive/negative pairs extracted correctly
- [ ] Hit rate in [0, 1] range
- [ ] Profit factor > 0 (if any trades)
- [ ] CAGR/Sortino/Calmar compute without NaN
- [ ] Rolling Sharpe has 30-period window
- [ ] Benchmark correlations in [-1, 1]

---

## 📝 Questions for User

1. **Historical Data Scope**: For IC series, should we:
   - Use only current run (limited data)
   - Query last N runs (e.g., 30 runs = ~30 data points)
   - Query all historical runs (most accurate but slow)

2. **Factor Access**: How to efficiently fetch all 158 factors?
   - Join all 6 `signals_*` tables per query
   - Cache factor structure from `factor_to_group.yaml`
   - Pre-compute and store in Phase 5?

3. **Performance Timeframe**: For global metrics (CAGR, etc.):
   - Use 30d returns only
   - Use all horizon returns (1d, 3d, 7d, etc.)
   - Compute separate metrics per horizon

4. **Factor Contributions Format**: The spec says:
   ```json
   { "technical": { "alpha_pct": 0.32, "vol_pct": 0.18 }, ... }
   ```
   - Is this already computed in Phase 4?
   - Or do we need to derive from factor scores in Phase 7?

---

**Next Step**: Answer these 4 questions, then implement the functions!
