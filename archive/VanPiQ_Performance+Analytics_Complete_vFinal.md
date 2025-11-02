# VanPiQ — Performance + Analytics Unified Spec (Developer-Facing)

**Scope:** Implement the refactored **Performance (per-run)** and **Analytics (global)** tabs and align the database with **column-only** additions (no new tables, no renames/drops).  
**Style:** Compact, imperative, implementation-oriented.  
**Percent style:** Store as **decimal fractions** (e.g., 0.1234 = 12.34%).  
**Benchmarks:** Support **SPY** and **QQQ** at the global level.  

---

## 1) PERFORMANCE TAB (Per-Run)

### 1.1 Purpose
- Per-run, ticker-focused. Show realized performance by horizon, countdown to next unlock, and quick quality signals.
- **Do not** include global analytics (Sharpe/Sortino/Calmar/IC/Heatmaps/Buckets).

### 1.2 Files & Routing
- Move:
  - `/frontend/src/dashboard/PerformanceTab.tsx` → `/frontend/src/performance/PerformanceTab.tsx`
  - `/frontend/src/dashboard/PerformanceCountdown.tsx` → `/frontend/src/performance/PerformanceCountdown.tsx`
- Create barrel `/frontend/src/performance/index.ts` exporting both.
- Update imports/routes; order: **Performance first**, **Analytics second**.

### 1.3 Layout
- **Header:** Ticker • Sector • MktCap • β • Baseline Date • Last Updated (compact `text-sm` / `text-xs`, `gap-2`).
- **Row A (Main):**
  - **Per‑Horizon Grid**: rows for `1D • 3D • 7D • 10D • 14D • 30D • 90D`. Columns:
    - VP Return (%)
    - SPY Return (%)
    - QQQ Return (%)
    - Alpha vs SPY (%)
    - Alpha vs QQQ (%)
    - Status (Completed / Pending)
  - **Countdown Card**: time to next unlock; auto-hide after 90D.
- **Row B (Insight):**
  - **Alpha Sparkline (Run-only)**: cumulative alpha vs selected benchmark (SPY/QQQ toggle). X = completed horizons in order.
  - **Horizon Quality Summary**: earliest completed, most recent, best/worst horizon, “Beating {BM}: X/Y”.
- **Row C (Optional):**
  - **Top Signal Contributors (latest completed horizon)**: top 3 with contribution % (only if attribution exists).
  - **Data Staleness**: latest price/factor/backtest refresh; warn if stale (>24h), otherwise hidden.

### 1.4 UX/Logic Rules
- Missing data → **Pending** (neutral). No error badges.
- Color only VP Return + Alpha cells (green/red by sign); numeric monospace, right-aligned.
- Alpha formula (tooltips): `alpha_vs_spy = return - spy_return` (analogous for QQQ).
- SPY/QQQ toggle affects Alpha Sparkline + “Beating X/Y” only.

### 1.5 Telemetry & A11y
- Emit counts of completed horizons, missing values, and pending total.
- Use ▲/▼ glyphs where color indicates sign; keyboard-nav for toggles/popovers.

---

## 2) ANALYTICS TAB (Global)

### 2.1 Purpose
- Global aggregates, cross-run intelligence. **All components obey global controls** (Score Bucket + Time Interval).

### 2.2 Global Controls
- Sticky header: **Score Bucket** (All/Top10%/etc.), **Time Interval** (1d/7d/30d/90d/custom).
- Persist to URL query params; hydrate on mount.
- Single fetch: `/analytics/global?bucket=...&interval=...` that returns all payloads required by subsections.

### 2.3 Sections (compact, data-first)
- **Global Performance Summary (cards):** CAGR • Volatility • Sharpe • Sortino • Calmar • Alpha/Beta vs SPY/QQQ.
- **Predictive Strength:** Rolling RankIC (line with 30-period MA), plus IC mean/std, Hit Rate (Top Decile), Profit Factor, Win/Loss Ratio.
- **Score Bucket Performance:** Average Return by Bucket, Win Rate by Bucket, Distribution counts.
- **Correlation Heatmap (All Signals):**
  - Modes: **Signals (default)** ~158×158, **Groups** (existing).
  - Features: diagonal masked, clustered ordering (toggle alphabetical), multi-select search/filter, threshold slider (|r|), abs toggle, group stripes, hover (A/B/r/n/group), click (rolling corr modal), export PNG/CSV, precomputed **top positive/negative pairs**.
- **Factor Contributions:** Alpha% and Vol% by group (normalized 0..1). Clicking a group filters heatmap axes.
- **Backtest vs Benchmarks:** cumulative series (VP/ SPY/ QQQ) + Rolling Sharpe (30d) sparkline + benchmark correlations; toggle log scale / relative return.
- **Meta (optional, collapsible):** Data coverage, turnover, OOS vs IS delta, drift score (deferred for now).

---

## 3) DATABASE — Column-Only Updates (No New Tables/Renames)

**Table:** `public.analytics` (keep as-is, additive columns only).  
**All new scalars:** `numeric`. **Series/arrays:** `jsonb`. **Percentages:** decimal fractions.

### 3.1 New Columns

#### A) Predictive Strength
- `ic_series jsonb` — `[{ "date":"YYYY-MM-DD", "ic": <numeric> }, ...]`
- `ic_mean numeric`
- `ic_std numeric`
- `hit_rate_top_decile numeric`        -- fraction 0..1
- `profit_factor numeric`
- `win_loss_ratio numeric`

#### B) Global Performance Summary (benchmarked)
- `cagr numeric`
- `volatility numeric`
- `sortino_ratio numeric`
- `calmar_ratio numeric`
- `alpha_vs_spy numeric`
- `beta_vs_spy numeric`
- `alpha_vs_qqq numeric`
- `beta_vs_qqq numeric`

#### C) Backtest Extras
- `rolling_sharpe_30d jsonb`           -- `[{ "date":"YYYY-MM-DD", "sharpe": <numeric> }, ...]`
- `benchmark_correlations jsonb`       -- `{ "SPY": 0.68, "QQQ": 0.57 }`

#### D) Signal-Level Correlations (All Signals Heatmap)
- `signal_correlations jsonb`          -- `[{ "i":"RSI_14", "j":"MACD", "r":0.42, "n":1284 }, ...]`
- `top_positive_pairs jsonb`           -- `[{ "i":"…", "j":"…", "r": … }, ...]` (r>0)
- `top_negative_pairs jsonb`           -- `[{ "i":"…", "j":"…", "r": … }, ...]` (r<0)

> Keep `factor_correlations jsonb` for **group-level** snapshots.

#### E) Factor Contributions (normalized, consistent with pipeline)
**Contract change only (no new column):** standardize `factor_contributions jsonb` to store both **alpha** and **volatility** contributions as **fractions** 0..1 per group. Example:
```json
{
  "technical":                 { "alpha_pct": 0.32, "vol_pct": 0.18 },
  "fundamental":               { "alpha_pct": 0.21, "vol_pct": 0.22 },
  "news_macro":                { "alpha_pct": 0.14, "vol_pct": 0.12 },
  "social_alternative":        { "alpha_pct": 0.09, "vol_pct": 0.15 },
  "risk_stability":            { "alpha_pct": 0.16, "vol_pct": 0.20 },
  "institutional_smart_money": { "alpha_pct": 0.08, "vol_pct": 0.13 }
}
```

### 3.2 Inline SQL (Additive Only)
```sql
ALTER TABLE public.analytics
  -- Predictive Strength
  ADD COLUMN IF NOT EXISTS ic_series jsonb,
  ADD COLUMN IF NOT EXISTS ic_mean numeric,
  ADD COLUMN IF NOT EXISTS ic_std numeric,
  ADD COLUMN IF NOT EXISTS hit_rate_top_decile numeric,
  ADD COLUMN IF NOT EXISTS profit_factor numeric,
  ADD COLUMN IF NOT EXISTS win_loss_ratio numeric,

  -- Global Performance Summary
  ADD COLUMN IF NOT EXISTS cagr numeric,
  ADD COLUMN IF NOT EXISTS volatility numeric,
  ADD COLUMN IF NOT EXISTS sortino_ratio numeric,
  ADD COLUMN IF NOT EXISTS calmar_ratio numeric,
  ADD COLUMN IF NOT EXISTS alpha_vs_spy numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_spy numeric,
  ADD COLUMN IF NOT EXISTS alpha_vs_qqq numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_qqq numeric,

  -- Backtest Extras
  ADD COLUMN IF NOT EXISTS rolling_sharpe_30d jsonb,
  ADD COLUMN IF NOT EXISTS benchmark_correlations jsonb,

  -- Signal-Level Correlations
  ADD COLUMN IF NOT EXISTS signal_correlations jsonb,
  ADD COLUMN IF NOT EXISTS top_positive_pairs jsonb,
  ADD COLUMN IF NOT EXISTS top_negative_pairs jsonb
;
```

**Notes:**
- No drops/renames. Backwards compatible.
- Continue writing bucket stats to `score_bucket_performance jsonb` and group-level to `factor_correlations jsonb`.
- FE reads normalized `factor_contributions` with `{alpha_pct, vol_pct}` per group (fractions).

---

## 4) API Contracts

### 4.1 Performance (Per-Run)
- **Source:** existing per-run endpoints + FE composition (no DB changes required for per-run in this spec).
- **Behavior:** populate grid/sparkline/summary using run’s realized horizon metrics; SPY/QQQ toggle on sparkline/summary.

### 4.2 Analytics (Global)
- **GET** `/analytics/global?bucket=<id>&interval=<id>` → returns:
  - `perf_summary`: `cagr, volatility, sharpe, sortino, calmar, alpha_vs_spy, beta_vs_spy, alpha_vs_qqq, beta_vs_qqq`
  - `predictive`: `ic_series, ic_mean, ic_std, hit_rate_top_decile, profit_factor, win_loss_ratio`
  - `buckets`: from `score_bucket_performance jsonb`
  - `correlations`: `signal_correlations, factor_correlations, top_positive_pairs, top_negative_pairs`
  - `factor_contributions`: unified alpha/vol schema (jsonb)
  - `backtest`: `backtest_cumulative_returns`, `rolling_sharpe_30d`, `benchmark_correlations`

---

## 5) Frontend Behavior Matrix (Controls → Sections)

| Control | Sections updated |
|---|---|
| Score Bucket | Bucket charts, distributions, any summary that depends on selection |
| Interval | Global perf summary, RankIC series window, backtest series window, bucket stats, correlations scope |
| Signals/Groups toggle | Heatmap data source + top pairs |
| Abs/Threshold/Search | Heatmap rendering only (no refetch) |

---

## 6) Pipeline Alignment Notes (Phases 3 & 4)

- **Phase 3 (normalize):** robust z-scoring, winsorization/clip, preserve factor→group structure.  
- **Phase 4 (score assemble):** apply group/factor weights → overall score; produce group-level contribution weights normalized to [0..1].  
- **This spec consumes those outputs** to compute aggregates (RankIC, bucket stats, contributions, etc.).  
- **No algorithm changes** required; only ensure the Analytics job writes the new columns/JSON shapes described above.  

---

## 7) QA / Done

- **Performance Tab**
  - Files moved; imports updated; nav order correct.
  - Horizon grid renders all horizons with Pending/Completed; correct alpha math.
  - Countdown accurate; hides post-90D.
  - Alpha Sparkline + Horizon Quality reflect SPY/QQQ toggle.
  - Contributors/staleness sections conditionally visible; compact spacing.

- **Analytics Tab**
  - Global controls persist via URL; one fetch drives all subsections.
  - Cards populate from new scalar columns; % formatted from fractions.
  - RankIC series + stats render; Score Bucket charts obey controls.
  - Signals heatmap supports search, threshold, abs toggle; top pairs read from JSON.
  - Factor contributions (alpha%/vol%) render; clicking a group filters the heatmap.
  - Backtest: cumulative + rolling Sharpe + benchmark correlations render.

- **DB**
  - New columns added; no drops/renames.
  - `factor_contributions` uses unified `{alpha_pct, vol_pct}` JSON contract (fractions).
  - Data backfilled/populated by Analytics job; FE shows expected results.

---

**End of Spec**
