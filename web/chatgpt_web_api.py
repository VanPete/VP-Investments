"""
chatgpt_web_api.py
Web API endpoints for ChatGPT integration with the VP Investments dashboard.
Adds optional caching, compression, and fast JSON responses.
Gracefully degrades if optional libs are not installed.
"""

import os
import sys
import json
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime
import glob

import pandas as pd
from flask import Flask, request, Response
from flask import redirect
from flask_cors import CORS

# Optional imports with safe fallbacks
try:
    from flask_caching import Cache  # type: ignore
except Exception:  # pragma: no cover
    class Cache:  # type: ignore
        def __init__(self, app=None, config=None):
            pass
        def cached(self, *args, **kwargs):
            def _decorator(func):
                return func
            return _decorator

try:
    from flask_compress import Compress  # type: ignore
except Exception:  # pragma: no cover
    class Compress:  # type: ignore
        def __init__(self, app=None):
            pass

try:
    import orjson  # type: ignore
    def _dumps(obj) -> bytes:
        return orjson.dumps(obj)
except Exception:  # pragma: no cover
    def _dumps(obj) -> bytes:
        return json.dumps(obj).encode("utf-8")
# Ensure project root is on sys.path so 'config' package is importable when running from /web
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.config import WEB_API_DEBUG, WEB_API_PORT

# Import our ChatGPT integrator (after we install openai)
# from processors.chatgpt_integrator import ChatGPTIntegrator, enhance_dataframe_with_chatgpt

app = Flask(__name__)
CORS(app)
Compress(app)

# Optional Redis cache backend
try:
    from config.config import REDIS_URL
    if REDIS_URL:
        cache = Cache(app, config={
            "CACHE_TYPE": "RedisCache",
            "CACHE_REDIS_URL": REDIS_URL,
            "CACHE_DEFAULT_TIMEOUT": 300
        })
    else:
        cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})
except Exception:
    cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})

logger = logging.getLogger(__name__)

# Optional Sentry
try:
    from config.config import SENTRY_DSN
    if SENTRY_DSN:
        import sentry_sdk  # type: ignore
        sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.0)
except Exception:
    pass

# Prometheus metrics (simple request counter)
try:
    from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    REQ_COUNT = Counter('vp_api_requests_total', 'Total API requests', ['endpoint'])

    @app.route('/metrics')
    def metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
except Exception:
    REQ_COUNT = None

def latest_final_csv() -> Optional[str]:
    try:
        candidates = glob.glob(os.path.join("outputs", "*", "Final Analysis.csv"))
        if not candidates:
            return None
        return sorted(candidates)[-1]
    except Exception:
        return None

def latest_signal_dir() -> Optional[str]:
    try:
        candidates = [d for d in glob.glob(os.path.join("outputs", "*")) if os.path.isdir(d)]
        if not candidates:
            return None
        return sorted(candidates)[-1]
    except Exception:
        return None

def latest_fetch_reliability_csv() -> Optional[str]:
    try:
        base = latest_signal_dir()
        if not base:
            return None
        p = os.path.join(base, 'References', 'Fetch_Reliability.csv')
        return p if os.path.exists(p) else None
    except Exception:
        return None

def latest_reliability_csv() -> Optional[str]:
    try:
        candidates = glob.glob(os.path.join("outputs", "*", "References", "Fetch_Reliability.csv"))
        if not candidates:
            return None
        return sorted(candidates)[-1]
    except Exception:
        return None

def read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def latest_signals_min() -> Optional[str]:
    d = latest_signal_dir()
    if not d:
        return None
    p = os.path.join(d, "signals_min.json")
    return p if os.path.exists(p) else None

def latest_signals_full() -> Optional[str]:
    d = latest_signal_dir()
    if not d:
        return None
    p = os.path.join(d, "signals_full.json")
    return p if os.path.exists(p) else None

def _latest_table(name: str) -> Optional[str]:
    d = latest_signal_dir()
    if not d:
        return None
    p = os.path.join("outputs", "tables", f"{name}.csv")
    # Prefer per-run table if present; else fall back to global outputs/tables
    per_run = os.path.join(d, "tables", f"{name}.csv")
    return per_run if os.path.exists(per_run) else (p if os.path.exists(p) else None)

def _json(data: Dict[str, Any], status: int = 200) -> Response:
    return Response(_dumps(data), status=status, mimetype="application/json")


@app.route('/api/outputs/final', methods=['GET'])
@cache.cached()
def get_final_analysis():
    """Return the latest Final Analysis.csv as JSON rows."""
    if REQ_COUNT:
        try:
            REQ_COUNT.labels('/api/outputs/final').inc()
        except Exception:
            pass
    path = latest_final_csv()
    if not path or not os.path.exists(path):
        return _json({"error": "No Final Analysis.csv found"}, 404)
    try:
        df = pd.read_csv(path)
        return _json({"rows": df.to_dict(orient="records"), "path": path})
    except Exception as e:
        logger.error(f"Failed reading final CSV: {e}")
        return _json({"error": "Failed to read CSV"}, 500)

@app.route('/api/outputs/signal/<sheet>', methods=['GET'])
@cache.cached()
def get_signal_sheet(sheet: str):
    """Return JSON for a specific signal CSV sheet (e.g., Signals, Correlations, Legend, Score_Breakdown, Trade_Notes, Summary_By_Sector, Summary_By_TradeType, Data_Quality, Clean)."""
    if REQ_COUNT:
        try:
            REQ_COUNT.labels('/api/outputs/signal').inc()
        except Exception:
            pass
    base = latest_signal_dir()
    if not base:
        return _json({"error": "No outputs directory found"}, 404)
    name_map = {
        "signals": "Signal_Report__Signals.csv",
        "correlations": "Signal_Report__Correlations.csv",
        "legend": "Signal_Report__Legend.csv",
        "score_breakdown": "Signal_Report__Score_Breakdown.csv",
        "trade_notes": "Signal_Report__Trade_Notes.csv",
        "summary_by_sector": "Signal_Report__Summary_By_Sector.csv",
        "summary_by_tradetype": "Signal_Report__Summary_By_TradeType.csv",
        "data_quality": "Signal_Report__Data_Quality.csv",
        "clean": "Signal_Report__Clean.csv",
    }
    filename = name_map.get(sheet.lower())
    if not filename:
        return _json({"error": "Unknown sheet"}, 400)
    path = os.path.join(base, filename)
    df = read_csv_if_exists(path)
    if df is None:
        return _json({"error": f"{filename} not found"}, 404)
    return _json({"rows": df.to_dict(orient="records"), "path": path})

@app.route('/', methods=['GET'])
def home():
    latest_dir = latest_signal_dir() or ""
    sheets = [
        ("Clean Signals", "/api/outputs/signal/clean"),
        ("Signals", "/api/outputs/signal/signals"),
        ("Correlations", "/api/outputs/signal/correlations"),
        ("Legend", "/api/outputs/signal/legend"),
        ("Score Breakdown", "/api/outputs/signal/score_breakdown"),
        ("Trade Notes", "/api/outputs/signal/trade_notes"),
        ("Summary by Sector", "/api/outputs/signal/summary_by_sector"),
        ("Summary by Trade Type", "/api/outputs/signal/summary_by_tradetype"),
        ("Data Quality", "/api/outputs/signal/data_quality"),
        ("Final Analysis (raw)", "/api/outputs/final"),
        ("signals_min.json", "/api/signals/min"),
        ("Fetch Reliability", "/api/observability/fetch-reliability"),
    ("Daily Turnover", "/api/metrics/daily-turnover"),
    ("Turnover Summary", "/api/metrics/turnover-summary"),
    ("Backtest Cost Summary", "/api/metrics/backtest-cost-summary"),
    ("Feature Correlations (tables)", "/api/metrics/feature-correlations"),
    ("Score Deciles (gross)", "/api/metrics/score-deciles"),
    ("Score Deciles (net)", "/api/metrics/score-deciles-net"),
    ]
    links = ''.join([f'<li><a href="{href}">{label}</a></li>' for label, href in sheets])
    return f"""
    <html><head><title>VP Investments API</title></head>
    <body>
      <h1>VP Investments API</h1>
      <p>Latest output dir: {latest_dir}</p>
      <ul>{links}</ul>
    </body></html>
    """
@app.route('/health', methods=['GET'])
def health():
    return _json({"status": "ok", "time": datetime.now().isoformat()})

@app.route('/api/observability/fetch-reliability', methods=['GET'])
@cache.cached()
def fetch_reliability():
    """Return aggregated fetch reliability metrics (canonical)."""
    if REQ_COUNT:
        try:
            REQ_COUNT.labels('/api/observability/fetch-reliability').inc()
        except Exception:
            pass
    path = latest_reliability_csv()
    if not path:
        return _json({"error": "Fetch_Reliability.csv not found"}, 404)
    try:
        df = pd.read_csv(path)
        return _json({"rows": df.to_dict(orient="records"), "path": path})
    except Exception as e:
        logger.error(f"Fetch_Reliability read failed: {e}")
        return _json({"error": "read failure"}, 500)

# Back-compat: redirect old metrics route to canonical observability path
@app.route('/api/metrics/fetch-reliability', methods=['GET'])
def fetch_reliability_redirect():
    return redirect('/api/observability/fetch-reliability', code=301)

@app.route('/api/metrics/daily-turnover', methods=['GET'])
@cache.cached()
def daily_turnover():
    path = _latest_table('daily_turnover')
    if not path:
        return _json({"error": "daily_turnover.csv not found"}, 404)
    try:
        df = pd.read_csv(path)
        return _json({"rows": df.to_dict(orient="records"), "path": path})
    except Exception as e:
        logger.error(f"Daily Turnover read failed: {e}")
        return _json({"error": "read failure"}, 500)

@app.route('/api/metrics/turnover-summary', methods=['GET'])
@cache.cached()
def turnover_summary():
    path = _latest_table('turnover_summary')
    if not path:
        return _json({"error": "turnover_summary.csv not found"}, 404)
    try:
        df = pd.read_csv(path)
        return _json({"rows": df.to_dict(orient="records"), "path": path})
    except Exception as e:
        logger.error(f"Turnover summary read failed: {e}")
        return _json({"error": "read failure"}, 500)

@app.route('/api/metrics/backtest-cost-summary', methods=['GET'])
@cache.cached()
def backtest_cost_summary():
    path = _latest_table('backtest_cost_summary')
    if not path:
        return _json({"error": "backtest_cost_summary.csv not found"}, 404)
    try:
        df = pd.read_csv(path)
        return _json({"rows": df.to_dict(orient="records"), "path": path})
    except Exception as e:
        logger.error(f"Backtest cost summary read failed: {e}")
        return _json({"error": "read failure"}, 500)

@app.route('/api/metrics/feature-correlations', methods=['GET'])
@cache.cached()
def feature_correlations():
    path = _latest_table('feature_correlations')
    if not path:
        return _json({"error": "feature_correlations.csv not found"}, 404)
    try:
        df = pd.read_csv(path)
        return _json({"rows": df.to_dict(orient="records"), "path": path})
    except Exception as e:
        logger.error(f"Feature correlations read failed: {e}")
        return _json({"error": "read failure"}, 500)

@app.route('/api/metrics/score-deciles', methods=['GET'])
@cache.cached()
def score_deciles():
    path = _latest_table('score_quantile_performance')
    if not path:
        return _json({"error": "score_quantile_performance.csv not found"}, 404)
    try:
        df = pd.read_csv(path)
        return _json({"rows": df.to_dict(orient="records"), "path": path})
    except Exception as e:
        logger.error(f"Score deciles read failed: {e}")
        return _json({"error": "read failure"}, 500)

@app.route('/api/metrics/score-deciles-net', methods=['GET'])
@cache.cached()
def score_deciles_net():
    path = _latest_table('score_quantile_performance_net')
    if not path:
        return _json({"error": "score_quantile_performance_net.csv not found"}, 404)
    try:
        df = pd.read_csv(path)
        return _json({"rows": df.to_dict(orient="records"), "path": path})
    except Exception as e:
        logger.error(f"Score deciles net read failed: {e}")
        return _json({"error": "read failure"}, 500)

@app.route('/api/signals/min', methods=['GET'])
@cache.cached()
def signals_min():
    path = latest_signals_min()
    if not path:
        return _json({"error": "signals_min.json not found"}, 404)
    try:
        # Support optional top_n slicing without re-computing artifacts
        top_n = request.args.get('top_n')
        if top_n is None:
            with open(path, 'rb') as f:
                data = f.read()
            # Light ETag based on size+mtime
            st = os.stat(path)
            etag = f'W/"{st.st_size}-{int(st.st_mtime)}"'
            resp = Response(data, mimetype='application/json')
            resp.headers['ETag'] = etag
            resp.headers['Cache-Control'] = 'public, max-age=120'
            return resp
        # Slice dynamically
        try:
            n = max(0, int(top_n))
        except Exception:
            n = 100
        with open(path, 'rb') as f:
            arr = json.loads(f.read().decode('utf-8'))
        arr = arr[:n]
        return _json({"rows": arr, "path": path, "top_n": n})
    except Exception as e:
        logger.error(f"signals_min read failed: {e}")
        return _json({"error": "read failure"}, 500)

@app.route('/api/signals/full', methods=['GET'])
@cache.cached()
def signals_full():
    path = latest_signals_full()
    if not path:
        return _json({"error": "signals_full.json not found"}, 404)
    try:
        # Optional top_n here as well for large payloads
        top_n = request.args.get('top_n')
        if top_n is None:
            with open(path, 'rb') as f:
                data = f.read()
            st = os.stat(path)
            etag = f'W/"{st.st_size}-{int(st.st_mtime)}"'
            resp = Response(data, mimetype='application/json')
            resp.headers['ETag'] = etag
            resp.headers['Cache-Control'] = 'public, max-age=60'
            return resp
        try:
            n = max(0, int(top_n))
        except Exception:
            n = 200
        with open(path, 'rb') as f:
            arr = json.loads(f.read().decode('utf-8'))
        arr = arr[:n]
        return _json({"rows": arr, "path": path, "top_n": n})
    except Exception as e:
        logger.error(f"signals_full read failed: {e}")
        return _json({"error": "read failure"}, 500)

@app.route('/api/chatgpt/analyze-signal', methods=['POST'])
def analyze_signal():
    """
    Analyze a single stock signal with ChatGPT commentary.
    """
    try:
        data = request.json or {}
        ticker = data.get('ticker')
        metrics = data.get('metrics', {})
        
        # integrator = ChatGPTIntegrator()
        # commentary = integrator.generate_signal_commentary(pd.Series(metrics))
        
        # Placeholder response until we install OpenAI
        response = {
            "ticker": ticker,
            "commentary": f"AI analysis for {ticker} would appear here with detailed insights about sentiment, momentum, and risk factors.",
            "risk_level": "Medium",
            "confidence": "High",
            "generated_at": datetime.now().isoformat()
        }
        return _json(response)
    
    except Exception as e:
        logger.error(f"Signal analysis error: {e}")
        return _json({"error": "Analysis failed"}, 500)

@app.route('/api/chatgpt/portfolio-insights', methods=['POST'])
def portfolio_insights():
    """
    Generate portfolio-level insights for the dashboard.
    """
    try:
        data = request.json or {}
        signals = data.get('signals', [])
        
        # Create DataFrame from signals
        df = pd.DataFrame(signals)
        
        # integrator = ChatGPTIntegrator()
        # insights = integrator.generate_portfolio_insights(df)
        
        # Placeholder response
        insights = {
            "summary": "Current market shows strong retail sentiment with 15 high-quality signals. Technology and healthcare sectors are showing momentum. Recommend moderate position sizing with focus on risk management.",
            "key_themes": ["Tech momentum", "Healthcare breakouts", "Retail sentiment surge"],
            "risk_factors": ["Market volatility", "Sector concentration"],
            "recommendation": "Selective positioning in top-tier signals",
            "generated_at": datetime.now().isoformat()
        }
        return _json(insights)
    
    except Exception as e:
        logger.error(f"Portfolio insights error: {e}")
        return _json({"error": "Analysis failed"}, 500)

@app.route('/api/chatgpt/explain-score', methods=['POST'])
def explain_score():
    """
    Explain why a signal received its score in plain English.
    """
    try:
        data = request.json or {}
        ticker = data.get('ticker')
        score = data.get('score', 0)
        factors = data.get('factors', {})
        
        # integrator = ChatGPTIntegrator()
        # explanation = integrator.explain_signal_scoring(pd.Series(factors))
        
        # Placeholder response
        explanation = f"{ticker} scored {score}/100 primarily due to strong Reddit sentiment ({factors.get('reddit_sentiment', 0):.2f}) and positive price momentum. The combination of social buzz and technical indicators suggests potential upward movement."
        return _json({
            "ticker": ticker,
            "score": score,
            "explanation": explanation,
            "generated_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Score explanation error: {e}")
        return _json({"error": "Explanation failed"}, 500)

@app.route('/api/chatgpt/market-outlook', methods=['GET'])
def market_outlook():
    """
    Generate current market outlook based on signal patterns.
    """
    try:
        # integrator = ChatGPTIntegrator()
        # outlook = integrator.generate_market_outlook({})
        
        # Placeholder response
        outlook = {
            "sentiment": "Bullish",
            "confidence": "Medium",
            "summary": "Alternative data signals indicate growing retail interest in growth stocks. Quality of signals is above average with several emerging opportunities in technology sector.",
            "themes": ["Tech innovation", "Retail participation", "Momentum building"],
            "generated_at": datetime.now().isoformat()
        }
        return _json(outlook)
    
    except Exception as e:
        logger.error(f"Market outlook error: {e}")
        return _json({"error": "Analysis failed"}, 500)

if __name__ == '__main__':
    # Prefer waitress if available for production-like serving; otherwise Flask dev server
    try:
        from waitress import serve  # type: ignore
        serve(app, host="0.0.0.0", port=WEB_API_PORT)
    except Exception:
        app.run(debug=WEB_API_DEBUG, port=WEB_API_PORT)
