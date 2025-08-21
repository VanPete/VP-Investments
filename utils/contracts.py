"""
Lightweight data contracts for normalized DB rows (optional validation).

Enable validation by setting DB_CONTRACTS_VALIDATE=1 in environment
or toggling in config.config.DB_CONTRACTS_VALIDATE.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


def _is_iso8601(s: Optional[str]) -> bool:
    if not s:
        return True
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except Exception:
        return False


class RunRow(BaseModel):
    run_id: str = Field(min_length=1)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    config_json: Optional[str] = None
    code_version: Optional[str] = None
    notes: Optional[str] = None

    @validator("started_at", "ended_at")
    def _ts(cls, v):
        assert _is_iso8601(v), "timestamps must be ISO-8601"
        return v


class PriceRow(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    date: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    adj_close: Optional[float]
    volume: Optional[float]


class FeatureRow(BaseModel):
    run_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1, max_length=10)
    key: str = Field(min_length=1)
    value: Optional[float]
    as_of: Optional[str]

    @validator("as_of")
    def _iso(cls, v):
        assert _is_iso8601(v), "as_of must be ISO-8601"
        return v


class LabelRow(BaseModel):
    run_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1, max_length=10)
    window: str = Field(min_length=1)
    fwd_return: Optional[float]
    beat_spy: Optional[int]
    ready_at: Optional[str]

    @validator("window")
    def _win(cls, v):
        assert v.endswith("D"), "window must end with 'D' (e.g., 3D)"
        return v

    @validator("ready_at")
    def _iso(cls, v):
        assert _is_iso8601(v), "ready_at must be ISO-8601"
        return v


class SignalNormRow(BaseModel):
    run_id: str
    ticker: str
    score: Optional[float]
    rank: Optional[int]
    trade_type: Optional[str]
    risk_level: Optional[str]
    reddit_score: Optional[float]
    news_score: Optional[float]
    financial_score: Optional[float]
    run_datetime: Optional[str]

    @validator("run_datetime")
    def _iso(cls, v):
        assert _is_iso8601(v), "run_datetime must be ISO-8601"
        return v


class ExperimentRow(BaseModel):
    exp_id: str
    run_id: Optional[str]
    profile: Optional[str]
    params_json: Optional[str]
    code_version: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    notes: Optional[str]

    @validator("started_at", "ended_at")
    def _iso(cls, v):
        assert _is_iso8601(v), "timestamps must be ISO-8601"
        return v


class MetricRow(BaseModel):
    run_id: Optional[str]
    name: str
    value: Optional[float]
    context_json: Optional[str]
    created_at: str

    @validator("created_at")
    def _iso(cls, v):
        assert _is_iso8601(v), "created_at must be ISO-8601"
        return v
