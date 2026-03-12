"""
specvel/adapters/fx.py

FX adapter — USD vs major currencies via FRED (free, same API key as
fixed income and macro).

FRED FX series are expressed as "units of foreign currency per 1 USD"
for most pairs (e.g. DEXJPUS = yen per dollar). We handle the sign
convention explicitly so signals can be read as either:

  USD_STRENGTH  — LONG means dollar rising vs the pair
  LOCAL_STRENGTH — LONG means foreign currency rising vs dollar

Both directions are reported in the backtest output so you can read
the signal from either perspective.

FRED series used (all free, daily):
  DEXUSEU  USD/EUR  (dollars per euro — inverted from most FRED pairs)
  DEXUSUK  USD/GBP  (dollars per pound — inverted)
  DEXJPUS  JPY/USD  (yen per dollar)
  DEXSZUS  CHF/USD  (francs per dollar)
  DEXCAUS  CAD/USD  (Canadian dollars per dollar)
  DEXAUS   AUD/USD  (Australian dollars per dollar... actually USD per AUD on FRED)

Sign conventions are normalised internally so all series are expressed
as USD_PER_FOREIGN (how many dollars buys one unit of foreign currency).
Rising = foreign currency strengthening vs USD.

Get your free FRED API key at:
    https://fred.stlouisfed.org/docs/api/api_key.html

Usage:
    from specvel.adapters.fx import FXAdapter
    adapter = FXAdapter(api_key="your_key_here")
    raw     = adapter.fetch("EURUSD", "2015-01-01", "2026-03-10")
    normed  = adapter.normalize(raw)
    # raw is always in USD-per-foreign-currency terms
"""

import time
import requests
import pandas as pd
import numpy as np

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ── Series registry ───────────────────────────────────────────────────────────
# Each entry: fred_id, label, invert
#   invert=False → series is already USD-per-foreign (rising = foreign stronger)
#   invert=True  → series is foreign-per-USD (rising = USD stronger) — we flip it

FX_SERIES = {
    # Internal ID  : (fred_id,    label,              invert)
    "EURUSD": ("DEXUSEU", "EUR/USD",           False),  # USD per EUR — no flip
    "GBPUSD": ("DEXUSUK", "GBP/USD",           False),  # USD per GBP — no flip
    "USDJPY": ("DEXJPUS", "USD/JPY (inverted)", True),  # JPY per USD — flip
    "USDCHF": ("DEXSZUS", "USD/CHF (inverted)", True),  # CHF per USD — flip
    "USDCAD": ("DEXCAUS", "USD/CAD (inverted)", True),  # CAD per USD — flip
    "AUDUSD": ("DEXAUS",  "AUD/USD",            False), # USD per AUD — no flip
}

# Default tickers to scan (all 6 majors)
DEFAULT_TICKERS = list(FX_SERIES.keys())


class FXAdapter:
    """
    FX adapter for SpecVel — USD vs 6 major currencies via FRED.

    All series are normalised to USD-per-foreign-currency convention:
      Positive velocity → foreign currency strengthening (USD weakening)
      Negative velocity → foreign currency weakening (USD strengthening)

    The backtest reports both USD_strength and LOCAL_strength signals.
    """
    source_name = "fx"

    # FX follows business cycle broadly (risk-on/off drives dollar)
    # but auto-detection works well given the daily structure
    CYCLE_METHOD = "auto"

    def __init__(self, api_key: str, tickers: list = None, sleep: float = 0.25):
        if not api_key or api_key == "your_key_here":
            raise ValueError(
                "FRED API key required. Get one free at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.api_key = api_key
        self.tickers = tickers or DEFAULT_TICKERS
        self.sleep   = sleep

    # ── Data fetching ─────────────────────────────────────────────────────────

    def fetch(self, ticker: str, start: str, end: str) -> pd.Series:
        """
        Fetch and return series in USD-per-foreign-currency convention.
        Rising value = foreign currency stronger vs USD.
        """
        if ticker not in FX_SERIES:
            raise ValueError(
                f"Unknown FX ticker '{ticker}'. "
                f"Available: {list(FX_SERIES.keys())}"
            )
        fred_id, label, invert = FX_SERIES[ticker]

        time.sleep(self.sleep)
        params = {
            "series_id":         fred_id,
            "observation_start": start,
            "observation_end":   end,
            "api_key":           self.api_key,
            "file_type":         "json",
        }
        r = requests.get(FRED_BASE, params=params, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            raise ValueError(f"No observations returned for {fred_id} ({ticker})")

        df = pd.DataFrame(obs)
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"].dropna()

        # Invert so all series are in USD-per-foreign convention
        if invert:
            s = 1.0 / s.replace(0, np.nan).dropna()

        s.name = label
        return s

    def fetch_usd_strength(self, ticker: str, start: str, end: str) -> pd.Series:
        """
        Fetch in USD-strength convention (inverse of fetch).
        Rising value = USD strengthening.
        Used internally for dual-direction reporting.
        """
        s = self.fetch(ticker, start, end)
        inverted = 1.0 / s.replace(0, np.nan).dropna()
        inverted.name = s.name.replace("USD", "USD↑") + " [USD str]"
        return inverted

    # ── Normalisation ─────────────────────────────────────────────────────────

    def normalize(self, series: pd.Series) -> pd.Series:
        """
        Percentage returns normalised to [0, 1].
        FX is better normalised on returns than levels to avoid
        long-run drift swamping the velocity signal.
        """
        s = series.dropna()
        if len(s) < 2:
            return series
        # Use log returns for symmetry, then min-max scale
        log_ret = np.log(s / s.shift(1)).dropna()
        mn, mx  = log_ret.min(), log_ret.max()
        if mx == mn:
            return pd.Series(0.5, index=log_ret.index)
        normed = (log_ret - mn) / (mx - mn)
        return normed.reindex(series.index)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def list_series(self) -> list:
        return list(self.tickers)

    def label(self, ticker: str) -> str:
        if ticker in FX_SERIES:
            return FX_SERIES[ticker][1]
        return ticker

    def label_usd(self, ticker: str) -> str:
        """Label for USD-strength direction."""
        labels = {
            "EURUSD": "USD vs EUR",
            "GBPUSD": "USD vs GBP",
            "USDJPY": "USD vs JPY",
            "USDCHF": "USD vs CHF",
            "USDCAD": "USD vs CAD",
            "AUDUSD": "USD vs AUD",
        }
        return labels.get(ticker, ticker)

    def label_local(self, ticker: str) -> str:
        """Label for local currency strength direction."""
        labels = {
            "EURUSD": "EUR vs USD",
            "GBPUSD": "GBP vs USD",
            "USDJPY": "JPY vs USD",
            "USDCHF": "CHF vs USD",
            "USDCAD": "CAD vs USD",
            "AUDUSD": "AUD vs USD",
        }
        return labels.get(ticker, ticker)
