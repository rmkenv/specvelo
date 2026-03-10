"""
specvel/adapters/fixed_income.py

Fixed income adapter — uses FRED API (free, requires free API key).
Covers Treasury yields, spreads, and credit indices.

Get your free FRED API key at:
    https://fred.stlouisfed.org/docs/api/api_key.html
    (takes 30 seconds, no credit card)

Usage:
    from specvel.adapters.fixed_income import FixedIncomeAdapter
    adapter = FixedIncomeAdapter(api_key="your_key_here")
    series  = adapter.fetch("DGS10", "2020-01-01", "2026-03-10")
"""
import time
import requests
import pandas as pd


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Key fixed income series available free on FRED
DEFAULT_SERIES = {
    # Treasury yields
    "DGS1MO":  "1M Treasury",
    "DGS3MO":  "3M Treasury",
    "DGS6MO":  "6M Treasury",
    "DGS1":    "1Y Treasury",
    "DGS2":    "2Y Treasury",
    "DGS5":    "5Y Treasury",
    "DGS10":   "10Y Treasury",
    "DGS20":   "20Y Treasury",
    "DGS30":   "30Y Treasury",
    # Spreads
    "T10Y2Y":  "10Y-2Y Spread",
    "T10Y3M":  "10Y-3M Spread",
    "T5YIEM":  "5Y Breakeven Inflation",
    "T10YIEM": "10Y Breakeven Inflation",
    # Credit
    "BAMLH0A0HYM2":    "HY OAS Spread",
    "BAMLC0A0CM":      "IG OAS Spread",
    "BAMLH0A0HYM2EY":  "HY Effective Yield",
    # Fed
    "DFEDTARU": "Fed Funds Upper Target",
    "DFEDTARL": "Fed Funds Lower Target",
    # Mortgage
    "MORTGAGE30US": "30Y Fixed Mortgage",
    "MORTGAGE15US": "15Y Fixed Mortgage",
}


class FixedIncomeAdapter:
    source_name = "fixed_income"

    def __init__(self, api_key: str, series: dict = None, sleep: float = 0.2):
        if not api_key or api_key == "your_key_here":
            raise ValueError(
                "FRED API key required. Get one free at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.api_key = api_key
        self.series  = series or DEFAULT_SERIES
        self.sleep   = sleep

    def fetch(self, series_id: str, start: str, end: str) -> pd.Series:
        time.sleep(self.sleep)
        params = {
            "series_id":         series_id,
            "observation_start": start,
            "observation_end":   end,
            "api_key":           self.api_key,
            "file_type":         "json",
        }
        r = requests.get(FRED_BASE, params=params, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            raise ValueError(f"No observations returned for {series_id}")
        df = pd.DataFrame(obs)
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"].dropna()
        s.name = self.series.get(series_id, series_id)
        return s

    def list_series(self) -> list:
        return list(self.series.keys())

    def normalize(self, series: pd.Series) -> pd.Series:
        """Min-max normalize — spreads and yields are already in natural units."""
        s = series.dropna()
        if s.empty:
            return series
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    def label(self, series_id: str) -> str:
        return self.series.get(series_id, series_id)
