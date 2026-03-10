"""
specvel/adapters/commodities.py

Commodities adapter — uses yfinance futures (free, no API key required).
Covers energy, metals, and soft commodities.

Usage:
    from specvel.adapters.commodities import CommoditiesAdapter
    adapter = CommoditiesAdapter()
    series  = adapter.fetch("CL=F", "2020-01-01", "2026-03-10")
"""
import time
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class CommoditiesAdapter:
    source_name = "commodities"

    DEFAULT_TICKERS = {
        # Energy
        "CL=F":  "WTI Crude Oil",
        "BZ=F":  "Brent Crude Oil",
        "NG=F":  "Natural Gas",
        "RB=F":  "RBOB Gasoline",
        "HO=F":  "Heating Oil",
        # Metals — precious
        "GC=F":  "Gold",
        "SI=F":  "Silver",
        "PA=F":  "Palladium",
        "PL=F":  "Platinum",
        # Metals — base
        "HG=F":  "Copper",
        "ALI=F": "Aluminum",
        # Softs
        "KC=F":  "Coffee",
        "CT=F":  "Cotton",
        "SB=F":  "Sugar",
        "CC=F":  "Cocoa",
        "OJ=F":  "Orange Juice",
        # Other
        "LBS=F": "Lumber",
        "ZC=F":  "Corn Futures",
        "ZS=F":  "Soy Futures",
        "ZW=F":  "Wheat Futures",
    }

    # Cycle method — commodities have auto-detectable supply/demand cycles
    CYCLE_METHOD = "auto"

    def __init__(self, tickers: dict = None, sleep: float = 0.3):
        if not HAS_YFINANCE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        self.tickers = tickers or self.DEFAULT_TICKERS
        self.sleep   = sleep

    def fetch(self, series_id: str, start: str, end: str) -> pd.Series:
        time.sleep(self.sleep)
        df = yf.download(series_id, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {series_id}")
        s = df["Close"].squeeze()
        s.name = self.tickers.get(series_id, series_id)
        return s.dropna()

    def list_series(self) -> list:
        return list(self.tickers.keys())

    def normalize(self, series: pd.Series) -> pd.Series:
        """Log-normalize commodity prices."""
        s = series.dropna()
        if s.empty or s.min() <= 0:
            mn, mx = s.min(), s.max()
            if mx == mn:
                return pd.Series(0.0, index=s.index)
            return (s - mn) / (mx - mn)
        log_s = np.log(s)
        mn, mx = log_s.min(), log_s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (log_s - mn) / (mx - mn)

    def label(self, series_id: str) -> str:
        return self.tickers.get(series_id, series_id)
