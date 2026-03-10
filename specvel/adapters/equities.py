"""
specvel/adapters/equities.py

Equities adapter — uses yfinance (free, no API key required).
Covers stocks, ETFs, sector funds, and major indices.

Usage:
    from specvel.adapters.equities import EquitiesAdapter
    adapter = EquitiesAdapter()
    series  = adapter.fetch("AAPL", "2020-01-01", "2026-03-10")
"""
import time
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class EquitiesAdapter:
    source_name = "equities"

    # Default universe — S&P 500 sectors + major indices + mega caps
    DEFAULT_TICKERS = {
        # Broad market
        "SPY":  "S&P 500",
        "QQQ":  "Nasdaq 100",
        "IWM":  "Russell 2000",
        "DIA":  "Dow Jones",
        "VTI":  "Total Market",
        # Sectors
        "XLK":  "Technology",
        "XLF":  "Financials",
        "XLE":  "Energy",
        "XLV":  "Healthcare",
        "XLI":  "Industrials",
        "XLU":  "Utilities",
        "XLB":  "Materials",
        "XLP":  "Cons Staples",
        "XLRE": "Real Estate",
        "XLY":  "Cons Disc",
        "XLC":  "Comm Services",
        # Mega caps
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "Nvidia",
        "GOOGL":"Alphabet",
        "AMZN": "Amazon",
        "META": "Meta",
        "TSLA": "Tesla",
        "BRK-B":"Berkshire",
        # Financials
        "JPM":  "JPMorgan",
        "GS":   "Goldman Sachs",
        "BAC":  "Bank of America",
        # Energy
        "XOM":  "ExxonMobil",
        "CVX":  "Chevron",
    }

    # Cycle method — equities follow the business cycle best
    CYCLE_METHOD = "business"

    def __init__(self, tickers: dict = None, sleep: float = 0.3):
        if not HAS_YFINANCE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        self.tickers = tickers or self.DEFAULT_TICKERS
        self.sleep   = sleep  # polite pause between fetches

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
        """Log-normalize prices — better for velocity on trending assets."""
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
