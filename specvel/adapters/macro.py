"""
specvel/adapters/macro.py

Macro adapter — uses FRED API (free, requires free API key).
Covers GDP, inflation, employment, PMI, housing, and sentiment.

Note on ISM PMI: ISM charges for their data directly.
We use NAPM (ISM Mfg PMI older series) and MANEMP as free proxies.
S&P Global Flash PMI is available free via pandas_datareader.

Get your free FRED API key at:
    https://fred.stlouisfed.org/docs/api/api_key.html

Usage:
    from specvel.adapters.macro import MacroAdapter
    adapter = MacroAdapter(api_key="your_key_here")
    series  = adapter.fetch("CPIAUCSL", "2015-01-01", "2026-03-10")
"""
import time
import requests
import pandas as pd


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# All free on FRED — ISM replaced with NAPM proxy
DEFAULT_SERIES = {
    # Inflation
    "CPIAUCSL":  "CPI All Items",
    "CPILFESL":  "Core CPI (ex Food/Energy)",
    "PCEPI":     "PCE Inflation",
    "PCEPILFE":  "Core PCE",
    "CPIENGSL":  "CPI Energy",
    "CPIFABSL":  "CPI Food",
    # Growth
    "GDP":       "Nominal GDP",
    "GDPC1":     "Real GDP",
    "A191RL1Q225SBEA": "Real GDP Growth QoQ",
    # Employment
    "UNRATE":    "Unemployment Rate",
    "PAYEMS":    "Nonfarm Payrolls",
    "ICSA":      "Initial Jobless Claims",
    "CCSA":      "Continued Jobless Claims",
    "MANEMP":    "Mfg Employment",
    # PMI proxy — NAPM is the historical ISM Mfg series on FRED
    "NAPM":      "ISM Mfg PMI (NAPM)",
    "NAPMNOI":   "ISM New Orders Index",
    "NAPMEI":    "ISM Employment Index",
    # Consumer
    "UMCSENT":   "UMich Consumer Sentiment",
    "UMCSI":     "UMich Current Conditions",
    "RETAILERS": "Retail Sales",
    "RSAFS":     "Advance Retail Sales",
    # Housing
    "HOUST":     "Housing Starts Total",
    "PERMIT":    "Building Permits",
    "CSUSHPISA": "Case-Shiller Home Price Index",
    "MORTGAGE30US": "30Y Mortgage Rate",
    # Production
    "INDPRO":    "Industrial Production",
    "TCU":       "Capacity Utilization",
    "DGORDER":   "Durable Goods Orders",
    # Money / credit
    "M2SL":      "M2 Money Supply",
    "TOTCI":     "Total Consumer Credit",
}


class MacroAdapter:
    source_name = "macro"

    # Cycle method — macro uses calendar seasons (CPI, retail sales are seasonal)
    CYCLE_METHOD = "calendar"

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
        """Min-max normalize macro series."""
        s = series.dropna()
        if s.empty:
            return series
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    def label(self, series_id: str) -> str:
        return self.series.get(series_id, series_id)
