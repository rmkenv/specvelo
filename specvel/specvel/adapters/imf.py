"""
specvel/adapters/imf.py

IMF Primary Commodity Price System (PCPS) adapter.
Free API, no key required.

Covers annual and monthly commodity price series from the IMF:
  Dubai crude, EU natural gas, fertilizer index, wheat, energy index,
  and others. Annual frequency is forward-filled to daily for alignment
  with the SpecVel signal stack.

This adapter is designed to serve as a SLOW SIGNAL layer — fundamental
commodity price dynamics operating on a multi-year cycle. It should be
combined with the fast financial signal (yfinance commodities) for a
two-layer conviction framework.

Usage:
    from specvel.adapters.imf import IMFAdapter
    adapter = IMFAdapter()
    series  = adapter.fetch("POILDUB", "2000-01-01", "2026-03-10")

API docs:
    https://dataservices.imf.org/REST/SDMX_JSON.svc/
"""

import time
import requests
import numpy as np
import pandas as pd

IMF_BASE    = "http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData"
IMF_DB      = "PCPS"          # Primary Commodity Prices
IMF_FREQ    = "A"             # A = annual, M = monthly
WORLD_CODE  = "W00"           # world aggregate

# ── Series registry ───────────────────────────────────────────────────────────
# Annual world-aggregate series from PCPS
# Series format:  {FREQ}.{AREA_CODE}.{SERIES_CODE}
IMF_SERIES = {
    # Energy
    "POILDUB":  "Dubai Crude ($/bbl)",
    "POILWTI":  "WTI Crude ($/bbl)",
    "POILBRE":  "Brent Crude ($/bbl)",
    "PNGASEU":  "EU Natural Gas ($/MMBtu)",
    "PNGASJP":  "Japan LNG ($/MMBtu)",
    "PCOALAU":  "Australian Coal ($/mt)",
    "PNRG":     "Energy Price Index (2016=100)",
    # Food & agriculture
    "PWHEAMT":  "Wheat ($/mt)",
    "PMAIZMT":  "Maize ($/mt)",
    "PSOYB":    "Soybeans ($/mt)",
    "PRICENPQ": "Rice ($/mt)",
    "PSUGAUSA": "US Sugar (cents/lb)",
    "PCOFFOBU": "Coffee (cents/lb)",
    "PFOOD":    "Food Price Index (2016=100)",
    # Fertilizers
    "PFERT":    "Fertilizer Index (2016=100)",
    "PUREA":    "Urea ($/mt)",
    "PDAP":     "DAP Fertilizer ($/mt)",
    # Metals
    "PIORECR":  "Iron Ore ($/dmt)",
    "PALUM":    "Aluminum ($/mt)",
    "PCOPP":    "Copper ($/mt)",
    "PNICK":    "Nickel ($/mt)",
    "PGOLD":    "Gold ($/troy oz)",
    "PSILVER":  "Silver (cents/troy oz)",
    # Aggregates
    "PINDU":    "Industrial Inputs Index",
    "PALL":     "All Commodity Index (2016=100)",
}

DEFAULT_SERIES = {
    "POILDUB": "Dubai Crude ($/bbl)",
    "PNGASEU": "EU Natural Gas ($/MMBtu)",
    "PFERT":   "Fertilizer Index (2016=100)",
    "PWHEAMT": "Wheat ($/mt)",
    "PMAIZMT": "Maize ($/mt)",
    "PNRG":    "Energy Price Index (2016=100)",
    "PFOOD":   "Food Price Index (2016=100)",
    "PCOPP":   "Copper ($/mt)",
}


class IMFAdapter:
    """
    IMF Primary Commodity Prices adapter.

    Fetches annual world-aggregate commodity price series from the IMF PCPS
    database. Annual data is forward-filled to daily frequency for alignment
    with the SpecVel financial signal stack.

    Role in SpecVel:
        SLOW SIGNAL / FUNDAMENTAL LAYER
        - Captures multi-year commodity supercycles
        - Complements fast financial signals (yfinance) which reflect
          market pricing rather than physical fundamentals
        - Use as regime filter: fundamental velocity should reinforce
          or dampen the fast financial signal conviction

    Cycle method:
        "auto" — IMF commodity cycles don't follow calendar or business
        cycles cleanly; let velocity curvature detect phases directly.
    """

    source_name  = "imf_commodity"
    CYCLE_METHOD = "auto"

    def __init__(
        self,
        series: dict  = None,
        freq:   str   = "A",    # "A" annual or "M" monthly
        sleep:  float = 0.5,    # IMF API is slow; be polite
    ):
        self.series = series or DEFAULT_SERIES
        self.freq   = freq
        self.sleep  = sleep

    # ── Data fetching ─────────────────────────────────────────────────────────

    def fetch(self, series_id: str, start: str, end: str) -> pd.Series:
        """
        Fetch and return a daily time series for the given IMF series code.

        Annual data is forward-filled to daily frequency so it aligns
        with other SpecVel adapters. The fill value is constant within
        each year until the next annual observation arrives — this is
        intentional: IMF annual data represents a slow-moving fundamental
        backdrop, not a daily signal.

        Parameters
        ----------
        series_id : str   IMF PCPS series code, e.g. "POILDUB"
        start     : str   'YYYY-MM-DD'
        end       : str   'YYYY-MM-DD'
        """
        if series_id not in IMF_SERIES and series_id not in (self.series or {}):
            raise ValueError(
                f"Unknown IMF series '{series_id}'. "
                f"Available: {list(IMF_SERIES.keys())}"
            )

        start_year = int(start[:4])
        end_year   = int(end[:4])

        time.sleep(self.sleep)
        url = (
            f"{IMF_BASE}/{IMF_DB}/"
            f"{self.freq}.{WORLD_CODE}.{series_id}"
            f"?startPeriod={start_year}&endPeriod={end_year}"
        )

        r = requests.get(url, timeout=30)
        r.raise_for_status()

        try:
            data   = r.json()["CompactData"]["DataSet"]["Series"]
            obs    = data.get("Obs", [])
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Unexpected IMF API response for {series_id}: {e}\n"
                f"URL: {url}"
            )

        if not isinstance(obs, list):
            obs = [obs]

        if not obs:
            raise ValueError(f"No observations returned for {series_id}")

        rows = []
        for o in obs:
            period = o.get("@TIME_PERIOD", "")
            value  = o.get("@OBS_VALUE")
            if value is None:
                continue
            try:
                year = int(period)
                rows.append({"date": pd.Timestamp(f"{year}-01-01"), "value": float(value)})
            except (ValueError, TypeError):
                continue

        if not rows:
            raise ValueError(f"No parseable observations for {series_id}")

        annual = pd.DataFrame(rows).set_index("date")["value"].sort_index()
        annual.name = self.series.get(series_id, IMF_SERIES.get(series_id, series_id))

        # Forward-fill annual to daily
        daily_idx = pd.date_range(start=start, end=end, freq="D")
        daily = annual.reindex(daily_idx.union(annual.index)).ffill()
        daily = daily.reindex(daily_idx)

        return daily.dropna()

    def fetch_monthly(self, series_id: str, start: str, end: str) -> pd.Series:
        """
        Fetch monthly frequency data where available.
        Falls back to annual if monthly is not available.
        """
        adapter = IMFAdapter(series=self.series, freq="M", sleep=self.sleep)
        try:
            return adapter._fetch_raw_monthly(series_id, start, end)
        except Exception:
            return self.fetch(series_id, start, end)

    def _fetch_raw_monthly(self, series_id: str, start: str, end: str) -> pd.Series:
        start_year = int(start[:4])
        end_year   = int(end[:4])
        time.sleep(self.sleep)
        url = (
            f"{IMF_BASE}/{IMF_DB}/"
            f"M.{WORLD_CODE}.{series_id}"
            f"?startPeriod={start_year}&endPeriod={end_year}"
        )
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()["CompactData"]["DataSet"]["Series"]
        obs  = data.get("Obs", [])
        if not isinstance(obs, list):
            obs = [obs]

        rows = []
        for o in obs:
            period = o.get("@TIME_PERIOD", "")
            value  = o.get("@OBS_VALUE")
            if value is None:
                continue
            try:
                date = pd.Period(period, freq="M").to_timestamp()
                rows.append({"date": date, "value": float(value)})
            except (ValueError, TypeError):
                continue

        if not rows:
            raise ValueError(f"No monthly data for {series_id}")

        monthly = pd.DataFrame(rows).set_index("date")["value"].sort_index()
        monthly.name = self.series.get(series_id, IMF_SERIES.get(series_id, series_id))

        daily_idx = pd.date_range(start=start, end=end, freq="D")
        daily = monthly.reindex(daily_idx.union(monthly.index)).ffill()
        return daily.reindex(daily_idx).dropna()

    # ── Normalisation ─────────────────────────────────────────────────────────

    def normalize(self, series: pd.Series) -> pd.Series:
        """
        Log-normalize commodity prices.
        Log transform stabilizes variance across commodity supercycles.
        """
        s = series.dropna()
        if s.empty or (s <= 0).any():
            mn, mx = s.min(), s.max()
            if mx == mn:
                return pd.Series(0.0, index=s.index)
            return (s - mn) / (mx - mn)
        log_s = np.log(s)
        mn, mx = log_s.min(), log_s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (log_s - mn) / (mx - mn)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def list_series(self) -> list:
        return list(self.series.keys())

    def label(self, series_id: str) -> str:
        return self.series.get(series_id, IMF_SERIES.get(series_id, series_id))

    @staticmethod
    def available_series() -> dict:
        """Return the full registry of available IMF PCPS series."""
        return dict(IMF_SERIES)
