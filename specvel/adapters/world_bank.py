"""
specvel/adapters/world_bank.py

World Bank Development Indicators (WDI) adapter.
Free API, no key required.

Covers annual development indicators for MENA countries and any other
World Bank country/indicator combination. Annual data is forward-filled
to daily frequency for alignment with the SpecVel signal stack.

Role in SpecVel:
    GEOPOLITICAL / STRUCTURAL REGIME LAYER
    - Captures long-run structural changes in food security, energy
      dependency, military posture, and economic stress by country
    - Used to construct composite stress indices for MENA countries
    - Fed into the GeopoliticalRegimeFilter as regime context signals

Usage:
    from specvel.adapters.world_bank import WorldBankAdapter
    adapter = WorldBankAdapter()
    series  = adapter.fetch("SAU_FERT", "2000-01-01", "2026-03-10")
    # Returns daily forward-filled Saudi Arabia fertilizer consumption

API docs:
    https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

import time
import requests
import numpy as np
import pandas as pd

WB_BASE = "https://api.worldbank.org/v2"

# ── Country registry ──────────────────────────────────────────────────────────
MENA_COUNTRIES = {
    "SAU": "Saudi Arabia",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "EGY": "Egypt",
    "ARE": "UAE",
    "JOR": "Jordan",
    "SYR": "Syria",
    "LBN": "Lebanon",
    "ISR": "Israel",
    "KWT": "Kuwait",
    "QAT": "Qatar",
    "BHR": "Bahrain",
    "OMN": "Oman",
    "YEM": "Yemen",
    "MAR": "Morocco",
    "TUN": "Tunisia",
    "LBY": "Libya",
    "DZA": "Algeria",
}

# ── Indicator registry ────────────────────────────────────────────────────────
WB_INDICATORS = {
    # Agriculture & food security
    "AG.CON.FERT.ZS":    "Fertilizer Use (kg/ha)",
    "AG.LND.ARBL.ZS":    "Arable Land (% land area)",
    "TM.VAL.FOOD.ZS.UN": "Food Imports (% merchandise)",
    "SN.ITK.DEFC.ZS":    "Undernourishment (%)",
    # Energy
    "EG.USE.PCAP.KG.OE": "Energy Use (kg oil eq/capita)",
    "EG.IMP.CONS.ZS":    "Energy Imports (% energy use)",
    "EG.ELC.ACCS.ZS":    "Electricity Access (%)",
    "EG.FEC.RNEW.ZS":    "Renewable Energy (% final)",
    # Security & instability
    "MS.MIL.XPND.GD.ZS": "Military Spend (% GDP)",
    "MS.MIL.XPND.ZS":    "Military Spend (% govt expenditure)",
    "MS.MIL.TOTL.P1":    "Armed Forces Personnel",
    "VC.IHR.PSRC.P5":    "Intentional Homicides (per 100k)",
    # Macroeconomic stress
    "FP.CPI.TOTL.ZG":    "Inflation CPI (%)",
    "BN.CAB.XOKA.GD.ZS": "Current Account (% GDP)",
    "GC.DOD.TOTL.GD.ZS": "Central Govt Debt (% GDP)",
    "NY.GDP.PCAP.KD.ZG": "GDP per Capita Growth (%)",
    "NY.GDP.MKTP.KD.ZG": "GDP Growth (%)",
    "SL.UEM.TOTL.ZS":    "Unemployment Rate (%)",
    # Population & demographics
    "SP.POP.TOTL":        "Population Total",
    "SP.URB.TOTL.IN.ZS":  "Urban Population (%)",
    "SP.DYN.CDRT.IN":     "Death Rate (per 1k)",
}

# Default indicator subset for MENA stress analysis
DEFAULT_INDICATORS = {
    "AG.CON.FERT.ZS":    "Fertilizer Use (kg/ha)",
    "TM.VAL.FOOD.ZS.UN": "Food Imports (% merchandise)",
    "EG.USE.PCAP.KG.OE": "Energy Use (kg oil eq/capita)",
    "EG.IMP.CONS.ZS":    "Energy Imports (% energy use)",
    "MS.MIL.XPND.GD.ZS": "Military Spend (% GDP)",
    "FP.CPI.TOTL.ZG":    "Inflation CPI (%)",
    "NY.GDP.MKTP.KD.ZG": "GDP Growth (%)",
}

# Default countries for MENA stress analysis
DEFAULT_COUNTRIES = {
    "SAU": "Saudi Arabia",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "EGY": "Egypt",
    "ARE": "UAE",
    "JOR": "Jordan",
}

# ── Internal series ID convention ────────────────────────────────────────────
# WorldBankAdapter uses composite IDs: "{ISO3}_{INDICATOR_SHORTCODE}"
# e.g. "SAU_FERT" → Saudi Arabia fertilizer use
#      "EGY_FOOD" → Egypt food imports
# But the adapter also accepts raw World Bank indicator codes if a
# country filter is set at construction time.

INDICATOR_SHORTCODES = {
    "AG.CON.FERT.ZS":    "FERT",
    "TM.VAL.FOOD.ZS.UN": "FOOD_IMP",
    "EG.USE.PCAP.KG.OE": "ENERGY_USE",
    "EG.IMP.CONS.ZS":    "ENERGY_IMP",
    "MS.MIL.XPND.GD.ZS": "MIL_GDP",
    "MS.MIL.XPND.ZS":    "MIL_GOVT",
    "FP.CPI.TOTL.ZG":    "CPI",
    "NY.GDP.MKTP.KD.ZG": "GDP_GROWTH",
    "NY.GDP.PCAP.KD.ZG": "GDPPC_GROWTH",
    "BN.CAB.XOKA.GD.ZS": "CURR_ACCT",
    "GC.DOD.TOTL.GD.ZS": "DEBT_GDP",
    "SL.UEM.TOTL.ZS":    "UNEMP",
    "SN.ITK.DEFC.ZS":    "UNDERNUT",
    "VC.IHR.PSRC.P5":    "HOMICIDE",
}

# Reverse lookup: shortcode → WB indicator code
_SHORT_TO_IND = {v: k for k, v in INDICATOR_SHORTCODES.items()}


class WorldBankAdapter:
    """
    World Bank WDI adapter for SpecVel.

    Fetches annual development indicators for a set of countries and
    forward-fills to daily frequency. Designed primarily for MENA
    geopolitical stress analysis, but works for any WB country/indicator.

    Series ID convention
    --------------------
    Two formats accepted:

    1. Composite:  "{ISO3}_{SHORTCODE}"
       e.g. "SAU_FERT", "EGY_FOOD_IMP", "IRN_MIL_GDP"
       Use list_series() to see all combinations.

    2. Raw WB code with single country:
       Construct with country="SAU" then fetch("AG.CON.FERT.ZS", ...)

    Aggregation
    -----------
    When fetch() receives a composite ID, it returns a single country
    series. fetch_panel() returns a DataFrame of all countries for one
    indicator — useful for divergence analysis.

    Role in SpecVel
    ---------------
    GEOPOLITICAL / STRUCTURAL REGIME LAYER
    Annual frequency → regime filter, not a trade trigger.
    Feed the output into GeopoliticalRegimeFilter.
    """

    source_name  = "world_bank"
    CYCLE_METHOD = "auto"

    def __init__(
        self,
        countries:  dict  = None,
        indicators: dict  = None,
        sleep:      float = 0.3,
    ):
        self.countries  = countries  or DEFAULT_COUNTRIES
        self.indicators = indicators or DEFAULT_INDICATORS
        self.sleep      = sleep

    # ── ID parsing ────────────────────────────────────────────────────────────

    def _parse_id(self, series_id: str):
        """
        Parse composite series_id "SAU_FERT" into (iso3, wb_indicator).
        Returns (None, series_id) if the ID looks like a raw WB code.
        """
        if "." in series_id:
            # Raw WB indicator code — need country set at construction
            return None, series_id

        # Try all country codes first
        for iso3 in list(self.countries.keys()) + list(MENA_COUNTRIES.keys()):
            if series_id.startswith(iso3 + "_"):
                shortcode = series_id[len(iso3) + 1:]
                wb_code   = _SHORT_TO_IND.get(shortcode)
                if wb_code:
                    return iso3, wb_code
                # Maybe it's the full WB code after the country prefix
                rest = series_id[len(iso3) + 1:]
                if "." in rest:
                    return iso3, rest

        raise ValueError(
            f"Cannot parse series_id '{series_id}'. "
            f"Expected format: '{{ISO3}}_{{SHORTCODE}}' e.g. 'SAU_FERT'. "
            f"Known shortcodes: {list(INDICATOR_SHORTCODES.values())}"
        )

    def _series_label(self, iso3: str, wb_code: str) -> str:
        country   = self.countries.get(iso3, MENA_COUNTRIES.get(iso3, iso3))
        indicator = self.indicators.get(wb_code, WB_INDICATORS.get(wb_code, wb_code))
        return f"{country} — {indicator}"

    # ── Fetching ──────────────────────────────────────────────────────────────

    def _fetch_wb_raw(
        self,
        iso3:       str,
        wb_code:    str,
        start_year: int,
        end_year:   int,
    ) -> pd.Series:
        """Fetch a single country/indicator from World Bank API."""
        time.sleep(self.sleep)
        url = (
            f"{WB_BASE}/country/{iso3}/indicator/{wb_code}"
            f"?format=json&per_page=100&date={start_year}:{end_year}"
        )
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        payload = r.json()
        if len(payload) < 2 or payload[1] is None:
            raise ValueError(f"No data returned for {iso3}/{wb_code}")

        rows = [
            {"year": int(d["date"]), "value": float(d["value"])}
            for d in payload[1]
            if d.get("value") is not None
        ]

        if not rows:
            raise ValueError(f"All null values for {iso3}/{wb_code}")

        s = (
            pd.DataFrame(rows)
            .set_index("year")["value"]
            .sort_index()
        )
        s.index = pd.to_datetime([f"{y}-01-01" for y in s.index])
        s.name  = self._series_label(iso3, wb_code)
        return s

    def fetch(self, series_id: str, start: str, end: str) -> pd.Series:
        """
        Fetch a single series and return daily forward-filled values.

        series_id format: "SAU_FERT" or "SAU_AG.CON.FERT.ZS"
        """
        iso3, wb_code = self._parse_id(series_id)

        if iso3 is None:
            raise ValueError(
                f"Raw WB indicator '{series_id}' requires ISO3 country prefix. "
                f"Use format '{{ISO3}}_{{CODE}}' e.g. 'SAU_{series_id}'"
            )

        start_year = int(start[:4])
        end_year   = int(end[:4])

        annual = self._fetch_wb_raw(iso3, wb_code, start_year, end_year)

        # Forward-fill annual to daily
        daily_idx = pd.date_range(start=start, end=end, freq="D")
        daily = annual.reindex(daily_idx.union(annual.index)).ffill()
        daily = daily.reindex(daily_idx)
        return daily.dropna()

    def fetch_panel(
        self,
        wb_code:   str,
        start:     str,
        end:       str,
        countries: dict = None,
    ) -> pd.DataFrame:
        """
        Fetch an indicator for multiple countries.
        Returns a daily DataFrame with country names as columns.

        Parameters
        ----------
        wb_code   : str   World Bank indicator code e.g. "AG.CON.FERT.ZS"
        start     : str   'YYYY-MM-DD'
        end       : str   'YYYY-MM-DD'
        countries : dict  {ISO3: label} — defaults to self.countries
        """
        countries = countries or self.countries
        cols = {}
        errors = []

        for iso3, name in countries.items():
            try:
                start_year = int(start[:4])
                end_year   = int(end[:4])
                annual = self._fetch_wb_raw(iso3, wb_code, start_year, end_year)
                daily_idx = pd.date_range(start=start, end=end, freq="D")
                daily = annual.reindex(daily_idx.union(annual.index)).ffill()
                cols[name] = daily.reindex(daily_idx)
            except Exception as e:
                errors.append(f"  {iso3}: {e}")

        if errors:
            print(f"  WorldBankAdapter.fetch_panel warnings ({wb_code}):")
            for err in errors:
                print(err)

        if not cols:
            raise ValueError(f"No data fetched for any country: {wb_code}")

        return pd.DataFrame(cols).dropna(how="all")

    # ── Normalisation ─────────────────────────────────────────────────────────

    def normalize(self, series: pd.Series) -> pd.Series:
        """Min-max normalize development indicators."""
        s = series.dropna()
        if s.empty:
            return series
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def list_series(self) -> list:
        """Return all {ISO3}_{SHORTCODE} combinations for default countries/indicators."""
        ids = []
        for iso3 in self.countries:
            for wb_code in self.indicators:
                sc = INDICATOR_SHORTCODES.get(wb_code)
                if sc:
                    ids.append(f"{iso3}_{sc}")
        return ids

    def label(self, series_id: str) -> str:
        try:
            iso3, wb_code = self._parse_id(series_id)
            return self._series_label(iso3, wb_code)
        except Exception:
            return series_id

    @staticmethod
    def available_indicators() -> dict:
        return dict(WB_INDICATORS)

    @staticmethod
    def available_countries() -> dict:
        return dict(MENA_COUNTRIES)
