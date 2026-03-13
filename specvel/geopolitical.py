"""
specvel/geopolitical.py

MENA Geopolitical Regime Filter for SpecVel.

This module builds a composite geopolitical stress index from:
  - IMF commodity price fundamentals (PCPS database)
  - World Bank development indicators for MENA countries (WDI)

and translates that index into a regime signal that modifies conviction
on existing SpecVel financial signals — particularly wheat, WTI crude,
natural gas, and fertilizer-linked names.

────────────────────────────────────────────────────────────────────────
TWO-LAYER SIGNAL FRAMEWORK
────────────────────────────────────────────────────────────────────────

Layer 1 — FAST (existing SpecVel):
    Daily financial velocity from yfinance/FRED.
    Detects what the market is pricing RIGHT NOW.

Layer 2 — SLOW (this module):
    Annual fundamental velocity from IMF + World Bank.
    Detects what the physical and geopolitical reality IS.

When both layers align → amplify conviction.
When they diverge    → flag regime-change risk, dampen conviction.

────────────────────────────────────────────────────────────────────────
STRESS INDEX COMPONENTS
────────────────────────────────────────────────────────────────────────

Commodity stress (IMF):
    - Dubai crude velocity         (energy price pressure)
    - EU natural gas velocity      (regional energy cost)
    - Fertilizer index velocity    (agricultural input cost)
    - Wheat price velocity         (food security indicator)
    - Food price index velocity    (broad food basket)

Country stress (World Bank, aggregated across MENA):
    - Military spending velocity   (conflict escalation signal)
    - Food import dependency vel.  (vulnerability to supply shocks)
    - Fertilizer use velocity      (agricultural modernization/stress)
    - Energy import dependency     (energy security exposure)
    - Inflation velocity           (economic stress)

Each component is velocity-scored (z-score of rate of change),
then combined into a composite index with configurable weights.

────────────────────────────────────────────────────────────────────────
REGIME MULTIPLIERS (applied to existing SpecVel signals)
────────────────────────────────────────────────────────────────────────

    Regime             Index   Wheat  WTI  Gas  Fert  Equities
    ─────────────────────────────────────────────────────────
    HIGH_STRESS   > +1.0       1.3   1.2  1.2  1.3    0.8
    ELEVATED      +0.3 to +1.0 1.1   1.1  1.1  1.1    0.9
    NEUTRAL       -0.3 to +0.3 1.0   1.0  1.0  1.0    1.0
    SUPPRESSED    -0.3 to -1.0 0.9   0.9  0.9  0.9    1.1
    LOW_STRESS    < -1.0       0.8   0.8  0.8  0.8    1.2

────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────

    from specvel.geopolitical import GeopoliticalRegimeFilter

    grf = GeopoliticalRegimeFilter()
    index = grf.build_index("2000-01-01", "2026-03-10")
    regime = grf.classify_regime(index)
    multipliers = grf.get_multipliers(regime)

    # Apply to an existing SpecVel signal dataframe
    modified_signals = grf.apply_to_signals(signal_df, index)

    # Full diagnostic report
    report = grf.run_report("2000-01-01", "2026-03-10")

────────────────────────────────────────────────────────────────────────
DEPENDENCIES
────────────────────────────────────────────────────────────────────────

    from specvel.adapters.imf        import IMFAdapter
    from specvel.adapters.world_bank import WorldBankAdapter
    (both free APIs, no keys required)
"""

import warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

try:
    from adapters.imf        import IMFAdapter
    from adapters.world_bank import WorldBankAdapter
except ImportError:
    try:
        from specvel.adapters.imf        import IMFAdapter
        from specvel.adapters.world_bank import WorldBankAdapter
    except ImportError:
        import importlib, sys
        _pkg = sys.modules.get("specvel") or importlib.import_module("specvel")
        IMFAdapter        = getattr(importlib.import_module("specvel.adapters.imf"), "IMFAdapter")
        WorldBankAdapter  = getattr(importlib.import_module("specvel.adapters.world_bank"), "WorldBankAdapter")


# ── Config ────────────────────────────────────────────────────────────────────

ROLL_WINDOW  = 5    # years rolling window for z-scoring (annual data)
MIN_OBS      = 4    # minimum observations before scoring

# Component weights for composite index
# Weights sum to 1.0 within each group; groups are averaged.
COMMODITY_WEIGHTS = {
    "POILDUB": 0.25,   # crude oil — most liquid, most traded
    "PNGASEU": 0.20,   # natural gas — MENA export + regional cost
    "PFERT":   0.25,   # fertilizer — direct link to food security
    "PWHEAMT": 0.20,   # wheat — primary MENA staple food
    "PFOOD":   0.10,   # broad food index — diversifier
}

COUNTRY_INDICATOR_WEIGHTS = {
    "MS.MIL.XPND.GD.ZS": 0.30,   # military spend — most direct conflict signal
    "TM.VAL.FOOD.ZS.UN":  0.25,   # food import dependency — vulnerability
    "AG.CON.FERT.ZS":     0.20,   # fertilizer use — agricultural stress
    "EG.IMP.CONS.ZS":     0.15,   # energy import dependency
    "FP.CPI.TOTL.ZG":     0.10,   # inflation — economic stress
}

COUNTRY_WEIGHTS = {
    "SAU": 0.25,   # Saudi Arabia — largest economy, oil price setter
    "IRN": 0.20,   # Iran — sanctions, nuclear, Gulf tension
    "EGY": 0.20,   # Egypt — wheat importer, regional anchor, Suez
    "IRQ": 0.15,   # Iraq — oil production, instability
    "ARE": 0.10,   # UAE — financial hub, Gulf stability
    "JOR": 0.10,   # Jordan — food/water stress, refugee pressure
}

# Group weights for final composite
GROUP_WEIGHTS = {
    "commodity": 0.55,   # commodity fundamentals drive near-term pricing
    "country":   0.45,   # geopolitical backdrop
}

# Regime thresholds (z-score of composite index)
REGIME_THRESHOLDS = {
    "HIGH_STRESS":  1.0,
    "ELEVATED":     0.3,
    "NEUTRAL":     -0.3,
    "SUPPRESSED":  -1.0,
    # Below SUPPRESSED → LOW_STRESS
}

# Signal multipliers by regime and asset class
REGIME_MULTIPLIERS = {
    "HIGH_STRESS":  {"wheat": 1.30, "crude": 1.20, "gas": 1.20, "fertilizer": 1.30, "food": 1.20, "equities": 0.80},
    "ELEVATED":     {"wheat": 1.10, "crude": 1.10, "gas": 1.10, "fertilizer": 1.10, "food": 1.10, "equities": 0.90},
    "NEUTRAL":      {"wheat": 1.00, "crude": 1.00, "gas": 1.00, "fertilizer": 1.00, "food": 1.00, "equities": 1.00},
    "SUPPRESSED":   {"wheat": 0.90, "crude": 0.90, "gas": 0.90, "fertilizer": 0.90, "food": 0.90, "equities": 1.10},
    "LOW_STRESS":   {"wheat": 0.80, "crude": 0.80, "gas": 0.80, "fertilizer": 0.80, "food": 0.80, "equities": 1.20},
}

# Ticker → asset class mapping for multiplier lookup
TICKER_CLASS = {
    "ZW=F":  "wheat",
    "CL=F":  "crude",
    "BZ=F":  "crude",
    "NG=F":  "gas",
    "UAN":   "fertilizer",
    "MOS":   "fertilizer",
    "NTR":   "fertilizer",
    "ZC=F":  "food",    # corn — food basket proxy
    "ZS=F":  "food",    # soybeans
    "SI=F":  "food",    # silver (industrial, lower weight)
    "SPY":   "equities",
    "QQQ":   "equities",
    "IWM":   "equities",
    "XLE":   "crude",
    "XLF":   "equities",
    "XLK":   "equities",
}


# ── Velocity helper (mirrors backtest.py logic) ───────────────────────────────

def _velocity_annual(series: pd.Series, win: int = 5) -> pd.Series:
    """
    Compute velocity on annual data.
    Uses smaller Savitzky-Golay window appropriate for ~25 annual obs.
    """
    s = series.dropna()
    if len(s) < win + 2:
        return pd.Series(np.nan, index=series.index)
    w = max(win if win % 2 == 1 else win + 1, 5)
    w = min(w, len(s) - 1 if (len(s) - 1) % 2 == 1 else len(s) - 2)
    w = max(w, 3)
    vals = s.values.astype(float)
    if len(vals) >= w:
        vals = savgol_filter(vals, window_length=w, polyorder=2)
    grad = np.gradient(vals)
    mx = np.abs(grad).max()
    if mx > 0:
        grad /= (mx + 1e-9)
    return pd.Series(grad, index=s.index).reindex(series.index)


def _zscore_rolling(series: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    """Rolling z-score on annual data."""
    roll_mean = series.rolling(window, min_periods=MIN_OBS).mean().shift(1)
    roll_std  = series.rolling(window, min_periods=MIN_OBS).std().shift(1)
    return (series - roll_mean) / roll_std.clip(lower=1e-9)


# ── Main class ────────────────────────────────────────────────────────────────

class GeopoliticalRegimeFilter:
    """
    MENA Geopolitical Stress Index and Regime Filter.

    Builds a composite stress index from IMF commodity fundamentals
    and World Bank MENA country indicators, then applies regime-dependent
    multipliers to existing SpecVel financial signals.
    """

    def __init__(
        self,
        imf_adapter:  IMFAdapter       = None,
        wb_adapter:   WorldBankAdapter  = None,
        commodity_weights:    dict = None,
        indicator_weights:    dict = None,
        country_weights:      dict = None,
        group_weights:        dict = None,
        verbose:              bool = True,
    ):
        self.imf = imf_adapter or IMFAdapter()
        self.wb  = wb_adapter  or WorldBankAdapter()

        self.commodity_weights = commodity_weights or COMMODITY_WEIGHTS
        self.indicator_weights = indicator_weights or COUNTRY_INDICATOR_WEIGHTS
        self.country_weights   = country_weights   or COUNTRY_WEIGHTS
        self.group_weights     = group_weights     or GROUP_WEIGHTS
        self.verbose           = verbose

    # ── Index construction ────────────────────────────────────────────────────

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def build_commodity_component(self, start: str, end: str) -> pd.Series:
        """
        Build weighted commodity velocity z-score component.
        Returns a daily series of commodity stress.
        """
        self._log("  Building commodity component (IMF PCPS)...")
        scores = {}

        for code, weight in self.commodity_weights.items():
            try:
                raw    = self.imf.fetch(code, start, end)
                normed = self.imf.normalize(raw)
                # Resample to annual for velocity (avoids ffill artifacts)
                annual = normed.resample("YE").last().dropna()
                if len(annual) < MIN_OBS + 2:
                    self._log(f"    {code}: insufficient data ({len(annual)} obs) — skipped")
                    continue
                vel   = _velocity_annual(annual)
                zscore = _zscore_rolling(vel)
                # Forward-fill back to daily
                daily_idx = pd.date_range(start=start, end=end, freq="D")
                daily_z = zscore.reindex(daily_idx.union(zscore.index)).ffill().reindex(daily_idx)
                scores[code] = (daily_z, weight)
                self._log(f"    {code}: OK  ({len(annual)} annual obs, "
                          f"z_mean={daily_z.dropna().mean():.2f}, "
                          f"z_std={daily_z.dropna().std():.2f})")
            except Exception as e:
                self._log(f"    {code}: FAILED — {e}")

        if not scores:
            self._log("  WARNING: No commodity data — returning zeros")
            daily_idx = pd.date_range(start=start, end=end, freq="D")
            return pd.Series(0.0, index=daily_idx, name="commodity_stress")

        total_weight = sum(w for _, w in scores.values())
        composite    = sum(z * (w / total_weight) for z, w in scores.values())
        composite.name = "commodity_stress"
        return composite

    def build_country_component(self, start: str, end: str) -> pd.Series:
        """
        Build weighted country stress velocity z-score component.
        Aggregates across MENA countries and development indicators.
        Returns a daily series of country/geopolitical stress.
        """
        self._log("  Building country component (World Bank WDI)...")
        indicator_scores = {}

        for wb_code, ind_weight in self.indicator_weights.items():
            country_scores = {}
            for iso3, ctry_weight in self.country_weights.items():
                try:
                    raw    = self.wb._fetch_wb_raw(iso3, wb_code,
                                                    int(start[:4]), int(end[:4]))
                    normed = self.wb.normalize(raw)
                    if len(normed) < MIN_OBS + 2:
                        continue
                    vel    = _velocity_annual(normed)
                    zscore = _zscore_rolling(vel)
                    country_scores[iso3] = (zscore, ctry_weight)
                except Exception as e:
                    self._log(f"    {iso3}/{wb_code}: skipped — {e}")

            if not country_scores:
                continue

            # Weighted average across countries for this indicator
            total_ctry_w = sum(w for _, w in country_scores.values())
            ind_composite = sum(
                z * (w / total_ctry_w) for z, w in country_scores.values()
            )

            # Forward-fill to daily
            daily_idx = pd.date_range(start=start, end=end, freq="D")
            daily_ind = ind_composite.reindex(daily_idx.union(ind_composite.index)).ffill()
            daily_ind = daily_ind.reindex(daily_idx)

            indicator_scores[wb_code] = (daily_ind, ind_weight)
            non_null = daily_ind.dropna()
            self._log(f"    {wb_code}: OK  ({len(country_scores)}/{len(self.country_weights)} countries, "
                      f"z_mean={non_null.mean():.2f})")

        if not indicator_scores:
            self._log("  WARNING: No country data — returning zeros")
            daily_idx = pd.date_range(start=start, end=end, freq="D")
            return pd.Series(0.0, index=daily_idx, name="country_stress")

        total_ind_w = sum(w for _, w in indicator_scores.values())
        composite   = sum(z * (w / total_ind_w) for z, w in indicator_scores.values())
        composite.name = "country_stress"
        return composite

    def build_index(self, start: str, end: str) -> pd.DataFrame:
        """
        Build the full composite MENA Geopolitical Stress Index.

        Returns a DataFrame with columns:
          - commodity_stress   : IMF commodity velocity component
          - country_stress     : World Bank country indicator component
          - composite_index    : weighted combination of both
          - composite_zscore   : rolling z-score of composite_index
          - regime             : regime label string

        The composite_zscore is the primary signal for regime classification.
        """
        self._log("\n── MENA Geopolitical Regime Filter ──")
        self._log(f"  Building index {start} → {end}")

        commodity = self.build_commodity_component(start, end)
        country   = self.build_country_component(start, end)

        # Align on shared index
        idx = commodity.index.intersection(country.index)
        if idx.empty:
            idx = commodity.index

        commodity = commodity.reindex(idx).fillna(0)
        country   = country.reindex(idx).fillna(0)

        gw_c = self.group_weights.get("commodity", 0.55)
        gw_k = self.group_weights.get("country",   0.45)

        composite = commodity * gw_c + country * gw_k

        # Z-score the composite on a rolling annual basis
        annual_composite = composite.resample("YE").mean().dropna()
        annual_z = _zscore_rolling(annual_composite, window=ROLL_WINDOW)

        daily_idx = pd.date_range(start=start, end=end, freq="D")
        comp_z_daily = annual_z.reindex(daily_idx.union(annual_z.index)).ffill().reindex(daily_idx)

        df = pd.DataFrame({
            "commodity_stress": commodity,
            "country_stress":   country,
            "composite_index":  composite,
            "composite_zscore": comp_z_daily,
        }, index=daily_idx)

        df["regime"] = df["composite_zscore"].apply(self._zscore_to_regime)

        self._log(f"\n  Index built: {len(df)} rows, "
                  f"{df['regime'].value_counts().to_dict()}")

        return df

    # ── Regime classification ─────────────────────────────────────────────────

    @staticmethod
    def _zscore_to_regime(z) -> str:
        if pd.isna(z):
            return "NEUTRAL"
        if z > REGIME_THRESHOLDS["HIGH_STRESS"]:
            return "HIGH_STRESS"
        if z > REGIME_THRESHOLDS["ELEVATED"]:
            return "ELEVATED"
        if z > REGIME_THRESHOLDS["NEUTRAL"]:
            return "NEUTRAL"
        if z > REGIME_THRESHOLDS["SUPPRESSED"]:
            return "SUPPRESSED"
        return "LOW_STRESS"

    def classify_regime(self, index_df: pd.DataFrame) -> pd.Series:
        """Return the regime series from an index DataFrame."""
        return index_df["regime"]

    # ── Signal modification ───────────────────────────────────────────────────

    def get_multiplier(self, regime: str, ticker: str) -> float:
        """
        Get the signal multiplier for a given regime and ticker.

        Parameters
        ----------
        regime  : str   one of HIGH_STRESS / ELEVATED / NEUTRAL / SUPPRESSED / LOW_STRESS
        ticker  : str   SpecVel ticker e.g. "ZW=F", "CL=F", "SPY"
        """
        asset_class = TICKER_CLASS.get(ticker, "equities")
        return REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS["NEUTRAL"]).get(
            asset_class, 1.0
        )

    def apply_to_signals(
        self,
        signal_df:  pd.DataFrame,
        index_df:   pd.DataFrame,
        ticker_col: str = "ticker",
        zscore_col: str = "zscore",
    ) -> pd.DataFrame:
        """
        Apply geopolitical regime multipliers to a SpecVel signal DataFrame.

        Takes the signal DataFrame produced by SpecVel's _build_signals()
        (which has a DatetimeIndex and columns including 'signal', 'zscore')
        and modifies the effective zscore by the regime multiplier.

        Parameters
        ----------
        signal_df   : pd.DataFrame   SpecVel signal output (DatetimeIndex)
        index_df    : pd.DataFrame   Output of build_index()
        ticker_col  : str            Column name identifying the ticker
        zscore_col  : str            Column name for the z-score to modify

        Returns
        -------
        pd.DataFrame — copy of signal_df with added columns:
            geo_regime          : regime label for that date
            geo_multiplier      : multiplier applied
            geo_adjusted_zscore : zscore × multiplier
        """
        df = signal_df.copy()

        # Align regime to signal dates
        regime_aligned = index_df["regime"].reindex(df.index, method="ffill")
        df["geo_regime"] = regime_aligned

        # Determine ticker
        if ticker_col in df.columns:
            # Each row may have a different ticker
            df["geo_multiplier"] = df.apply(
                lambda row: self.get_multiplier(row["geo_regime"], row[ticker_col]),
                axis=1
            )
        elif hasattr(signal_df, "name"):
            # Single ticker
            ticker = signal_df.name
            df["geo_multiplier"] = df["geo_regime"].apply(
                lambda r: self.get_multiplier(r, ticker)
            )
        else:
            df["geo_multiplier"] = 1.0

        if zscore_col in df.columns:
            df["geo_adjusted_zscore"] = df[zscore_col] * df["geo_multiplier"]

        return df

    # ── Divergence analysis ───────────────────────────────────────────────────

    def compute_divergence(
        self,
        index_df:      pd.DataFrame,
        financial_vel: pd.Series,
        ticker:        str,
    ) -> pd.DataFrame:
        """
        Compute divergence between the geopolitical fundamental signal
        and the fast financial velocity signal for a given ticker.

        Divergence = financial velocity z-score direction OPPOSITE to
        the geopolitical regime direction.

        Returns a DataFrame with:
          - geo_regime        : regime label
          - geo_zscore        : composite z-score
          - fin_zscore        : financial velocity z-score
          - divergence        : True when signals diverge
          - divergence_type   : "GEO_BULLISH_FIN_BEARISH" | "GEO_BEARISH_FIN_BULLISH" | "ALIGNED"
        """
        geo_z   = index_df["composite_zscore"].reindex(financial_vel.index, method="ffill")
        regime  = index_df["regime"].reindex(financial_vel.index, method="ffill")

        df = pd.DataFrame({
            "geo_regime":  regime,
            "geo_zscore":  geo_z,
            "fin_zscore":  financial_vel,
            "ticker":      ticker,
        })

        # Divergence: geo says stress (positive z) but fin says no signal or bearish
        geo_bullish = df["geo_zscore"] > REGIME_THRESHOLDS["ELEVATED"]
        geo_bearish = df["geo_zscore"] < REGIME_THRESHOLDS["NEUTRAL"]
        fin_bullish = df["fin_zscore"] > 0.3
        fin_bearish = df["fin_zscore"] < -0.3

        df["divergence"] = (geo_bullish & fin_bearish) | (geo_bearish & fin_bullish)

        conditions = [
            geo_bullish & fin_bearish,
            geo_bearish & fin_bullish,
        ]
        choices = [
            "GEO_BULLISH_FIN_BEARISH",
            "GEO_BEARISH_FIN_BULLISH",
        ]
        df["divergence_type"] = np.select(conditions, choices, default="ALIGNED")

        return df

    # ── Country stress breakdown ──────────────────────────────────────────────

    def country_stress_breakdown(
        self,
        start: str,
        end:   str,
    ) -> pd.DataFrame:
        """
        Build a per-country stress score for the most recent year.
        Returns a DataFrame with countries as rows and indicators as columns.
        Used for identifying which countries are driving the composite index.
        """
        self._log("\n  Computing country stress breakdown...")
        end_year   = int(end[:4])
        start_year = int(start[:4])
        rows = []

        from adapters.world_bank import MENA_COUNTRIES
        for iso3 in self.country_weights:
            country_name = MENA_COUNTRIES.get(iso3, self.wb.countries.get(iso3, iso3))
            row = {"country": country_name, "iso3": iso3}
            for wb_code, _ in self.indicator_weights.items():
                try:
                    raw    = self.wb._fetch_wb_raw(iso3, wb_code, start_year, end_year)
                    normed = self.wb.normalize(raw)
                    vel    = _velocity_annual(normed)
                    zscore = _zscore_rolling(vel)
                    latest = zscore.dropna().iloc[-1] if not zscore.dropna().empty else np.nan
                    label  = self.wb.indicators.get(wb_code, wb_code).split(" ")[0]
                    row[label] = round(latest, 2)
                except Exception:
                    label = self.wb.indicators.get(wb_code, wb_code).split(" ")[0]
                    row[label] = np.nan
            rows.append(row)

        df = pd.DataFrame(rows).set_index("country")
        df["composite"] = df.drop(columns=["iso3"]).mean(axis=1)
        df = df.sort_values("composite", ascending=False)
        return df

    # ── Full diagnostic report ────────────────────────────────────────────────

    def run_report(
        self,
        start: str = "2000-01-01",
        end:   str = "2026-03-13",
    ) -> dict:
        """
        Full diagnostic report. Returns a dict with:
          - index_df           : full daily index DataFrame
          - regime_summary     : regime value counts and stats
          - country_breakdown  : per-country stress scores
          - current_regime     : most recent regime label
          - current_zscore     : most recent composite z-score
          - regime_history     : annual regime by year
        """
        print("\n" + "="*65)
        print("  SPECVEL MENA GEOPOLITICAL REGIME FILTER — REPORT")
        print("="*65)

        # Build index
        index_df = self.build_index(start, end)

        # Current state
        recent   = index_df.dropna(subset=["composite_zscore"])
        if not recent.empty:
            current_z      = recent["composite_zscore"].iloc[-1]
            current_regime = recent["regime"].iloc[-1]
            current_date   = recent.index[-1].strftime("%Y-%m-%d")
        else:
            current_z, current_regime, current_date = 0.0, "NEUTRAL", end

        print(f"\n  Current regime ({current_date}): {current_regime}  (z = {current_z:.2f})")

        # Regime summary
        regime_counts = index_df["regime"].value_counts()
        print(f"\n  Regime distribution ({start} → {end}):")
        total = len(index_df)
        for regime, count in regime_counts.items():
            bar = "█" * int(30 * count / total)
            print(f"    {regime:<18} {bar}  ({count/total:.0%})")

        # Annual breakdown
        print(f"\n  Annual regime history:")
        annual = index_df["composite_zscore"].resample("YE").mean().dropna()
        for date, z in annual.items():
            regime = self._zscore_to_regime(z)
            bar    = "+" * max(0, int(z * 5)) if z > 0 else "-" * max(0, int(-z * 5))
            print(f"    {date.year}  {z:>+6.2f}  [{bar:<15}]  {regime}")

        # Country breakdown
        print(f"\n  Country stress breakdown (latest year velocity z-scores):")
        try:
            breakdown = self.country_stress_breakdown(start, end)
            print(breakdown.drop(columns=["iso3"], errors="ignore").to_string())
        except Exception as e:
            print(f"    Country breakdown failed: {e}")
            breakdown = pd.DataFrame()

        # Multiplier implications
        print(f"\n  Signal multipliers for current regime ({current_regime}):")
        mults = REGIME_MULTIPLIERS.get(current_regime, REGIME_MULTIPLIERS["NEUTRAL"])
        for asset, mult in sorted(mults.items(), key=lambda x: -x[1]):
            bar = "↑" * max(0, int((mult - 1.0) * 20)) if mult > 1 else "↓" * max(0, int((1.0 - mult) * 20))
            print(f"    {asset:<12} {mult:.2f}×  {bar}")

        print("\n" + "="*65)

        return {
            "index_df":        index_df,
            "regime_summary":  regime_counts,
            "country_breakdown": breakdown,
            "current_regime":  current_regime,
            "current_zscore":  current_z,
            "current_date":    current_date,
            "multipliers":     mults,
        }


# ── Convenience functions ─────────────────────────────────────────────────────

def build_mena_stress_index(
    start:   str = "2000-01-01",
    end:     str = "2026-03-10",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Shorthand: build the MENA stress index with default settings.
    Returns the full index DataFrame.
    """
    grf = GeopoliticalRegimeFilter(verbose=verbose)
    return grf.build_index(start, end)


def get_current_regime(
    start:   str = "2000-01-01",
    end:     str = "2026-03-10",
    verbose: bool = False,
) -> tuple:
    """
    Shorthand: return (regime_label, zscore) for the most recent date.
    """
    grf   = GeopoliticalRegimeFilter(verbose=verbose)
    index = grf.build_index(start, end)
    recent = index.dropna(subset=["composite_zscore"])
    if recent.empty:
        return "NEUTRAL", 0.0
    z = recent["composite_zscore"].iloc[-1]
    return GeopoliticalRegimeFilter._zscore_to_regime(z), z


def apply_geo_filter_to_backtest(
    signal_df:  pd.DataFrame,
    ticker:     str,
    start:      str = "2000-01-01",
    end:        str = "2026-03-10",
    verbose:    bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper: build index and apply to a signal DataFrame.
    Returns the signal_df with geo_regime, geo_multiplier, geo_adjusted_zscore columns.
    """
    grf   = GeopoliticalRegimeFilter(verbose=verbose)
    index = grf.build_index(start, end)
    return grf.apply_to_signals(signal_df, index, ticker_col="ticker")


# ── CLI / standalone demo ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="MENA Geopolitical Regime Filter")
    p.add_argument("--start",   type=str, default="2000-01-01")
    p.add_argument("--end",     type=str, default="2026-03-10")
    p.add_argument("--save",    type=str, default=None,
                   help="Path to save index CSV, e.g. results/geo_index.csv")
    p.add_argument("--quiet",   action="store_true")
    a = p.parse_args()

    grf    = GeopoliticalRegimeFilter(verbose=not a.quiet)
    report = grf.run_report(start=a.start, end=a.end)

    if a.save:
        import os
        os.makedirs(os.path.dirname(a.save) if os.path.dirname(a.save) else ".", exist_ok=True)
        report["index_df"].to_csv(a.save)
        print(f"\nIndex saved to {a.save}")
