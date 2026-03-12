"""
specvel/signal_runner.py

Live daily signal runner. For each configured instrument, computes the
current SpecVel v2 signal and outputs:

  - Direction:     LONG / SHORT / NEUTRAL
  - Z-score:       current velocity surprise magnitude
  - Phase:         green_up / peak / senescence / dormant
  - Conviction:    LOW (0.5x) / HIGH (1.0x)
  - Target exit:   calendar date based on per-asset optimal hold period
  - Warning:       phase transition flag if active

Optimal hold periods derived from Run 9 backtest (forward return decay analysis):

  Equities (SPY/QQQ/XLK/XLF)  →  10–20 days
  IWM                          →  20–45 days
  XLE                          →  30–45 days
  WTI Crude                    →  25–35 days
  Silver                       →  45–90 days
  Corn                         →  20–45 days
  Wheat                        →  5–20 days
  Gold                         →  20–45 days
  2Y Treasury                  →  60–90 days
  10Y Treasury                 →  60–90 days
  HY Credit                    →  30–45 days

Usage:
    python specvel/signal_runner.py
    python specvel/signal_runner.py --fred_key YOUR_KEY --include_fi --include_macro
    python specvel/signal_runner.py --output signals.csv
"""

import sys, os, warnings, argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ── Per-asset optimal hold periods (trading days) ─────────────────────────────
# Derived from Run 9 forward-return decay analysis.
# Format: ticker → (min_days, max_days, label)

HOLD_PERIODS = {
    # Equities
    "SPY":          (10, 20,  "10–20d"),
    "QQQ":          (10, 20,  "10–20d"),
    "IWM":          (20, 45,  "20–45d"),
    "XLE":          (30, 45,  "30–45d"),
    "XLF":          (10, 20,  "10–20d"),
    "XLK":          (10, 20,  "10–20d"),
    # Commodities
    "CL=F":         (25, 35,  "25–35d"),
    "GC=F":         (20, 45,  "20–45d"),
    "SI=F":         (45, 90,  "45–90d"),
    "ZC=F":         (20, 45,  "20–45d"),
    "ZW=F":         (5,  20,  "5–20d"),
    # Fixed income
    "DGS2":         (60, 90,  "60–90d"),
    "DGS10":        (60, 90,  "60–90d"),
    "BAMLH0A0HYM2": (30, 45,  "30–45d"),
    # Macro (informational — use as regime context)
    "CPIAUCSL":     (None, None, "regime context"),
    "CPILFESL":     (None, None, "regime context"),
    "UNRATE":       (None, None, "regime context"),
    "INDPRO":       (None, None, "regime context"),
}

# Default for unlisted tickers
DEFAULT_HOLD = (20, 45, "20–45d")

# ── Signal parameters (must match backtest.py) ────────────────────────────────

THRESHOLDS = {
    "equities":     0.75,
    "commodities":  1.25,
    "fixed_income": 1.00,
    "macro":        0.50,
    "default":      0.75,
}

ROLL        = 120
MIN_HISTORY = 60
LOOKBACK    = "2020-01-01"   # data window for live run

PHASE_CONSISTENCY = {
    ("green_up",   "LONG"):  1.25,
    ("green_up",   "SHORT"): 0.70,
    ("peak",       "LONG"):  0.70,
    ("peak",       "SHORT"): 1.25,
    ("senescence", "LONG"):  0.70,
    ("senescence", "SHORT"): 1.25,
    ("dormant",    "LONG"):  1.25,
    ("dormant",    "SHORT"): 0.70,
}

NBER_RECESSIONS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


# ── Core signal computation (mirrors backtest._build_signals version=v2) ──────

def _velocity(series: pd.Series, win: int = 7) -> pd.Series:
    v = series.dropna().values.astype(float)
    if len(v) < win + 2:
        return pd.Series(np.nan, index=series.index)
    w = max(win if win % 2 == 1 else win + 1, 5)
    if len(v) >= w:
        v = savgol_filter(v, window_length=w, polyorder=2)
    grad = np.gradient(v)
    mx = np.abs(grad).max()
    if mx > 0:
        grad /= (mx + 1e-9)
    return pd.Series(grad, index=series.dropna().index).reindex(series.index)


def _phase_auto(vel: pd.Series) -> pd.Series:
    v     = vel.dropna()
    rank  = v.rolling(min(ROLL, len(v)), min_periods=20).rank(pct=True)
    accel = v.diff()
    phase = pd.Series("dormant", index=v.index)
    phase[rank >= 0.70]                  = "peak"
    phase[(rank < 0.70) & (accel > 0)]  = "green_up"
    phase[(rank < 0.70) & (accel <= 0)] = "senescence"
    phase[rank <= 0.30]                 = "dormant"
    return phase.reindex(vel.index, fill_value="dormant")


def _phase_business(vel: pd.Series) -> pd.Series:
    in_rec = pd.Series(False, index=vel.index)
    for s, e in NBER_RECESSIONS:
        in_rec[(vel.index >= s) & (vel.index <= e)] = True
    v     = vel.dropna()
    rank  = v.rolling(min(ROLL, len(v)), min_periods=20).rank(pct=True)
    accel = v.diff()
    phase = pd.Series("green_up", index=v.index)
    phase[rank >= 0.65]                 = "peak"
    phase[(accel <= 0) & (rank < 0.65)] = "senescence"
    in_rv = in_rec.reindex(v.index, fill_value=False)
    phase[in_rv & (rank >= 0.40)]       = "senescence"
    phase[in_rv & (rank <  0.40)]       = "dormant"
    return phase.reindex(vel.index, fill_value="dormant")


def _phase_calendar(vel: pd.Series) -> pd.Series:
    q_map = {1: "green_up", 2: "peak", 3: "senescence", 4: "dormant"}
    return vel.index.to_series().dt.quarter.map(q_map).reindex(vel.index, fill_value="dormant")


def _get_phases(vel, cycle_method):
    if cycle_method == "business": return _phase_business(vel)
    if cycle_method == "calendar": return _phase_calendar(vel)
    return _phase_auto(vel)


def compute_signal(normed: pd.Series, cycle_method: str, threshold: float) -> dict:
    """
    Compute the current SpecVel v2 signal for a single series.
    Returns a dict with all signal fields.
    """
    vel = _velocity(normed).dropna()
    if len(vel) < MIN_HISTORY:
        return None

    roll_mean = vel.rolling(ROLL, min_periods=MIN_HISTORY).mean()
    roll_std  = vel.rolling(ROLL, min_periods=MIN_HISTORY).std()
    phases    = _get_phases(vel, cycle_method)

    # Raw z-score (no lookahead issue in live mode — we use all history)
    raw_z = (vel - roll_mean) / roll_std.clip(lower=1e-9)

    # Phase-consistency multiplier on current bar only
    cur_phase = phases.iloc[-1]
    cur_raw_z = float(raw_z.iloc[-1])
    cur_vel   = float(vel.iloc[-1])
    cur_accel = float(vel.diff().iloc[-1])

    raw_direction = ("LONG"  if cur_raw_z >=  threshold else
                     "SHORT" if cur_raw_z <= -threshold else "NEUTRAL")

    mult   = PHASE_CONSISTENCY.get((cur_phase, raw_direction), 1.0)
    zscore = float(np.clip(cur_raw_z * mult, -6, 6))

    signal = ("LONG"  if zscore >=  threshold else
              "SHORT" if zscore <= -threshold else "NEUTRAL")

    # Conviction tier
    abs_z      = abs(zscore)
    conviction = "HIGH" if abs_z >= 1.5 else ("LOW" if signal != "NEUTRAL" else "NONE")
    weight     = 1.0 if conviction == "HIGH" else (0.5 if conviction == "LOW" else 0.0)

    # Transition warning
    warning = ""
    if   cur_phase == "peak"       and cur_accel < -0.01: warning = "Peak → Senescence"
    elif cur_phase == "dormant"    and cur_accel >  0.01: warning = "Dormant → Green-up"
    elif cur_phase == "senescence" and cur_accel >  0.01: warning = "Senescence → Recovery"
    elif cur_phase == "green_up"   and cur_accel <  0.0 and float(vel.rolling(5).mean().iloc[-1]) < 0:
        warning = "Green-up stalling"

    # Phase age
    phase_age = 0
    for p in reversed(phases.values):
        if p == cur_phase: phase_age += 1
        else: break

    return {
        "as_of":       str(vel.index[-1])[:10],
        "signal":      signal,
        "zscore":      round(zscore, 3),
        "raw_zscore":  round(cur_raw_z, 3),
        "phase":       cur_phase,
        "phase_age":   phase_age,
        "conviction":  conviction,
        "weight":      weight,
        "mult":        round(mult, 2),
        "velocity":    round(cur_vel, 5),
        "warning":     warning,
    }


def _trading_days_ahead(n: int) -> str:
    """Approximate calendar date n trading days from today."""
    today  = datetime.today()
    days   = 0
    target = today
    while days < n:
        target += timedelta(days=1)
        if target.weekday() < 5:   # Mon–Fri
            days += 1
    return target.strftime("%Y-%m-%d")


# ── Main runner ───────────────────────────────────────────────────────────────

def run(
    fred_key:       str  = None,
    include_fi:     bool = False,
    include_macro:  bool = False,
    output:         str  = None,
    today:          str  = None,
):
    from adapters.equities    import EquitiesAdapter
    from adapters.commodities import CommoditiesAdapter

    END = today or datetime.today().strftime("%Y-%m-%d")

    adapter_groups = [
        (EquitiesAdapter(), {
            "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
            "XLE": "Energy",  "XLF": "Financials",  "XLK": "Technology",
        }, "business", "equities"),
        (CommoditiesAdapter(), {
            "CL=F": "WTI Crude", "GC=F": "Gold", "SI=F": "Silver",
            "ZC=F": "Corn",      "ZW=F": "Wheat",
        }, "auto", "commodities"),
    ]

    if include_fi and fred_key:
        from adapters.fixed_income import FixedIncomeAdapter
        adapter_groups.append((
            FixedIncomeAdapter(api_key=fred_key),
            {"DGS2": "2Y Treasury", "DGS10": "10Y Treasury", "BAMLH0A0HYM2": "HY Credit"},
            "business", "fixed_income"
        ))

    if include_macro and fred_key:
        from adapters.macro import MacroAdapter
        adapter_groups.append((
            MacroAdapter(api_key=fred_key),
            {"CPIAUCSL": "CPI", "CPILFESL": "Core CPI",
             "UNRATE": "Unemployment", "INDPRO": "Ind Production"},
            "calendar", "macro"
        ))

    rows = []
    print(f"\n{'='*90}")
    print(f"  SPECVEL LIVE SIGNALS  —  as of {END}")
    print(f"{'='*90}")
    print(f"  {'TICKER':<14} {'LABEL':<18} {'SIGNAL':<8} {'Z':>7} {'PHASE':<12} "
          f"{'CONV':<5} {'HOLD':>8}  {'TARGET EXIT':<13}  WARNING")
    print(f"  {'─'*85}")

    for adapter, tickers, cycle_method, asset_class in adapter_groups:
        threshold = THRESHOLDS.get(asset_class, THRESHOLDS["default"])
        print(f"\n  ── {asset_class.upper().replace('_',' ')}  (threshold ±{threshold}) ──")

        for ticker, label in tickers.items():
            try:
                raw    = adapter.fetch(ticker, LOOKBACK, END)
                normed = adapter.normalize(raw)
                sig    = compute_signal(normed, cycle_method, threshold)

                if sig is None:
                    print(f"  {ticker:<14} {label:<18} — insufficient data")
                    continue

                # Hold period and target exit date
                hold_min, hold_max, hold_label = HOLD_PERIODS.get(ticker, DEFAULT_HOLD)

                if hold_min is None:
                    exit_date  = "regime context"
                    exit_label = "—"
                elif sig["signal"] == "NEUTRAL":
                    exit_date  = "—"
                    exit_label = hold_label
                else:
                    # Use midpoint of optimal hold range
                    hold_mid   = (hold_min + hold_max) // 2
                    exit_date  = _trading_days_ahead(hold_mid)
                    exit_label = hold_label

                warn_flag = "⚠ " + sig["warning"] if sig["warning"] else ""

                sig_display = sig["signal"]
                if sig["signal"] == "LONG":    sig_display = "▲ LONG"
                elif sig["signal"] == "SHORT":  sig_display = "▼ SHORT"
                else:                           sig_display = "— NEUT"

                print(f"  {ticker:<14} {label:<18} {sig_display:<8} "
                      f"{sig['zscore']:>+7.3f} {sig['phase']:<12} "
                      f"{sig['conviction']:<5} {exit_label:>8}  {exit_date:<13}  {warn_flag}")

                rows.append({
                    "as_of":        sig["as_of"],
                    "ticker":       ticker,
                    "label":        label,
                    "asset_class":  asset_class,
                    "signal":       sig["signal"],
                    "zscore":       sig["zscore"],
                    "raw_zscore":   sig["raw_zscore"],
                    "phase_mult":   sig["mult"],
                    "phase":        sig["phase"],
                    "phase_age":    sig["phase_age"],
                    "conviction":   sig["conviction"],
                    "weight":       sig["weight"],
                    "hold_range":   hold_label,
                    "target_exit":  exit_date if sig["signal"] != "NEUTRAL" else "",
                    "warning":      sig["warning"],
                    "threshold":    threshold,
                })

            except Exception as e:
                print(f"  {ticker:<14} {label:<18} — error: {e}")

    print(f"\n{'='*90}\n")

    df = pd.DataFrame(rows)

    # Summary counts
    if not df.empty:
        longs   = (df["signal"] == "LONG").sum()
        shorts  = (df["signal"] == "SHORT").sum()
        neutral = (df["signal"] == "NEUTRAL").sum()
        hi_conv = (df["conviction"] == "HIGH").sum()
        warns   = (df["warning"] != "").sum()
        print(f"  Summary: {longs} LONG  |  {shorts} SHORT  |  {neutral} NEUTRAL")
        print(f"           {hi_conv} HIGH conviction  |  {warns} transition warnings\n")

    if output:
        df.to_csv(output, index=False)
        print(f"  Signals saved to: {output}\n")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SpecVel live signal runner")
    p.add_argument("--fred_key",     type=str,  default=None)
    p.add_argument("--include_fi",   action="store_true", help="Include fixed income (needs FRED key)")
    p.add_argument("--include_macro",action="store_true", help="Include macro (needs FRED key)")
    p.add_argument("--output",       type=str,  default=None, help="Save signals to CSV path")
    p.add_argument("--date",         type=str,  default=None, help="Override today's date YYYY-MM-DD")
    a = p.parse_args()

    run(
        fred_key=a.fred_key,
        include_fi=a.include_fi,
        include_macro=a.include_macro,
        output=a.output,
        today=a.date,
    )
