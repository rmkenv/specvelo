"""
specvel/cycle.py

The TRUE spectral velocity formula applied to financial data.

In crop monitoring, the signal wasn't just "NDVI is moving" —
it was "NDVI is moving FASTER OR SLOWER than expected at this
point in the growing season." That timing surprise IS the alpha.

This module ports that exact concept to finance:

  velocity_surprise = current_velocity - expected_velocity_for_this_cycle_phase

Three cycle detection methods — adapter picks which fits:
  - AUTO     : detected from the data itself via changepoint + phase logic
  - BUSINESS : NBER expansion/contraction dates (requires FRED key)
  - CALENDAR : fixed Q1-Q4 seasonal structure

Outputs per series:
  - cycle_phase      : green_up / peak / senescence / dormant
  - phase_age        : how many periods into this phase
  - velocity_surprise: current_vel - historical_mean_vel_for_this_phase
  - surprise_zscore  : surprise / historical_std_vel_for_this_phase
  - surprise_signal  : ACCELERATING / DECELERATING / ON_TREND
  - conviction_boost : ±1 or ±2 added to base signal score
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

from core import compute_velocity


# ── Phase labels (mirror crop phenology) ─────────────────────────────────────
PHASE_GREEN_UP   = "green_up"    # velocity rising, below peak — early expansion
PHASE_PEAK       = "peak"        # velocity at/near maximum — late expansion
PHASE_SENESCENCE = "senescence"  # velocity falling from peak — contraction onset
PHASE_DORMANT    = "dormant"     # velocity at/near minimum — trough / base


# ── NBER recession dates (hardcoded — update as needed) ───────────────────────
NBER_RECESSIONS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Phase detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_phase(vel: pd.Series, window: int = 10) -> pd.Series:
    """
    Assign a cycle phase label to each period based on velocity pattern.
    Mirrors crop phenology detection from the ag system.

    Logic:
      - Compute rolling percentile rank of velocity
      - green_up   : vel rising AND rank < 0.75
      - peak       : rank >= 0.75
      - senescence : vel falling AND rank > 0.25
      - dormant    : rank <= 0.25
    """
    v = vel.dropna()
    if len(v) < window * 2:
        return pd.Series(PHASE_GREEN_UP, index=vel.index)

    # Rolling percentile rank of velocity
    rank = v.rolling(window=min(window * 4, len(v)), min_periods=window) \
             .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    # First derivative of velocity = acceleration
    accel = pd.Series(np.gradient(v.values), index=v.index)

    phases = []
    for i in range(len(v)):
        r = rank.iloc[i] if not np.isnan(rank.iloc[i]) else 0.5
        a = accel.iloc[i]

        if r >= 0.70:
            phases.append(PHASE_PEAK)
        elif r <= 0.30:
            phases.append(PHASE_DORMANT)
        elif a > 0:
            phases.append(PHASE_GREEN_UP)
        else:
            phases.append(PHASE_SENESCENCE)

    return pd.Series(phases, index=v.index).reindex(vel.index, fill_value=PHASE_DORMANT)


def detect_phase_business_cycle(
    vel:        pd.Series,
    recession_dates: list = None,
) -> pd.Series:
    """
    Assign phases using NBER business cycle dates.
    Recession = senescence/dormant, Expansion = green_up/peak.
    """
    dates = recession_dates or NBER_RECESSIONS
    phases = pd.Series(index=vel.index, dtype=str)

    # Mark recession periods
    in_recession = pd.Series(False, index=vel.index)
    for start, end in dates:
        mask = (vel.index >= start) & (vel.index <= end)
        in_recession[mask] = True

    # Within expansion: use velocity rank to distinguish green_up vs peak
    v     = vel.dropna()
    rank  = v.rolling(window=min(60, len(v)), min_periods=10) \
              .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    accel = pd.Series(np.gradient(v.fillna(0).values), index=v.index)

    result = []
    for idx in vel.index:
        if idx not in v.index:
            result.append(PHASE_DORMANT)
            continue
        if in_recession.get(idx, False):
            r = rank.get(idx, 0.5)
            result.append(PHASE_DORMANT if (r if not np.isnan(r) else 0.5) < 0.4
                          else PHASE_SENESCENCE)
        else:
            r = rank.get(idx, 0.5)
            r = r if not np.isnan(r) else 0.5
            a = accel.get(idx, 0.0)
            if r >= 0.65:
                result.append(PHASE_PEAK)
            elif a > 0:
                result.append(PHASE_GREEN_UP)
            else:
                result.append(PHASE_SENESCENCE)

    return pd.Series(result, index=vel.index)


def detect_phase_calendar(vel: pd.Series) -> pd.Series:
    """
    Assign phases based on calendar quarter.
    Q1=green_up, Q2=peak, Q3=senescence, Q4=dormant
    Useful for macro series with strong seasonal structure (CPI, retail sales).
    """
    phase_map = {1: PHASE_GREEN_UP, 2: PHASE_PEAK,
                 3: PHASE_SENESCENCE, 4: PHASE_DORMANT}
    quarters  = vel.index.to_series().dt.quarter
    return quarters.map(phase_map).reindex(vel.index, fill_value=PHASE_DORMANT)


# ─────────────────────────────────────────────────────────────────────────────
# Historical phase baseline
# ─────────────────────────────────────────────────────────────────────────────

def build_phase_baseline(
    vel:    pd.Series,
    phases: pd.Series,
) -> dict:
    """
    For each phase, compute the historical distribution of velocity.
    This is the 'seasonal norm' — the expected velocity given the phase.

    Returns dict: phase → {mean, std, median, p25, p75, n}
    """
    baseline = {}
    for phase in [PHASE_GREEN_UP, PHASE_PEAK, PHASE_SENESCENCE, PHASE_DORMANT]:
        mask = phases == phase
        v_phase = vel[mask].dropna()
        if len(v_phase) < 3:
            baseline[phase] = {
                "mean": 0.0, "std": 1.0, "median": 0.0,
                "p25": -0.1, "p75": 0.1, "n": 0
            }
        else:
            baseline[phase] = {
                "mean":   float(v_phase.mean()),
                "std":    float(v_phase.std()) or 1.0,
                "median": float(v_phase.median()),
                "p25":    float(v_phase.quantile(0.25)),
                "p75":    float(v_phase.quantile(0.75)),
                "n":      len(v_phase),
            }
    return baseline


# ─────────────────────────────────────────────────────────────────────────────
# Velocity surprise — the core specvel insight
# ─────────────────────────────────────────────────────────────────────────────

def compute_velocity_surprise(
    series:       pd.Series,
    cycle_method: str  = "auto",   # "auto", "business", "calendar"
    smooth:       bool = True,
    window:       int  = 7,
    recession_dates: list = None,
) -> dict:
    """
    THE CORE SPECVEL FUNCTION for financial data.

    Computes velocity surprise: how much faster or slower is this series
    moving RIGHT NOW compared to what history says it should be doing
    at this point in its cycle.

    Parameters
    ----------
    series       : normalized pd.Series
    cycle_method : how to define the cycle
    smooth       : apply SG smoothing
    window       : SG window
    recession_dates : override NBER dates for business cycle method

    Returns
    -------
    dict with full surprise analysis + phase info
    """
    vel = compute_velocity(series, smooth=smooth, window=window)
    v   = vel.dropna()

    if len(v) < 20:
        return _empty_surprise()

    # ── Detect cycle phases ───────────────────────────────────────────────────
    if cycle_method == "business":
        phases = detect_phase_business_cycle(vel, recession_dates)
    elif cycle_method == "calendar":
        phases = detect_phase_calendar(vel)
    else:  # auto
        phases = detect_phase(vel)

    # ── Build historical baseline per phase ───────────────────────────────────
    baseline = build_phase_baseline(v, phases.reindex(v.index))

    # ── Current state ─────────────────────────────────────────────────────────
    current_vel   = float(v.iloc[-1])
    current_phase = str(phases.reindex(v.index).iloc[-1])
    phase_base    = baseline.get(current_phase, baseline[PHASE_GREEN_UP])

    # ── Velocity surprise ─────────────────────────────────────────────────────
    expected_vel     = phase_base["mean"]
    surprise         = current_vel - expected_vel
    surprise_zscore  = surprise / (phase_base["std"] + 1e-9)

    # ── Phase age — how many consecutive periods in current phase ─────────────
    phase_series = phases.reindex(v.index).fillna(PHASE_DORMANT)
    phase_age    = 0
    for p in reversed(phase_series.values):
        if p == current_phase:
            phase_age += 1
        else:
            break

    # ── Surprise signal label ─────────────────────────────────────────────────
    if surprise_zscore >= 1.5:
        surprise_signal = "STRONG ACCELERATING 🚀"
        conviction_boost = +2
    elif surprise_zscore >= 0.75:
        surprise_signal = "ACCELERATING ↑"
        conviction_boost = +1
    elif surprise_zscore <= -1.5:
        surprise_signal = "STRONG DECELERATING 🛑"
        conviction_boost = -2
    elif surprise_zscore <= -0.75:
        surprise_signal = "DECELERATING ↓"
        conviction_boost = -1
    else:
        surprise_signal = "ON TREND →"
        conviction_boost = 0

    # ── Phase transition warning ──────────────────────────────────────────────
    # If we're deep in a phase, a transition may be imminent
    accel     = float(np.gradient(v.values)[-1])
    phase_pct = phase_age / max(len(v), 1)

    if current_phase == PHASE_PEAK and accel < -0.01:
        transition_warning = "⚠ Peak → Senescence transition possible"
    elif current_phase == PHASE_GREEN_UP and accel < 0 and phase_age > 5:
        transition_warning = "⚠ Green-up stalling — watch for peak"
    elif current_phase == PHASE_SENESCENCE and accel > 0.01:
        transition_warning = "⚠ Senescence → Recovery possible"
    elif current_phase == PHASE_DORMANT and accel > 0.01:
        transition_warning = "⚠ Dormant → Green-up starting"
    else:
        transition_warning = ""

    return {
        # Phase info
        "cycle_method":       cycle_method,
        "current_phase":      current_phase,
        "phase_age":          phase_age,
        "transition_warning": transition_warning,
        # Velocity
        "current_velocity":   round(current_vel, 6),
        "expected_velocity":  round(expected_vel, 6),
        "velocity_surprise":  round(surprise, 6),
        "surprise_zscore":    round(surprise_zscore, 4),
        # Baseline for this phase
        "phase_vel_mean":     round(phase_base["mean"], 6),
        "phase_vel_std":      round(phase_base["std"], 6),
        "phase_vel_p25":      round(phase_base["p25"], 6),
        "phase_vel_p75":      round(phase_base["p75"], 6),
        "phase_n_samples":    phase_base["n"],
        # Signal
        "surprise_signal":    surprise_signal,
        "conviction_boost":   conviction_boost,
        # Full series for charting
        "_vel":               vel,
        "_phases":            phases,
        "_baseline":          baseline,
    }


def _empty_surprise() -> dict:
    return {
        "cycle_method": "auto", "current_phase": PHASE_DORMANT,
        "phase_age": 0, "transition_warning": "",
        "current_velocity": 0.0, "expected_velocity": 0.0,
        "velocity_surprise": 0.0, "surprise_zscore": 0.0,
        "phase_vel_mean": 0.0, "phase_vel_std": 1.0,
        "phase_vel_p25": 0.0, "phase_vel_p75": 0.0,
        "phase_n_samples": 0,
        "surprise_signal": "ON TREND →", "conviction_boost": 0,
        "_vel": pd.Series(dtype=float),
        "_phases": pd.Series(dtype=str),
        "_baseline": {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_cycle_scan(
    adapter,
    start:        str,
    end:          str,
    cycle_method: str = "auto",
    top_n:        int = 20,
    verbose:      bool = True,
) -> pd.DataFrame:
    """
    Run full cycle + velocity surprise scan across all series in an adapter.
    Returns DataFrame ranked by absolute surprise_zscore.
    """
    import sys
    from features import build_features
    from signals  import compute_signal

    series_list = adapter.list_series()
    if verbose:
        print(f"\nCycle scan: {len(series_list)} series "
              f"[{adapter.source_name}] method={cycle_method}")

    rows   = []
    errors = []

    for i, sid in enumerate(series_list):
        if verbose:
            pct = int((i + 1) / len(series_list) * 100)
            sys.stdout.write(f"\r  {pct:3d}%  {sid:<25}")
            sys.stdout.flush()

        try:
            raw    = adapter.fetch(sid, start, end)
            if len(raw.dropna()) < 20:
                continue
            normed = adapter.normalize(raw)
            label  = adapter.label(sid) if hasattr(adapter, "label") else sid

            # Base velocity features + signal
            feats  = build_features(normed, adapter.source_name, sid,
                                    label=label)
            sig    = compute_signal(feats, adapter.source_name)

            # Cycle surprise — the specvel layer
            surp   = compute_velocity_surprise(normed, cycle_method=cycle_method)

            # Boost base conviction with surprise
            boosted_conviction = np.clip(
                sig["conviction"] + surp["conviction_boost"], -4, 4
            )

            rows.append({
                "adapter":            adapter.source_name,
                "series_id":          sid,
                "label":              label,
                "as_of_date":         str(normed.dropna().index[-1])[:10],
                # Base signal
                "base_conviction":    sig["conviction"],
                "base_signal":        sig["signal"],
                # Cycle layer
                "cycle_phase":        surp["current_phase"],
                "phase_age":          surp["phase_age"],
                "velocity_surprise":  surp["velocity_surprise"],
                "surprise_zscore":    surp["surprise_zscore"],
                "surprise_signal":    surp["surprise_signal"],
                "conviction_boost":   surp["conviction_boost"],
                "transition_warning": surp["transition_warning"],
                # Final
                "conviction":         int(boosted_conviction),
                "expected_velocity":  surp["expected_velocity"],
                "current_velocity":   surp["current_velocity"],
                "phase_vel_mean":     surp["phase_vel_mean"],
                "phase_vel_std":      surp["phase_vel_std"],
            })

        except Exception as e:
            errors.append((sid, str(e)))
            continue

    if verbose:
        print(f"\r  Done. {len(rows)} processed, {len(errors)} errors.")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("surprise_zscore", key=abs, ascending=False)
    return df.head(top_n).reset_index(drop=True)


def print_cycle_leaderboard(df: pd.DataFrame, title: str = "SPECVEL CYCLE SCAN"):
    """Pretty-print the cycle leaderboard."""
    if df.empty:
        print("No results.")
        return

    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    print(f"{'#':<4} {'LABEL':<24} {'PHASE':<12} {'AGE':>4} "
          f"{'SURP_Z':>7} {'SURPRISE':<26} {'BASE':>5} {'BOOST':>6} "
          f"{'FINAL':>6}  WARNING")
    print('-'*110)

    for i, row in df.iterrows():
        warn  = str(row.get("transition_warning", ""))[:30]
        label = str(row.get("label", ""))[:23]
        phase = str(row.get("cycle_phase", ""))[:11]
        surp  = str(row.get("surprise_signal", ""))[:25]

        print(
            f"{i+1:<4} {label:<24} {phase:<12} "
            f"{row.get('phase_age',0):>4} "
            f"{row.get('surprise_zscore',0):>+7.3f} "
            f"{surp:<26} "
            f"{row.get('base_conviction',0):>+5} "
            f"{row.get('conviction_boost',0):>+6} "
            f"{row.get('conviction',0):>+6}  {warn}"
        )

    print(f"{'='*110}")
    acc  = (df["surprise_zscore"] >= 0.75).sum()
    dec  = (df["surprise_zscore"] <= -0.75).sum()
    warn = (df["transition_warning"] != "").sum()
    print(f"  Accelerating: {acc}  |  Decelerating: {dec}  |  "
          f"Transition warnings: {warn}")
    print(f"{'='*110}\n")
