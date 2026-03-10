"""
specvel/signals.py

Rule-based LONG/SHORT/NEUTRAL signal from velocity features.
Uses per-asset-class thresholds — macro series move slowly,
equities move faster, so the bars are calibrated accordingly.

No ML required — pure velocity logic, immediately useful from day one.
An ML layer can be added later once enough history accumulates.
"""
import pandas as pd


# ── Per-asset-class thresholds ────────────────────────────────────────────────
# Tune these in configs/ or override at runtime
SIGNAL_THRESHOLDS = {
    "default": {
        "long_vel_z":    0.5,   # velocity z-score above this → LONG component
        "short_vel_z":  -0.5,   # velocity z-score below this → SHORT component
        "long_mom":      0.55,  # momentum (fraction positive periods) → LONG bonus
        "short_mom":     0.45,  # momentum below this → SHORT bonus
        "accel_bonus":   True,  # award +1 if acceleration confirms direction
        "min_periods":   10,    # minimum periods before signaling
    },
    "equities": {
        "long_vel_z":    0.4,
        "short_vel_z":  -0.4,
        "long_mom":      0.55,
        "short_mom":     0.45,
        "accel_bonus":   True,
        "min_periods":   10,
    },
    "commodities": {
        "long_vel_z":    0.5,
        "short_vel_z":  -0.5,
        "long_mom":      0.55,
        "short_mom":     0.45,
        "accel_bonus":   True,
        "min_periods":   10,
    },
    "fixed_income": {
        "long_vel_z":    0.4,   # rates move meaningfully but not wildly
        "short_vel_z":  -0.4,
        "long_mom":      0.53,
        "short_mom":     0.47,
        "accel_bonus":   True,
        "min_periods":   8,
    },
    "macro": {
        "long_vel_z":    0.3,   # macro series update monthly — lower bar
        "short_vel_z":  -0.3,
        "long_mom":      0.52,
        "short_mom":     0.48,
        "accel_bonus":   False, # acceleration less meaningful on monthly data
        "min_periods":   6,
    },
}


def compute_signal(features: dict, asset_class: str = "default") -> dict:
    """
    Compute LONG/SHORT/NEUTRAL signal from a feature dict.

    Scoring:
        Velocity direction   ±2 points
        Momentum confirmation ±1 point
        Acceleration bonus   ±1 point
        ─────────────────────────────
        Max possible          ±4

    Conviction labels:
        ±4 → STRONG LONG / STRONG SHORT
        ±3 → LONG / SHORT
        ±2 → LEAN LONG / LEAN SHORT
        ±1 → WEAK LONG / WEAK SHORT
         0 → NEUTRAL

    Parameters
    ----------
    features    : dict from features.build_features()
    asset_class : one of 'equities', 'commodities', 'fixed_income', 'macro', 'default'

    Returns
    -------
    dict with signal, conviction score, and component breakdown
    """
    thresholds = SIGNAL_THRESHOLDS.get(asset_class,
                                        SIGNAL_THRESHOLDS["default"])

    # Insufficient data guard
    if features.get("n_periods", 0) < thresholds["min_periods"]:
        return {
            "signal":     "NEUTRAL",
            "conviction": 0,
            "reason":     f"insufficient data ({features.get('n_periods', 0)} periods)",
            "components": {},
        }

    vel_z  = float(features.get("velocity_zscore", 0.0) or 0.0)
    mom    = float(features.get("momentum_score",  0.5) or 0.5)
    accel  = features.get("acceleration")
    accel  = float(accel) if accel is not None else 0.0

    score      = 0
    components = {}

    # ── Component 1: Velocity direction (±2) ─────────────────────────────────
    if vel_z >= thresholds["long_vel_z"]:
        score += 2
        components["velocity"] = f"+2 (vel_z={vel_z:+.3f})"
    elif vel_z <= thresholds["short_vel_z"]:
        score -= 2
        components["velocity"] = f"-2 (vel_z={vel_z:+.3f})"
    else:
        components["velocity"] = f"0 (vel_z={vel_z:+.3f}, inconclusive)"

    # ── Component 2: Momentum confirmation (±1) ───────────────────────────────
    if score > 0 and mom >= thresholds["long_mom"]:
        score += 1
        components["momentum"] = f"+1 (mom={mom:.3f}, confirms)"
    elif score < 0 and mom <= thresholds["short_mom"]:
        score -= 1
        components["momentum"] = f"-1 (mom={mom:.3f}, confirms)"
    elif score > 0 and mom <= thresholds["short_mom"]:
        score = max(0, score - 1)
        components["momentum"] = f"-1 (mom={mom:.3f}, CONTRADICTS — score reduced)"
    elif score < 0 and mom >= thresholds["long_mom"]:
        score = min(0, score + 1)
        components["momentum"] = f"+1 (mom={mom:.3f}, CONTRADICTS — score reduced)"
    else:
        components["momentum"] = f"0 (mom={mom:.3f})"

    # ── Component 3: Acceleration bonus (±1) ──────────────────────────────────
    if thresholds["accel_bonus"]:
        if accel > 0 and score > 0:
            score += 1
            components["acceleration"] = f"+1 (accel={accel:+.5f}, accelerating)"
        elif accel < 0 and score < 0:
            score -= 1
            components["acceleration"] = f"-1 (accel={accel:+.5f}, accelerating)"
        else:
            components["acceleration"] = f"0 (accel={accel:+.5f})"
    else:
        components["acceleration"] = "0 (not used for this asset class)"

    # ── Conviction label ───────────────────────────────────────────────────────
    if score >= 4:
        signal = "STRONG LONG 🟢"
    elif score == 3:
        signal = "LONG 🟢"
    elif score == 2:
        signal = "LEAN LONG 🟡"
    elif score == 1:
        signal = "WEAK LONG"
    elif score == 0:
        signal = "NEUTRAL ⚪"
    elif score == -1:
        signal = "WEAK SHORT"
    elif score == -2:
        signal = "LEAN SHORT 🟡"
    elif score == -3:
        signal = "SHORT 🔴"
    else:
        signal = "STRONG SHORT 🔴"

    return {
        "signal":     signal,
        "conviction": score,
        "vel_zscore": round(vel_z, 4),
        "momentum":   round(mom, 4),
        "components": components,
        "reason":     " | ".join(
            f"{k}: {v}" for k, v in components.items()
        ),
    }
