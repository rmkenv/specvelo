"""
specvel/features.py

Builds the full feature vector for one series.
Combines velocity stats, level stats, and trend features
into a flat dict ready for signals, anomaly detection, or ML.
"""
import numpy as np
import pandas as pd
from core import compute_velocity, velocity_summary


def build_features(
    series:       pd.Series,
    adapter_name: str,
    series_id:    str,
    label:        str  = None,
    lookback:     int  = 20,
    smooth:       bool = True,
    window:       int  = 7,
) -> dict:
    """
    Full feature vector for one time series.

    Parameters
    ----------
    series       : normalized pd.Series (output of adapter.normalize())
    adapter_name : string identifier for the data source
    series_id    : ticker / FRED code / identifier
    label        : human-readable name (falls back to series_id)
    lookback     : periods for recent vs historical comparison
    smooth       : apply SG smoothing in velocity computation
    window       : SG smoothing window

    Returns
    -------
    flat dict suitable for a DataFrame row
    """
    s = series.dropna()
    if s.empty or len(s) < 4:
        return {}

    vel     = compute_velocity(series, smooth=smooth, window=window)
    summary = velocity_summary(series, vel, lookback=lookback)

    if not summary:
        return {}

    # Trend — linear slope over lookback window
    recent = s.tail(lookback)
    if len(recent) > 2:
        x     = np.arange(len(recent))
        slope = float(np.polyfit(x, recent.values, 1)[0])
    else:
        slope = np.nan

    # Regime detection — are we above or below the long-run mean?
    long_mean      = float(s.mean())
    above_mean     = float(s.iloc[-1]) > long_mean

    # Rolling velocity trend — is velocity itself accelerating?
    vel_clean      = vel.dropna()
    if len(vel_clean) >= lookback:
        vel_recent = vel_clean.tail(lookback)
        vel_slope  = float(np.polyfit(np.arange(len(vel_recent)),
                                      vel_recent.values, 1)[0])
    else:
        vel_slope = np.nan

    # 52-week (or full history) high/low distance
    window_52w = min(252, len(s))
    s_52w      = s.tail(window_52w)
    pct_from_52w_high = float((s.iloc[-1] - s_52w.max()) / (s_52w.max() + 1e-9))
    pct_from_52w_low  = float((s.iloc[-1] - s_52w.min()) / (s_52w.min() + 1e-9))

    return {
        "adapter":            adapter_name,
        "series_id":          series_id,
        "label":              label or series_id,
        "as_of_date":         str(s.index[-1])[:10],
        "trend_slope":        round(slope, 6) if not np.isnan(slope) else None,
        "vel_slope":          round(vel_slope, 6) if not np.isnan(vel_slope) else None,
        "above_mean":         above_mean,
        "pct_from_52w_high":  round(pct_from_52w_high, 4),
        "pct_from_52w_low":   round(pct_from_52w_low, 4),
        **summary,
    }
