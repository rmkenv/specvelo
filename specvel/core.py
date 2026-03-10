"""
specvel/core.py

The spectral velocity formula — domain-agnostic.

Input:  any normalized pd.Series with DatetimeIndex
Output: velocity series + scalar summary stats

The formula applies Savitzky-Golay smoothing (preserves peaks/troughs
better than a moving average) then computes the first derivative.
The result is a velocity series that captures rate-of-change while
filtering out noise.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def compute_velocity(
    series:    pd.Series,
    smooth:    bool = True,
    window:    int  = 7,
    poly:      int  = 2,
    normalize: bool = True,
) -> pd.Series:
    """
    Compute spectral velocity as the smoothed first derivative of a series.

    Parameters
    ----------
    series    : pd.Series — normalized time series (DatetimeIndex)
    smooth    : apply Savitzky-Golay smoothing before differentiation
    window    : smoothing window length (must be odd and >= poly+2)
    poly      : polynomial order for Savitzky-Golay filter
    normalize : normalize output velocity to [-1, +1]

    Returns
    -------
    pd.Series of velocity values, same index as input
    """
    s = series.dropna().copy()

    if len(s) < max(window + 1, 4):
        return pd.Series(np.nan, index=series.index)

    values = s.values.astype(float)

    if smooth and len(values) >= window:
        # Ensure window is odd and at least poly+2
        w = window if window % 2 == 1 else window + 1
        w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
        if len(values) >= w:
            values = savgol_filter(values, window_length=w, polyorder=poly)

    # First derivative — velocity
    vel = np.gradient(values)

    if normalize and len(vel) > 0:
        max_abs = np.abs(vel).max()
        if max_abs > 0:
            vel = vel / (max_abs + 1e-9)

    return pd.Series(vel, index=s.index).reindex(series.index)


def velocity_summary(
    series:   pd.Series,
    vel:      pd.Series,
    lookback: int = 20,
) -> dict:
    """
    Compute scalar summary stats from a velocity series.

    Parameters
    ----------
    series   : original normalized series
    vel      : velocity series from compute_velocity()
    lookback : number of recent periods for current vs historical comparison

    Returns
    -------
    dict of scalar metrics
    """
    recent_vel = vel.dropna().tail(lookback)
    full_vel   = vel.dropna()

    if len(full_vel) < 2:
        return {}

    current_vel  = float(recent_vel.iloc[-1]) if not recent_vel.empty else np.nan
    vel_mean     = float(full_vel.mean())
    vel_std      = float(full_vel.std())
    vel_zscore   = (current_vel - vel_mean) / (vel_std + 1e-9)

    # Acceleration — second derivative of recent velocity
    acceleration = float(np.gradient(recent_vel.values)[-1]) if len(recent_vel) > 1 else np.nan

    # Momentum — fraction of recent periods with positive velocity
    momentum = float((recent_vel > 0).mean()) if not recent_vel.empty else 0.5

    # Peak detection on full velocity history
    peak_vel      = float(full_vel.max())
    trough_vel    = float(full_vel.min())
    peak_date     = vel.idxmax()
    pct_from_peak = (current_vel - peak_vel) / (abs(peak_vel) + 1e-9)

    # Consistency — how many of last lookback periods agree with current direction
    direction        = 1 if current_vel >= 0 else -1
    consistency      = float((np.sign(recent_vel) == direction).mean()) if not recent_vel.empty else 0.5

    # Level stats on original series
    s            = series.dropna()
    current_val  = float(s.iloc[-1]) if not s.empty else np.nan
    level_zscore = float((current_val - s.mean()) / (s.std() + 1e-9)) if not s.empty else 0.0
    pct_rank     = float(pd.Series(s).rank(pct=True).iloc[-1]) if not s.empty else 0.5

    return {
        "current_velocity":   round(current_vel, 6),
        "velocity_zscore":    round(vel_zscore, 4),
        "velocity_mean":      round(vel_mean, 6),
        "velocity_std":       round(vel_std, 6),
        "acceleration":       round(acceleration, 6) if not np.isnan(acceleration) else None,
        "momentum_score":     round(momentum, 4),
        "consistency":        round(consistency, 4),
        "peak_velocity":      round(peak_vel, 6),
        "trough_velocity":    round(trough_vel, 6),
        "peak_date":          str(peak_date)[:10] if peak_date is not None else None,
        "pct_from_peak":      round(pct_from_peak, 4),
        "current_value":      round(current_val, 6) if not np.isnan(current_val) else None,
        "level_zscore":       round(level_zscore, 4),
        "pct_rank":           round(pct_rank, 4),
        "n_periods":          len(full_vel),
    }
