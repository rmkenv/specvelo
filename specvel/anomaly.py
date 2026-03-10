"""
specvel/anomaly.py

Regime change and anomaly detection for any velocity series.
Uses IsolationForest on rolling velocity features + optional
changepoint detection via ruptures.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

from core import compute_velocity


ANOMALY_THRESHOLD = -0.05   # IsolationForest score below this = flagged


def detect_anomaly(
    series:        pd.Series,
    contamination: float = 0.1,
    smooth:        bool  = True,
    window:        int   = 7,
) -> dict:
    """
    Detect whether the current velocity pattern is anomalous vs its own history.

    Parameters
    ----------
    series        : normalized pd.Series
    contamination : expected fraction of anomalies (IsolationForest param)
    smooth        : apply smoothing in velocity computation
    window        : smoothing window

    Returns
    -------
    dict with score (float), flag (bool), severity, and changepoint count
    """
    vel = compute_velocity(series, smooth=smooth, window=window).dropna()

    if len(vel) < 10:
        return {
            "score":          0.0,
            "flag":           False,
            "severity":       "normal",
            "n_changepoints": 0,
        }

    # Build rolling feature matrix
    vel_df = pd.DataFrame({
        "vel":        vel,
        "vel_lag1":   vel.shift(1),
        "vel_lag2":   vel.shift(2),
        "vel_abs":    vel.abs(),
        "vel_sq":     vel ** 2,
    }).dropna()

    if len(vel_df) < 8:
        return {
            "score":          0.0,
            "flag":           False,
            "severity":       "normal",
            "n_changepoints": 0,
        }

    X = vel_df.values
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    score = float(model.score_samples(X[-1].reshape(1, -1))[0])
    flag  = score < ANOMALY_THRESHOLD

    # Severity tiers
    if score < -0.20:
        severity = "extreme"
    elif score < -0.12:
        severity = "significant"
    elif score < ANOMALY_THRESHOLD:
        severity = "moderate"
    else:
        severity = "normal"

    # Changepoint detection
    n_cp = 0
    if HAS_RUPTURES and len(vel) >= 8:
        try:
            algo = rpt.Pelt(model="rbf").fit(vel.values.reshape(-1, 1))
            cps  = algo.predict(pen=1.0)
            n_cp = max(0, len(cps) - 1)
        except Exception:
            n_cp = _derivative_changepoints(vel.values)
    else:
        n_cp = _derivative_changepoints(vel.values)

    return {
        "score":          round(score, 4),
        "flag":           flag,
        "severity":       severity,
        "n_changepoints": n_cp,
    }


def _derivative_changepoints(values: np.ndarray) -> int:
    """Fallback: count sign changes in the derivative."""
    if len(values) < 3:
        return 0
    diff    = np.diff(values)
    signs   = np.sign(diff)
    changes = int(np.sum(np.diff(signs) != 0))
    return changes
