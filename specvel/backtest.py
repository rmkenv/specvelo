"""
specvel/backtest.py  —  v2 (improved signal)

Three improvements over v1 based on backtest results:

  FIX 1 — Unconditional rolling z-score for the surprise signal
           Phase-conditional baseline hurt Test 4 by ~2%.
           Phase labels are kept for transition warnings only.

  FIX 2 — Per-asset-class thresholds
           Equities: ±0.75 (default)
           Fixed income: ±1.0  (cleaner, less frequent)
           Commodities: ±1.25  (noisy — only fire on strong signals)
           Macro: ±0.50        (monthly data, need sensitivity)

  FIX 3 — Conviction-weighted signals
           Instead of binary LONG/SHORT, position size is proportional
           to |zscore| capped at 3σ. Monotonic IC → monotonic sizing.
           Reported as effective L/S spread using weights.

Tests:
  1 — Signal Return Backtest
  2 — Information Coefficient
  3 — Phase Transition Accuracy  [skipped in fast mode]
  4 — Specvel v1 vs v2 vs Naive  [extended test 4]
  5 — Stability Across Regimes   [skipped in fast mode]

Usage:
    python specvel/backtest.py --fast
    python specvel/backtest.py
    python specvel/backtest.py --fred_key KEY --include_macro
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from scipy.stats  import spearmanr
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ── Config ────────────────────────────────────────────────────────────────────

EQUITY_TICKERS = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "XLE": "Energy",  "XLF": "Financials",  "XLK": "Technology",
}
COMMODITY_TICKERS = {
    "CL=F": "WTI Crude", "GC=F": "Gold", "SI=F": "Silver",
    "ZC=F": "Corn",      "ZW=F": "Wheat",
}
EQUITY_TICKERS_FAST    = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "XLE": "Energy"}
COMMODITY_TICKERS_FAST = {"CL=F": "WTI Crude", "GC=F": "Gold"}

STABILITY_PERIODS = [
    ("2015-01-01", "2018-01-01", "Pre-2018 Bull"),
    ("2018-01-01", "2020-03-01", "Late Cycle"),
    ("2020-03-01", "2022-01-01", "COVID Recovery"),
    ("2022-01-01", "2024-01-01", "Rate Hike Cycle"),
    ("2024-01-01", "2026-03-10", "Recent"),
]

NBER_RECESSIONS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

FORWARD_PERIODS = [5, 10, 20, 60]
MIN_HISTORY     = 60
ROLL            = 120

# FIX 2: per-asset-class thresholds
THRESHOLDS = {
    "equities":    0.75,
    "commodities": 1.25,
    "fixed_income": 1.00,
    "macro":       0.50,
    "default":     0.75,
}


# ── Velocity ──────────────────────────────────────────────────────────────────

def _velocity(series: pd.Series, win: int = 7) -> pd.Series:
    v = series.dropna().values.astype(float)
    if len(v) < win + 2:
        return pd.Series(np.nan, index=series.index)
    w = win if win % 2 == 1 else win + 1
    w = max(w, 5)
    if len(v) >= w:
        v = savgol_filter(v, window_length=w, polyorder=2)
    grad = np.gradient(v)
    mx   = np.abs(grad).max()
    if mx > 0:
        grad /= (mx + 1e-9)
    return pd.Series(grad, index=series.dropna().index).reindex(series.index)


# ── Phase labels (kept for warnings + reporting only) ─────────────────────────

def _phase_labels_auto(vel: pd.Series) -> pd.Series:
    v     = vel.dropna()
    rank  = v.rolling(min(ROLL, len(v)), min_periods=20).rank(pct=True)
    accel = v.diff()
    phase = pd.Series("dormant", index=v.index)
    phase[rank >= 0.70]                  = "peak"
    phase[(rank < 0.70) & (accel > 0)]  = "green_up"
    phase[(rank < 0.70) & (accel <= 0)] = "senescence"
    phase[rank <= 0.30]                 = "dormant"
    return phase.reindex(vel.index, fill_value="dormant")


def _phase_labels_business(vel: pd.Series) -> pd.Series:
    in_rec = pd.Series(False, index=vel.index)
    for s, e in NBER_RECESSIONS:
        in_rec[(vel.index >= s) & (vel.index <= e)] = True
    v     = vel.dropna()
    rank  = v.rolling(min(ROLL, len(v)), min_periods=20).rank(pct=True)
    accel = v.diff()
    phase = pd.Series("green_up", index=v.index)
    phase[rank >= 0.65]                      = "peak"
    phase[(accel <= 0) & (rank < 0.65)]      = "senescence"
    in_rv = in_rec.reindex(v.index, fill_value=False)
    phase[in_rv & (rank >= 0.40)]            = "senescence"
    phase[in_rv & (rank <  0.40)]            = "dormant"
    return phase.reindex(vel.index, fill_value="dormant")


def _phase_labels_calendar(vel: pd.Series) -> pd.Series:
    q_map = {1: "green_up", 2: "peak", 3: "senescence", 4: "dormant"}
    return vel.index.to_series().dt.quarter.map(q_map).reindex(vel.index, fill_value="dormant")


def _get_phases(vel: pd.Series, cycle_method: str) -> pd.Series:
    if cycle_method == "business": return _phase_labels_business(vel)
    if cycle_method == "calendar": return _phase_labels_calendar(vel)
    return _phase_labels_auto(vel)


# ── Core signal builder ───────────────────────────────────────────────────────

def _build_signals(
    normed:        pd.Series,
    raw:           pd.Series,
    cycle_method:  str,
    threshold:     float,
    version:       str = "v2",   # "v1" | "v2" | "naive"
) -> pd.DataFrame:
    """
    version="v2"    FIX 1 — unconditional rolling zscore + phase for warnings
    version="v1"    old phase-conditional baseline (for comparison in Test 4)
    version="naive" plain velocity zscore, no phase at all
    """
    vel = _velocity(normed).dropna()
    if len(vel) < MIN_HISTORY + max(FORWARD_PERIODS) + 10:
        return pd.DataFrame()

    # Unconditional rolling baseline — always computed
    roll_mean = vel.rolling(ROLL, min_periods=MIN_HISTORY).mean().shift(1)
    roll_std  = vel.rolling(ROLL, min_periods=MIN_HISTORY).std().shift(1)

    if version == "naive":
        zscore  = (vel - roll_mean) / roll_std.clip(lower=1e-9)
        phases  = pd.Series("n/a", index=vel.index)
        warning = pd.Series(False, index=vel.index)

    elif version == "v1":
        # Old: phase-conditional baseline
        phases     = _get_phases(vel, cycle_method)
        ph_mean    = pd.Series(np.nan, index=vel.index)
        ph_std     = pd.Series(np.nan, index=vel.index)
        for ph in ["green_up", "peak", "senescence", "dormant"]:
            mask   = (phases == ph)
            ph_v   = vel.where(mask)
            pm     = ph_v.rolling(ROLL, min_periods=10).mean().shift(1)
            ps     = ph_v.rolling(ROLL, min_periods=10).std().shift(1)
            ph_mean = ph_mean.where(~mask, pm)
            ph_std  = ph_std.where(~mask, ps)
        ph_mean = ph_mean.fillna(roll_mean)
        ph_std  = ph_std.fillna(roll_std)
        zscore  = (vel - ph_mean) / ph_std.clip(lower=1e-9)
        accel   = vel.diff()
        warning = (
            ((phases == "peak")       & (accel < -0.01)) |
            ((phases == "dormant")    & (accel >  0.01)) |
            ((phases == "senescence") & (accel >  0.01)) |
            ((phases == "green_up")   & (accel <  0.0) & (vel.rolling(5).mean() < 0))
        )

    else:  # v2 — FIX 1: unconditional zscore, phases for warnings only
        zscore = (vel - roll_mean) / roll_std.clip(lower=1e-9)
        phases = _get_phases(vel, cycle_method)
        accel  = vel.diff()
        warning = (
            ((phases == "peak")       & (accel < -0.01)) |
            ((phases == "dormant")    & (accel >  0.01)) |
            ((phases == "senescence") & (accel >  0.01)) |
            ((phases == "green_up")   & (accel <  0.0) & (vel.rolling(5).mean() < 0))
        )

    # ── FIX 2: per-asset-class threshold applied here ─────────────────────────
    signal = pd.Series("NEUTRAL", index=vel.index)
    signal[zscore >=  threshold] = "LONG"
    signal[zscore <= -threshold] = "SHORT"

    # ── FIX 3: conviction weight = |zscore| / 3, capped at 1.0 ───────────────
    weight = (zscore.abs() / 3.0).clip(upper=1.0)
    # Signed weight: positive for LONG, negative for SHORT, 0 for NEUTRAL
    signed_weight = weight * zscore.apply(lambda z: 1 if z >= threshold else (-1 if z <= -threshold else 0))

    df = pd.DataFrame({
        "date":            vel.index,
        "surprise_zscore": zscore.round(4),
        "phase":           phases,
        "signal":          signal,
        "weight":          signed_weight.round(4),
        "has_warning":     warning,
    })

    df = df.iloc[MIN_HISTORY: -max(FORWARD_PERIODS)].copy()
    df = df.dropna(subset=["surprise_zscore"]).reset_index(drop=True)

    # Forward returns
    raw_idx = raw.index
    raw_arr = raw.values
    for fp in FORWARD_PERIODS:
        fwd_vals = []
        for d in df["date"].values:
            pos = raw_idx.get_loc(d) if d in raw_idx else -1
            if pos == -1 or pos + fp >= len(raw_arr):
                fwd_vals.append(np.nan)
            else:
                p0 = raw_arr[pos]
                pf = raw_arr[pos + fp]
                fwd_vals.append((pf - p0) / p0 if p0 != 0 else np.nan)
        df[f"fwd_{fp}d"] = fwd_vals

    return df


# ── L/S helpers ───────────────────────────────────────────────────────────────

def _ls_binary(df, fwd_col):
    """Classic binary L/S spread."""
    l = df[df["signal"] == "LONG"][fwd_col].mean()
    s = df[df["signal"] == "SHORT"][fwd_col].mean()
    return (l - s) * 100


def _ls_weighted(df, fwd_col):
    """FIX 3: weighted L/S — position proportional to conviction."""
    clean = df[["weight", fwd_col]].dropna()
    if clean.empty:
        return np.nan
    # Weighted return = sum(weight * return) / sum(|weight|)
    num   = (clean["weight"] * clean[fwd_col]).sum()
    denom = clean["weight"].abs().sum()
    return (num / denom) * 100 if denom > 0 else np.nan


# ── Test 1 ────────────────────────────────────────────────────────────────────

def test1_signal_returns(adapter, tickers, start, end, cycle_method, asset_class="default"):
    threshold = THRESHOLDS.get(asset_class, THRESHOLDS["default"])
    fwd_cols  = [f"fwd_{fp}d" for fp in FORWARD_PERIODS]
    rows = []

    print(f"\n{'─'*70}")
    print(f"  TEST 1 — Signal Returns  [{cycle_method}]  threshold=±{threshold}  {start}→{end}")
    print(f"{'─'*70}")

    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df     = _build_signals(normed, raw, cycle_method, threshold, version="v2")
            if df.empty:
                print(f"  {label:<22} — no results"); continue

            print(f"\n  {label} ({ticker})  —  {len(df)} obs  "
                  f"[LONG:{(df['signal']=='LONG').sum()} "
                  f"SHORT:{(df['signal']=='SHORT').sum()} "
                  f"NEUTRAL:{(df['signal']=='NEUTRAL').sum()}]")
            print(f"  {'SIGNAL':<10} {'N':>5}  " + "  ".join(f"{c:>10}" for c in fwd_cols))
            print(f"  {'─'*57}")

            for sig in ["LONG", "NEUTRAL", "SHORT"]:
                sub   = df[df["signal"] == sig]
                means = [sub[c].mean()*100 if not sub.empty else np.nan for c in fwd_cols]
                print(f"  {sig:<10} {len(sub):>5}  " +
                      "  ".join(f"{m:>+9.2f}%" if not np.isnan(m) else f"{'n/a':>10}" for m in means))

            print(f"  {'─'*57}")
            for c in fwd_cols:
                binary   = _ls_binary(df, c)
                weighted = _ls_weighted(df, c)
                flag     = "✓" if binary > 0.3 else ("✗" if binary < 0 else "~")
                print(f"  L/S {c}: binary={binary:>+6.2f}%  weighted={weighted:>+6.2f}%  {flag}")

            rows.append({
                "ticker": ticker, "label": label, "n_obs": len(df),
                "threshold": threshold,
                **{f"ls_{c}":   _ls_binary(df, c)   for c in fwd_cols if c in df.columns},
                **{f"wls_{c}":  _ls_weighted(df, c) for c in fwd_cols if c in df.columns},
            })
        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    return pd.DataFrame(rows)


# ── Test 2 ────────────────────────────────────────────────────────────────────

def test2_ic(adapter, tickers, start, end, cycle_method, asset_class="default", fwd_col="fwd_20d"):
    threshold = THRESHOLDS.get(asset_class, THRESHOLDS["default"])
    print(f"\n{'─'*70}")
    print(f"  TEST 2 — IC [{fwd_col}]  threshold=±{threshold}")
    print(f"{'─'*70}")
    print(f"  {'TICKER':<22} {'IC':>8} {'P-VAL':>8} {'ICIR':>8} {'IC+%':>8}  STATUS")
    print(f"  {'─'*62}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df     = _build_signals(normed, raw, cycle_method, threshold, version="v2")
            if df.empty or fwd_col not in df.columns: continue

            clean = df[["surprise_zscore", fwd_col]].dropna()
            if len(clean) < 30: continue

            ic, pval = spearmanr(clean["surprise_zscore"], clean[fwd_col])
            ric      = clean["surprise_zscore"].rolling(52).corr(clean[fwd_col])
            icir     = ric.mean() / (ric.std() + 1e-9)
            pct_pos  = (ric > 0).mean()

            status = ("✓ STRONG" if ic>0.08 and pval<0.05 else
                      "✓ USEFUL" if ic>0.04 and pval<0.10 else
                      "~ WEAK"   if ic>0 else "✗ INVERSE")

            print(f"  {label:<22} {ic:>+8.4f} {pval:>8.4f} {icir:>+8.4f} {pct_pos:>7.1%}  {status}")
            rows.append({"ticker":ticker,"label":label,"ic":ic,"p_value":pval,
                         "icir":icir,"pct_positive":pct_pos,"n":len(clean),"status":status})
        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Avg IC:   {df_out['ic'].mean():>+.4f}")
        print(f"  Avg ICIR: {df_out['icir'].mean():>+.4f}")
        print(f"  IC > 0:   {(df_out['ic']>0).mean():.0%}")
    return df_out


# ── Test 3 ────────────────────────────────────────────────────────────────────

def test3_transitions(adapter, tickers, start, end, cycle_method, asset_class="default", lookahead=10):
    threshold = THRESHOLDS.get(asset_class, THRESHOLDS["default"])
    print(f"\n{'─'*70}")
    print(f"  TEST 3 — Transition Accuracy  lookahead={lookahead}")
    print(f"{'─'*70}")
    print(f"  {'TICKER':<22} {'WARNINGS':>9} {'HITS':>6} {'HIT RATE':>10}  STATUS")
    print(f"  {'─'*57}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df     = _build_signals(normed, raw, cycle_method, threshold, version="v2")
            if df.empty: continue

            warned = df[df["has_warning"]].copy()
            if len(warned) < 5:
                print(f"  {label:<22} — too few warnings ({len(warned)})"); continue

            hits = sum(
                1 for pos in warned.index
                if (lambda f: len(f) > 1 and f.iloc[0] != f.iloc[-1])(
                    df.loc[pos: pos+lookahead, "phase"])
            )

            hit_rate = hits / len(warned)
            status   = ("✓ STRONG" if hit_rate>0.65 else
                        "✓ USEFUL" if hit_rate>0.50 else "✗ BELOW RANDOM")
            print(f"  {label:<22} {len(warned):>9} {hits:>6} {hit_rate:>9.1%}   {status}")
            rows.append({"ticker":ticker,"label":label,"n_warnings":len(warned),
                         "hits":hits,"hit_rate":hit_rate,"status":status})
        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Avg hit rate: {df_out['hit_rate'].mean():.1%}")
    return df_out


# ── Test 4 — v2 vs v1 vs naive ────────────────────────────────────────────────

def test4_comparison(adapter, tickers, start, end, cycle_method, asset_class="default", fwd_col="fwd_20d"):
    threshold = THRESHOLDS.get(asset_class, THRESHOLDS["default"])
    print(f"\n{'─'*70}")
    print(f"  TEST 4 — v2 vs v1 vs Naive [{fwd_col}]  threshold=±{threshold}")
    print(f"{'─'*70}")
    print(f"  {'TICKER':<22} {'V2 (new)':>10} {'V1 (old)':>10} {'NAIVE':>10} {'V2-NAIVE':>10}  STATUS")
    print(f"  {'─'*68}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)

            df_v2    = _build_signals(normed, raw, cycle_method, threshold, version="v2")
            df_v1    = _build_signals(normed, raw, cycle_method, threshold, version="v1")
            df_naive = _build_signals(normed, raw, cycle_method, threshold, version="naive")

            if any(d.empty for d in [df_v2, df_v1, df_naive]) or fwd_col not in df_v2.columns:
                continue

            v2    = _ls_binary(df_v2,    fwd_col)
            v1    = _ls_binary(df_v1,    fwd_col)
            naive = _ls_binary(df_naive, fwd_col)
            v2w   = _ls_weighted(df_v2,  fwd_col)  # weighted version
            delta = v2 - naive

            status = ("✓ BEATS NAIVE" if delta > 0.2  else
                      "~ SIMILAR"     if delta > -0.2  else
                      "✗ BELOW NAIVE")

            print(f"  {label:<22} {v2:>+9.2f}% {v1:>+9.2f}% {naive:>+9.2f}% {delta:>+9.2f}%  {status}")
            print(f"  {'':22}  weighted={v2w:>+8.2f}%")

            rows.append({"ticker":ticker,"label":label,
                         "v2_ls":v2,"v1_ls":v1,"naive_ls":naive,
                         "v2_weighted_ls":v2w,"delta_vs_naive":delta,"status":status})
        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Avg v2 L/S:       {df_out['v2_ls'].mean():>+.2f}%")
        print(f"  Avg v2 weighted:  {df_out['v2_weighted_ls'].mean():>+.2f}%")
        print(f"  Avg v1 L/S:       {df_out['v1_ls'].mean():>+.2f}%")
        print(f"  Avg naive L/S:    {df_out['naive_ls'].mean():>+.2f}%")
        print(f"  Avg delta:        {df_out['delta_vs_naive'].mean():>+.2f}%")
        print(f"  v2 beats naive:   {(df_out['delta_vs_naive']>0).mean():.0%}")
    return df_out


# ── Test 5 ────────────────────────────────────────────────────────────────────

def test5_stability(adapter, tickers, cycle_method, asset_class="default", fwd_col="fwd_20d"):
    threshold = THRESHOLDS.get(asset_class, THRESHOLDS["default"])
    print(f"\n{'─'*70}")
    print(f"  TEST 5 — Stability  threshold=±{threshold}  [{fwd_col}]")
    print(f"{'─'*70}")
    print(f"  {'TICKER':<18} " + "  ".join(f"{p[2][:13]:>13}" for p in STABILITY_PERIODS))
    print(f"  {'─'*90}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, "2014-01-01", "2026-03-10")
            normed = adapter.normalize(raw)
            df     = _build_signals(normed, raw, cycle_method, threshold, version="v2")
            if df.empty: continue
            df["date"] = pd.to_datetime(df["date"])

            period_results = {}
            for s, e, pl in STABILITY_PERIODS:
                sub = df[(df["date"] >= s) & (df["date"] <= e)]
                if len(sub) < 30 or fwd_col not in sub.columns:
                    period_results[pl] = np.nan; continue
                period_results[pl] = _ls_binary(sub, fwd_col)

            values = [period_results.get(p[2], np.nan) for p in STABILITY_PERIODS]
            n_pos  = sum(1 for v in values if not np.isnan(v) and v > 0)
            status = ("✓" if n_pos>=4 else ("~" if n_pos>=3 else "✗"))

            line = f"  {label[:17]:<18} "
            for v in values:
                line += (f"{'n/a':>13}  " if np.isnan(v) else f"{v:>+11.2f}%  ")
            print(line + f" {status} ({n_pos}/5)")

            row = {"ticker":ticker,"label":label,"n_positive":n_pos}
            row.update({p[2]: period_results.get(p[2]) for p in STABILITY_PERIODS})
            rows.append(row)
        except Exception as e:
            print(f"  {label:<18} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Avg positive periods:  {df_out['n_positive'].mean():.1f}/5")
        print(f"  % tickers ≥4 positive: {(df_out['n_positive']>=4).mean():.0%}")
    return df_out


# ── Scorecard ─────────────────────────────────────────────────────────────────

def print_scorecard(t1, t2, t3, t4, t5):
    print(f"\n{'='*70}\n  SPECVEL v2 VALIDATION SCORECARD\n{'='*70}")
    passes = []

    if not t1.empty:
        lsc    = [c for c in t1.columns if c.startswith("ls_")]
        wlsc   = [c for c in t1.columns if c.startswith("wls_")]
        avg    = t1[lsc].mean().mean()  if lsc  else np.nan
        wavg   = t1[wlsc].mean().mean() if wlsc else np.nan
        ppos   = (t1[lsc]>0).mean().mean() if lsc else np.nan
        print(f"\n  Test 1 — Signal Returns")
        print(f"    Avg L/S (binary):   {avg:>+6.2f}%  {'✓' if avg>0.3 else '✗'}  (>+0.30%)")
        print(f"    Avg L/S (weighted): {wavg:>+6.2f}%  {'✓' if wavg>0.3 else '✗'}  (>+0.30%)")
        print(f"    % positive spreads: {ppos:>6.1%}  {'✓' if ppos>0.55 else '✗'}  (>55%)")
        passes.append(not np.isnan(avg) and avg > 0.3)

    if not t2.empty:
        ic, icir = t2["ic"].mean(), t2["icir"].mean()
        pp       = (t2["ic"]>0).mean()
        print(f"\n  Test 2 — Information Coefficient")
        print(f"    Avg IC:   {ic:>+7.4f}  {'✓' if ic>0.04 else '✗'}  (>0.04)")
        print(f"    Avg ICIR: {icir:>+7.4f}  {'✓' if icir>0.3 else '✗'}  (>0.30)")
        print(f"    IC > 0:   {pp:>6.1%}  {'✓' if pp>0.55 else '✗'}  (>55%)")
        passes.append(ic > 0.04)

    if not t3.empty:
        hr = t3["hit_rate"].mean()
        print(f"\n  Test 3 — Transition Accuracy")
        print(f"    Avg hit rate: {hr:>6.1%}  {'✓' if hr>0.50 else '✗'}  (>50%)")
        passes.append(hr > 0.50)

    if not t4.empty:
        d  = t4["delta_vs_naive"].mean()
        dw = t4["v2_weighted_ls"].mean() - t4["naive_ls"].mean()
        pb = (t4["delta_vs_naive"]>0).mean()
        print(f"\n  Test 4 — v2 vs Naive")
        print(f"    Avg delta (binary):   {d:>+6.2f}%  {'✓' if d>0 else '✗'}  (>0%)")
        print(f"    Avg delta (weighted): {dw:>+6.2f}%  {'✓' if dw>0 else '✗'}  (>0%)")
        print(f"    % beats naive:        {pb:>6.1%}  {'✓' if pb>0.55 else '✗'}  (>55%)")
        passes.append(d > 0)

    if not t5.empty:
        ap = t5["n_positive"].mean()
        ps = (t5["n_positive"]>=4).mean()
        print(f"\n  Test 5 — Stability")
        print(f"    Avg pos periods: {ap:>5.1f}/5  {'✓' if ap>=3.5 else '✗'}  (≥3.5)")
        print(f"    % stable:        {ps:>6.1%}  {'✓' if ps>0.55 else '✗'}  (>55%)")
        passes.append(ap >= 3.5)

    n = sum(passes)
    v = ("🟢 SIGNAL VALIDATED"  if n>=4 else
         "🟡 SIGNAL PROMISING"  if n>=3 else
         "🔴 SIGNAL NEEDS WORK" if n>=2 else
         "⚫ NOT CONFIRMED")
    print(f"\n  {'─'*55}")
    print(f"  Tests passed: {n}/{len(passes)}")
    print(f"  Verdict:      {v}")
    print(f"{'='*70}\n")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tests(fred_key=None, include_macro=False, save_results=None, fast=False):
    from adapters.equities    import EquitiesAdapter
    from adapters.commodities import CommoditiesAdapter

    if fast:
        START, END = "2015-01-01", "2026-03-10"
        eq_t       = EQUITY_TICKERS_FAST
        cm_t       = COMMODITY_TICKERS_FAST
        print("FAST MODE: 5 tickers, Tests 1+2+4 only")
    else:
        START, END = "2015-01-01", "2026-03-10"
        eq_t       = EQUITY_TICKERS
        cm_t       = COMMODITY_TICKERS

    adapters = [
        (EquitiesAdapter(),    eq_t, "business", "equities",    "EQUITIES"),
        (CommoditiesAdapter(), cm_t, "auto",     "commodities", "COMMODITIES"),
    ]

    if include_macro and fred_key:
        from adapters.fixed_income import FixedIncomeAdapter
        from adapters.macro        import MacroAdapter
        fi = ({"DGS2":"2Y Tsy","DGS10":"10Y Tsy"} if fast else
              {"DGS2":"2Y Tsy","DGS10":"10Y Tsy","T10Y2Y":"Spread","BAMLH0A0HYM2":"HY"})
        mc = ({"CPIAUCSL":"CPI","UNRATE":"Unemployment"} if fast else
              {"CPIAUCSL":"CPI","CPILFESL":"Core CPI","UNRATE":"Unemp","INDPRO":"IndProd"})
        adapters += [
            (FixedIncomeAdapter(api_key=fred_key), fi, "business", "fixed_income", "FIXED INCOME"),
            (MacroAdapter(api_key=fred_key),       mc, "calendar", "macro",        "MACRO"),
        ]

    a1,a2,a3,a4,a5 = [],[],[],[],[]

    for adapter, tickers, cm, asset_class, name in adapters:
        print(f"\n\n{'#'*70}\n#  {name}  (threshold ±{THRESHOLDS.get(asset_class,'?')})\n{'#'*70}")

        t1 = test1_signal_returns(adapter, tickers, START, END, cm, asset_class)
        t2 = test2_ic(adapter, tickers, START, END, cm, asset_class)
        t3 = test3_transitions(adapter, tickers, START, END, cm, asset_class) if not fast else pd.DataFrame()
        t4 = test4_comparison(adapter, tickers, START, END, cm, asset_class)
        t5 = test5_stability(adapter, tickers, cm, asset_class) if not fast else pd.DataFrame()

        for df, lst in [(t1,a1),(t2,a2),(t3,a3),(t4,a4),(t5,a5)]:
            if not df.empty:
                df["adapter"] = name
                lst.append(df)

    T1 = pd.concat(a1, ignore_index=True) if a1 else pd.DataFrame()
    T2 = pd.concat(a2, ignore_index=True) if a2 else pd.DataFrame()
    T3 = pd.concat(a3, ignore_index=True) if a3 else pd.DataFrame()
    T4 = pd.concat(a4, ignore_index=True) if a4 else pd.DataFrame()
    T5 = pd.concat(a5, ignore_index=True) if a5 else pd.DataFrame()

    print_scorecard(T1, T2, T3, T4, T5)

    if save_results:
        os.makedirs(save_results, exist_ok=True)
        for lbl, df in [("t1_signal_returns",T1),("t2_ic",T2),
                        ("t3_transitions",T3),("t4_comparison",T4),
                        ("t5_stability",T5)]:
            if not df.empty:
                path = os.path.join(save_results, f"backtest_{lbl}.csv")
                df.to_csv(path, index=False)
                print(f"Saved {path}")

    return T1, T2, T3, T4, T5


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fred_key",      type=str,  default=None)
    p.add_argument("--include_macro", action="store_true")
    p.add_argument("--save",          type=str,  default="results/")
    p.add_argument("--fast",          action="store_true")
    a = p.parse_args()
    run_all_tests(fred_key=a.fred_key, include_macro=a.include_macro,
                  save_results=a.save, fast=a.fast)
