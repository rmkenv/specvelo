"""
specvel/backtest.py

Validates the spectral velocity cycle signal across 5 tests:

  Test 1 — Signal Return Backtest
           Does high surprise_zscore predict above-average forward returns?

  Test 2 — Information Coefficient (IC)
           Spearman correlation between surprise_zscore and forward return.
           IC > 0.05 = useful, IC > 0.10 = strong.

  Test 3 — Phase Transition Accuracy
           Do transition_warnings actually precede phase changes?

  Test 4 — Specvel vs Naive Momentum
           Does cycle conditioning beat plain velocity z-score?

  Test 5 — Stability Across Time Periods
           Does the signal hold up in each market regime?

Usage (Colab):
    !git clone https://github.com/rmkenv/specvelo.git
    !pip install yfinance scipy scikit-learn ruptures -q
    import sys; sys.path.insert(0, '/content/specvelo/specvel')
    from backtest import run_all_tests
    run_all_tests()

Usage (local):
    cd specvel
    python specvel/backtest.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Path setup — works whether run directly or imported from Colab ────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core    import compute_velocity
from cycle   import compute_velocity_surprise, detect_phase


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Tickers to validate — equities and commodities need no API key
EQUITY_TICKERS = {
    "SPY":  "S&P 500",
    "QQQ":  "Nasdaq 100",
    "IWM":  "Russell 2000",
    "XLE":  "Energy Sector",
    "XLF":  "Financials",
    "XLK":  "Technology",
}

COMMODITY_TICKERS = {
    "CL=F": "WTI Crude",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "ZC=F": "Corn",
    "ZW=F": "Wheat",
}

# Time periods for stability test
STABILITY_PERIODS = [
    ("2015-01-01", "2018-01-01", "Pre-2018 Bull"),
    ("2018-01-01", "2020-03-01", "Late Cycle + Correction"),
    ("2020-03-01", "2022-01-01", "COVID Recovery"),
    ("2022-01-01", "2024-01-01", "Rate Hike Cycle"),
    ("2024-01-01", "2026-03-10", "Recent"),
]

FORWARD_PERIODS   = [5, 10, 20, 60]   # trading days ahead
ZSCORE_THRESHOLD  = 0.75              # |z| above this = signal
MIN_HISTORY       = 60                # periods needed before signaling
MIN_PHASE_SAMPLES = 5                 # minimum phase observations for baseline


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward(
    normed:          pd.Series,
    raw:             pd.Series,
    cycle_method:    str,
    forward_periods: list,
    zscore_threshold: float,
    use_cycle:       bool = True,
) -> pd.DataFrame:
    """
    Walk forward through history one period at a time.
    At each step, compute signal using only past data (no lookahead),
    then record forward returns.

    Parameters
    ----------
    normed           : normalized series (output of adapter.normalize())
    raw              : original unnormalized series (for return calculation)
    cycle_method     : 'auto', 'business', or 'calendar'
    forward_periods  : list of N-day horizons to measure returns
    zscore_threshold : |zscore| above this = LONG or SHORT signal
    use_cycle        : if False, use plain velocity zscore (naive baseline)

    Returns
    -------
    DataFrame with one row per observation point
    """
    results = []
    max_fp  = max(forward_periods)

    for i in range(MIN_HISTORY, len(normed) - max_fp - 1):
        hist       = normed.iloc[:i]
        raw_future = raw.iloc[i: i + max_fp + 1]

        if raw_future.empty or len(raw_future) < 2:
            continue

        p0 = float(raw_future.iloc[0])
        if p0 == 0:
            continue

        date = hist.index[-1]

        # ── Signal computation ────────────────────────────────────────────────
        if use_cycle:
            surp = compute_velocity_surprise(hist, cycle_method=cycle_method)
            if not surp or surp.get("phase_n_samples", 0) < MIN_PHASE_SAMPLES:
                continue
            zscore = surp["surprise_zscore"]
            phase  = surp["current_phase"]
            has_warning = bool(surp.get("transition_warning", ""))
            boost  = surp.get("conviction_boost", 0)
        else:
            # Naive: plain velocity z-score, no cycle conditioning
            vel = compute_velocity(hist).dropna()
            if len(vel) < 20:
                continue
            cur_vel = float(vel.iloc[-1])
            zscore  = (cur_vel - vel.mean()) / (vel.std() + 1e-9)
            phase   = "n/a"
            has_warning = False
            boost   = 0

        signal = (
            "LONG"    if zscore >=  zscore_threshold else
            "SHORT"   if zscore <= -zscore_threshold else
            "NEUTRAL"
        )

        row = {
            "date":            date,
            "surprise_zscore": round(zscore, 4),
            "phase":           phase,
            "signal":          signal,
            "conviction_boost": boost,
            "has_warning":     has_warning,
        }

        for fp in forward_periods:
            if fp < len(raw_future):
                fwd_price = float(raw_future.iloc[fp])
                row[f"fwd_{fp}d"] = (fwd_price - p0) / p0
            else:
                row[f"fwd_{fp}d"] = np.nan

        results.append(row)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Signal Return Backtest
# ─────────────────────────────────────────────────────────────────────────────

def test1_signal_returns(
    adapter,
    tickers:      dict,
    start:        str,
    end:          str,
    cycle_method: str,
) -> pd.DataFrame:
    """
    For each ticker, compute average forward returns by signal bucket.
    Returns summary DataFrame.
    """
    fwd_cols = [f"fwd_{fp}d" for fp in FORWARD_PERIODS]
    all_rows = []

    print(f"\n{'─'*65}")
    print(f"  TEST 1 — Signal Return Backtest")
    print(f"  Method: {cycle_method}  |  {start} → {end}")
    print(f"{'─'*65}")

    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)

            if len(normed.dropna()) < MIN_HISTORY + max(FORWARD_PERIODS) + 10:
                print(f"  {label:<22} — insufficient data, skipping")
                continue

            df = _walk_forward(normed, raw, cycle_method,
                               FORWARD_PERIODS, ZSCORE_THRESHOLD)
            if df.empty:
                continue

            print(f"\n  {label} ({ticker})  —  {len(df)} observations")
            print(f"  {'SIGNAL':<10} {'N':>5}  " +
                  "  ".join(f"{c:>10}" for c in fwd_cols))
            print(f"  {'─'*55}")

            for sig in ["LONG", "NEUTRAL", "SHORT"]:
                sub   = df[df["signal"] == sig]
                means = [sub[c].mean() * 100 if c in sub.columns and not sub.empty
                         else np.nan for c in fwd_cols]
                print(f"  {sig:<10} {len(sub):>5}  " +
                      "  ".join(f"{m:>+9.2f}%" if not np.isnan(m) else f"{'n/a':>10}"
                                for m in means))

            # L/S spread
            print(f"  {'─'*55}")
            for c in fwd_cols:
                l_ret  = df[df["signal"] == "LONG"][c].mean()
                s_ret  = df[df["signal"] == "SHORT"][c].mean()
                spread = (l_ret - s_ret) * 100
                flag   = "✓" if spread > 0.3 else ("✗" if spread < 0 else "~")
                print(f"  L/S spread {c}: {spread:>+6.2f}%  {flag}")

            all_rows.append({
                "ticker": ticker,
                "label":  label,
                "n_obs":  len(df),
                **{f"ls_{c}": (df[df["signal"]=="LONG"][c].mean() -
                               df[df["signal"]=="SHORT"][c].mean()) * 100
                   for c in fwd_cols if c in df.columns},
            })

        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Information Coefficient
# ─────────────────────────────────────────────────────────────────────────────

def test2_information_coefficient(
    adapter,
    tickers:      dict,
    start:        str,
    end:          str,
    cycle_method: str,
    fwd_col:      str = "fwd_20d",
) -> pd.DataFrame:
    """
    Spearman rank correlation between surprise_zscore and forward return.
    IC > 0.05 = useful signal.
    IC Information Ratio (ICIR) > 0.5 = consistent signal.
    """
    print(f"\n{'─'*65}")
    print(f"  TEST 2 — Information Coefficient (IC)  [{fwd_col}]")
    print(f"  Method: {cycle_method}  |  {start} → {end}")
    print(f"{'─'*65}")
    print(f"  {'TICKER':<22} {'IC':>8} {'P-VAL':>8} {'ICIR':>8} "
          f"{'IC+%':>8} {'STATUS'}")
    print(f"  {'─'*60}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df     = _walk_forward(normed, raw, cycle_method,
                                   FORWARD_PERIODS, ZSCORE_THRESHOLD)

            if df.empty or fwd_col not in df.columns:
                continue

            clean = df[["surprise_zscore", fwd_col]].dropna()
            if len(clean) < 30:
                continue

            ic, pval = spearmanr(clean["surprise_zscore"], clean[fwd_col])

            # Rolling IC (52-period window)
            rolling_ic = clean["surprise_zscore"].rolling(52).corr(clean[fwd_col])
            ic_mean    = rolling_ic.mean()
            ic_std     = rolling_ic.std()
            icir       = ic_mean / (ic_std + 1e-9)
            pct_pos    = (rolling_ic > 0).mean()

            status = ("✓ STRONG"  if ic > 0.08 and pval < 0.05 else
                      "✓ USEFUL"  if ic > 0.04 and pval < 0.10 else
                      "~ WEAK"    if ic > 0     else
                      "✗ INVERSE")

            print(f"  {label:<22} {ic:>+8.4f} {pval:>8.4f} {icir:>+8.4f} "
                  f"{pct_pos:>7.1%}  {status}")

            rows.append({
                "ticker": ticker, "label": label,
                "ic": ic, "p_value": pval,
                "ic_mean": ic_mean, "icir": icir,
                "pct_positive": pct_pos, "n": len(clean),
                "status": status,
            })

        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Average IC:   {df_out['ic'].mean():>+.4f}")
        print(f"  Average ICIR: {df_out['icir'].mean():>+.4f}")
        print(f"  % Positive IC: {(df_out['ic'] > 0).mean():.0%}")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Phase Transition Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def test3_transition_accuracy(
    adapter,
    tickers:      dict,
    start:        str,
    end:          str,
    cycle_method: str,
    lookahead:    int = 10,
) -> pd.DataFrame:
    """
    When a transition_warning fires, does a phase change actually follow
    within `lookahead` periods?
    Hit rate > 0.50 beats random.
    Hit rate > 0.65 is strong.
    """
    print(f"\n{'─'*65}")
    print(f"  TEST 3 — Phase Transition Warning Accuracy")
    print(f"  Lookahead: {lookahead} periods  |  {start} → {end}")
    print(f"{'─'*65}")
    print(f"  {'TICKER':<22} {'WARNINGS':>9} {'HITS':>6} "
          f"{'HIT RATE':>10}  STATUS")
    print(f"  {'─'*55}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df     = _walk_forward(normed, raw, cycle_method,
                                   FORWARD_PERIODS, ZSCORE_THRESHOLD)

            if df.empty or "has_warning" not in df.columns:
                continue

            warned = df[df["has_warning"]].copy().reset_index(drop=True)
            if len(warned) < 5:
                print(f"  {label:<22} — too few warnings ({len(warned)}), skipping")
                continue

            hits = 0
            for i, row in warned.iterrows():
                # Find position in full df
                pos = df[df["date"] == row["date"]].index
                if pos.empty:
                    continue
                pos = pos[0]
                future = df.loc[pos: pos + lookahead, "phase"]
                if len(future) > 1 and future.iloc[0] != future.iloc[-1]:
                    hits += 1

            hit_rate = hits / len(warned)
            status   = ("✓ STRONG" if hit_rate > 0.65 else
                        "✓ USEFUL" if hit_rate > 0.50 else
                        "✗ BELOW RANDOM")

            print(f"  {label:<22} {len(warned):>9} {hits:>6} "
                  f"{hit_rate:>9.1%}   {status}")

            rows.append({
                "ticker": ticker, "label": label,
                "n_warnings": len(warned),
                "hits": hits, "hit_rate": hit_rate,
                "status": status,
            })

        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Average hit rate: {df_out['hit_rate'].mean():.1%}")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Specvel vs Naive Momentum
# ─────────────────────────────────────────────────────────────────────────────

def test4_vs_naive(
    adapter,
    tickers:      dict,
    start:        str,
    end:          str,
    cycle_method: str,
    fwd_col:      str = "fwd_20d",
) -> pd.DataFrame:
    """
    Compare specvel cycle signal L/S spread vs plain velocity z-score.
    Positive delta = cycle conditioning adds value over naive momentum.
    """
    print(f"\n{'─'*65}")
    print(f"  TEST 4 — Specvel vs Naive Momentum  [{fwd_col}]")
    print(f"  Method: {cycle_method}  |  {start} → {end}")
    print(f"{'─'*65}")
    print(f"  {'TICKER':<22} {'SPECVEL L/S':>12} {'NAIVE L/S':>10} "
          f"{'DELTA':>8}  STATUS")
    print(f"  {'─'*60}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)

            # Specvel signal
            df_sv = _walk_forward(normed, raw, cycle_method,
                                  FORWARD_PERIODS, ZSCORE_THRESHOLD,
                                  use_cycle=True)

            # Naive signal
            df_nv = _walk_forward(normed, raw, cycle_method,
                                  FORWARD_PERIODS, ZSCORE_THRESHOLD,
                                  use_cycle=False)

            if df_sv.empty or df_nv.empty or fwd_col not in df_sv.columns:
                continue

            def ls_spread(df):
                l = df[df["signal"] == "LONG"][fwd_col].mean()
                s = df[df["signal"] == "SHORT"][fwd_col].mean()
                return (l - s) * 100

            sv_ls = ls_spread(df_sv)
            nv_ls = ls_spread(df_nv)
            delta = sv_ls - nv_ls

            status = ("✓ BETTER"  if delta > 0.2  else
                      "~ SIMILAR" if delta > -0.2  else
                      "✗ WORSE")

            print(f"  {label:<22} {sv_ls:>+11.2f}% {nv_ls:>+9.2f}% "
                  f"{delta:>+7.2f}%  {status}")

            rows.append({
                "ticker": ticker, "label": label,
                "specvel_ls": sv_ls, "naive_ls": nv_ls,
                "delta": delta, "status": status,
            })

        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Average specvel L/S: {df_out['specvel_ls'].mean():>+.2f}%")
        print(f"  Average naive L/S:   {df_out['naive_ls'].mean():>+.2f}%")
        print(f"  Average delta:       {df_out['delta'].mean():>+.2f}%")
        pct_better = (df_out["delta"] > 0).mean()
        print(f"  Specvel better in:   {pct_better:.0%} of tickers")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Stability Across Time Periods
# ─────────────────────────────────────────────────────────────────────────────

def test5_stability(
    adapter,
    tickers:      dict,
    cycle_method: str,
    fwd_col:      str = "fwd_20d",
) -> pd.DataFrame:
    """
    Run the signal across each market regime period.
    A real signal holds up across at least 4 of 5 periods.
    """
    print(f"\n{'─'*65}")
    print(f"  TEST 5 — Stability Across Market Regimes  [{fwd_col}]")
    print(f"  Method: {cycle_method}")
    print(f"{'─'*65}")

    # Header
    period_labels = [p[2][:16] for p in STABILITY_PERIODS]
    header = f"  {'TICKER':<18} " + "  ".join(f"{l:>14}" for l in period_labels)
    print(header)
    print(f"  {'─'*max(65, len(header)-2)}")

    rows = []
    for ticker, label in tickers.items():
        period_results = {}
        try:
            raw    = adapter.fetch(ticker, "2014-01-01", "2026-03-10")
            normed = adapter.normalize(raw)

            for start, end, period_label in STABILITY_PERIODS:
                try:
                    n_start = normed[start:end]
                    r_start = raw[start:end]

                    if len(n_start.dropna()) < MIN_HISTORY + max(FORWARD_PERIODS) + 5:
                        period_results[period_label] = np.nan
                        continue

                    df = _walk_forward(n_start, r_start, cycle_method,
                                       FORWARD_PERIODS, ZSCORE_THRESHOLD)

                    if df.empty or fwd_col not in df.columns:
                        period_results[period_label] = np.nan
                        continue

                    l  = df[df["signal"] == "LONG"][fwd_col].mean()
                    s  = df[df["signal"] == "SHORT"][fwd_col].mean()
                    ls = (l - s) * 100
                    period_results[period_label] = ls

                except Exception:
                    period_results[period_label] = np.nan

            values = [period_results.get(p[2], np.nan) for p in STABILITY_PERIODS]
            n_pos  = sum(1 for v in values if not np.isnan(v) and v > 0)
            status = ("✓" if n_pos >= 4 else ("~" if n_pos >= 3 else "✗"))

            line = f"  {label[:17]:<18} "
            for v in values:
                if np.isnan(v):
                    line += f"{'n/a':>14}  "
                else:
                    flag  = "+" if v > 0 else "-"
                    line += f"{v:>+12.2f}%  "
            line += f" {status} ({n_pos}/5)"
            print(line)

            row = {"ticker": ticker, "label": label, "n_positive": n_pos}
            row.update({p[2]: period_results.get(p[2]) for p in STABILITY_PERIODS})
            rows.append(row)

        except Exception as e:
            print(f"  {label:<18} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        avg_pos = df_out["n_positive"].mean()
        print(f"\n  Average positive periods: {avg_pos:.1f} / 5")
        print(f"  % tickers with ≥4 positive: "
              f"{(df_out['n_positive'] >= 4).mean():.0%}")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Summary scorecard
# ─────────────────────────────────────────────────────────────────────────────

def print_scorecard(
    t1: pd.DataFrame,
    t2: pd.DataFrame,
    t3: pd.DataFrame,
    t4: pd.DataFrame,
    t5: pd.DataFrame,
):
    """Print a one-page summary of all test results."""
    print(f"\n{'='*65}")
    print(f"  SPECVEL VALIDATION SCORECARD")
    print(f"{'='*65}")

    # Test 1
    if not t1.empty:
        ls_cols = [c for c in t1.columns if c.startswith("ls_")]
        avg_ls  = t1[ls_cols].mean().mean() if ls_cols else np.nan
        pct_pos = (t1[ls_cols] > 0).mean().mean() if ls_cols else np.nan
        t1_pass = avg_ls > 0.3 and pct_pos > 0.55
        print(f"\n  Test 1 — Signal Returns")
        print(f"    Avg L/S spread:     {avg_ls:>+6.2f}%  "
              f"({'✓' if avg_ls > 0.3 else '✗'}  target > +0.30%)")
        print(f"    % positive spreads: {pct_pos:>6.1%}  "
              f"({'✓' if pct_pos > 0.55 else '✗'}  target > 55%)")

    # Test 2
    if not t2.empty:
        avg_ic   = t2["ic"].mean()
        avg_icir = t2["icir"].mean()
        pct_pos  = (t2["ic"] > 0).mean()
        print(f"\n  Test 2 — Information Coefficient")
        print(f"    Average IC:         {avg_ic:>+7.4f}  "
              f"({'✓' if avg_ic > 0.04 else '✗'}  target > 0.04)")
        print(f"    Average ICIR:       {avg_icir:>+7.4f}  "
              f"({'✓' if avg_icir > 0.3 else '✗'}  target > 0.30)")
        print(f"    % positive IC:      {pct_pos:>6.1%}  "
              f"({'✓' if pct_pos > 0.55 else '✗'}  target > 55%)")

    # Test 3
    if not t3.empty:
        avg_hr = t3["hit_rate"].mean()
        print(f"\n  Test 3 — Transition Warning Accuracy")
        print(f"    Avg hit rate:       {avg_hr:>6.1%}  "
              f"({'✓' if avg_hr > 0.50 else '✗'}  target > 50%)")

    # Test 4
    if not t4.empty:
        avg_delta = t4["delta"].mean()
        pct_better = (t4["delta"] > 0).mean()
        print(f"\n  Test 4 — Specvel vs Naive Momentum")
        print(f"    Avg delta vs naive: {avg_delta:>+6.2f}%  "
              f"({'✓' if avg_delta > 0 else '✗'}  target > 0%)")
        print(f"    % better than naive:{pct_better:>6.1%}  "
              f"({'✓' if pct_better > 0.55 else '✗'}  target > 55%)")

    # Test 5
    if not t5.empty:
        avg_pos    = t5["n_positive"].mean()
        pct_stable = (t5["n_positive"] >= 4).mean()
        print(f"\n  Test 5 — Stability Across Regimes")
        print(f"    Avg positive periods:{avg_pos:>5.1f}/5  "
              f"({'✓' if avg_pos >= 3.5 else '✗'}  target ≥ 3.5/5)")
        print(f"    % tickers stable:   {pct_stable:>6.1%}  "
              f"({'✓' if pct_stable > 0.55 else '✗'}  target > 55%)")

    # Overall verdict
    passes = []
    if not t1.empty:
        ls_cols = [c for c in t1.columns if c.startswith("ls_")]
        passes.append(t1[ls_cols].mean().mean() > 0.3 if ls_cols else False)
    if not t2.empty:
        passes.append(t2["ic"].mean() > 0.04)
    if not t3.empty:
        passes.append(t3["hit_rate"].mean() > 0.50)
    if not t4.empty:
        passes.append(t4["delta"].mean() > 0)
    if not t5.empty:
        passes.append(t5["n_positive"].mean() >= 3.5)

    n_pass = sum(passes)
    verdict = ("🟢 SIGNAL VALIDATED"     if n_pass >= 4 else
               "🟡 SIGNAL PROMISING"     if n_pass >= 3 else
               "🔴 SIGNAL NEEDS WORK"    if n_pass >= 2 else
               "⚫ SIGNAL NOT CONFIRMED")

    print(f"\n  {'─'*50}")
    print(f"  Tests passed: {n_pass}/{len(passes)}")
    print(f"  Verdict:      {verdict}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests(
    fred_key:       str  = None,
    include_macro:  bool = False,
    save_results:   str  = None,   # path to save CSV summary e.g. 'results/'
):
    """
    Run all 5 validation tests on equities and commodities.
    Optionally include fixed income and macro if fred_key is provided.

    Parameters
    ----------
    fred_key      : FRED API key for fixed income / macro tests
    include_macro : also test fixed income and macro adapters
    save_results  : directory path to save CSV outputs
    """
    from adapters.equities    import EquitiesAdapter
    from adapters.commodities import CommoditiesAdapter

    START = "2015-01-01"
    END   = "2026-03-10"

    adapters_to_test = [
        (EquitiesAdapter(),    EQUITY_TICKERS,    "business", "EQUITIES"),
        (CommoditiesAdapter(), COMMODITY_TICKERS, "auto",     "COMMODITIES"),
    ]

    if include_macro and fred_key:
        from adapters.fixed_income import FixedIncomeAdapter
        from adapters.macro        import MacroAdapter

        FI_TICKERS  = {"DGS2": "2Y Treasury", "DGS10": "10Y Treasury",
                       "T10Y2Y": "10Y-2Y Spread", "BAMLH0A0HYM2": "HY Spread"}
        MAC_TICKERS = {"CPIAUCSL": "CPI", "CPILFESL": "Core CPI",
                       "UNRATE": "Unemployment", "INDPRO": "Ind Production"}

        adapters_to_test += [
            (FixedIncomeAdapter(api_key=fred_key), FI_TICKERS,  "business", "FIXED INCOME"),
            (MacroAdapter(api_key=fred_key),       MAC_TICKERS, "calendar", "MACRO"),
        ]

    all_t1, all_t2, all_t3, all_t4, all_t5 = [], [], [], [], []

    for adapter, tickers, cycle_method, name in adapters_to_test:
        print(f"\n\n{'#'*65}")
        print(f"#  {name}")
        print(f"{'#'*65}")

        t1 = test1_signal_returns(adapter, tickers, START, END, cycle_method)
        t2 = test2_information_coefficient(adapter, tickers, START, END, cycle_method)
        t3 = test3_transition_accuracy(adapter, tickers, START, END, cycle_method)
        t4 = test4_vs_naive(adapter, tickers, START, END, cycle_method)
        t5 = test5_stability(adapter, tickers, cycle_method)

        for df, lst in [(t1,all_t1),(t2,all_t2),(t3,all_t3),(t4,all_t4),(t5,all_t5)]:
            if not df.empty:
                df["adapter"] = name
                lst.append(df)

    # Combine across adapters
    T1 = pd.concat(all_t1, ignore_index=True) if all_t1 else pd.DataFrame()
    T2 = pd.concat(all_t2, ignore_index=True) if all_t2 else pd.DataFrame()
    T3 = pd.concat(all_t3, ignore_index=True) if all_t3 else pd.DataFrame()
    T4 = pd.concat(all_t4, ignore_index=True) if all_t4 else pd.DataFrame()
    T5 = pd.concat(all_t5, ignore_index=True) if all_t5 else pd.DataFrame()

    print_scorecard(T1, T2, T3, T4, T5)

    # Save results
    if save_results:
        os.makedirs(save_results, exist_ok=True)
        for name, df in [("t1_signal_returns",T1),("t2_ic",T2),
                         ("t3_transitions",T3),("t4_vs_naive",T4),
                         ("t5_stability",T5)]:
            if not df.empty:
                path = os.path.join(save_results, f"backtest_{name}.csv")
                df.to_csv(path, index=False)
                print(f"Saved {path}")

    return T1, T2, T3, T4, T5


# ─────────────────────────────────────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Specvel backtest validation")
    parser.add_argument("--fred_key",      type=str, default=None,
                        help="FRED API key (optional, enables macro/FI tests)")
    parser.add_argument("--include_macro", action="store_true",
                        help="Include fixed income and macro adapter tests")
    parser.add_argument("--save",          type=str, default="results/",
                        help="Directory to save CSV results (default: results/)")
    args = parser.parse_args()

    run_all_tests(
        fred_key=args.fred_key,
        include_macro=args.include_macro,
        save_results=args.save,
    )
