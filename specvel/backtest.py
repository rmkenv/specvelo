"""
specvel/backtest.py

Vectorized backtest — completely self-contained, zero calls to cycle.py.
All phase detection and z-score logic is reimplemented here using
fast numpy/pandas operations only. No Python-level rolling loops.

Tests:
  1 — Signal Return Backtest
  2 — Information Coefficient (IC + ICIR)
  3 — Phase Transition Accuracy   [skipped in fast mode]
  4 — Specvel vs Naive Momentum
  5 — Stability Across Regimes    [skipped in fast mode]

Usage:
    python specvel/backtest.py --fast          # CI ~2min
    python specvel/backtest.py                 # full ~10min
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from scipy.stats    import spearmanr
from scipy.signal   import savgol_filter

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

FORWARD_PERIODS  = [5, 10, 20, 60]
ZSCORE_THRESHOLD = 0.75
MIN_HISTORY      = 60
ROLL             = 120    # rolling baseline window


# ── Fast numpy-only building blocks ──────────────────────────────────────────

def _velocity(series: pd.Series, win: int = 7) -> pd.Series:
    """Savitzky-Golay smooth then gradient. Pure numpy."""
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


def _phase_labels_auto(vel: pd.Series) -> pd.Series:
    """
    Fast phase detection via rolling percentile rank.
    Uses vectorized rolling.rank() — no Python lambda loops.
    """
    v    = vel.dropna()
    roll = min(ROLL, len(v))
    # rank() is vectorized in pandas
    rank = v.rolling(roll, min_periods=20).rank(pct=True)
    accel = v.diff()

    phase = pd.Series("dormant", index=v.index)
    phase[rank >= 0.70]                            = "peak"
    phase[(rank < 0.70) & (accel > 0)]             = "green_up"
    phase[(rank < 0.70) & (accel <= 0)]            = "senescence"
    phase[rank <= 0.30]                            = "dormant"
    return phase.reindex(vel.index, fill_value="dormant")


def _phase_labels_business(vel: pd.Series) -> pd.Series:
    """NBER recession = senescence/dormant, expansion = green_up/peak."""
    in_rec = pd.Series(False, index=vel.index)
    for s, e in NBER_RECESSIONS:
        in_rec[(vel.index >= s) & (vel.index <= e)] = True

    v     = vel.dropna()
    roll  = min(ROLL, len(v))
    rank  = v.rolling(roll, min_periods=20).rank(pct=True)
    accel = v.diff()

    phase = pd.Series("green_up", index=v.index)
    phase[rank >= 0.65]              = "peak"
    phase[rank < 0.65]               = "green_up"
    phase[(accel <= 0) & (rank < 0.65)] = "senescence"

    in_rec_v = in_rec.reindex(v.index, fill_value=False)
    phase[in_rec_v & (rank >= 0.40)] = "senescence"
    phase[in_rec_v & (rank < 0.40)]  = "dormant"

    return phase.reindex(vel.index, fill_value="dormant")


def _phase_labels_calendar(vel: pd.Series) -> pd.Series:
    """Q1=green_up, Q2=peak, Q3=senescence, Q4=dormant."""
    q_map = {1: "green_up", 2: "peak", 3: "senescence", 4: "dormant"}
    return vel.index.to_series().dt.quarter.map(q_map).reindex(vel.index, fill_value="dormant")


def _get_phases(vel: pd.Series, cycle_method: str) -> pd.Series:
    if cycle_method == "business":
        return _phase_labels_business(vel)
    elif cycle_method == "calendar":
        return _phase_labels_calendar(vel)
    else:
        return _phase_labels_auto(vel)


def _vectorized_surprise(
    normed:       pd.Series,
    cycle_method: str,
    use_cycle:    bool = True,
) -> pd.DataFrame:
    """
    Build the full signal DataFrame in one vectorized pass.
    No per-row Python calls. O(n) not O(n²).
    """
    vel = _velocity(normed).dropna()
    if len(vel) < MIN_HISTORY + max(FORWARD_PERIODS) + 10:
        return pd.DataFrame()

    if use_cycle:
        phases = _get_phases(vel, cycle_method)

        # Phase-conditional rolling mean/std using fast groupby trick
        phase_mean = pd.Series(np.nan, index=vel.index)
        phase_std  = pd.Series(np.nan, index=vel.index)

        for ph in ["green_up", "peak", "senescence", "dormant"]:
            mask      = (phases == ph)
            ph_vel    = vel.where(mask)   # NaN when not in this phase
            pm        = ph_vel.rolling(ROLL, min_periods=10).mean().shift(1)
            ps        = ph_vel.rolling(ROLL, min_periods=10).std().shift(1)
            phase_mean = phase_mean.where(~mask, pm)
            phase_std  = phase_std.where(~mask, ps)

        # Fall back to unconditional stats for sparse phases
        fallback_mean = vel.rolling(ROLL, min_periods=MIN_HISTORY).mean().shift(1)
        fallback_std  = vel.rolling(ROLL, min_periods=MIN_HISTORY).std().shift(1)
        phase_mean    = phase_mean.fillna(fallback_mean)
        phase_std     = phase_std.fillna(fallback_std)

        zscore = (vel - phase_mean) / phase_std.clip(lower=1e-9)

        # Transition warning: rapid acceleration change near phase boundary
        accel   = vel.diff()
        warning = (
            ((phases == "peak")       & (accel < -0.01)) |
            ((phases == "dormant")    & (accel >  0.01)) |
            ((phases == "senescence") & (accel >  0.01)) |
            ((phases == "green_up")   & (accel <  0.0) & (vel.rolling(5).mean() < 0))
        )
    else:
        phases  = pd.Series("n/a", index=vel.index)
        fallback_mean = vel.rolling(ROLL, min_periods=MIN_HISTORY).mean().shift(1)
        fallback_std  = vel.rolling(ROLL, min_periods=MIN_HISTORY).std().shift(1)
        zscore  = (vel - fallback_mean) / fallback_std.clip(lower=1e-9)
        warning = pd.Series(False, index=vel.index)

    signal = pd.Series("NEUTRAL", index=vel.index)
    signal[zscore >=  ZSCORE_THRESHOLD] = "LONG"
    signal[zscore <= -ZSCORE_THRESHOLD] = "SHORT"

    df = pd.DataFrame({
        "date":            vel.index,
        "surprise_zscore": zscore.round(4),
        "phase":           phases,
        "signal":          signal,
        "has_warning":     warning,
    })

    # Drop warmup rows and last rows that have no forward data
    df = df.iloc[MIN_HISTORY: -max(FORWARD_PERIODS)].copy()
    df = df.dropna(subset=["surprise_zscore"]).reset_index(drop=True)
    return df


def _add_forward_returns(df: pd.DataFrame, raw: pd.Series) -> pd.DataFrame:
    """Attach forward returns to the signal DataFrame."""
    raw_a = raw.reindex(df["date"]).ffill()
    for fp in FORWARD_PERIODS:
        fwd = raw.reindex(df["date"]).shift(-fp) if hasattr(raw.index, 'name') else \
              raw.reindex(df["date"]).ffill()
        # align by date arithmetic
        fwd_vals = []
        dates    = df["date"].values
        raw_arr  = raw.values
        raw_idx  = raw.index
        for d in dates:
            pos = raw_idx.get_loc(d) if d in raw_idx else -1
            if pos == -1 or pos + fp >= len(raw_arr):
                fwd_vals.append(np.nan)
            else:
                p0 = raw_arr[pos]
                pf = raw_arr[pos + fp]
                fwd_vals.append((pf - p0) / p0 if p0 != 0 else np.nan)
        df[f"fwd_{fp}d"] = fwd_vals
    return df


def _build(normed, raw, cycle_method, use_cycle=True):
    df = _vectorized_surprise(normed, cycle_method, use_cycle)
    if df.empty:
        return df
    return _add_forward_returns(df, raw)


# ── Test 1 ────────────────────────────────────────────────────────────────────

def test1_signal_returns(adapter, tickers, start, end, cycle_method):
    fwd_cols = [f"fwd_{fp}d" for fp in FORWARD_PERIODS]
    rows = []
    print(f"\n{'─'*65}")
    print(f"  TEST 1 — Signal Returns  [{cycle_method}]  {start}→{end}")
    print(f"{'─'*65}")

    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df     = _build(normed, raw, cycle_method)
            if df.empty:
                print(f"  {label:<22} — no results"); continue

            print(f"\n  {label} ({ticker})  —  {len(df)} obs")
            print(f"  {'SIGNAL':<10} {'N':>5}  " + "  ".join(f"{c:>10}" for c in fwd_cols))
            print(f"  {'─'*55}")
            for sig in ["LONG", "NEUTRAL", "SHORT"]:
                sub   = df[df["signal"] == sig]
                means = [sub[c].mean()*100 if not sub.empty else np.nan for c in fwd_cols]
                print(f"  {sig:<10} {len(sub):>5}  " +
                      "  ".join(f"{m:>+9.2f}%" if not np.isnan(m) else f"{'n/a':>10}" for m in means))
            print(f"  {'─'*55}")
            for c in fwd_cols:
                ls = (df[df["signal"]=="LONG"][c].mean() - df[df["signal"]=="SHORT"][c].mean()) * 100
                print(f"  L/S {c}: {ls:>+6.2f}%  {'✓' if ls>0.3 else('✗' if ls<0 else '~')}")

            rows.append({"ticker": ticker, "label": label, "n_obs": len(df),
                **{f"ls_{c}": (df[df["signal"]=="LONG"][c].mean() -
                               df[df["signal"]=="SHORT"][c].mean())*100
                   for c in fwd_cols if c in df.columns}})
        except Exception as e:
            print(f"  {label:<22} — error: {e}")
    return pd.DataFrame(rows)


# ── Test 2 ────────────────────────────────────────────────────────────────────

def test2_ic(adapter, tickers, start, end, cycle_method, fwd_col="fwd_20d"):
    print(f"\n{'─'*65}")
    print(f"  TEST 2 — Information Coefficient [{fwd_col}]")
    print(f"{'─'*65}")
    print(f"  {'TICKER':<22} {'IC':>8} {'P-VAL':>8} {'ICIR':>8} {'IC+%':>8}  STATUS")
    print(f"  {'─'*60}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df     = _build(normed, raw, cycle_method)
            if df.empty or fwd_col not in df.columns: continue

            clean = df[["surprise_zscore", fwd_col]].dropna()
            if len(clean) < 30: continue

            ic, pval   = spearmanr(clean["surprise_zscore"], clean[fwd_col])
            ric        = clean["surprise_zscore"].rolling(52).corr(clean[fwd_col])
            icir       = ric.mean() / (ric.std() + 1e-9)
            pct_pos    = (ric > 0).mean()

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

def test3_transitions(adapter, tickers, start, end, cycle_method, lookahead=10):
    print(f"\n{'─'*65}")
    print(f"  TEST 3 — Phase Transition Accuracy  lookahead={lookahead}")
    print(f"{'─'*65}")
    print(f"  {'TICKER':<22} {'WARNINGS':>9} {'HITS':>6} {'HIT RATE':>10}  STATUS")
    print(f"  {'─'*55}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df     = _build(normed, raw, cycle_method)
            if df.empty: continue

            warned = df[df["has_warning"]].copy()
            if len(warned) < 5:
                print(f"  {label:<22} — too few warnings ({len(warned)})"); continue

            hits = 0
            for pos in warned.index:
                future = df.loc[pos: pos+lookahead, "phase"]
                if len(future) > 1 and future.iloc[0] != future.iloc[-1]:
                    hits += 1

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


# ── Test 4 ────────────────────────────────────────────────────────────────────

def test4_vs_naive(adapter, tickers, start, end, cycle_method, fwd_col="fwd_20d"):
    print(f"\n{'─'*65}")
    print(f"  TEST 4 — Specvel vs Naive Momentum [{fwd_col}]")
    print(f"{'─'*65}")
    print(f"  {'TICKER':<22} {'SPECVEL L/S':>12} {'NAIVE L/S':>10} {'DELTA':>8}  STATUS")
    print(f"  {'─'*60}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df_sv  = _build(normed, raw, cycle_method, use_cycle=True)
            df_nv  = _build(normed, raw, cycle_method, use_cycle=False)
            if df_sv.empty or df_nv.empty or fwd_col not in df_sv.columns: continue

            def ls(df):
                l = df[df["signal"]=="LONG"][fwd_col].mean()
                s = df[df["signal"]=="SHORT"][fwd_col].mean()
                return (l - s) * 100

            sv, nv = ls(df_sv), ls(df_nv)
            delta  = sv - nv
            status = ("✓ BETTER" if delta>0.2 else ("~ SIMILAR" if delta>-0.2 else "✗ WORSE"))
            print(f"  {label:<22} {sv:>+11.2f}% {nv:>+9.2f}% {delta:>+7.2f}%  {status}")
            rows.append({"ticker":ticker,"label":label,"specvel_ls":sv,"naive_ls":nv,
                         "delta":delta,"status":status})
        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Avg specvel L/S: {df_out['specvel_ls'].mean():>+.2f}%")
        print(f"  Avg naive L/S:   {df_out['naive_ls'].mean():>+.2f}%")
        print(f"  Avg delta:       {df_out['delta'].mean():>+.2f}%")
        print(f"  Specvel better:  {(df_out['delta']>0).mean():.0%}")
    return df_out


# ── Test 5 ────────────────────────────────────────────────────────────────────

def test5_stability(adapter, tickers, cycle_method, fwd_col="fwd_20d"):
    print(f"\n{'─'*65}")
    print(f"  TEST 5 — Stability Across Regimes [{fwd_col}]")
    print(f"{'─'*65}")
    print(f"  {'TICKER':<18} " + "  ".join(f"{p[2][:13]:>13}" for p in STABILITY_PERIODS))
    print(f"  {'─'*90}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, "2014-01-01", "2026-03-10")
            normed = adapter.normalize(raw)
            df     = _build(normed, raw, cycle_method)   # compute once
            if df.empty: continue
            df["date"] = pd.to_datetime(df["date"])

            period_results = {}
            for s, e, pl in STABILITY_PERIODS:
                sub = df[(df["date"] >= s) & (df["date"] <= e)]
                if len(sub) < 30 or fwd_col not in sub.columns:
                    period_results[pl] = np.nan; continue
                ls = (sub[sub["signal"]=="LONG"][fwd_col].mean() -
                      sub[sub["signal"]=="SHORT"][fwd_col].mean()) * 100
                period_results[pl] = ls

            values = [period_results.get(p[2], np.nan) for p in STABILITY_PERIODS]
            n_pos  = sum(1 for v in values if not np.isnan(v) and v > 0)
            status = ("✓" if n_pos>=4 else ("~" if n_pos>=3 else "✗"))
            line   = f"  {label[:17]:<18} "
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
    print(f"\n{'='*65}\n  SPECVEL VALIDATION SCORECARD\n{'='*65}")
    passes = []

    if not t1.empty:
        lsc  = [c for c in t1.columns if c.startswith("ls_")]
        avg  = t1[lsc].mean().mean() if lsc else np.nan
        ppos = (t1[lsc]>0).mean().mean() if lsc else np.nan
        print(f"\n  Test 1 — Signal Returns")
        print(f"    Avg L/S spread:     {avg:>+6.2f}%  {'✓' if avg>0.3 else '✗'}  (>+0.30%)")
        print(f"    % positive spreads: {ppos:>6.1%}  {'✓' if ppos>0.55 else '✗'}  (>55%)")
        passes.append(not np.isnan(avg) and avg > 0.3)

    if not t2.empty:
        ic   = t2["ic"].mean()
        icir = t2["icir"].mean()
        pp   = (t2["ic"]>0).mean()
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
        d  = t4["delta"].mean()
        pb = (t4["delta"]>0).mean()
        print(f"\n  Test 4 — Specvel vs Naive")
        print(f"    Avg delta:       {d:>+6.2f}%  {'✓' if d>0 else '✗'}  (>0%)")
        print(f"    % better naive:  {pb:>6.1%}  {'✓' if pb>0.55 else '✗'}  (>55%)")
        passes.append(d > 0)

    if not t5.empty:
        ap = t5["n_positive"].mean()
        ps = (t5["n_positive"]>=4).mean()
        print(f"\n  Test 5 — Stability")
        print(f"    Avg pos periods: {ap:>5.1f}/5  {'✓' if ap>=3.5 else '✗'}  (≥3.5)")
        print(f"    % stable:        {ps:>6.1%}  {'✓' if ps>0.55 else '✗'}  (>55%)")
        passes.append(ap >= 3.5)

    n  = sum(passes)
    v  = ("🟢 SIGNAL VALIDATED"  if n>=4 else
          "🟡 SIGNAL PROMISING"  if n>=3 else
          "🔴 SIGNAL NEEDS WORK" if n>=2 else
          "⚫ NOT CONFIRMED")
    print(f"\n  {'─'*50}")
    print(f"  Tests passed: {n}/{len(passes)}")
    print(f"  Verdict:      {v}")
    print(f"{'='*65}\n")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tests(fred_key=None, include_macro=False, save_results=None, fast=False):
    from adapters.equities    import EquitiesAdapter
    from adapters.commodities import CommoditiesAdapter

    if fast:
        START, END   = "2020-01-01", "2026-03-10"
        eq_t, cm_t   = EQUITY_TICKERS_FAST, COMMODITY_TICKERS_FAST
        print("FAST MODE: 2020→now, 5 tickers, Tests 1+2+4 only (~2 min)")
    else:
        START, END   = "2015-01-01", "2026-03-10"
        eq_t, cm_t   = EQUITY_TICKERS, COMMODITY_TICKERS

    adapters = [
        (EquitiesAdapter(),    eq_t, "business", "EQUITIES"),
        (CommoditiesAdapter(), cm_t, "auto",     "COMMODITIES"),
    ]

    if include_macro and fred_key:
        from adapters.fixed_income import FixedIncomeAdapter
        from adapters.macro        import MacroAdapter
        fi = ({"DGS2":"2Y Tsy","DGS10":"10Y Tsy"} if fast else
              {"DGS2":"2Y Tsy","DGS10":"10Y Tsy","T10Y2Y":"Spread","BAMLH0A0HYM2":"HY"})
        mc = ({"CPIAUCSL":"CPI","UNRATE":"Unemployment"} if fast else
              {"CPIAUCSL":"CPI","CPILFESL":"Core CPI","UNRATE":"Unemp","INDPRO":"IndProd"})
        adapters += [
            (FixedIncomeAdapter(api_key=fred_key), fi, "business", "FIXED INCOME"),
            (MacroAdapter(api_key=fred_key),       mc, "calendar", "MACRO"),
        ]

    a1,a2,a3,a4,a5 = [],[],[],[],[]

    for adapter, tickers, cm, name in adapters:
        print(f"\n\n{'#'*65}\n#  {name}\n{'#'*65}")
        t1 = test1_signal_returns(adapter, tickers, START, END, cm)
        t2 = test2_ic(adapter, tickers, START, END, cm)
        t3 = test3_transitions(adapter, tickers, START, END, cm) if not fast else pd.DataFrame()
        t4 = test4_vs_naive(adapter, tickers, START, END, cm)
        t5 = test5_stability(adapter, tickers, cm) if not fast else pd.DataFrame()

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
                        ("t3_transitions",T3),("t4_vs_naive",T4),
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
