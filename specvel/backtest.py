"""
specvel/backtest.py

Validates the spectral velocity cycle signal across 5 tests:

  Test 1 — Signal Return Backtest
  Test 2 — Information Coefficient (IC)
  Test 3 — Phase Transition Accuracy
  Test 4 — Specvel vs Naive Momentum
  Test 5 — Stability Across Time Periods

Usage (Colab / local):
    from backtest import run_all_tests
    run_all_tests()                        # full run
    run_all_tests(fast=True)               # fast CI mode (~5 min)

Usage (CLI):
    python specvel/backtest.py                          # full
    python specvel/backtest.py --fast                   # fast
    python specvel/backtest.py --fast --fred_key KEY --include_macro
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core  import compute_velocity
from cycle import compute_velocity_surprise


# ── Configuration ─────────────────────────────────────────────────────────────

EQUITY_TICKERS = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "XLE": "Energy",  "XLF": "Financials",  "XLK": "Technology",
}
COMMODITY_TICKERS = {
    "CL=F": "WTI Crude", "GC=F": "Gold", "SI=F": "Silver",
    "ZC=F": "Corn",      "ZW=F": "Wheat",
}

# Fast mode — 3 tickers, 3yr window, skip Tests 3+5
EQUITY_TICKERS_FAST    = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "XLE": "Energy"}
COMMODITY_TICKERS_FAST = {"CL=F": "WTI Crude", "GC=F": "Gold"}

STABILITY_PERIODS = [
    ("2015-01-01", "2018-01-01", "Pre-2018 Bull"),
    ("2018-01-01", "2020-03-01", "Late Cycle"),
    ("2020-03-01", "2022-01-01", "COVID Recovery"),
    ("2022-01-01", "2024-01-01", "Rate Hike Cycle"),
    ("2024-01-01", "2026-03-10", "Recent"),
]

FORWARD_PERIODS  = [5, 10, 20, 60]
ZSCORE_THRESHOLD = 0.75
MIN_HISTORY      = 60
MIN_PHASE_SAMPLES = 5


# ── Core walk-forward engine ───────────────────────────────────────────────────

def _walk_forward(normed, raw, cycle_method, use_cycle=True):
    results = []
    max_fp  = max(FORWARD_PERIODS)

    for i in range(MIN_HISTORY, len(normed) - max_fp - 1):
        hist       = normed.iloc[:i]
        raw_future = raw.iloc[i: i + max_fp + 1]

        if raw_future.empty or len(raw_future) < 2:
            continue
        p0 = float(raw_future.iloc[0])
        if p0 == 0:
            continue

        if use_cycle:
            surp = compute_velocity_surprise(hist, cycle_method=cycle_method)
            if not surp or surp.get("phase_n_samples", 0) < MIN_PHASE_SAMPLES:
                continue
            zscore      = surp["surprise_zscore"]
            phase       = surp["current_phase"]
            has_warning = bool(surp.get("transition_warning", ""))
            boost       = surp.get("conviction_boost", 0)
        else:
            vel = compute_velocity(hist).dropna()
            if len(vel) < 20:
                continue
            cur_vel     = float(vel.iloc[-1])
            zscore      = (cur_vel - vel.mean()) / (vel.std() + 1e-9)
            phase       = "n/a"
            has_warning = False
            boost       = 0

        signal = ("LONG"  if zscore >=  ZSCORE_THRESHOLD else
                  "SHORT" if zscore <= -ZSCORE_THRESHOLD else "NEUTRAL")

        row = {
            "date": hist.index[-1],
            "surprise_zscore": round(zscore, 4),
            "phase": phase,
            "signal": signal,
            "conviction_boost": boost,
            "has_warning": has_warning,
        }
        for fp in FORWARD_PERIODS:
            if fp < len(raw_future):
                row[f"fwd_{fp}d"] = (float(raw_future.iloc[fp]) - p0) / p0
            else:
                row[f"fwd_{fp}d"] = np.nan

        results.append(row)

    return pd.DataFrame(results)


# ── Test 1 — Signal Return Backtest ───────────────────────────────────────────

def test1_signal_returns(adapter, tickers, start, end, cycle_method):
    fwd_cols = [f"fwd_{fp}d" for fp in FORWARD_PERIODS]
    all_rows = []

    print(f"\n{'─'*65}")
    print(f"  TEST 1 — Signal Return Backtest  [{cycle_method}]  {start}→{end}")
    print(f"{'─'*65}")

    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            if len(normed.dropna()) < MIN_HISTORY + max(FORWARD_PERIODS) + 10:
                print(f"  {label:<22} — insufficient data")
                continue
            df = _walk_forward(normed, raw, cycle_method)
            if df.empty:
                continue

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
                print(f"  L/S {c}: {ls:>+6.2f}%  {'✓' if ls > 0.3 else ('✗' if ls < 0 else '~')}")

            all_rows.append({
                "ticker": ticker, "label": label, "n_obs": len(df),
                **{f"ls_{c}": (df[df["signal"]=="LONG"][c].mean() -
                               df[df["signal"]=="SHORT"][c].mean()) * 100
                   for c in fwd_cols if c in df.columns},
            })
        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    return pd.DataFrame(all_rows)


# ── Test 2 — Information Coefficient ──────────────────────────────────────────

def test2_information_coefficient(adapter, tickers, start, end, cycle_method, fwd_col="fwd_20d"):
    print(f"\n{'─'*65}")
    print(f"  TEST 2 — Information Coefficient [{fwd_col}]  [{cycle_method}]")
    print(f"{'─'*65}")
    print(f"  {'TICKER':<22} {'IC':>8} {'P-VAL':>8} {'ICIR':>8} {'IC+%':>8}  STATUS")
    print(f"  {'─'*60}")

    rows = []
    for ticker, label in tickers.items():
        try:
            raw    = adapter.fetch(ticker, start, end)
            normed = adapter.normalize(raw)
            df     = _walk_forward(normed, raw, cycle_method)
            if df.empty or fwd_col not in df.columns:
                continue
            clean = df[["surprise_zscore", fwd_col]].dropna()
            if len(clean) < 30:
                continue

            ic, pval   = spearmanr(clean["surprise_zscore"], clean[fwd_col])
            rolling_ic = clean["surprise_zscore"].rolling(52).corr(clean[fwd_col])
            ic_mean    = rolling_ic.mean()
            ic_std     = rolling_ic.std()
            icir       = ic_mean / (ic_std + 1e-9)
            pct_pos    = (rolling_ic > 0).mean()

            status = ("✓ STRONG" if ic > 0.08 and pval < 0.05 else
                      "✓ USEFUL" if ic > 0.04 and pval < 0.10 else
                      "~ WEAK"   if ic > 0 else "✗ INVERSE")

            print(f"  {label:<22} {ic:>+8.4f} {pval:>8.4f} {icir:>+8.4f} {pct_pos:>7.1%}  {status}")
            rows.append({"ticker": ticker, "label": label, "ic": ic,
                         "p_value": pval, "ic_mean": ic_mean, "icir": icir,
                         "pct_positive": pct_pos, "n": len(clean), "status": status})
        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Average IC:    {df_out['ic'].mean():>+.4f}")
        print(f"  Average ICIR:  {df_out['icir'].mean():>+.4f}")
        print(f"  % Positive IC: {(df_out['ic'] > 0).mean():.0%}")
    return df_out


# ── Test 3 — Phase Transition Accuracy ────────────────────────────────────────

def test3_transition_accuracy(adapter, tickers, start, end, cycle_method, lookahead=10):
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
            df     = _walk_forward(normed, raw, cycle_method)
            if df.empty:
                continue

            warned = df[df["has_warning"]].copy().reset_index(drop=True)
            if len(warned) < 5:
                print(f"  {label:<22} — too few warnings ({len(warned)}), skipping")
                continue

            hits = 0
            for i, row in warned.iterrows():
                pos = df[df["date"] == row["date"]].index
                if pos.empty:
                    continue
                pos    = pos[0]
                future = df.loc[pos: pos + lookahead, "phase"]
                if len(future) > 1 and future.iloc[0] != future.iloc[-1]:
                    hits += 1

            hit_rate = hits / len(warned)
            status   = ("✓ STRONG" if hit_rate > 0.65 else
                        "✓ USEFUL" if hit_rate > 0.50 else "✗ BELOW RANDOM")

            print(f"  {label:<22} {len(warned):>9} {hits:>6} {hit_rate:>9.1%}   {status}")
            rows.append({"ticker": ticker, "label": label, "n_warnings": len(warned),
                         "hits": hits, "hit_rate": hit_rate, "status": status})
        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Average hit rate: {df_out['hit_rate'].mean():.1%}")
    return df_out


# ── Test 4 — Specvel vs Naive Momentum ────────────────────────────────────────

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

            df_sv = _walk_forward(normed, raw, cycle_method, use_cycle=True)
            df_nv = _walk_forward(normed, raw, cycle_method, use_cycle=False)

            if df_sv.empty or df_nv.empty or fwd_col not in df_sv.columns:
                continue

            def ls(df):
                l = df[df["signal"] == "LONG"][fwd_col].mean()
                s = df[df["signal"] == "SHORT"][fwd_col].mean()
                return (l - s) * 100

            sv_ls  = ls(df_sv)
            nv_ls  = ls(df_nv)
            delta  = sv_ls - nv_ls
            status = ("✓ BETTER" if delta > 0.2 else ("~ SIMILAR" if delta > -0.2 else "✗ WORSE"))

            print(f"  {label:<22} {sv_ls:>+11.2f}% {nv_ls:>+9.2f}% {delta:>+7.2f}%  {status}")
            rows.append({"ticker": ticker, "label": label,
                         "specvel_ls": sv_ls, "naive_ls": nv_ls,
                         "delta": delta, "status": status})
        except Exception as e:
            print(f"  {label:<22} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Avg specvel L/S: {df_out['specvel_ls'].mean():>+.2f}%")
        print(f"  Avg naive L/S:   {df_out['naive_ls'].mean():>+.2f}%")
        print(f"  Avg delta:       {df_out['delta'].mean():>+.2f}%")
        print(f"  Specvel better:  {(df_out['delta'] > 0).mean():.0%} of tickers")
    return df_out


# ── Test 5 — Stability Across Regimes ─────────────────────────────────────────

def test5_stability(adapter, tickers, cycle_method, fwd_col="fwd_20d"):
    print(f"\n{'─'*65}")
    print(f"  TEST 5 — Stability Across Regimes [{fwd_col}]")
    print(f"{'─'*65}")

    period_labels = [p[2][:14] for p in STABILITY_PERIODS]
    print(f"  {'TICKER':<18} " + "  ".join(f"{l:>13}" for l in period_labels))
    print(f"  {'─'*90}")

    rows = []
    for ticker, label in tickers.items():
        period_results = {}
        try:
            raw    = adapter.fetch(ticker, "2014-01-01", "2026-03-10")
            normed = adapter.normalize(raw)

            for start, end, plabel in STABILITY_PERIODS:
                try:
                    ns = normed[start:end]
                    rs = raw[start:end]
                    if len(ns.dropna()) < MIN_HISTORY + max(FORWARD_PERIODS) + 5:
                        period_results[plabel] = np.nan
                        continue
                    df = _walk_forward(ns, rs, cycle_method)
                    if df.empty or fwd_col not in df.columns:
                        period_results[plabel] = np.nan
                        continue
                    ls = (df[df["signal"]=="LONG"][fwd_col].mean() -
                          df[df["signal"]=="SHORT"][fwd_col].mean()) * 100
                    period_results[plabel] = ls
                except Exception:
                    period_results[plabel] = np.nan

            values = [period_results.get(p[2], np.nan) for p in STABILITY_PERIODS]
            n_pos  = sum(1 for v in values if not np.isnan(v) and v > 0)
            status = ("✓" if n_pos >= 4 else ("~" if n_pos >= 3 else "✗"))

            line = f"  {label[:17]:<18} "
            for v in values:
                line += (f"{'n/a':>13}  " if np.isnan(v) else f"{v:>+11.2f}%  ")
            print(line + f" {status} ({n_pos}/5)")

            row = {"ticker": ticker, "label": label, "n_positive": n_pos}
            row.update({p[2]: period_results.get(p[2]) for p in STABILITY_PERIODS})
            rows.append(row)
        except Exception as e:
            print(f"  {label:<18} — error: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(f"\n  Avg positive periods:  {df_out['n_positive'].mean():.1f}/5")
        print(f"  % tickers ≥4 positive: {(df_out['n_positive'] >= 4).mean():.0%}")
    return df_out


# ── Scorecard ─────────────────────────────────────────────────────────────────

def print_scorecard(t1, t2, t3, t4, t5):
    print(f"\n{'='*65}")
    print(f"  SPECVEL VALIDATION SCORECARD")
    print(f"{'='*65}")

    passes = []

    if not t1.empty:
        ls_cols = [c for c in t1.columns if c.startswith("ls_")]
        avg_ls  = t1[ls_cols].mean().mean() if ls_cols else np.nan
        pct_pos = (t1[ls_cols] > 0).mean().mean() if ls_cols else np.nan
        print(f"\n  Test 1 — Signal Returns")
        print(f"    Avg L/S spread:     {avg_ls:>+6.2f}%  {'✓' if avg_ls > 0.3 else '✗'}  (target > +0.30%)")
        print(f"    % positive spreads: {pct_pos:>6.1%}  {'✓' if pct_pos > 0.55 else '✗'}  (target > 55%)")
        passes.append(avg_ls > 0.3)

    if not t2.empty:
        avg_ic   = t2["ic"].mean()
        avg_icir = t2["icir"].mean()
        pct_pos  = (t2["ic"] > 0).mean()
        print(f"\n  Test 2 — Information Coefficient")
        print(f"    Average IC:         {avg_ic:>+7.4f}  {'✓' if avg_ic > 0.04 else '✗'}  (target > 0.04)")
        print(f"    Average ICIR:       {avg_icir:>+7.4f}  {'✓' if avg_icir > 0.3 else '✗'}  (target > 0.30)")
        print(f"    % positive IC:      {pct_pos:>6.1%}  {'✓' if pct_pos > 0.55 else '✗'}  (target > 55%)")
        passes.append(avg_ic > 0.04)

    if not t3.empty:
        avg_hr = t3["hit_rate"].mean()
        print(f"\n  Test 3 — Transition Warning Accuracy")
        print(f"    Avg hit rate:       {avg_hr:>6.1%}  {'✓' if avg_hr > 0.50 else '✗'}  (target > 50%)")
        passes.append(avg_hr > 0.50)

    if not t4.empty:
        avg_delta  = t4["delta"].mean()
        pct_better = (t4["delta"] > 0).mean()
        print(f"\n  Test 4 — Specvel vs Naive Momentum")
        print(f"    Avg delta vs naive: {avg_delta:>+6.2f}%  {'✓' if avg_delta > 0 else '✗'}  (target > 0%)")
        print(f"    % better than naive:{pct_better:>6.1%}  {'✓' if pct_better > 0.55 else '✗'}  (target > 55%)")
        passes.append(avg_delta > 0)

    if not t5.empty:
        avg_pos    = t5["n_positive"].mean()
        pct_stable = (t5["n_positive"] >= 4).mean()
        print(f"\n  Test 5 — Stability Across Regimes")
        print(f"    Avg positive periods:{avg_pos:>5.1f}/5  {'✓' if avg_pos >= 3.5 else '✗'}  (target ≥ 3.5)")
        print(f"    % tickers stable:   {pct_stable:>6.1%}  {'✓' if pct_stable > 0.55 else '✗'}  (target > 55%)")
        passes.append(avg_pos >= 3.5)

    n_pass  = sum(passes)
    verdict = ("🟢 SIGNAL VALIDATED"  if n_pass >= 4 else
               "🟡 SIGNAL PROMISING"  if n_pass >= 3 else
               "🔴 SIGNAL NEEDS WORK" if n_pass >= 2 else
               "⚫ NOT CONFIRMED")

    print(f"\n  {'─'*50}")
    print(f"  Tests passed: {n_pass}/{len(passes)}")
    print(f"  Verdict:      {verdict}")
    print(f"{'='*65}\n")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_tests(
    fred_key:      str  = None,
    include_macro: bool = False,
    save_results:  str  = None,
    fast:          bool = False,
):
    """
    Run all validation tests.

    Parameters
    ----------
    fred_key      : FRED API key (enables fixed income + macro tests)
    include_macro : include fixed income and macro adapters
    save_results  : directory to save CSV results
    fast          : CI mode — fewer tickers, 3yr history, skip Tests 3+5
    """
    from adapters.equities    import EquitiesAdapter
    from adapters.commodities import CommoditiesAdapter

    if fast:
        START      = "2022-01-01"
        END        = "2026-03-10"
        eq_tickers = EQUITY_TICKERS_FAST
        cm_tickers = COMMODITY_TICKERS_FAST
        print("FAST MODE: 3yr history, reduced tickers, Tests 1+2+4 only")
    else:
        START      = "2015-01-01"
        END        = "2026-03-10"
        eq_tickers = EQUITY_TICKERS
        cm_tickers = COMMODITY_TICKERS

    adapters_to_test = [
        (EquitiesAdapter(),    eq_tickers, "business", "EQUITIES"),
        (CommoditiesAdapter(), cm_tickers, "auto",     "COMMODITIES"),
    ]

    if include_macro and fred_key:
        from adapters.fixed_income import FixedIncomeAdapter
        from adapters.macro        import MacroAdapter

        fi_tickers  = ({"DGS2": "2Y Treasury", "DGS10": "10Y Treasury"}
                       if fast else
                       {"DGS2": "2Y Treasury", "DGS10": "10Y Treasury",
                        "T10Y2Y": "10Y-2Y Spread", "BAMLH0A0HYM2": "HY Spread"})
        mac_tickers = ({"CPIAUCSL": "CPI", "UNRATE": "Unemployment"}
                       if fast else
                       {"CPIAUCSL": "CPI", "CPILFESL": "Core CPI",
                        "UNRATE": "Unemployment", "INDPRO": "Ind Production"})

        adapters_to_test += [
            (FixedIncomeAdapter(api_key=fred_key), fi_tickers,  "business", "FIXED INCOME"),
            (MacroAdapter(api_key=fred_key),       mac_tickers, "calendar", "MACRO"),
        ]

    all_t1, all_t2, all_t3, all_t4, all_t5 = [], [], [], [], []

    for adapter, tickers, cycle_method, name in adapters_to_test:
        print(f"\n\n{'#'*65}\n#  {name}\n{'#'*65}")

        t1 = test1_signal_returns(adapter, tickers, START, END, cycle_method)
        t2 = test2_information_coefficient(adapter, tickers, START, END, cycle_method)
        t3 = test3_transition_accuracy(adapter, tickers, START, END, cycle_method) if not fast else pd.DataFrame()
        t4 = test4_vs_naive(adapter, tickers, START, END, cycle_method)
        t5 = test5_stability(adapter, tickers, cycle_method) if not fast else pd.DataFrame()

        for df, lst in [(t1,all_t1),(t2,all_t2),(t3,all_t3),(t4,all_t4),(t5,all_t5)]:
            if not df.empty:
                df["adapter"] = name
                lst.append(df)

    T1 = pd.concat(all_t1, ignore_index=True) if all_t1 else pd.DataFrame()
    T2 = pd.concat(all_t2, ignore_index=True) if all_t2 else pd.DataFrame()
    T3 = pd.concat(all_t3, ignore_index=True) if all_t3 else pd.DataFrame()
    T4 = pd.concat(all_t4, ignore_index=True) if all_t4 else pd.DataFrame()
    T5 = pd.concat(all_t5, ignore_index=True) if all_t5 else pd.DataFrame()

    print_scorecard(T1, T2, T3, T4, T5)

    if save_results:
        os.makedirs(save_results, exist_ok=True)
        for label, df in [("t1_signal_returns",T1),("t2_ic",T2),
                          ("t3_transitions",T3),("t4_vs_naive",T4),
                          ("t5_stability",T5)]:
            if not df.empty:
                path = os.path.join(save_results, f"backtest_{label}.csv")
                df.to_csv(path, index=False)
                print(f"Saved {path}")

    return T1, T2, T3, T4, T5


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Specvel backtest validation")
    parser.add_argument("--fred_key",      type=str,  default=None)
    parser.add_argument("--include_macro", action="store_true")
    parser.add_argument("--save",          type=str,  default="results/")
    parser.add_argument("--fast",          action="store_true",
                        help="CI mode: fewer tickers, 3yr history, Tests 1+2+4 only")
    args = parser.parse_args()

    run_all_tests(
        fred_key=args.fred_key,
        include_macro=args.include_macro,
        save_results=args.save,
        fast=args.fast,
    )
