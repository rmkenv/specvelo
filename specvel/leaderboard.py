"""
specvel/leaderboard.py

Scans all series across one or more adapters, computes velocity
features + signals, and returns a ranked DataFrame.

Usage:
    from specvel.adapters.equities import EquitiesAdapter
    from specvel.leaderboard import run_leaderboard, print_leaderboard

    adapter = EquitiesAdapter()
    df = run_leaderboard(adapter, "2023-01-01", "2026-03-10",
                         asset_class="equities", top_n=20)
    print_leaderboard(df)
"""
import sys
import pandas as pd
from features import build_features
from signals  import compute_signal
from anomaly  import detect_anomaly


def run_leaderboard(
    adapter,
    start:       str,
    end:         str,
    asset_class: str = "default",
    top_n:       int = 20,
    lookback:    int = 20,
    smooth:      bool = True,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Fetch all series from adapter, compute velocity + signals,
    return ranked DataFrame sorted by absolute conviction score.

    Parameters
    ----------
    adapter     : any adapter with .fetch(), .list_series(), .normalize()
    start       : 'YYYY-MM-DD' — start of history window
    end         : 'YYYY-MM-DD' — end of history window
    asset_class : threshold profile ('equities', 'commodities',
                  'fixed_income', 'macro', 'default')
    top_n       : return top N by absolute conviction
    lookback    : lookback window for velocity stats
    smooth      : apply SG smoothing in velocity computation
    verbose     : print progress

    Returns
    -------
    pd.DataFrame — one row per series, sorted by |conviction|
    """
    series_list = adapter.list_series()
    if verbose:
        print(f"\nScanning {len(series_list)} series from "
              f"[{adapter.source_name}] ({start} → {end})...")

    rows = []
    errors = []

    for i, series_id in enumerate(series_list):
        if verbose:
            pct = int((i + 1) / len(series_list) * 100)
            sys.stdout.write(f"\r  {pct:3d}%  {series_id:<25}")
            sys.stdout.flush()

        try:
            raw    = adapter.fetch(series_id, start, end)
            if raw.dropna().empty or len(raw.dropna()) < 10:
                continue

            normed = adapter.normalize(raw)
            label  = adapter.label(series_id) if hasattr(adapter, "label") else series_id
            feats  = build_features(normed, adapter.source_name, series_id,
                                    label=label, lookback=lookback, smooth=smooth)
            if not feats:
                continue

            sig  = compute_signal(feats, asset_class)
            anom = detect_anomaly(normed, smooth=smooth)

            rows.append({
                **feats,
                "signal":         sig["signal"],
                "conviction":     sig["conviction"],
                "vel_zscore":     sig["vel_zscore"],
                "momentum":       sig["momentum"],
                "reason":         sig["reason"],
                "anomaly_score":  anom["score"],
                "anomaly_flag":   anom["flag"],
                "anomaly_sev":    anom["severity"],
                "n_changepoints": anom["n_changepoints"],
            })
        except Exception as e:
            errors.append((series_id, str(e)))
            continue

    if verbose:
        print(f"\r  Done. {len(rows)} series processed, {len(errors)} errors.")
        if errors:
            for sid, err in errors[:5]:
                print(f"    Skipped {sid}: {err}")
            if len(errors) > 5:
                print(f"    ... and {len(errors)-5} more")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("conviction", key=abs, ascending=False)
    return df.head(top_n).reset_index(drop=True)


def run_multi_leaderboard(
    adapters:    list,
    start:       str,
    end:         str,
    asset_classes: dict = None,
    top_n:       int  = 30,
    lookback:    int  = 20,
) -> pd.DataFrame:
    """
    Run leaderboard across multiple adapters and combine results.

    Parameters
    ----------
    adapters      : list of adapter instances
    asset_classes : dict mapping adapter.source_name → asset_class string
                    e.g. {"equities": "equities", "macro": "macro"}
    """
    frames = []
    for adapter in adapters:
        ac = (asset_classes or {}).get(adapter.source_name, "default")
        try:
            df = run_leaderboard(adapter, start, end,
                                 asset_class=ac, top_n=top_n,
                                 lookback=lookback)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  Error running {adapter.source_name}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("conviction", key=abs, ascending=False)
    return combined.head(top_n).reset_index(drop=True)


def print_leaderboard(df: pd.DataFrame, title: str = "SPECVEL LEADERBOARD"):
    """Pretty-print the leaderboard to console."""
    if df.empty:
        print("No results to display.")
        return

    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    print(f"{'#':<4} {'LABEL':<24} {'SIGNAL':<18} {'CONV':>5} "
          f"{'VEL_Z':>7} {'MOM':>6} {'ANOM':>10}  AS_OF")
    print('-'*100)

    for i, row in df.iterrows():
        anom_str = f"⚠ {row.get('anomaly_sev','')[:4].upper()}" \
                   if row.get("anomaly_flag") else "  ok"
        label    = str(row.get("label", row.get("series_id", "")))[:23]
        signal   = str(row.get("signal", ""))[:17]
        as_of    = str(row.get("as_of_date", ""))[:10]

        print(
            f"{i+1:<4} {label:<24} {signal:<18} "
            f"{row.get('conviction', 0):>+5} "
            f"{row.get('vel_zscore', 0):>7.3f} "
            f"{row.get('momentum', 0):>6.3f} "
            f"{anom_str:>10}  {as_of}"
        )

    print(f"{'='*100}")
    longs  = (df["conviction"] >= 2).sum()
    shorts = (df["conviction"] <= -2).sum()
    anoms  = df["anomaly_flag"].sum() if "anomaly_flag" in df.columns else 0
    print(f"  LONG (≥+2): {longs}  |  SHORT (≤-2): {shorts}  |  "
          f"Anomalies: {anoms}")
    print(f"{'='*100}\n")


def save_leaderboard(df: pd.DataFrame, path: str):
    """Save leaderboard to CSV."""
    # Drop component dicts before saving — not CSV-friendly
    save_cols = [c for c in df.columns if c != "components"]
    df[save_cols].to_csv(path, index=False)
    print(f"Saved leaderboard to {path}")
