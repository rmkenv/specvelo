# backtest.py — add this to the repo

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/content/specvelo/specvel')

from adapters.equities import EquitiesAdapter
from cycle import compute_velocity_surprise
from core import compute_velocity

def backtest_surprise_signal(
    adapter,
    series_id:    str,
    start:        str,
    end:          str,
    cycle_method: str   = 'auto',
    forward_periods: list = [5, 10, 20, 60],   # trading days ahead
    zscore_threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Walk forward through history. At each point, record:
      - surprise_zscore at time T
      - actual return at T+5, T+10, T+20, T+60
    Then split by signal (LONG/SHORT/NEUTRAL) and compare returns.
    """
    raw    = adapter.fetch(series_id, start, end)
    normed = adapter.normalize(raw)

    results = []
    min_history = 60  # need enough history to build baseline

    for i in range(min_history, len(normed) - max(forward_periods)):
        # Only use data up to point i — no lookahead
        hist    = normed.iloc[:i]
        future  = normed.iloc[i:i + max(forward_periods) + 1]

        surp = compute_velocity_surprise(hist, cycle_method=cycle_method)
        if not surp or surp['phase_n_samples'] < 5:
            continue

        sz    = surp['surprise_zscore']
        phase = surp['current_phase']
        date  = hist.index[-1]

        # Forward returns on RAW (unnormalized) price
        raw_future = raw.reindex(future.index).dropna()
        if raw_future.empty:
            continue
        p0 = float(raw_future.iloc[0])

        row = {
            'date':           date,
            'surprise_zscore': sz,
            'phase':          phase,
            'signal':         'LONG' if sz >= zscore_threshold
                              else ('SHORT' if sz <= -zscore_threshold
                              else 'NEUTRAL'),
        }

        for fp in forward_periods:
            if fp < len(raw_future):
                fwd_ret = (float(raw_future.iloc[fp]) - p0) / p0
                row[f'fwd_{fp}d'] = fwd_ret

        results.append(row)

    return pd.DataFrame(results)


def print_backtest_summary(df: pd.DataFrame, series_id: str):
    """Show average forward returns by signal bucket."""
    fwd_cols = [c for c in df.columns if c.startswith('fwd_')]

    print(f"\n{'='*65}")
    print(f"  BACKTEST RESULTS — {series_id}")
    print(f"  {len(df)} observations  |  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    print(f"{'='*65}")
    print(f"{'SIGNAL':<10} {'N':>5}  " +
          "  ".join(f"{c:>10}" for c in fwd_cols))
    print('-'*65)

    for sig in ['LONG', 'NEUTRAL', 'SHORT']:
        sub = df[df['signal'] == sig]
        if sub.empty:
            continue
        means = [sub[c].mean() * 100 for c in fwd_cols if c in sub.columns]
        print(f"{sig:<10} {len(sub):>5}  " +
              "  ".join(f"{m:>+9.2f}%" for m in means))

    # Long - Short spread (the key number)
    print('-'*65)
    for c in fwd_cols:
        if c not in df.columns:
            continue
        l_ret = df[df['signal'] == 'LONG'][c].mean()
        s_ret = df[df['signal'] == 'SHORT'][c].mean()
        spread = (l_ret - s_ret) * 100
        print(f"  L/S spread {c}: {spread:+.2f}%")

    print(f"{'='*65}\n")


# Run it
adapter = EquitiesAdapter()
for ticker in ['SPY', 'QQQ', 'XLE', 'XLF', 'GC=F', 'CL=F']:
    try:
        df = backtest_surprise_signal(
            adapter, ticker,
            start='2015-01-01', end='2026-03-10',
            cycle_method='business',
            forward_periods=[5, 10, 20, 60]
        )
        print_backtest_summary(df, ticker)
    except Exception as e:
        print(f"{ticker}: {e}")
