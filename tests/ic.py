from scipy.stats import spearmanr

def information_coefficient(df: pd.DataFrame, fwd_col: str = 'fwd_20d') -> dict:
    """
    Spearman rank correlation between surprise_zscore and forward return.
    IC > 0.05 = useful signal
    IC > 0.10 = strong signal
    """
    clean = df[['surprise_zscore', fwd_col]].dropna()
    ic, pval = spearmanr(clean['surprise_zscore'], clean[fwd_col])

    # Rolling IC — does it hold up over time?
    df2 = clean.copy()
    df2['product'] = df2['surprise_zscore'].rank() * df2[fwd_col].rank()
    rolling_ic = df2['surprise_zscore'].rolling(52).corr(df2[fwd_col])

    return {
        'ic':       round(ic, 4),
        'p_value':  round(pval, 4),
        'ic_mean':  round(rolling_ic.mean(), 4),
        'ic_std':   round(rolling_ic.std(), 4),
        'icir':     round(rolling_ic.mean() / rolling_ic.std(), 4),  # IC / std — want > 0.5
        'pct_positive': round((rolling_ic > 0).mean(), 3),
    }

# Run it
df = backtest_surprise_signal(adapter, 'SPY', '2015-01-01', '2026-03-10')
ic = information_coefficient(df, 'fwd_20d')
print(ic)
# Good result looks like: {'ic': 0.08, 'p_value': 0.001, 'icir': 0.72, 'pct_positive': 0.68}
