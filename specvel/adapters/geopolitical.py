"""
specvel/adapters/geopolitical.py

Shim — imports from specvel.geopolitical (the canonical module).
Kept for backward compatibility with any code importing from adapters/.
"""
try:
    from geopolitical import (
        GeopoliticalRegimeFilter,
        build_mena_stress_index,
        get_current_regime,
        apply_geo_filter_to_backtest,
        REGIME_MULTIPLIERS,
        TICKER_CLASS,
    )
except ImportError:
    from specvel.geopolitical import (
        GeopoliticalRegimeFilter,
        build_mena_stress_index,
        get_current_regime,
        apply_geo_filter_to_backtest,
        REGIME_MULTIPLIERS,
        TICKER_CLASS,
    )

__all__ = [
    "GeopoliticalRegimeFilter",
    "build_mena_stress_index",
    "get_current_regime",
    "apply_geo_filter_to_backtest",
    "REGIME_MULTIPLIERS",
    "TICKER_CLASS",
]
