"""
specvel/adapters/base.py

Base adapter — all data sources implement this interface.
The engine only ever calls .fetch(), .list_series(), and .normalize().
Adding a new data source = subclass this + implement those 2 methods.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseAdapter(ABC):

    # Human-readable name shown in leaderboard output
    source_name: str = "unknown"

    @abstractmethod
    def fetch(self, series_id: str, start: str, end: str) -> pd.Series:
        """
        Fetch a single named time series.

        Parameters
        ----------
        series_id : str   — ticker, FRED code, or any string identifier
        start     : str   — 'YYYY-MM-DD'
        end       : str   — 'YYYY-MM-DD'

        Returns
        -------
        pd.Series with DatetimeIndex, float values, named with series_id
        """
        ...

    @abstractmethod
    def list_series(self) -> list[str]:
        """
        Return all available series IDs for this adapter.
        Used by the leaderboard scanner to iterate all assets.
        """
        ...

    def normalize(self, series: pd.Series) -> pd.Series:
        """
        Normalize series before velocity computation.
        Default: min-max scaling over full history → [0, 1].
        Override for domain-specific normalization (e.g. log prices).
        """
        s = series.dropna()
        if s.empty:
            return series
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.0, index=series.index)
        return (s - mn) / (mx - mn)

    def label(self, series_id: str) -> str:
        """
        Human-readable label for a series ID.
        Override to provide friendly names.
        """
        return series_id
