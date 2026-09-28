"""
Stage 6 — Backtesting.  [Built in Phase 6]

The honesty layer: a walk-forward engine that at each match uses only data that
existed before it, with explicit lookahead guards. Produces the bet history that
CLV and the headline metrics are computed from. See docs/methodology.md section 6.
"""
from __future__ import annotations


def walk_forward(df, config=None):
    """Run a leakage-free walk-forward backtest over the match history.

    Not yet implemented — arrives in Phase 6.

    Parameters
    ----------
    df : pandas.DataFrame
        Full match history (the engine slices it by date internally).
    config : dict | None
        Run configuration.
    """
    raise NotImplementedError("backtest.walk_forward is scheduled for Phase 6.")
