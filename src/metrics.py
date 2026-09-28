"""
Metrics.  [Built in Phase 6]

Definitions for the numbers we report: proper scoring rules (Brier, log-loss)
for probability quality, and betting performance (ROI, yield, hit rate by
confidence band, a Sharpe-like ratio, maximum drawdown). See
docs/methodology.md sections 4 and 6.4.
"""
from __future__ import annotations


def summarise(results):
    """Summarise backtest results into the headline metrics table.

    Not yet implemented — arrives in Phase 6.

    Parameters
    ----------
    results : object
        Output of the walk-forward backtest.
    """
    raise NotImplementedError("metrics.summarise is scheduled for Phase 6.")
