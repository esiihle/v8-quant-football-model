"""
Stage 6 — Closing-line value (CLV).  [Built in Phase 6]

The headline metric. Compares the odds we took against the closing odds;
consistent positive CLV is the strongest available evidence of a real,
repeatable edge. See docs/methodology.md section 6.3.
"""
from __future__ import annotations


def compute_clv(bets, closing_odds):
    """Compute closing-line value for a set of placed bets.

    Not yet implemented — arrives in Phase 6.

    Parameters
    ----------
    bets : object
        The bets placed during the backtest (odds taken, stake, outcome).
    closing_odds : object
        Closing odds for the same fixtures/markets.
    """
    raise NotImplementedError("clv.compute_clv is scheduled for Phase 6.")
