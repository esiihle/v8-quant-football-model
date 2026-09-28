"""
Stage 5 — Staking.  [Built in Phase 5]

Converts (model probability, market odds) into decisions: flags value bets,
sizes them with fractional Kelly, and simulates the bankroll to expose realistic
drawdowns. See docs/methodology.md section 5.
"""
from __future__ import annotations


def stake(probs, odds, config=None):
    """Decide and size bets from model probabilities and market odds.

    Not yet implemented — arrives in Phase 5.

    Parameters
    ----------
    probs : object
        Calibrated model probabilities.
    odds : object
        Market odds to bet into.
    config : dict | None
        Run configuration (kelly_fraction, min_edge, starting_bankroll).
    """
    raise NotImplementedError("staking.stake is scheduled for Phase 5.")
