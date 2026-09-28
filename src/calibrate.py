"""
Stage 4 — Probability calibration.  [Built in Phase 4]

Maps raw model probabilities to calibrated ones (isotonic or Platt) and proves
the result with reliability diagrams and proper scoring rules (Brier, log-loss).
See docs/methodology.md section 4.
"""
from __future__ import annotations


def calibrate(probs, outcomes, config=None):
    """Fit and apply a calibration map to raw model probabilities.

    Not yet implemented — arrives in Phase 4.

    Parameters
    ----------
    probs : object
        Raw model probabilities.
    outcomes : object
        Realised outcomes, for fitting the calibration map.
    config : dict | None
        Run configuration.
    """
    raise NotImplementedError("calibrate.calibrate is scheduled for Phase 4.")
