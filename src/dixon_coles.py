"""
Stage 3 — The Dixon-Coles model.  [Built in Phase 3]

A bivariate Poisson goal model with the low-score dependence correction (rho),
fitted by time-weighted maximum likelihood. From the fitted score matrix we
derive every market probability. See docs/methodology.md section 3.
"""
from __future__ import annotations


def fit(features, config=None):
    """Fit Dixon-Coles parameters (attack, defence, home advantage, rho).

    Not yet implemented — arrives in Phase 3.

    Parameters
    ----------
    features : object
        Feature matrix from Stage 2.
    config : dict | None
        Run configuration.
    """
    raise NotImplementedError("dixon_coles.fit is scheduled for Phase 3.")


def predict(model, fixtures, config=None):
    """Turn fitted parameters into market probabilities for given fixtures.

    Not yet implemented — arrives in Phase 3.
    """
    raise NotImplementedError("dixon_coles.predict is scheduled for Phase 3.")
