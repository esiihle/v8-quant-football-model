"""
Stage 2 — Feature engineering.  [Built in Phase 2]

Turns raw match results into predictive signal: team attack/defence strengths
with Bayesian shrinkage for thin samples, expected-goals proxies, exponential
time decay, and home advantage. See docs/methodology.md section 2.
"""
from __future__ import annotations


def build_features(df, config=None):
    """Build the model feature matrix from ingested matches.

    Not yet implemented — arrives in Phase 2. The signature is fixed now so the
    pipeline runner and tests can wire against it.

    Parameters
    ----------
    df : pandas.DataFrame
        Ingested, validated matches.
    config : dict | None
        Run configuration.
    """
    raise NotImplementedError("features.build_features is scheduled for Phase 2.")
