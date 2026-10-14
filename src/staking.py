r"""
Stage 5 — Staking.  [Built in Phase 5, 14 Oct 2026]

Turns ``(calibrated probability, market odds)`` into decisions: which bets carry
value, how much to stake on each, and what the bankroll would have done.

Two probabilities, two different jobs
-------------------------------------
Bookmaker odds do not sum to 100%; the excess is the margin. That leaves two
market probabilities for every outcome, and conflating them is the subtlest
mistake in this whole pipeline:

* **Raw implied**, :math:`1/o` — what the price pays. A bet has positive expected
  value **if and only if** the model's probability exceeds this. The margin is a
  real cost and must be overcome, not removed.
* **De-vigged ("fair")** — the market's honest estimate of the true probability,
  with the margin stripped out.

A bet can beat the de-vigged probability and still lose money, because the payout
is at the actual odds. At odds of 3.50 with a 5% book: raw implied 0.2857, fair
0.2721. A model saying 0.278 disagrees with the market in the right direction and
has expected value of **-2.7%**.

So:

1. **The value test uses the raw price**, which is what Kelly already does —
   :math:`f^{*} > 0` exactly when :math:`p > 1/o`. ``staking.min_edge`` is
   measured there too.
2. **The de-vigged probability is a diagnostic**: it measures how far the model
   disagrees with the market's own view, which is what Phase 6's closing-line
   value is built on. It never decides whether to bet.
3. **Size with fractional Kelly**, then simulate the bankroll.

(The first version of this module had this backwards — it tested value against
the de-vigged probability, which is the permissive direction and would have
flagged negative-EV bets as value. `test_a_bet_beating_fair_but_not_raw_is_not_value`
pins the corrected behaviour.)

The Kelly criterion, and the assumption it rests on
---------------------------------------------------
For one bet at decimal odds :math:`o` (net odds :math:`b = o - 1`), win
probability :math:`p`, :math:`q = 1 - p`:

.. math::
    f^{*} = \frac{bp - q}{b} = \frac{po - 1}{o - 1}

This is the fraction of the bankroll that maximises expected *logarithmic*
growth, which is the same as maximising the long-run geometric growth rate.

**Kelly assumes you know p exactly.** You do not — :math:`p` here is the output
of a fitted model, and it carries estimation error. That is not a footnote, it is
the most important fact in this module, because the penalty for overestimating
:math:`p` is sharply asymmetric:

* staking :math:`f^{*}` maximises growth;
* staking roughly :math:`2f^{*}` gives **zero** expected log growth — all of the
  edge converted into variance and nothing else;
* staking beyond that loses money *even when the edge is real*.

The "2x" figure deserves precision, because it is usually quoted as an identity
and is not one. It is exact in the small-edge (Gaussian / continuous-rebalancing)
limit, where growth is :math:`f\mu - f^2\sigma^2/2` and vanishes at
:math:`f = 2\mu/\sigma^2`. For discrete bets the zero-growth point is *near*
2x and depends on the odds — measured in the tests, it runs from about 1.95x at
even money with a large edge to 2.10x on a 5.0 shot, converging on exactly 2.000x
as the edge shrinks. `test_zero_growth_point_is_near_two_times_kelly` records
those numbers.

Either way the practical conclusion stands: if your probabilities are 20% too
optimistic, full Kelly is already betting about double what it should, and you
are at the zero-growth boundary with a genuine edge in hand. That — not variance
reduction — is the argument for staking a fraction.

``staking.kelly_fraction`` defaults to 0.25. Quarter-Kelly survives being wrong
about :math:`p` by a factor of four before reaching the zero-growth boundary.

See docs/methodology.md section 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# Probabilities are kept inside this range before any division or logarithm.
EPSILON = 1e-9


class StakingError(Exception):
    """Raised when staking cannot be computed from the inputs given."""


# ===========================================================================
# De-vigging the market
# ===========================================================================

def overround(odds):
    """The bookmaker's margin: how far the implied probabilities exceed 1.

    ``sum(1/odds) - 1``. A typical 1X2 market runs 2% to 7%. This is the hurdle
    any bet has to clear before it is worth anything, and the reason a "5% edge"
    against raw odds is often no edge at all.
    """
    o = np.asarray(odds, dtype=float)
    return float(np.sum(1.0 / o) - 1.0)


def devig_proportional(odds):
    """Scale the implied probabilities so they sum to 1.

    .. math:: p_i = \\frac{1/o_i}{\\sum_j 1/o_j}

    Simple, fast, and it makes one strong assumption: that the bookmaker applies
    its margin *proportionally* across outcomes. Bookmakers do not — they load
    more margin onto longshots, because that is where recreational money goes.
    So this method systematically **overstates longshot probabilities**, which is
    exactly the direction that creates phantom value on big-price bets.
    """
    o = np.asarray(odds, dtype=float)
    if np.any(o <= 1.0):
        raise StakingError(f"Decimal odds must exceed 1.0; got {o.tolist()}")

    implied = 1.0 / o
    return implied / implied.sum()


def devig_shin(odds, max_insider_share=0.35):
    """Shin's method: de-vig while correcting the favourite-longshot bias.

    Shin (1993) models the book as containing a proportion *z* of insider money.
    The bookmaker protects itself by loading margin onto longshots, so unwinding
    that gives:

    .. math::
        p_i = \\frac{\\sqrt{z^2 + 4(1-z)\\,\\pi_i^2 / \\Pi} - z}{2(1-z)},
        \\qquad \\pi_i = 1/o_i,\\quad \\Pi = \\sum_j \\pi_j

    with *z* solved numerically so the probabilities sum to 1.

    The practical consequence: Shin assigns **lower** probabilities to longshots
    than proportional de-vigging does. Since a naive model tends to find its
    "best" value on big prices, this is the difference between a backtest with a
    fake edge on 8.0 shots and one without.

    Falls back to proportional when the book has no margin to unwind (z would be
    0) or when the solve fails, and says which was used.

    Returns
    -------
    (numpy.ndarray, dict)
        The probabilities and a note on the method and the fitted z.
    """
    o = np.asarray(odds, dtype=float)
    if np.any(o <= 1.0):
        raise StakingError(f"Decimal odds must exceed 1.0; got {o.tolist()}")

    implied = 1.0 / o
    book = implied.sum()

    # No margin (or a negative one) means there is nothing for Shin to unwind.
    if book <= 1.0 + 1e-12:
        return implied / book, {"method": "proportional", "z": 0.0,
                                "reason": "book has no margin to unwind"}

    def probabilities(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * implied ** 2 / book) - z) / (2 * (1 - z))

    def total_minus_one(z):
        return probabilities(z).sum() - 1.0

    try:
        # total_minus_one is positive at z -> 0 (where it reduces to the raw book)
        # and decreasing in z, so a root is bracketed on (0, max_insider_share).
        z = brentq(total_minus_one, 1e-9, max_insider_share, xtol=1e-10)
        result = probabilities(z)
        result = result / result.sum()      # tidy up the last 1e-12
        return result, {"method": "shin", "z": float(z)}
    except (ValueError, RuntimeError):
        # Happens on a badly-formed book. Falling back is better than guessing,
        # and saying so is better than falling back quietly.
        return implied / book, {"method": "proportional", "z": 0.0,
                                "reason": "Shin solve did not converge"}


def market_probabilities(odds, method="shin"):
    """De-vig a set of odds by the configured method.

    Returns
    -------
    (numpy.ndarray, dict)
    """
    if method == "proportional":
        return devig_proportional(odds), {"method": "proportional", "z": 0.0}
    if method == "shin":
        return devig_shin(odds)
    raise StakingError(f"Unknown de-vig method '{method}'. Use 'shin' or 'proportional'.")


# ===========================================================================
# Kelly
# ===========================================================================

def kelly_fraction(probability, odds):
    """The growth-optimal fraction of bankroll for a single bet.

    .. math:: f^{*} = \\frac{bp - q}{b} = \\frac{po - 1}{o - 1}

    Returns a NEGATIVE number when the bet has no edge — which is the formula
    telling you to take the other side, and in a market where you cannot, to
    stake nothing. Callers clip at zero rather than the formula doing it, so the
    size of the *negative* edge stays available for diagnostics.
    """
    p = np.asarray(probability, dtype=float)
    o = np.asarray(odds, dtype=float)

    if np.any(o <= 1.0):
        raise StakingError("Decimal odds must exceed 1.0.")

    return (p * o - 1.0) / (o - 1.0)


def expected_log_growth(probability, odds, fraction):
    """Expected log growth from staking `fraction` of bankroll.

    .. math::
        E[\\ln] = p \\ln(1 + f(o-1)) + (1-p)\\ln(1 - f)

    The quantity Kelly maximises. Worth having as a function rather than a
    comment, because it is what makes the 2x-Kelly result checkable: at
    :math:`f = 2f^{*}` this returns 0, and beyond it goes negative while the edge
    is still real.
    """
    p = float(probability)
    o = float(odds)
    f = float(fraction)

    if f >= 1.0:
        return -np.inf            # staking the whole bankroll risks ruin outright
    if f <= 0.0:
        return 0.0                # no bet, no growth

    return p * np.log(1 + f * (o - 1)) + (1 - p) * np.log(1 - f)


def stake_size(model_probability, odds, config=None):
    """Decide whether to bet, and how much, for one outcome.

    Returns
    -------
    dict
        Edge, the full and fractional Kelly stakes, the capped stake actually
        used, and the reason when nothing is staked.
    """
    config = config or {}
    cfg = config.get("staking", {})
    fraction = float(cfg.get("kelly_fraction", 0.25))
    min_edge = float(cfg.get("min_edge", 0.02))
    max_stake = float(cfg.get("max_stake_fraction", 0.05))
    devig_method = cfg.get("devig_method", "shin")

    p_model = float(model_probability)
    o = float(odds)

    # Edge against the raw price, which is both the EV break-even and Kelly's own
    # threshold. A single outcome cannot be de-vigged on its own anyway — the
    # whole market is needed for that — and de-vigging is not what decides a bet.
    implied = 1.0 / o
    edge = p_model - implied

    full_kelly = float(kelly_fraction(p_model, o))
    fractional = max(full_kelly * fraction, 0.0)

    # The cap is a risk limit, not part of Kelly. It exists because a single
    # model probability can be badly wrong, and no one bet should be able to
    # matter that much.
    capped = min(fractional, max_stake)

    reason = None
    if edge < min_edge:
        capped, reason = 0.0, f"edge {edge:.4f} below minimum {min_edge}"
    elif full_kelly <= 0:
        capped, reason = 0.0, "no Kelly edge at these odds"

    return {
        "model_probability": p_model,
        "market_implied": float(implied),
        "edge": float(edge),
        "full_kelly": full_kelly,
        "fractional_kelly": float(fractional),
        "stake_fraction": float(capped),
        "capped": bool(fractional > max_stake),
        "no_bet_reason": reason,
    }


# ===========================================================================
# Finding bets across a set of fixtures
# ===========================================================================

OUTCOME_COLUMNS = {"home_win": ("OddsH", "H"),
                   "draw": ("OddsD", "D"),
                   "away_win": ("OddsA", "A")}


def find_value_bets(predictions, config=None):
    """Scan calibrated predictions and return every bet worth placing.

    `predictions` must carry the model's 1X2 probabilities, the odds we would bet
    into (``OddsH/D/A``) and the realised ``Result``. One row per bet, not per
    match, because two outcomes of the same match can both look like value — and
    if they do, that is itself worth seeing.

    The whole market is de-vigged together, which is the only way to do it
    correctly: a single price carries no information about the margin.
    """
    config = config or {}
    cfg = config.get("staking", {})
    devig_method = cfg.get("devig_method", "shin")

    required = ["OddsH", "OddsD", "OddsA"] + list(OUTCOME_COLUMNS)
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise StakingError(f"find_value_bets needs columns {missing}")

    rows = []
    for _, match in predictions.iterrows():
        odds = np.array([match["OddsH"], match["OddsD"], match["OddsA"]],
                        dtype=float)
        if not np.all(np.isfinite(odds)) or np.any(odds <= 1.0):
            continue                      # unpriced or impossible: not bettable

        fair, devig_note = market_probabilities(odds, devig_method)
        book_margin = overround(odds)

        for index, (outcome, (odds_column, code)) in enumerate(
                OUTCOME_COLUMNS.items()):
            p_model = float(match[outcome])
            price = float(match[odds_column])
            raw = 1.0 / price

            # EDGE is measured against the RAW price, because that is what the
            # bet pays. p > 1/o is exactly the condition for positive expected
            # value, and exactly when Kelly turns positive — the two agree by
            # construction, which is the point.
            edge = p_model - raw

            # DISAGREEMENT is measured against the de-vigged price: how far the
            # model departs from the market's own honest estimate. It is a
            # diagnostic (and the basis of Phase 6's CLV), never a bet trigger.
            # A bet can have positive disagreement and negative edge, and that
            # bet loses money.
            disagreement = p_model - float(fair[index])

            full_kelly = float(kelly_fraction(p_model, price))

            decision = {
                "MatchID": match.get("MatchID"),
                "MatchDate": match.get("MatchDate"),
                "HomeTeam": match.get("HomeTeam"),
                "AwayTeam": match.get("AwayTeam"),
                "outcome": outcome,
                "odds": price,
                "model_probability": p_model,
                "market_raw": raw,
                "market_fair": float(fair[index]),
                "book_margin": book_margin,
                "devig_method": devig_note["method"],
                "edge": edge,                      # vs the price: decides bets
                "disagreement": disagreement,      # vs fair value: diagnostic
                "full_kelly": full_kelly,
                "won": bool(match.get("Result") == code)
                        if match.get("Result") is not None else None,
            }
            rows.append(decision)

    bets = pd.DataFrame(rows)
    if bets.empty:
        return bets

    min_edge = float(cfg.get("min_edge", 0.02))
    fraction = float(cfg.get("kelly_fraction", 0.25))
    max_stake = float(cfg.get("max_stake_fraction", 0.05))

    bets["stake_fraction"] = np.clip(bets["full_kelly"] * fraction, 0.0, max_stake)
    bets["capped"] = (bets["full_kelly"] * fraction) > max_stake
    bets["is_value"] = (bets["edge"] >= min_edge) & (bets["full_kelly"] > 0)
    bets.loc[~bets["is_value"], "stake_fraction"] = 0.0

    return bets.sort_values("MatchDate").reset_index(drop=True)


# ===========================================================================
# Bankroll simulation
# ===========================================================================

def simulate_bankroll(bets, config=None):
    """Walk the bet history forward and record what the bankroll did.

    Two details that decide whether the simulation means anything:

    **Bets on the same date are sized against the same bankroll.** At the moment
    you place Saturday's bets you do not know Saturday's results, so sizing the
    third bet off the first two's winnings would be using information you did not
    have. Grouping by date removes that.

    **Total exposure on a single date is capped** at
    ``staking.max_daily_exposure``. Kelly sizes each bet as if it were the only
    one; ten simultaneous "independent" bets on one afternoon are not independent
    enough for that to be safe, and without a cap a single bad weekend can take
    a third of the bankroll.

    Returns
    -------
    (pandas.DataFrame, dict)
        The per-bet ledger and the summary — final bankroll, ROI, yield, hit
        rate, maximum drawdown.
    """
    config = config or {}
    cfg = config.get("staking", {})
    bankroll = float(cfg.get("starting_bankroll", 1000))
    max_daily = float(cfg.get("max_daily_exposure", 0.15))

    placed = bets[bets["stake_fraction"] > 0].copy()
    if placed.empty:
        return pd.DataFrame(), {
            "bets": 0, "starting_bankroll": bankroll, "final_bankroll": bankroll,
            "roi": 0.0, "yield": 0.0, "hit_rate": float("nan"),
            "max_drawdown": 0.0, "turnover": 0.0,
            "note": "no value bets found at these settings",
        }

    if placed["won"].isna().any():
        raise StakingError(
            "Some bets have no recorded outcome, so the bankroll cannot be "
            "simulated. Pass predictions that include the realised Result."
        )

    ledger = []
    peak = bankroll
    max_drawdown = 0.0
    turnover = 0.0

    for date, days_bets in placed.groupby("MatchDate", sort=True):
        bankroll_at_open = bankroll

        # Scale the day's stakes down together if they would exceed the daily cap,
        # which preserves their relative sizing.
        wanted = days_bets["stake_fraction"].sum()
        scale = min(1.0, max_daily / wanted) if wanted > 0 else 0.0

        day_profit = 0.0
        for _, bet in days_bets.iterrows():
            stake = bankroll_at_open * bet["stake_fraction"] * scale
            profit = stake * (bet["odds"] - 1.0) if bet["won"] else -stake

            day_profit += profit
            turnover += stake

            ledger.append({
                "MatchDate": date,
                "MatchID": bet["MatchID"],
                "outcome": bet["outcome"],
                "odds": bet["odds"],
                "edge": bet["edge"],
                "stake": stake,
                "stake_fraction_used": bet["stake_fraction"] * scale,
                "day_scaled": scale < 1.0,
                "won": bet["won"],
                "profit": profit,
                "bankroll_after_day": None,     # filled in below
            })

        bankroll += day_profit
        for row in ledger[-len(days_bets):]:
            row["bankroll_after_day"] = bankroll

        peak = max(peak, bankroll)
        # Drawdown as a share of the previous peak, which is how a bankroll
        # drop is actually experienced — a 30% fall from a high is 30% whatever
        # the starting figure was.
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak)

        if bankroll <= 0:
            break                            # ruin: stop, and report it

    frame = pd.DataFrame(ledger)
    starting = float(cfg.get("starting_bankroll", 1000))

    summary = {
        "bets": int(len(frame)),
        "starting_bankroll": starting,
        "final_bankroll": float(bankroll),
        # ROI is on the bankroll; yield is on money staked. They answer
        # different questions and are routinely confused.
        "roi": float((bankroll - starting) / starting),
        "yield": float(frame["profit"].sum() / turnover) if turnover > 0 else 0.0,
        "hit_rate": float(frame["won"].mean()),
        "average_odds": float(frame["odds"].mean()),
        "average_edge": float(frame["edge"].mean()),
        "turnover": float(turnover),
        "max_drawdown": float(max_drawdown),
        "ruined": bool(bankroll <= 0),
        "days_scaled_by_cap": int(frame["day_scaled"].sum()),
    }
    return frame, summary


def staking_report(bets, ledger, summary, config=None):
    """Readable summary of what the staking rule did and what it cost."""
    cfg = (config or {}).get("staking", {})
    lines = ["Staking (fractional Kelly)", "-" * 26]
    lines.append(f"  {'Kelly fraction':<30} {cfg.get('kelly_fraction', 0.25):>10}")
    lines.append(f"  {'minimum edge':<30} {cfg.get('min_edge', 0.02):>10}")
    lines.append(f"  {'max stake per bet':<30} "
                 f"{cfg.get('max_stake_fraction', 0.05):>10}")
    lines.append(f"  {'max exposure per day':<30} "
                 f"{cfg.get('max_daily_exposure', 0.15):>10}")
    lines.append(f"  {'de-vig method':<30} {cfg.get('devig_method', 'shin'):>10}")

    if not bets.empty:
        lines.append("")
        lines.append(f"  {'opportunities examined':<30} {len(bets):>10,}")
        lines.append(f"  {'flagged as value':<30} "
                     f"{int(bets['is_value'].sum()):>10,}")
        lines.append(f"  {'mean book margin':<30} "
                     f"{bets['book_margin'].mean():>10.2%}")
        lines.append(f"  {'mean edge vs price':<30} "
                     f"{bets['edge'].mean():>10.2%}   <- decides bets")
        lines.append(f"  {'mean disagreement vs fair':<30} "
                     f"{bets['disagreement'].mean():>10.2%}   <- diagnostic only")

    if summary["bets"] == 0:
        lines.append("")
        lines.append(f"  {summary.get('note', 'no bets placed')}")
        lines.append("  No bets is a legitimate result. A model with no edge over")
        lines.append("  a de-vigged market SHOULD stake nothing.")
        return "\n".join(lines)

    lines.append("")
    lines.append("  Bankroll:")
    lines.append(f"    {'bets placed':<28} {summary['bets']:>10,}")
    lines.append(f"    {'starting':<28} {summary['starting_bankroll']:>10,.2f}")
    lines.append(f"    {'final':<28} {summary['final_bankroll']:>10,.2f}")
    lines.append(f"    {'ROI (on bankroll)':<28} {summary['roi']:>10.2%}")
    lines.append(f"    {'yield (on turnover)':<28} {summary['yield']:>10.2%}")
    lines.append(f"    {'hit rate':<28} {summary['hit_rate']:>10.2%}")
    lines.append(f"    {'average odds':<28} {summary['average_odds']:>10.2f}")
    lines.append(f"    {'average edge':<28} {summary['average_edge']:>10.2%}")
    lines.append(f"    {'turnover':<28} {summary['turnover']:>10,.2f}")
    lines.append(f"    {'maximum drawdown':<28} {summary['max_drawdown']:>10.2%}")

    if summary["ruined"]:
        lines.append("")
        lines.append("  RUINED — the bankroll reached zero. Whatever the ROI said")
        lines.append("  before that point, this staking plan is not usable.")
    if summary["days_scaled_by_cap"]:
        lines.append(f"    ({summary['days_scaled_by_cap']} days scaled down by "
                     "the daily exposure cap)")

    lines.append("")
    lines.append("  Reminder: these numbers come from a FIXED model fitted on all")
    lines.append("  the data. They are a staking sanity check, not a result —")
    lines.append("  Phase 6's walk-forward backtest produces the honest version.")
    return "\n".join(lines)


def stake(probs, odds, config=None):
    """Backwards-compatible entry point kept from the Phase 0 scaffold.

    New code should use `find_value_bets` followed by `simulate_bankroll`.
    """
    raise StakingError(
        "staking.stake is superseded: use find_value_bets(predictions, config) "
        "then simulate_bankroll(bets, config)."
    )
