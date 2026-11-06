# V8 — Study Guide

*How the pipeline works, why each choice was made, what went wrong, and how to
say it to a panel. Written 6 Nov 2026.*

This guide is for one reader: the person who has to explain this project in an
interview without notes. It follows the pipeline stage by stage. Each stage has
the same six parts:

1. **What it does**, in two sentences.
2. **The code**, quoted from the repository. Every block is an exact excerpt,
   and `tests/test_docs.py` fails if the source changes and this guide does not.
3. **The maths**, derived rather than asserted.
4. **Trade-off box**: what was chosen, what else could have been, and why.
5. **The bug story**: what broke at this stage and how it was caught.
6. **Interview drill**: the questions a panel is likely to ask, with answers
   short enough to say out loud.

Every figure is from `docs/charts/numbers.json` (2 seasons, fingerprint
`eddedac58a664dd8`) or `docs/charts/epl_2012_2025/numbers.json` (13 seasons,
`9ed0f574b163c749`), and the same tests check it.

```
ingest → features → Dixon-Coles → calibration → staking → walk-forward → CLV → benchmark
                                                                ↑
                         simulation study: does the machinery work at all?
```

---

## 0. The pitch

**Thirty seconds.** "I built a football prediction and betting system to the
standard a bank's model-validation team would apply. The finding is that the
model has no edge over the market. What makes the project worth discussing is
how I know that: a walk-forward backtest with a lookahead guard on every
window, closing-line value split into the bookmaker's margin and the market's
movement, a head-to-head comparison of probabilities on the same matches, and a
simulation study showing the machinery can detect an edge of about five
probability points when one exists. Along the way I found and pinned eleven
bugs, several of which made the results look better than they were."

**Why a bank should care.** Replace "match" with "loan" and "bookmaker" with
"the market's price of credit", and every stage has a credit-risk twin:

| V8 | Credit risk |
|---|---|
| Dixon-Coles probabilities | PD model |
| Calibration (Platt, reliability, ECE) | PD calibration to observed default rates |
| Walk-forward backtest, lookahead guard | Out-of-time validation, no use of future data |
| Benchmark against the market | Benchmark against a challenger model or external rating |
| Kelly with caps | Limit setting and concentration caps |
| Simulation study | Validating the validation: can the test detect a known effect? |
| Docs tests against `numbers.json` | Model documentation that cannot drift from the model |

---

## 1. Ingest: getting data you can trust

**What it does.** Reads football-data.co.uk season files into one canonical
table: dates parsed, duplicates dropped, results checked against goals, odds
outside a plausible range voided. Because that site is blocked on the author's
network, `scripts/rebuild_raw_from_mirrors.py` rebuilds the files from two
GitHub mirrors of it.

**The cross-mirror audit.** One mirror has the odds, the other has the shots.
Both carry the score, which gives a free audit: two independent copies of every
result must agree before anything is written.

<!-- source: scripts/rebuild_raw_from_mirrors.py -->
```python
    disagree = joined[(joined["FTHG"] != joined["fthg"]) | (joined["FTAG"] != joined["ftag"])]
```

All 13 seasons, 4,940 matches, passed with zero disagreements.

**The silent failure the verifier catches.** If a "closing odds" column is a
copy of the opening price, the pipeline runs happily and reports CLV of exactly
zero, which looks like a measurement and isn't one.
`scripts/verify_raw_data.py` refuses a file where more than half the rows have
closing odds identical to opening.

> **Trade-off: rebuild from mirrors vs find another source.**
> *Chosen:* rebuild football-data's own format from two mirrors of it.
> *Alternative:* switch to a different dataset with different columns.
> *Why:* the pipeline was built and tested against one schema. Changing source
> changes everything downstream; rebuilding the original changes nothing, and
> the audit means the mirrors are checked, not trusted.

> **Bug story: the numbers that were never real.** The README once reported
> ROI −28.5% on 413 bets. Those figures came from an earlier session and had
> never been reproduced. A companion project had just been caught publishing
> synthetic results as real, so the README carried an "Awaiting regeneration"
> notice, enforced by a test. When the real data arrived, the old figures did
> not reproduce: the real run gave 433 bets and ROI −82.60%. They were
> withdrawn, not edited. **Rule since then: no number enters a document unless a
> command regenerates it.**

**Drill.**

- *How do you know your data is right?* "Two independent mirrors had to agree
  on all 4,940 scores before a file was written, and a verifier rejects closing
  odds that are a copy of opening odds, which is the failure that would make
  CLV read zero without any error."
- *Why not just download it?* "The site is blocked on my network. I rebuilt the
  exact original format from two GitHub mirrors, so nothing downstream changed,
  and I audited the join instead of trusting it."

---

## 2. Features: what the model is allowed to know

**What it does.** Turns match history into team attack and defence strengths
as of a date. Three ideas: older matches count less (time decay), small samples
are pulled toward average (shrinkage), and goals are blended with a
shots-on-target proxy (expected goals).

### 2.1 Time decay

A match $t$ days old gets weight $w = 0.5^{t/H}$ with half-life $H = 180$ days.
Matches dated on or after the forecast date get weight zero, and that is where
the lookahead guard lives. Since 27 Oct, weights below 1% are also set to zero:

<!-- source: src/features.py -->
```python
    # age < 0 means the match is in the future relative to as_of: weight 0.
    # A weight below min_weight is too small to matter: weight 0 as well.
    keep = (age_days > 0) & (weights >= float(min_weight))
    return np.where(keep, weights, 0.0)
```

**Maths: the cutoff discards exactly 1% of the weight.** For an unbroken
fixture list the weight is a continuous decay, and the share older than $t^*$ is

$$\frac{\int_{t^*}^{\infty} 0.5^{t/H}\,dt}{\int_0^{\infty} 0.5^{t/H}\,dt} = 0.5^{t^*/H}.$$

Setting this to 0.01 gives $t^* = H \log_2 100 \approx 1{,}196$ days, about 3.3
seasons. So the cutoff *is* its own tail share. It cut the 13-season run from
67 minutes to 11, and on 2 seasons all 37 headline figures came out
bit-identical with and without it.

### 2.2 Shrinkage

<!-- source: src/features.py -->
```python
    n_matches = np.asarray(n_matches, dtype=float)
    weight = n_matches / (n_matches + float(k))
    return weight * np.asarray(estimate, dtype=float) + (1.0 - weight) * float(prior)
```

**Maths: why $n/(n+k)$ is Bayesian.** Suppose a team's true strength $\theta$
has prior $\mathcal N(\mu_0, \tau^2)$ (the league), and each match gives a noisy
reading with variance $\sigma^2$. After $n$ matches with mean $\bar x$, the
posterior mean is

$$E[\theta \mid \bar x] = \frac{n/\sigma^2}{n/\sigma^2 + 1/\tau^2}\,\bar x + \frac{1/\tau^2}{n/\sigma^2 + 1/\tau^2}\,\mu_0
= \frac{n}{n+k}\,\bar x + \frac{k}{n+k}\,\mu_0, \qquad k = \frac{\sigma^2}{\tau^2}.$$

So $k$ is the ratio of match-to-match noise to genuine spread between teams.
With $k = 8$: zero matches gives the league average, 8 matches gives halfway,
and 38 matches gives 83% the team's own record. With decay, $n$ is the *sum of
weights*, the effective sample size, not a count of rows.

### 2.3 The expected-goals proxy

<!-- source: src/features.py -->
```python
    out["HomeAttackGoals"] = (1 - weight) * out["HomeGoals"] + weight * home_xg
    out["AwayAttackGoals"] = (1 - weight) * out["AwayGoals"] + weight * away_xg
```

Goals are rare and noisy; shots on target are plentiful. The exploration
measured a correlation of 0.584 between home shots on target and home goals on 2
seasons, strong enough to help and far enough from 1 that goals still matter.
The blend is 50/50.

> **Trade-off: half-life 180 days.**
> *Chosen:* 180 days, so last season counts about a quarter as much as this one.
> *Alternatives:* no decay (stale squads count fully), or a short half-life
> (fast to react, but a handful of matches dominate).
> *Why:* a squad changes over a summer but not over a month. 180 days keeps
> roughly one season of real influence. It is a stated assumption, not a
> tuned optimum, and it is in `config.yaml` where a reviewer can see it.

> **Bug story #1: lookahead in the xG proxy.** The conversion rate (goals per
> shot on target) was first measured over the whole file. So an August match
> was scored using a rate computed partly from May. The lookahead test caught
> it. The fix was structural: `blend_goals(matches, history, config)` takes
> `history` as a **required** argument, so the caller has to say which past it
> means. *A fix that makes the mistake impossible beats one that makes it
> unlikely.*

**Drill.**

- *What's shrinkage and why use it?* "A team with three matches has a noisy
  record, so I pull it toward the league average in proportion to how little
  data backs it: weight $n/(n+8)$. It's the posterior mean under a normal
  prior, with $k$ the ratio of match noise to real spread between teams."
- *How did you stop future data leaking into features?* "Weights are zero for
  any match on or after the forecast date, the shots conversion rate takes its
  history as a required argument, and the backtest checks every window. The
  xG leak was a real bug, caught by that test."

---

## 3. Dixon-Coles: the model

**What it does.** Models each side's goals as Poisson, with a rate set by
attack, the opponent's defence and home advantage, plus a correction for the
four lowest scores. Parameters are fitted by maximising a time-weighted,
penalised likelihood.

**Maths: the model.** For a match between home team $h$ and away team $a$:

$$\lambda = e^{\alpha_h + \beta_a + \gamma}, \qquad \mu = e^{\alpha_a + \beta_h}$$

$$P(X = x, Y = y) = \tau_{\lambda,\mu}(x, y)\cdot\frac{\lambda^x e^{-\lambda}}{x!}\cdot\frac{\mu^y e^{-\mu}}{y!}$$

Here $\alpha$ is attack, $\beta$ is how many goals a team *concedes* (higher
means leakier), and $\gamma$ is home advantage. On 2 seasons the fitted
$\gamma = 0.0970$, a home scoring multiplier of $e^{0.097} = 1.102$.

### 3.1 The τ correction

<!-- source: src/dixon_coles.py -->
```python
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
```

**Maths: τ moves probability between the four cells without creating any.**
Write $P_{xy}$ for the independent Poisson probabilities. With
$c = e^{-\lambda-\mu}$: $P_{00} = c$, $P_{01} = c\mu$, $P_{10} = c\lambda$,
$P_{11} = c\lambda\mu$. The total change from τ is

$$-\lambda\mu\rho\,P_{00} + \lambda\rho\,P_{01} + \mu\rho\,P_{10} - \rho\,P_{11}
= c\rho\,(-\lambda\mu + \lambda\mu + \lambda\mu - \lambda\mu) = 0.$$

So the distribution still sums to one. With $\rho < 0$, 0-0 and 1-1 become
**more** likely and 1-0 and 0-1 **less**. (The exploration notebook once said
it raised all four. It does not.)

**What the data says.** Against a league-average Poisson, 13 seasons show 1-1
eleven per cent *rarer* than independence, which would argue against
Dixon-Coles. Against each match's own expected goals, 1-1 is +0.5%: the
"effect" was strong and weak teams mixed together. What survives is 0-1 rarer
by 12.7% [−21.7%, −3.5%] and $\rho$ negative in 10 of 13 seasons (sign test
p = 0.046). That is partial support, consistent with the walk-forward fitting
$\rho = -0.0065$, close to zero.

### 3.2 Fitting: the penalty that keeps the optimiser alive

<!-- source: src/dixon_coles.py -->
```python
    violation = float(np.sum(np.clip(INVALID_FLOOR - correction, 0.0, None)))
    if violation > 0:
        return INVALID_PENALTY * (1.0 + violation)
```

> **Bug story #2: the optimiser that died while reporting success.** A
> parameter set giving a negative τ used to return `+inf`. L-BFGS-B estimates
> its gradient by finite differences, so one infinite probe gives `inf − inf =
> nan`. The gradient became useless and the optimiser stopped moving, **while
> still reporting `success: True`**. Parameter recovery on simulated data came
> back at r = 0.21. The fix is a large *finite* penalty that grows with the size
> of the violation, so there is always a downhill direction back into the valid
> region. (The function's docstring kept describing `+inf` until 6 Nov, when
> writing this guide caught it.)

> **Bug story #3: ρ pinned at its bound.** The box for $\rho$ was computed from
> the largest $\lambda$ the parameter box allowed, $e^{4} \approx 54.6$ goals a
> match. That squeezed the box to [−0.018, 0.0003], so $\rho$ sat on its lower
> bound while every start agreed. It was caught because **the bias did not
> shrink as the sample grew**: noise averages out, a binding constraint does
> not. The fix sizes the bound from goals actually observed.

> **Trade-off: Dixon-Coles vs alternatives.**
> *Chosen:* Dixon-Coles, about 40 interpretable parameters.
> *Alternatives:* bivariate Poisson (a covariance term, harder to fit), Elo
> (one number per team, no scoreline distribution), or gradient boosting (more
> flexible, but needs many features and hides its reasoning).
> *Why:* every parameter means something a reviewer can check (an attack
> multiplier, a home edge), it produces a full scoreline distribution, and it
> is the standard benchmark in the literature. For a validation exercise, an
> interpretable model whose failure is clear beats a flexible one whose success
> is ambiguous.

**Drill.**

- *Explain Dixon-Coles in one breath.* "Each side's goals are Poisson with a
  rate from its attack, the opponent's defence and a home term, and a
  correction parameter ρ adjusts the four lowest scores, where independent
  Poisson is known to be wrong."
- *How did you know the fit was right?* "Parameter recovery: I simulated a
  league with known strengths and checked the fit got them back. That's how I
  caught two optimiser bugs where the fit reported success but was wrong."
- *Is the low-score correction justified?* "Partly. I measured it against a
  null that accounts for team strength. 0-1 is significantly rarer and ρ is
  negative in 10 of 13 seasons, but 0-0 and 1-1 aren't significantly
  different, so ρ fits near zero. It does little work on modern EPL data."

---

## 4. Calibration: are the probabilities honest?

**What it does.** Checks whether events the model calls 30% happen about 30%
of the time, and fits a correction map if not. The map is fitted on the
earlier half of the data and judged on the later half.

<!-- source: src/calibrate.py -->
```python
        X = _log_odds(p).reshape(-1, 1)
        fit = LogisticRegression(solver="lbfgs", C=1e6)   # C large = unpenalised
        fit.fit(X, o)
```

**Maths: Platt scaling** fits $\operatorname{logit}(p') = a\,\operatorname{logit}(p) + b$.
A slope $a < 1$ means the model is overconfident (its probabilities are too
spread out); $b$ shifts the level. Isotonic regression fits any monotone step
function instead: more flexible, but with more to overfit.

**Maths: expected calibration error and its noise floor.** Bin the forecasts,
then $\text{ECE} = \sum_b \frac{n_b}{n}\,\lvert\bar p_b - \bar y_b\rvert$.
ECE is **never zero on finite data**, because each bin's realised frequency is
itself an estimate. To read it, draw outcomes *from the forecaster's own
probabilities*: that is the world in which it is perfectly calibrated. Score
ECE many times, and take the 95th percentile as the floor.

On 2 seasons the model's ECE is 0.0613 on 341 matches, below its floor of
0.0802: the reliability diagram **cannot show miscalibration either way**. On
13 seasons, with walk-forward probabilities on 4,722 matches, the model's
0.0237 is above its floor and both markets are below theirs, so there the model
is detectably miscalibrated.

> **Trade-off: Platt vs isotonic.**
> *Chosen:* Platt, after comparing both out of sample.
> *Alternative:* isotonic, which can fix any monotone distortion.
> *Why:* on 2 seasons isotonic made log-loss much worse out of sample: it
> overfits a step function to a few hundred matches. Two parameters can't
> overfit like that. On 13 seasons Platt improved the model out of sample; on 2
> it didn't, which points at the short window, and the report says so rather
> than tuning until it passes.

**Drill.**

- *Your ECE is 0.06. Is the model miscalibrated?* "On 341 matches a perfectly
  calibrated model scores up to about 0.08 by chance alone, so 0.06 can't show
  miscalibration. On 13 seasons it can, and there the model is miscalibrated
  and the market isn't."
- *Why fit calibration on the earlier half only?* "Fitted and judged on the
  same matches it would always look good. That's the same reason the backtest
  is walk-forward."

---

## 5. Staking: when to bet and how much

**What it does.** De-vigs the market to find its fair probabilities, decides
whether the model's probability beats the price, and sizes the stake by
fractional Kelly with caps.

### 5.1 Removing the margin: Shin's method

<!-- source: src/staking.py -->
```python
    def probabilities(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * implied ** 2 / book) - z) / (2 * (1 - z))
```

Shin models a share $z$ of insider money. Bookmakers protect themselves by
loading margin onto longshots, so unwinding it gives longshots *lower* fair
probabilities than the proportional method does. That matters because a naive
model finds its "best value" on long prices.

### 5.2 Edge against the price you're paid at

<!-- source: src/staking.py -->
```python
    implied = 1.0 / o
    edge = p_model - implied
```

**Maths: the break-even is the raw price.** A unit stake at decimal odds $o$
returns $o$ with probability $p$, so expected profit is $p\,o - 1$. That is
positive exactly when $p > 1/o$, the raw implied probability. The de-vigged
probability is what the market *thinks*; the raw one is what you're *paid at*.

> **Bug story #4: value tested against the wrong price.** Edge was first
> computed against the de-vigged probability. But payouts happen at the actual
> odds, which include the margin, so bets were placed that had negative
> expected value. Now `edge` against the raw price decides bets, and
> `disagreement` against the fair price is diagnostic only.

### 5.3 Kelly

<!-- source: src/staking.py -->
```python
    return (p * o - 1.0) / (o - 1.0)
```

**Maths: deriving Kelly.** With net odds $b = o - 1$, staking a fraction $f$
grows the bankroll by $1 + fb$ on a win and $1 - f$ on a loss. Expected log
growth is

$$G(f) = p \ln(1 + fb) + (1-p)\ln(1 - f).$$

Setting $G'(f) = \frac{pb}{1+fb} - \frac{1-p}{1-f} = 0$ gives

$$f^* = \frac{pb - (1-p)}{b} = \frac{p\,o - 1}{o - 1}.$$

**Why a quarter.** Near the optimum, $G(f) \approx G(f^*)\left[1 - (1 - f/f^*)^2\right]$.
Quarter Kelly keeps about $1 - 0.75^2 = 43.75\%$ of the maximum growth with
$0.25^2 = 6.25\%$ of the variance. Double Kelly gives roughly zero growth
(exactly zero in this quadratic approximation; on a 40% bet at 3.0 the exact
figures are 44.5% kept and +0.0007 growth at double Kelly). And if
$p$ is overestimated, which a model always risks, full Kelly is already past the
optimum.

> **Trade-off: quarter Kelly plus caps.**
> *Chosen:* 25% Kelly, at most 5% of bankroll per bet and 15% per day.
> *Alternatives:* flat stakes (simple, ignores edge size) or full Kelly
> (growth-optimal only if the probabilities are exactly right).
> *Why:* Kelly assumes you know $p$; a model only estimates it. The caps are
> risk limits, not part of Kelly: no single wrong probability should be able to
> matter much. *(Credit-risk twin: single-name and concentration limits.)*

**Drill.**

- *Why fractional Kelly?* "Kelly maximises growth if your probabilities are
  right. Mine are estimates, and overestimating the edge pushes full Kelly past
  the optimum. A quarter keeps about 44% of the growth with about 6% of the
  variance."
- *Why compare with the raw price, not the fair one?* "You're paid at the raw
  price. Expected profit is $p\,o - 1$, positive only when $p$ beats $1/o$.
  Comparing with the fair price ignores the margin you actually pay."

---

## 6. Walk-forward: the honest test

**What it does.** Steps through time. Every 14 days it refits the model and the
calibration map on matches **strictly before** the window, prices the window,
places bets, and moves on. A guard checks every window of every run.

<!-- source: src/backtest.py -->
```python
    offending = train_dates[train_dates >= window_start]
    if len(offending) > 0:
        raise LookaheadError(
```

The boundary is strict (`>=`): two matches on the same Saturday cannot inform
each other, because neither has happened when the bets are placed.

**The single most useful comparison in the project.** Same model, same
settings, same data:

| | Fitted on everything (in-sample) | Walk-forward |
|---|---|---|
| ROI | +603.26% | −82.60% |
| Yield | +34.07% | −15.84% [−28.57%, −3.47%] |

The only difference is whether the model saw the matches it bet on. That gap
is what a lookahead-free boundary costs an impressive-looking backtest.

> **Trade-off: refit every 14 days.**
> *Chosen:* 14 days, with at least 200 training matches.
> *Alternatives:* refit every match (most current, very slow) or once a season
> (fast, stale).
> *Why:* team strength changes over weeks, not days, and the 13-season run
> already takes 11 minutes. *(Credit-risk twin: out-of-time validation with a
> rolling estimation window.)*

**Drill.**

- *How do you know there's no lookahead?* "The guard runs on every window of
  every real run and raises, not warns. A leak that only appears at one season
  boundary would never show up in a unit test, so the check lives in the
  pipeline itself."
- *What's the most important number in the project?* "The in-sample vs
  walk-forward gap: +603% vs −83% ROI on the same model. It shows exactly what
  testing on data the model has seen is worth."

---

## 7. Closing-line value: was the model right before the market was?

**What it does.** Compares the price taken with the closing price, the market's
best estimate at kick-off. Profit is noisy; CLV is a much faster signal of
whether the model is finding prices the market later agrees were wrong.

**Three ways to measure it.** Odds ratio (taken ÷ closing − 1), beat rate
(share of bets where the price moved your way), and **probability points**, the
authoritative one:

<!-- source: src/clv.py -->
```python
            taken_implied = 1.0 / float(bet["odds"])
            points[position] = float(fair_closing[index]) - taken_implied
```

**Maths: the decomposition.** Probability points compares the closing *fair*
probability with the *raw* price taken, so it charges the bookmaker's margin.
Adding and subtracting the opening fair probability splits it exactly:

$$\underbrace{p^{\text{fair}}_{\text{close}} - p^{\text{raw}}_{\text{taken}}}_{\text{CLV}}
= \underbrace{\left(p^{\text{fair}}_{\text{open}} - p^{\text{raw}}_{\text{taken}}\right)}_{-\text{ margin paid}}
+ \underbrace{\left(p^{\text{fair}}_{\text{close}} - p^{\text{fair}}_{\text{open}}\right)}_{\text{market movement}}$$

<!-- source: src/clv.py -->
```python
    points = with_clv["clv_probability_points"].astype(float)
    margin_paid = (with_clv["market_fair"] - with_clv["market_raw"]).astype(float)
    movement = points - margin_paid           # = fair_close - fair_open, exactly
```

Only the **movement** measures information. On 2 seasons:
**CLV −1.87% = margin −1.89% + movement +0.02% [−0.22%, +0.27%]**. Every bit of
the negative CLV is the margin. On 13 seasons the movement is +0.02%
[−0.05%, +0.09%], pinned near zero. The market moved neither toward nor away
from the model's picks: that's what a model with no information looks like.

> **Bug story #5: the CLV verdict that was inverted.** The odds ratio is
> asymmetric: a halved price scores −50%, a doubled one +100%. Its mean was once
> carried positive by a single +108% outlier while most bets lost to the close.
> Probability points is now authoritative, and disagreement between measures is
> flagged, never resolved in the flattering direction.

> **Bug story #11: negative CLV blamed on the market.** The report said the
> market "moves AWAY from" the model's prices. Decomposed, the movement was
> zero: it was all margin. The verdict is now chosen from the movement. A
> follow-on: if the market moves toward a bet by *less* than the margin, the
> odds ratio is positive while points is negative. That disagreement is the
> margin, not asymmetry, and the report now says which cause applies.

> **Trade-off: CLV as the headline, not profit.**
> *Chosen:* CLV with its decomposition.
> *Alternative:* yield, the number everyone actually cares about.
> *Why:* yield on a few hundred bets has an interval tens of points wide. CLV
> measures every bet against a sharp benchmark, so it has far more power. The
> 13-season yield interval [−15.20%, +3.44%] still includes zero; the CLV
> movement interval is pinned within about a tenth of a point.

**Drill.**

- *Your CLV is −1.87%. So the market moves against you?* "No. CLV splits
  exactly into the margin paid and the market's movement after the bet. The
  margin is −1.89% and the movement is +0.02%, indistinguishable from zero. The
  market didn't move against me; it just never agreed with me."
- *Why is the beat rate only 34%?* "I bet into Bet365, which charges about
  twice Pinnacle's margin, and measure against Pinnacle's close. So my price is
  shorter even when both books agree on the probability. With zero movement,
  the beat rate is measuring the gap between bookmakers."

---

## 8. The model against the market, and the yield

**What it does.** Scores the model's walk-forward probabilities against both
markets' de-vigged probabilities, match by match, on every priced match.

<!-- source: src/benchmark.py -->
```python
    difference = np.asarray(model_scores, float) - np.asarray(market_scores, float)
```

**Maths: why pairing matters.** For match $i$, $d_i = L_i^{\text{model}} - L_i^{\text{market}}$. Then

$$\operatorname{Var}(\bar d) = \frac{\sigma_m^2 + \sigma_k^2 - 2\rho\,\sigma_m\sigma_k}{n}.$$

An upset is expensive for every forecaster, so the two scores are strongly
correlated ($\rho$ high) and the paired variance is far smaller than the
unpaired $(\sigma_m^2 + \sigma_k^2)/n$. A test shows the paired interval is
less than half as wide.

**The result** (13 seasons, 4,722 matches; positive = market better):

| | log-loss | Brier |
|---|---|---|
| vs opening (Bet365) | +0.0248 [+0.0188, +0.0308] | +0.0169 [+0.0129, +0.0209] |
| vs closing (Pinnacle) | +0.0284 [+0.0220, +0.0347] | +0.0192 [+0.0149, +0.0234] |

The opening line is the fair comparison, since it reflects about the same
information as the model, and the model still loses.

### 8.1 The yield, defined once

<!-- source: src/metrics.py -->
```python
        idx = rng.integers(0, len(num), len(num))   # one index -> both columns
        draws[i] = num[idx].sum() / den[idx].sum()
```

**Maths: yield is a ratio of sums.** $\text{yield} = \sum \text{profit}_i / \sum \text{stake}_i$,
a *stake-weighted* average of per-bet returns. The plain mean of
$\text{profit}_i / \text{stake}_i$ weights a 10-unit bet and a 1-unit bet
equally. Resampling whole bets keeps each bet's profit and stake together.

> **Bug story #10: two yields under one name.** On the same 433 bets the
> backtest printed −15.84% (profit ÷ turnover) and the headline printed
> −10.08% (the plain mean). The confidence interval was attached to the wrong
> one and *included* zero. On the right number it excludes zero: **the
> conclusion flipped**. The old test passed throughout because it only checked
> the interval's low end was below its high end, which is true of either. *A
> test that can't tell two candidate answers apart pins neither.*

**Drill.**

- *CLV already said no edge. Why build a second test?* "CLV asks whether my
  bets anticipated the market. The benchmark asks whether my probabilities are
  better than the market's on every match, bet or not. They're different
  questions with the same answer, which makes the conclusion much harder to
  argue with."
- *Your 13-season yield interval includes zero, but ROI is −99.98%. How?* "Yield is the
  average bet; ROI compounds. Betting a quarter of Kelly on a model with no edge
  loses a steady fraction of whatever is left, and over four thousand bets that
  compounds to nothing, even when the average bet isn't significantly negative."

---

## 9. Validating the validator: the simulation study

**What it does.** Builds simulated markets where the truth is known *by
construction*: an efficient book, and books mispriced by 2.0, 4.9 and 8.3
probability points. It then runs the real pipeline on each. If it finds edges
in an efficient book, or misses large ones, nothing else in the project can be
trusted.

**The result.** The pipeline finds an edge once the book is wrong by about
**4.9 probability points** on 760 matches, and reports nothing on an efficient
book or for a model betting random probabilities. So the real-data finding
means "no edge of detectable size", not "no edge".

**The calibration.** How much of the opening price's error the closing price
corrects was **measured, not chosen**: the statistic is the market movement on
the longest 25% of opening prices, with no model involved. Real EPL 2012–25
gives +0.12%; the simulator matches it when the close corrects 40% of the
opening error.

<!-- source: scripts/simulation_study.py -->
```python
                open_error = rng.normal(0, OPENING_NOISE, 3)
```

> **Bug story: the control that passed for the wrong reason.** Decomposing the
> simulated CLV showed a model betting *random numbers* getting the market
> moving toward its bets, +0.46 points. It passed only because the margin hid
> it. Two causes: the simulator loaded margin proportionally while the pipeline
> unwinds it with Shin, and the simulated close corrected 100% of the opening
> error, so any rule that bet long prices profited for free. Both fixed; the
> 4.9-point threshold survived at every setting the real data allows.

**Drill.**

- *How do you know your backtest would find an edge if there was one?* "I
  planted known mispricings in simulated markets. It detects about 5
  probability points on two seasons and reports nothing on an efficient book or
  for a random model."
- *Isn't 40% a fudge factor?* "No, it's measured. I computed a statistic that
  involves no model on 13 real seasons and chose the setting that reproduces
  it. Every value from 20% to 50% fits the data, and the threshold is the same
  across all of them."

---

## 10. Reproducibility and controls

**The fingerprint.** Every run prints a hash of every result-affecting setting.
Same fingerprint, same environment ⇒ same numbers.

**Across machines.** Windows and Linux with identical libraries agreed to two
decimals, but one long-run bet sat exactly on the 2% minimum edge and fell on
opposite sides: 4,309 bets vs 4,310. *A threshold rule turns a rounding-level
difference into a different decision.* So published figures come from one
designated machine, and the docs tests compare within a stated tolerance.

**Controls that refuse, not warn.**

| Control | What it stops |
|---|---|
| Lookahead guard on every window | A leak at one season boundary |
| `make_charts.py` refuses sample data into `docs/charts/` | Sample charts committed as results (bug #8) |
| Docs tests against `numbers.json` | A README figure no command regenerates |
| Notebook test (fingerprint, real data, every cell run) | A stale or sample notebook on GitHub |
| `verify_raw_data.py` copy check | CLV measured against itself |
| Cross-mirror score audit | A wrong join between two sources |

> **Bug story #8: the stamp that was walked past.** Sample charts were once
> committed as results. The ILLUSTRATIVE stamp fired correctly and was ignored.
> The stamp is now a **refusal**: `make_charts.py` won't write into
> `docs/charts/` from sample data. *A control that prevents beats a control that
> warns.* The same principle replaced the notebook's manual "Run All" with
> `scripts/run_notebook.py` after CI caught a notebook saved without outputs.

**Drill.**

- *Your numbers differ between machines. Is the pipeline unreliable?* "No. One
  bet sat exactly on the 2% threshold and floating-point differences put it on
  opposite sides. The point estimate didn't change. Published figures come from
  one machine, and tests compare every quoted number against its output."
- *How do you stop your documentation going stale?* "Every figure in the README
  and methodology is checked by a test against the generated numbers file. If a
  rerun changes a number, CI fails until the document is updated."

---

## 11. The bug list, as one story

Eleven bugs, each pinned by a regression test. The pattern a panel should hear:
**most of them made the results look better than they were.**

| # | Stage | Bug | Direction |
|---|---|---|---|
| 1 | Features | xG conversion rate measured on future matches | flattering |
| 2 | Model | `+inf` objective stalled the optimiser while reporting success | silent |
| 3 | Model | ρ pinned at a bound sized from an impossible 54.6-goal match | silent |
| 4 | Staking | Edge measured against the fair price, not the price paid | flattering |
| 5 | CLV | Odds-ratio mean carried by one outlier | flattering |
| 6 | CLV | CLV by band reported only in the misleading measure | hid a diagnostic |
| 7 | Tests | A regex read a comment as an import | false alarm |
| 8 | Charts | Sample charts committed as results | flattering |
| 9 | Simulation | Decimal odds below 1.00 | impossible input |
| 10 | Metrics | Two yields under one name; interval on the wrong one | flattering |
| 11 | CLV | Negative CLV blamed on the market, not the margin | wrong diagnosis |

**The one-line lesson:** "Every bug that flatters you is one you'll be tempted
not to look for. So I built tests whose job is to catch good news that isn't
true."

---

## 12. Final drill: the hard questions

Answer each in under thirty seconds.

1. **"So the model doesn't work. Why should we care?"** "Because the
   validation does. I can show exactly why it doesn't work and how big an edge
   it could have missed, which is what a validation team does for a model that
   might go into production."
2. **"What would you do next to find an edge?"** "Information the market prices
   late: team news, lineups, injuries. The model only sees past results, which
   the market already has. I'd also test smaller leagues, where the market is
   less efficient."
3. **"What's the weakest part of your work?"** "Two seasons can only rule out an
   edge of about 5 probability points, and the simulator's realism rests on one
   calibrated parameter. The 13-season run tightens the first; the second I
   measured rather than assumed, but it's still one statistic."
4. **"Why should we trust your numbers?"** "Every figure in the documents is
   regenerated by a command and checked by a test, the backtest refuses to run
   with lookahead, and I validated the validator on simulated markets where the
   right answer was known."
5. **"How does this relate to credit risk?"** "Same discipline: a probability
   model, calibration checked against outcomes, out-of-time validation, a
   benchmark against a challenger, and limits on exposure. Swap matches for
   loans and the market for observed defaults."
