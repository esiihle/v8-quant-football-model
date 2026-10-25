# Methodology — V8 Quantitative Football Model

> **Living document.** This is the mathematical spine of the project. It is
> scaffolded now and filled in as each phase lands (see `ROADMAP.md`). Each
> section carries a **Status** tag so the reader always knows what is settled and
> what is pending. Nothing here is reported as a *result* until it comes from a
> leakage-free walk-forward backtest.

---

## 1. Notation & data
**Status: 🟡 skeleton — finalised in Phase 1.**

- Matches indexed by $m$; home/away goals $x_m, y_m$; match date $t_m$.
- Teams indexed by $i$; each team has an **attack** parameter $\alpha_i$ and a
  **defence** parameter $\beta_i$. Global **home advantage** $\gamma$.
- Odds: decimal odds $o$ for each outcome; market-implied probability
  $\tilde{p} = 1/o$ (before de-vig); closing odds captured separately for CLV.

### 1.1 Data source (settled 30 Sep 2026)

**football-data.co.uk**, one CSV per league-season, downloaded from
`https://www.football-data.co.uk/mmz4281/{season}/{league}.csv`. It was chosen
over alternatives for one decisive reason: it publishes **both pre-match and
closing odds** in the same file. Closing odds are what make closing-line value
measurable, and CLV is the headline claim of this project — a source without
them would quietly remove the project's main result.

**Coverage at present:** English Premier League (`E0`), seasons 2023/24 and
2024/25. One league first, to prove the pipeline end to end before scaling out.

### 1.2 Fields we rely on

| Group | Columns | Used for |
|---|---|---|
| Identity | `Div`, `Date`, `HomeTeam`, `AwayTeam` | joining, ordering, time-decay weights |
| Result | `FTHG`, `FTAG`, `FTR` | the Dixon-Coles likelihood (§3) |
| Shots | `HS`, `AS`, `HST`, `AST` | xG proxy (§2.3) |
| Odds (bet into) | `B365H`, `B365D`, `B365A` | value detection and staking (§5) |
| Closing odds | `PSCH`, `PSCD`, `PSCA` | closing-line value (§6.3) |

Everything else in the file (half-time scores, cards, corners, referee, the
other bookmakers, Asian handicap lines) is carried through untouched for now and
either used or dropped explicitly in Phase 1 — never dropped by accident.

### 1.3 Known quirks of the source

Observed when loading the real files, and handled in `src/ingest.py`:

- **Dates are day-first**, and the year is 2-digit in some seasons and 4-digit
  in others, so dates are parsed with `dayfirst=True, format="mixed"`.
- **Trailing empty columns and rows** appear in several files (spreadsheet
  padding) and are dropped on load.
- **Column sets differ between seasons** — a column present in one season may be
  absent in another, which shows up as a block of nulls exactly the size of one
  season. The shape report surfaces this rather than hiding it.
- **Occasional missing odds** on individual matches; these rows are simply not
  bettable and will be excluded at the staking stage, not silently imputed.
- Encoding is not always UTF-8, so the loader falls back to latin-1.

### 1.4 The canonical table (Phase 1, 2 Oct 2026)

Raw files are never read by anything downstream. Phase 1 validates and cleans
them into one canonical table (`data/processed/matches.csv`), defined in
`src/canonical.py`:

`MatchID, MatchDate, League, Season, HomeTeam, AwayTeam, HomeGoals, AwayGoals,
Result, HomeShots, AwayShots, HomeShotsOT, AwayShotsOT, OddsH, OddsD, OddsA,
CloseH, CloseD, CloseA`

Rules applied, each counted in the data-quality report:

| Problem | What happens | Why |
|---|---|---|
| Required column missing | run stops | nothing downstream can be trusted |
| Date won't parse | row dropped | a match with no date can't be ordered in time |
| Goals impossible (<0 or >15) | row dropped | a data error, not a result |
| `FTR` disagrees with the goals | **result recomputed**, mismatch counted | goals are the primary record |
| Team name has stray whitespace | trimmed | otherwise one team becomes two |
| Odds outside 1.01–1000 | price voided (NaN), row kept | the match still trains the model; it just isn't bettable |
| Same date + same two teams | duplicate dropped | otherwise that match is double-counted in the likelihood |
| More than 5% of rows unusable | run stops | quiet degradation is worse than a loud failure |

The table is sorted by date, because Phase 6 walks forward through time and an
out-of-order frame is the easiest possible way to leak the future into a
backtest. `MatchID` is built from league, season, date and both teams, so it is
stable across re-downloads and lets a bet in the backtest be traced to a fixture.

Still to be settled: the exact de-vig method (proportional vs. Shin), before
Phase 5 staking.

## 2. Feature engineering  *(built 9 Oct 2026)*

One row of model inputs per match: each team's attack and defence strength,
estimated from recent form, shrunk toward the league average, using goals
blended with a shots-based expected-goals proxy.

### 2.0 The rule that governs the whole stage

**A match's features may use only matches played strictly before it.**

Every estimate takes an `as_of` date and filters on it; the filter lives in one
place (`time_weights`) rather than being trusted to each caller. A model that
has seen the second half of the season while rating a team in October will look
excellent in a backtest and lose money in practice.

This is not theoretical. The first version of this module fitted the xG
conversion rate over the whole file, so an August match was scored partly on
matches played in May. `test_features_do_not_use_the_future` caught it, and
`blend_goals` now takes `history` as a required argument so the mistake is hard
to repeat.

### 2.1 Attack / defence strengths

Relative to the league, where 1.0 is exactly average: attack 1.25 means a quarter
more goals than a typical team; defence 0.80 means a fifth fewer conceded (lower
is better). Home and away appearances are pooled, because home advantage is a
separate parameter fitted in Phase 3 — folding it in here would count it twice.

### 2.2 Bayesian shrinkage

$$\hat{\theta}_i^{\text{shrunk}} = w_i\,\hat{\theta}_i + (1 - w_i)\,\bar{\theta},
\qquad w_i = \frac{n_i}{n_i + k}$$

with $k$ = `features.shrinkage_k` (currently 8). A team with 0 matches is given
the prior exactly; with 8 it sits halfway; with 38 it is 83% its own record.
Trust is earned gradually, with no threshold to argue about.

$n_i$ is the **effective** sample size — the sum of time-decay weights, not a row
count. Ten matches from last season carry far less than ten matches' worth of
evidence, and the shrinkage should know that.

This is the direct analogue of a thin-file adjustment or a low-default-portfolio
treatment in credit risk: same formula, same argument, different quantity.

### 2.3 Expected-goals proxy

Goals are the truth but are rare and noisy — a 1-0 win and a 1-0 loss look
identical in the table and nothing alike on the pitch. Shots on target are far
more plentiful. The feature blends them:

    attack_goals = (1 - w) * goals + w * (shots_on_target * conversion_rate)

with $w$ = `features.xg_weight` (currently 0.5). The conversion rate is measured
**from past matches only**, separately for home and away sides, so it reflects
this league and era rather than a constant from a paper — and so that home
advantage is not smuggled in a second time.

Where shot data is absent (the bundled sample, some older seasons), the feature
falls back to actual goals and says so in the run report rather than inventing a
figure.

### 2.4 Time decay

$$w(\Delta t) = 0.5^{\,\Delta t / H}$$

with $H$ = `features.half_life_days` (currently 180, about half a season). A
match played today counts 1.000, one a season old 0.500, two seasons old 0.250.
Anything dated on or after `as_of` gets weight 0 — that is where the lookahead
guard lives.

### 2.5 Warm-up rows

A match played before either team has `features.min_prior_matches` of history is
flagged `HasSufficientHistory = False`, not deleted. Phase 3 decides whether to
train on them. One place decides what enters the model.

A team never seen before (a promoted side) is given the prior, not a null: an
unknown team is an average team until it shows otherwise.

### 2.6 Output

`data/processed/features.csv`, one row per match, in `FEATURE_COLUMNS` order,
carrying the strengths, the history counts, the outcome, and both odds sets for
Phases 5 and 6.

## 3. The Dixon-Coles model  *(built 10 Oct 2026)*

### 3.1 The model

Each team carries an attack parameter $\alpha_i$ and a defence parameter
$\beta_i$; there is one global home advantage $\gamma$ and one dependence
parameter $\rho$. For a fixture (home $i$, away $j$):

$$\lambda = \exp(\alpha_i + \beta_j + \gamma), \qquad
  \mu     = \exp(\alpha_j + \beta_i)$$

$$P(X = x, Y = y) = \tau_{\rho}(x, y)\;
  \frac{\lambda^{x} e^{-\lambda}}{x!}\,\frac{\mu^{y} e^{-\mu}}{y!}$$

Parameters live on a log scale, so $e^{\alpha}$ reads as a multiplier: attack
1.30 means a team scores 30% more than league average.

### 3.2 The low-score correction

Independent Poisson is almost right, and measurably wrong in one place — the
four lowest scorelines. Real football produces more 0-0 and 1-1 draws and fewer
1-0 and 0-1 results than independence predicts, which §2 of
`notebooks/01_exploration.ipynb` measures on our own data before any model is
fitted. Dixon and Coles (1997) correct it with one parameter:

$$\tau_{\rho}(x,y) = \begin{cases}
  1 - \lambda\mu\rho & x = 0, y = 0 \\
  1 + \lambda\rho    & x = 0, y = 1 \\
  1 + \mu\rho        & x = 1, y = 0 \\
  1 - \rho           & x = 1, y = 1 \\
  1                  & \text{otherwise}
\end{cases}$$

A negative $\rho$ raises 0-0 and 1-1 while lowering 1-0 and 0-1 — the direction
the data asks for. The correction is **mass-preserving**: the four adjustments
cancel exactly, so the distribution still sums to 1. That is a property of the
construction, not of our code, and
`test_tau_correction_preserves_total_probability` checks it to 1e-12.

Validity requires
$\max(-1/\lambda, -1/\mu) \le \rho \le \min(1/(\lambda\mu), 1)$; outside that
range the model assigns a negative probability to a scoreline.

### 3.3 Fitting: penalised, time-weighted maximum likelihood

$$\mathcal{L} = \sum_{m} \phi(t_m)\,\log P(X = x_m, Y = y_m)
  \; - \; \kappa \sum_i (\alpha_i^2 + \beta_i^2)$$

with $\phi$ the exponential time-decay weight from §2.4 and $\kappa$ =
`model.penalty`.

**Identifiability.** $\lambda$ and $\mu$ depend only on sums such as
$\alpha_i + \beta_j$, so adding a constant to every attack and subtracting it
from every defence changes nothing. One constraint fixes it: the attack
parameters are forced to sum to zero, and the optimiser works with $n-1$ free
attacks.

**The penalty is Phase 2's shrinkage, restated.** An L2 term pulls a team's
parameters toward 0, which on this scale is exactly league average, and it binds
hardest where there is least evidence. Same argument as the Bayesian shrinkage in
§2.2, expressed in the language of an optimiser, and the direct analogue of a
ridge-penalised PD model in credit risk.

**Phase 2 seeds the optimiser.** The shrunk strengths are converted to this log
scale and used as the first starting point. A cheap, stable estimate is a far
better start than a flat guess; the phases are complementary, not competing.

**Multi-start.** The fit runs from `model.n_starts` starting points and reports
the spread of the objective across them. Agreement is the evidence that the
optimum is global rather than lucky; disagreement raises a warning in the run
report, because a fit nobody can reproduce is not a fit.

### 3.4 Two bugs worth recording

Both were invisible in the output and both were caught by tests, not by reading
the numbers. The roadmap predicted this phase would be where numerical care was
needed, and it was right.

**An infinite objective breaks the optimiser.** The first version returned
`+inf` for an invalid parameter set. L-BFGS-B estimates its gradient by finite
differences, so a single infinite probe gives `inf - inf = nan`, the gradient
becomes unusable, and the optimiser stops improving *while still reporting
success*. Parameter recovery came back at $r = 0.21$ and nothing raised an
error. An invalid point now costs a large but finite amount that grows with the
size of the violation, so there is always a route back downhill.

**A "safe" bound became a binding constraint.** The $\rho$ box was sized from the
largest $\lambda$ the strength box could theoretically produce —
$e^{1.5+1.5+1.0} \approx 55$ goals a match. That squeezed the box to
$[-0.018,\ 0.0003]$, and $\rho$ sat pinned on the lower edge while the fit
reported success and the starts agreed with each other. The tell was that **the
bias did not shrink as the sample grew**: noise averages out, a binding
constraint does not. The box is now sized from the goals actually observed, the
exact per-fixture constraint is left to the smooth penalty, and the fit reports
`rho_at_bound` so the same failure would announce itself.

### 3.5 From parameters to market probabilities

The fit gives a full score matrix $P(X=x, Y=y)$ truncated at `model.max_goals`
and renormalised for the lost tail. Every market is then a sum over cells:

| Market | Cells |
|---|---|
| home win / draw / away win | $x>y$ / $x=y$ / $x<y$ |
| over / under 2.5 goals | $x+y\ge3$ / $x+y\le2$ |
| both teams to score | $x\ge1 \wedge y\ge1$ |
| correct score | the cell itself |

This is the real payoff of modelling the whole scoreline distribution rather than
the result alone: one fit prices every market, and they are automatically
consistent with one another.

A fixture involving a team absent from the training data is priced at
league-average parameters rather than rejected — a promoted side must be
priceable on debut — and flagged `KnownTeams = False`, so Phase 5 can decline to
stake on a fixture the model does not really know.

## 4. Calibration  *(built 13 Oct 2026)*

A calibrated probability means what it says: of all the matches where the model
said 60%, about 60% should happen.

### 4.1 Why this phase exists

Phase 5 stakes money by comparing the model's probability with the market's. A
probability that is five points too high across a whole band of matches does not
look wrong — **it looks like edge**. The staking rule will bet it, every time,
and lose. Calibration is what earns the right to treat the model's output as a
probability, and therefore the right to compare it to a price.

### 4.2 The discipline that makes it meaningful

A calibration map fitted on the same predictions it is then scored on will always
look like an improvement, because it has seen the answers. So the map is fitted
on the chronologically **earlier** share of matches (`calibration.fit_fraction`)
and judged on the **later** ones.

Not a random split. Calibration is applied forward in time, so it has to be
tested forward in time. `test_in_sample_calibration_flatters_itself`
demonstrates the difference rather than asserting it: fitting and scoring on the
same matches shows a larger gain than the honest split does, every time.

### 4.3 Proper scoring rules

Both are reported, because they punish differently and the difference is
informative.

$$\text{Brier} = \frac{1}{N}\sum_i\sum_k (p_{ik} - o_{ik})^2
\qquad
\text{LogLoss} = -\frac{1}{N}\sum_i \ln p_{i,\text{actual}}$$

A rule is **proper** when the only way to minimise it is to report your true
beliefs — an improper rule can be gamed by shading predictions toward the
extremes.

* **Brier** is bounded and forgiving: being wrong at 99% costs a little under
  twice being wrong at 70%.
* **Log-loss** is unbounded: a 1% probability on something that happens costs
  about 4.6, where Brier charges under 1.

A model that looks fine on Brier and poor on log-loss is making occasional
catastrophic calls, and for a Kelly-staked system that is the whole game — one
confident wrong bet at full stake can end a bankroll.

Every score is reported against a **base-rates-only** predictor. Beating that is
the minimum bar for claiming the model knows anything, and it is easier to fail
than people expect.

### 4.4 The two methods, and letting the data choose

* **Platt scaling** — a one-dimensional logistic regression on the model's
  log-odds. Two parameters: a slope (below 1 corrects overconfidence) and an
  intercept (a level shift). It can only stretch or shift the existing ordering,
  so it cannot overfit much.
* **Isotonic regression** — any non-decreasing step function. Far more flexible,
  so it can fix shapes Platt cannot, and on a small sample it will happily fit
  noise.

`compare_methods` scores both out of sample. On one or two seasons the first real
run came out exactly as the theory predicts:

| method | log-loss improvement (out of sample) |
|---|---|
| Platt | **+0.0018** |
| isotonic | **−0.0129** |

Isotonic made the model *worse*. With ~336 matches in the fitting window, split
three ways across outcomes, its extra freedom fits noise. That is the argument
for Platt on this dataset, measured rather than asserted.

### 4.5 Three outcomes, not two

Calibration is a binary technique; football has three outcomes. Each of home /
draw / away is calibrated separately and the three are renormalised to sum to 1.

That renormalisation is a genuine approximation, so `renormalisation_cost`
measures it: how far the three calibrated probabilities were from summing to 1
before being forced to. A mean total near 1.00 is unremarkable; a large
deviation would mean the per-outcome maps disagree with each other and the
approach needs rethinking rather than patching. The two-sided markets
(over/under, BTTS) need no such approximation.

### 4.6 Expected calibration error

$$\text{ECE} = \sum_b \frac{n_b}{N}\,\big|\bar p_b - \bar o_b\big|$$

The average gap between what the model said and what happened, weighted by band
population. It is a **diagnostic, not a scoring rule**: a model that always
predicts the base rate has near-zero ECE and is useless. It is always read beside
Brier or log-loss, never alone.

The reliability table is always printed with its **counts**, because a gap has to
be judged against sampling noise. A band of 200 matches at a realised rate near
0.5 carries a standard error of 0.035, so a gap of 0.06 there is ordinary while
the same gap in a 2,000-match band would be real. The test checks each band
against three standard errors of its own count rather than a flat tolerance.

### 4.7 What a small improvement means

On data simulated *from* the Dixon-Coles model the calibration gain is tiny, and
it should be: the model is already almost correct, so there is nothing
systematic to remove. On real matches the model is misspecified — real football
is not exactly bivariate Poisson — and the gain should be larger. If it is not,
that is a finding worth reporting rather than tuning away, and the run report
says so in those terms.

## 5. Staking  *(built 14 Oct 2026)*

### 5.1 Two market probabilities, two different jobs

Bookmaker odds do not sum to 100%. That leaves two market probabilities per
outcome, and conflating them is the subtlest error in this pipeline:

| | Meaning | Used for |
|---|---|---|
| **Raw implied**, $1/o$ | what the price pays | **the value test** — EV is positive iff $p > 1/o$ |
| **De-vigged ("fair")** | the market's honest estimate | a diagnostic, and the basis of CLV in §6 |

A bet can beat the de-vigged probability and still lose money, because the payout
is at the actual odds. At odds of 3.50 in a 4.9% book: raw implied 0.2857,
de-vigged fair 0.272. A model saying **0.278** disagrees with the market in the
right direction and has expected value of **−2.7%**.

So the value test is measured against the **raw price**, which is also exactly
where Kelly turns positive — the two agree by construction, and
`test_edge_and_kelly_agree_on_when_to_bet` asserts they never diverge. The
de-vigged probability is reported separately as `disagreement`, and never
triggers a bet.

**This module shipped with that backwards.** The first version tested value
against the de-vigged probability — the permissive direction — and would have
flagged negative-EV bets as value, every time the model agreed slightly more with
fair value than the price allowed.
`test_a_bet_beating_fair_but_not_raw_is_not_value` pins the corrected behaviour.

### 5.2 De-vigging: proportional vs Shin *(the open decision from §1, now settled)*

**Proportional** scales implied probabilities to sum to 1 and assumes the margin
is spread evenly. Bookmakers do not spread it evenly — they load more onto
longshots, because that is where recreational money goes. So proportional
de-vigging **overstates longshot probabilities**.

**Shin (1993)** models the book as containing a proportion $z$ of insider money
and unwinds the margin accordingly:

$$p_i = \frac{\sqrt{z^2 + 4(1-z)\,\pi_i^2/\Pi} - z}{2(1-z)},
\qquad \pi_i = 1/o_i,\ \Pi = \sum_j \pi_j$$

with $z$ solved numerically so the probabilities sum to 1. On a 5.7% book priced
1.45 / 4.2 / 8.0, Shin shades the longshot **down** and the favourite **up**
relative to proportional.

**Shin is the default.** It matters for CLV in §6 rather than for bet selection
(§5.1), but a biased estimate of fair value would bias the headline metric.
Falls back to proportional, with the reason recorded, when the book has no margin
to unwind.

### 5.3 The Kelly criterion, and the assumption underneath it

$$f^{*} = \frac{bp - q}{b} = \frac{po - 1}{o - 1}$$

the fraction maximising expected *logarithmic* growth — verified numerically in
`test_kelly_maximises_expected_log_growth` rather than asserted.

**Kelly assumes $p$ is known exactly.** It is not: $p$ is a fitted model's output
and carries estimation error. The penalty for overestimating it is sharply
asymmetric:

$$E[\ln] = p\ln(1 + f(o-1)) + (1-p)\ln(1-f)$$

* at $f^{*}$, growth is maximised;
* at roughly $2f^{*}$, growth is **zero** — all the edge converted to variance;
* beyond that, the bet loses money *with a real edge in hand*.

**The "2x" figure deserves precision, because it is usually quoted as an identity
and is not one.** It is exact only in the small-edge (Gaussian /
continuous-rebalancing) limit, where growth is $f\mu - f^2\sigma^2/2$ and
vanishes at $f = 2\mu/\sigma^2$. For discrete bets the zero-growth point is
*near* 2x and depends on the odds. Measured in the test suite:

| p | odds | zero growth at |
|---|---|---|
| 0.600 | 2.0 | 1.947× Kelly |
| 0.550 | 2.2 | 1.975× |
| 0.400 | 3.0 | 2.039× |
| 0.250 | 5.0 | 2.098× |
| 0.505 | 2.0 | **2.000×** (tiny edge: the limit) |

The practical conclusion is unchanged. A model whose probabilities are 20% too
optimistic is already staking roughly double what it should at "full Kelly" —
sitting on the zero-growth boundary with a genuine advantage. **That, not variance
reduction, is the argument for fractional Kelly.**
`staking.kelly_fraction` defaults to 0.25, which survives being wrong about $p$
by a factor of four before reaching that boundary.

### 5.4 Risk limits that are not part of Kelly

Kelly sizes each bet as though it were the only one. Two caps are imposed on top,
and they are explicitly *risk management*, not theory:

* `max_stake_fraction` (5%) — no single bet can matter more than this, because a
  single model probability can be badly wrong;
* `max_daily_exposure` (15%) — ten simultaneous bets on one afternoon are not
  independent enough for Kelly's arithmetic to be safe. A crowded day is scaled
  down proportionally, preserving relative sizing.

### 5.5 Bankroll simulation

Two details decide whether the simulation means anything:

**Bets on the same date are sized against the same bankroll.** When Saturday's
bets are placed, Saturday's results are unknown, so sizing the third bet off the
first two's winnings uses information that did not exist.
`test_same_day_bets_are_sized_off_the_same_bankroll` enforces it.

**ROI and yield are different numbers** and are routinely confused: ROI is on the
bankroll, yield is on money staked. A plan can show a large ROI with a thin yield
by churning volume, and the risk profiles are nothing alike. Both are reported,
alongside **maximum drawdown** as a share of the running peak — which is how a
bankroll fall is actually experienced.

### 5.6 What these numbers are not

The staking run uses a model fitted on **all** the data. It is a sanity check
that the plumbing works and the caps bind, not a result. §6's walk-forward
backtest produces the honest version, and the run report says so every time it
prints.

## 6. Backtesting & evaluation  *(built 18 Oct 2026)*

**The honesty layer.** Everything before this phase produced numbers that were,
in one way or another, fitted on data they were then measured against. This is
what turns them into a forecast.

### 6.1 Walk-forward protocol

Step forward through time in refit windows. At each step:

1. fit the Dixon-Coles model on matches **strictly before** the window opens;
2. fit the calibration map on matches strictly before the window opens;
3. price the matches inside the window with that frozen model;
4. stake at the odds that were **available**, never the close;
5. record the odds taken and the closing odds, for CLV.

Then move the window and refit. Nothing inside a window ever influences the model
that priced it. The window is expanding, so each refit sees everything the last
one did plus what has since been played
(`test_every_window_has_more_training_data_than_the_last`).

**Why a cadence rather than refitting every match.** Refitting before every
single match is the purest version and is also what a real bettor could not do —
the model would change between two matches kicking off together.
`backtest.refit_every_n_days` is both honest and realistic: it is what someone
running this weekly would actually have. The cost is a model up to one window
stale when it prices the last match in that window, which is a **conservative**
error, not a flattering one.

### 6.2 The lookahead guard

`assert_no_lookahead` runs on **every window of every run** and raises rather than
warns. It is not a test-only helper: a leak that only appears on real data at a
particular season boundary would never be caught by a unit test with synthetic
fixtures.

The boundary is **strict**. A match on the same date as the window's first match
is excluded from training, because two matches kicking off the same afternoon
cannot inform each other — at the moment the bets are placed, neither has
happened. `test_guard_treats_the_same_day_as_a_leak` pins it.

Three leaks this phase prevents, in increasing order of subtlety:

| Leak | Where it hides |
|---|---|
| **Model** | fitting on matches after the bet — guarded per window |
| **Calibration** | a map fitted on all the data has seen the outcomes it is calibrating; the effect is invisible, probabilities simply look better than they were |
| **Odds** | staking at the closing price, which did not exist when the decision was made — `test_bets_are_struck_at_the_available_odds_not_the_close` checks every recorded price against the source opening price |

### 6.3 Closing-line value — the headline metric

CLV compares the odds taken against the **closing** odds, the last price before
kick-off, after every injury report and team sheet has been absorbed. The closing
line is the market's most efficient estimate, and that is exactly why beating it
means something.

**Why CLV rather than ROI.** ROI is what you earn, so it is the number that
matters — and it is almost useless for deciding whether a model *works*, because
its variance is enormous. A 3% edge into 3.0 odds has a per-bet standard
deviation around 1.4 units; establishing it takes **thousands** of bets. CLV does
not depend on results at all: every bet measures whether the price moved your
way, not whether the ball went in. Removing the largest source of noise means a
few hundred bets say more about edge than a few thousand bets of ROI do.

Three expressions, all reported:

| Expression | Definition | What it is for |
|---|---|---|
| **Odds ratio** | $O_t/O_c - 1$ | intuitive, the usual quote |
| **Probability points** | de-vigged closing prob − taken implied prob | **authoritative** for averaging |
| **Beat rate** | share of bets where the price moved your way | robust, needs no averaging |

**Probability points is authoritative**, because the odds ratio is asymmetric:
a price halving scores −50% while a doubling scores +100%, so its mean is
pulled upward by its own right tail. A handful of large price collapses can
carry the ratio positive while most bets lose to the close. When the two
measures' intervals exclude zero in opposite directions, the report flags it
and never lets the flattering measure stand as the verdict
(`test_disagreeing_clv_measures_are_flagged_not_papered_over`).

> **Correction (25 Oct 2026).** This section previously illustrated the point
> with a worked example (ratio +1.82%, points −0.67%) from a session whose
> numbers were never reproduced from `data/raw/`. When the real 2023–25 run
> was made, the two measures *agreed*, so the example is withdrawn. The
> argument stands on the arithmetic alone. Real figures are in §6.5.

#### Decomposing CLV: margin paid and market movement *(added 25 Oct 2026)*

Probability points compares the closing **fair** probability with the **raw**
implied probability of the price taken, and the raw price includes the
bookmaker's margin. So a bettor with no information at all, whose prices the
market never moves against, still shows negative CLV equal to the margin. An
identity separates the two:

$$
\underbrace{p^{\text{fair}}_{\text{close}} - p^{\text{raw}}_{\text{taken}}}_{\text{CLV points}}
\;=\;
\underbrace{\big(p^{\text{fair}}_{\text{open}} - p^{\text{raw}}_{\text{taken}}\big)}_{-\,\text{margin paid}}
\;+\;
\underbrace{\big(p^{\text{fair}}_{\text{close}} - p^{\text{fair}}_{\text{open}}\big)}_{\text{market movement}}
$$

Only the **movement** measures information: how far the market's fair view
travelled toward the bet once it had heard everything. `clv.decompose_clv`
reports both parts with intervals, and the report's verdict is chosen from the
movement:

| movement interval | verdict |
|---|---|
| excludes zero, positive | real information; whether it covers the margin decides profit |
| includes zero | no detectable information: negative CLV is the margin alone |
| excludes zero, negative | the market corrects *against* the model |

The same decomposition explains a second trap. If the market does move toward
the bets by less than the margin, the price shortens, so the **odds ratio is
positive** while **probability points is negative**. The measures disagree, but
not because of asymmetry. The ratio compares price with price, so the margin
roughly cancels, while points charges it. The report now says which cause
applies (`test_movement_that_does_not_cover_the_margin_is_named`).

The beat rate has a related blind spot when the price taken and the close come
from different books. On the EPL data, Bet365's margin is about 5.4% against
Pinnacle's 2.9% (as `verify_raw_data.py` reports), so Bet365's price is shorter
than Pinnacle's close even when the two books agree on the probability. A beat
rate below 50% then measures the gap between the books, not the market moving.

**CLV by claimed edge** is the sharpest diagnostic available. If the model has
real information, the bets where it claimed the *most* edge should show the
*most* CLV. A flat profile means the edge estimate carries no signal even when
the average is positive: the model is right on average and cannot tell you
where.

### 6.4 Supporting metrics, all with intervals

**Nothing in this project reports a betting metric without a bootstrap
confidence interval.** A yield of +3% over 200 bets and +3% over 20,000 bets are
entirely different claims, and a bare point estimate hides which one you have.
Percentile bootstrap rather than a normal approximation, because bet-level
returns are violently skewed — most bets lose a unit, a few win several.

* **Yield** (profit ÷ turnover) and **ROI** (on bankroll) — different questions,
  routinely confused. Yield is a *ratio of sums*, so it is a stake-weighted
  average of per-bet returns, and its interval comes from
  `metrics.bootstrap_ratio_ci`, which resamples whole bets (profit and stake
  together) and recomputes the ratio. The unweighted mean of per-bet returns is
  reported beside it as **mean return per bet**, a diagnostic. *(Corrected 25
  Oct 2026: until then the headline "yield" was that unweighted mean, while the
  backtest report printed profit ÷ turnover under the same name. On the EPL run
  they were −10.08% and −15.84%; the gap is real information, since it means
  Kelly's larger stakes did worse than its smaller ones.)*
* **Maximum drawdown**, against the running peak, which is how a fall is
  experienced.
* **Sharpe-like ratio**, named that on purpose: no risk-free rate is subtracted,
  and it assumes near-normal returns which bet returns are not. It therefore
  understates tail risk — the risk that actually ends bankrolls — and is read
  beside the drawdown, which makes no such assumption.
* **Performance by claimed edge**, the same diagnostic as for CLV.

### 6.5 What the first honest run said

*Regenerated 25 Oct 2026 from `data/raw/` (EPL 2023/24 and 2024/25, 760
matches), run fingerprint `c6c0bb194dcd9ba2`. The authoritative copy of every
figure is `docs/charts/numbers.json`. The table previously here came from an
earlier session and was never reproduced, so it has been replaced, not
amended.*

| | Phase 5 (model fitted on everything) | Phase 6 (walk-forward) |
|---|---|---|
| bets | 260 | 433 |
| ROI | +603.26% | **−82.60%** |
| yield (profit ÷ turnover) | +34.07% | **−15.84%** [−28.57%, −3.47%] |
| CLV, probability points | — | **−1.87%** [−2.11%, −1.63%] |
| of which: margin paid | — | −1.89% |
| of which: market movement | — | **+0.02%** [−0.22%, +0.27%] |

The in-sample figure is an artefact of a model that had seen the matches it was
betting on. The walk-forward figure is the forecast, and its yield interval
excludes zero, so the loss is established. CLV says the model does not beat the
close, and the decomposition says why: the market's movement after each bet is
indistinguishable from zero, so the negative CLV is the margin and nothing else.
Read against §8.6, this means no edge larger than about 5 probability points.
It does not mean no edge.

Reporting this rather than tuning until it looks better is the deliverable.

## 7. Hardening & reproducibility  *(built 20 Oct 2026)*

Phases 1–6 built a model. Phase 7 makes the claim *"these are the numbers this
code produces"* checkable by someone other than its author. Three mechanisms,
each answering a specific way a quantitative result quietly stops being true.

### 7.1 Pinned dependencies

`requirements.txt` pins every dependency with `==`, not `>=`. The reason is
specific rather than hygienic: a model's output depends on its libraries. A
minor-version change to SciPy's `minimize` can move the L-BFGS-B stopping point,
which moves the fitted rho, which moves every probability downstream. With `>=`,
a clone six months from now installs different libraries and produces different
numbers, and nobody can say which run was wrong.

`test_every_dependency_is_pinned` asserts the `==`. A second test,
`test_requirements_cover_everything_the_code_imports`, walks every module in
`src/` and `scripts/` with `ast`, collects the third-party imports and asserts
each one is declared — the failure mode where code works on the author's machine
because a package happened to be installed there, and fails on a clean clone.

> **A bug in the test itself, worth recording.** The first version found imports
> by matching `^\s*(from|import)\s+(\w+)` against raw lines, and failed with a
> missing dependency called `average` — picked up from the comment
> `# Average ranks so tied scores...` in `src/metrics.py`. A regex cannot tell
> code from prose; a parser can. The test now walks `ast.Import` and
> `ast.ImportFrom` nodes, which by construction only sees real imports.

### 7.2 Continuous integration

`.github/workflows/tests.yml` runs on every push and pull request, on Python
3.11 **and** 3.12 with `fail-fast: false`, so a version-specific break is
visible rather than masked. Each job installs the pinned requirements, runs
`pytest -v --durations=10`, and then runs the pipeline end-to-end on the bundled
sample with `--no-save`.

The pipeline run matters as much as the test suite. Unit tests can all pass
while the stages no longer fit together — a changed return signature, a renamed
column — and the pipeline run is the only check that the thing a reader is
invited to run actually runs.

### 7.3 Structured logging and the run fingerprint

`src/logging_setup.py` separates two outputs that were previously one. The
reports on **stdout** are for a person to read; the log on **stderr** (and
optionally a file, via `--log-file`) is for working out why window 7 produced no
bets. Loggers are namespaced `v8.<module>`, and `configure()` clears existing
handlers so calling it twice does not double every line.

`run_fingerprint(config)` is the reproducibility mechanism. It hashes the
result-affecting config blocks — `seed`, `data`, `schema`, `validation`,
`features`, `model`, `calibration`, `staking`, `backtest` — as sorted JSON and
takes the first 16 characters of the SHA-256. Every run prints it:

```
run fingerprint: c6c0bb194dcd9ba2   (same fingerprint => same numbers)
```

The properties the tests pin:

* **Stable** — the same config hashes identically across runs and processes.
* **Sensitive** — changing `kelly_fraction` from 0.25 to 0.20 changes the hash.
* **Cosmetically invariant** — reordering keys, or editing a comment, does not.
  A fingerprint that changed on a whitespace edit would be ignored within a week.

This converts "it reproduces" from an assertion into something falsifiable. If
a reported ROI and a rerun disagree while the fingerprints match, the code is
non-deterministic and that is a bug. If the fingerprints differ, the settings
differed and the comparison was never valid. Either way the question is
answerable, which it was not before.

**What the fingerprint does not promise** *(added 25 Oct 2026)*. It fixes the
*settings*, not the arithmetic underneath them. The same fingerprint on the
same data, run on Windows and on Linux with identical pinned libraries, gave
identical bet counts, ROI and CLV to two decimal places, but walk-forward
turnover of 5,214.99 against 5,215.11. L-BFGS-B's result depends on the
floating-point behaviour of the platform's maths library, and a difference in
the eighth significant figure of one fitted parameter can nudge a Kelly stake
by a cent. So the precise claim is: *same fingerprint, same environment ⇒ same
numbers; same fingerprint across platforms ⇒ same numbers to reported
precision*. `numbers.json` is therefore generated on one designated machine
(the author's), and the documentation tests compare at the precision the
documents quote.

### 7.4 What is deliberately *not* hardened

* **No retry logic or network code.** The ingest stage reads local files and
  tells the user where to download them. Downloading inside the pipeline would
  make a run depend on a third-party site being up, which is the opposite of
  reproducible.
* **No caching of fitted models.** Refitting is a few seconds and a cache is a
  place for stale parameters to hide — the exact class of bug Phase 6 exists to
  catch.

## 8. Reporting  *(built 21 Oct 2026)*

The last phase produces nothing new about football. It is about the gap between
a result and a reader, which is where quantitative work is usually lost.

### 8.1 Charts as a build artefact, not a drawing

`scripts/make_charts.py` regenerates every chart in the README from one pipeline
run, and writes `docs/charts/numbers.json` alongside them — the headline figures
plus the Phase 7 run fingerprint. The README quotes numbers from that file and
`tests/test_docs.py` checks the two agree, to within half of the last decimal
place quoted.

This closes a loop that is usually left open. A README is the one file nobody
re-reads after the code changes, so its numbers drift out of date silently and
the drift is only ever found by a reader — in this case, an interviewer. Here a
stale number fails the test suite.

The five charts, and the question each answers:

| Chart | Question |
|---|---|
| in-sample vs walk-forward | what does an honest test boundary cost? |
| CLV distribution | is the model beating the closing line? |
| CLV by claimed edge | does the model know *where* it is right? |
| reliability diagram | is the model honest when it says 70%? |
| bankroll curve | what would holding this have felt like? |

### 8.2 The stamp

Charts built on the bundled 12-match sample are meaningless, and a meaningless
chart that looks real is worse than none. When the raw data is absent,
`make_charts.py` falls back to the sample, stamps **ILLUSTRATIVE** across every
chart, and records `"real_data": false` in `numbers.json`. A test
(`test_committed_charts_came_from_real_data`) refuses the commit.

The stamp is applied in `finish()`, which every chart is saved through, so it
cannot be forgotten on one. That is the same structural argument as the
`history` argument in `blend_goals` (§2): make the safe path the only path
rather than remembering to take it.

### 8.3 A chart that was reporting the wrong measure

Building the CLV-by-edge chart surfaced a defect in `clv_by_band`. It aggregated
`mean_clv_ratio` and nothing else — the odds-ratio measure that §6 establishes is
asymmetric and can read positive while the model loses to the close. The single
sharpest diagnostic in the project was only available in the units the project
had already declared misleading.

`clv_by_band` now reports `mean_clv_points` first, and
`test_clv_bands_report_the_authoritative_measure` holds it there. The general
lesson is worth stating plainly: **deciding which measure is authoritative is not
finished until every place that reports it has been changed.** A correct verdict
in one function and the old measure in the function beside it is not a fixed bug;
it is a bug with a narrower blast radius.

### 8.4 Two charting mistakes, both visible only on the rendered image

* **ROI and yield on one axis.** The first version put both on a shared scale.
  ROI ran to +409% and yield to +2.2%, so yield rendered as an invisible sliver
  against the axis. Two measures of different scale get two panels — never two
  y-axes on one chart, which invites the eye to read a relationship that is not
  there. They are now small multiples with independent scales.
* **A layout call that silently disabled the layout engine.**
  `fig.autofmt_xdate()` calls `subplots_adjust` internally, which switches
  constrained layout off; the bankroll chart's subtitle printed straight through
  its title while four other charts were fine. Neither fault raises, logs, or
  fails a unit test. Both were found by opening the PNG and looking at it, which
  is the only check that catches them.

### 8.5 The stamp became a refusal  *(22 Oct 2026)*

§8.2 described the ILLUSTRATIVE stamp as the guard against committing
sample-built charts as results. On 21 October that guard was tested in
production and lost: three stamped charts from the 12-match sample were
committed, and the README was published referencing two more that the sample
run had skipped entirely — so the repository's front page carried three
meaningless charts and two broken images.

Nothing malfunctioned. The stamp appeared, `numbers.json` recorded
`"real_data": false`, and `test_committed_charts_came_from_real_data` would
have failed. **Every control worked and the wrong thing happened anyway**,
because all of them were advisory: a warning is something a person can walk
past, and at the end of a long session a person will.

The guard is now a refusal. `make_charts.py` will not write into `docs/charts/`
from sample data; it exits with a non-zero status and names the two legitimate
alternatives (`--out` somewhere else, or `--allow-sample` asked for by name).
The check runs *before* any computation, from the filesystem alone, so a run
that cannot produce publishable output says so in a second rather than after a
full walk-forward backtest.

The distinction is worth carrying beyond this project:

> **A control that reports a problem is not the same as a control that prevents
> one.** Whether the difference matters depends entirely on whether anyone is
> reading at the moment it fires — which is exactly when they are least likely
> to be.

The same reasoning produced the required `history` argument in `blend_goals`
(§2) and the finite `INVALID_PENALTY` in §3: make the wrong outcome
unrepresentable rather than merely visible.

### 8.6 Validating the instrument  *(built 23 Oct 2026)*

Every section above reports a number. None of them establishes that the
machinery producing those numbers can tell signal from noise — and a backtester
that reports "+4% CLV" is worthless unless it would have reported nothing had
there been nothing there.

`scripts/simulation_study.py` answers that prior question. It generates markets
whose properties are known by construction, because the data-generating process
is written in the file: a bivariate Poisson world with Dixon-Coles dependence,
priced by a bookmaker with a stated margin, estimation noise, and — in the
biased arms — a deliberate understatement of home advantage that the closing
line then corrects toward the truth.

#### The result

| mispricing planted | measured CLV | detected? |
|---|---|---|
| none (efficient book) | −1.59% [−1.77%, −1.38%] | no |
| 2.0 probability points | −0.92% [−1.13%, −0.71%] | no |
| 4.9 probability points | +0.74% [+0.48%, +0.97%] | **yes** |
| 8.3 probability points | +3.34% [+3.03%, +3.64%] | **yes** |
| *control:* random model, efficient book | −1.20% [−1.31%, −1.09%] | no |

Two properties are established, and the second matters more:

1. **The instrument responds.** Measured CLV is monotonic in the planted
   mispricing and crosses zero where it should.
2. **The instrument does not hallucinate.** It reports no edge on an efficient
   book, and none for a model betting random probabilities. A backtester that
   invents edges is worse than one that finds none.

#### Why the threshold is the deliverable

**On 760 matches this pipeline can establish an edge once the book is wrong by
about 5 probability points, and not below that.** That is a statement about
statistical power, and it is what makes any real-data result interpretable: a
measured CLV near zero means *no edge of detectable size*, not *no edge*. Those
are different claims, and only one of them is supportable.

It also sets the bar for the project's own real-data run. If that run reports a
CLV inside ±5 points, the honest reading is that the question is unresolved at
this sample size — which is a more useful thing to be able to say than either
"it works" or "it doesn't".

*Applied 25 Oct 2026 (§6.5):* the real run's market movement was +0.02 points
[−0.22, +0.27], well inside that band. The honest reading is the one written
here in advance: no edge of detectable size on two seasons.

One caution when comparing the two. The simulation's efficient book showed
−1.59 points of CLV and the real run −1.87, and it is tempting to call them
the same. Both are mostly margin, but the simulated book's margin is a setting
in `simulation_study.py` while the real one is Bet365's. The like-for-like
comparison is between the *movement* components, and the simulation does not
yet report its CLV decomposed.

#### A bug the study found in itself

The first version of `price()` could emit decimal odds **below 1.00** — an
impossible price, and one `validation.min_odds` would reject had it arrived in a
real file. The simulator was producing data its own pipeline would refuse.

Caught by `test_the_simulated_odds_form_a_coherent_book`, which applies to the
simulator exactly the checks `verify_raw_data.py` applies to a download: three
prices must imply probabilities summing to a little over one, and the closing
book must be thinner than the opening one. Fixed by capping any single outcome
at 0.92 before the margin is loaded, then renormalising.

The general point is the one running through §8.5 and §10 of the companion
project: **a tool that cannot fail loudly will eventually fail quietly.** A
simulator held to a weaker standard than the data it stands in for is exactly
that kind of tool.

## References
- Dixon, M. J. & Coles, S. G. (1997). *Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market.* Applied Statistics, 46(2).
- Kelly, J. L. (1956). *A New Interpretation of Information Rate.*
- Additional references added as the methodology is written up.
