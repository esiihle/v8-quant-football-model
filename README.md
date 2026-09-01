# ⚽ V8 — Quantitative Football Model

**A production-style football match-prediction and staking system: statistical goal modelling, calibrated probabilities, Kelly staking, and honest edge measurement via closing-line value.**

![status](https://img.shields.io/badge/status-in%20active%20development-orange)
![python](https://img.shields.io/badge/python-3.11+-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## The problem

Bookmaker odds are an efficient estimate of match probabilities. Beating them is
genuinely hard, and most "profitable" betting models are overfit backtests that
quietly use information they wouldn't have had at bet time. V8 is built to do the
opposite: **model match outcomes rigorously, size bets sensibly, and prove any
edge honestly** — with the same discipline used to validate credit-risk and
quantitative-finance models.

## The method

A clean, one-directional pipeline:

1. **Ingest & validate** — historical results and odds against a strict schema.
2. **Feature engineering** — team attack/defence strengths with **Bayesian
   shrinkage** for thin samples, **expected-goals proxies**, exponential **time
   decay**, and home advantage.
3. **Model** — **Dixon-Coles**: a bivariate Poisson goal model with the
   low-score dependence correction, fitted by time-weighted maximum likelihood.
4. **Calibrate** — isotonic / Platt scaling, validated with reliability diagrams,
   Brier score and log-loss.
5. **Stake** — value detection vs. market-implied probabilities, sized with
   **fractional Kelly**, with full bankroll simulation.
6. **Validate** — **walk-forward backtesting** with strict lookahead guards, and
   **closing-line value (CLV)** as the primary proof of edge.

Full detail in [`docs/OVERVIEW.md`](docs/OVERVIEW.md) and
[`docs/methodology.md`](docs/methodology.md).

## Results

> 🚧 **In active development.** This section will be populated as each stage lands.
> No numbers are reported until they come from a leakage-free walk-forward
> backtest — placeholder metrics would defeat the entire point of the project.

Planned headline outputs:

- **CLV distribution** — the lead chart. Consistent positive CLV is the core
  claim of edge.
- **Reliability diagram** — showing predicted vs. realised probabilities.
- **Bankroll curve** — with drawdowns, under fractional Kelly.
- **Backtest summary table** — ROI, yield, hit rate by confidence band,
  Sharpe-like ratio, max drawdown, with confidence intervals.

## Why this project

Every component is a transferable quant/risk technique in disguise:

| This repo | The real-world skill |
|---|---|
| Dixon-Coles MLE | Parametric modelling of count data |
| Bayesian shrinkage | Regularisation for thin samples |
| Probability calibration | PD calibration in credit risk |
| Kelly staking | Position sizing under uncertainty |
| Closing-line value | Alpha vs. consensus / benchmark |
| Walk-forward + lookahead guards | Out-of-time model validation |

## Repository structure

```
src/            core pipeline (ingest, features, dixon_coles, calibrate, staking, backtest, clv, metrics)
tests/          unit tests for the fragile numerics
scripts/        config-driven end-to-end run
notebooks/      exploration
docs/           OVERVIEW, ROADMAP, methodology
```

## Quickstart

```bash
git clone https://github.com/<your-username>/v8-quant-football-model.git
cd v8-quant-football-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py --config config.yaml
```

*(Runnable end-to-end from the first working milestone — see the roadmap.)*

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the dated, week-by-week build plan
from inception to presentation.

## Author

**Sihle** — BSc Computer Science (Wits), data analyst targeting quantitative and
credit-risk roles. This is a portfolio project built in the open.

## License

MIT — see [`LICENSE`](LICENSE).
