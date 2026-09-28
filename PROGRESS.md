# Progress Log — V8 Quantitative Football Model

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus `docs/ROADMAP.md`)
> and you know exactly where the project stands. Newest entry at the top.

**Current phase:** Phase 1 — Data pipeline & validation (next)
**Overall status:** 🟢 Phase 0 scaffolding built; pipeline runs end-to-end; tests green.
**Next action:** Implement strict schema validation in `src/ingest.py` (the
`PHASE 1` marker) — types, date parsing, duplicate fixtures, goal/result checks —
and a data-quality report.

---

## Open decisions
- [ ] **Real data source:** confirm league(s)/seasons to pull (football-data.co.uk
      is the likely source; sample currently stands in for it).
- [ ] **De-vig method** for market-implied probabilities (proportional vs. Shin) —
      settle before Phase 5 staking.

## Phase checklist
- [x] Documentation (OVERVIEW, README, ROADMAP, methodology, LICENSE, .gitignore)
- [x] Phase 0 — Foundations & scaffolding
- [ ] Phase 1 — Data pipeline & validation
- [ ] Phase 2 — Feature engineering
- [ ] Phase 3 — Dixon-Coles model
- [ ] Phase 4 — Probability calibration
- [ ] Phase 5 — Staking (Kelly)
- [ ] Phase 6 — Backtest & CLV
- [ ] Phase 7 — Hardening
- [ ] Phase 8 — Docs & presentation

## How to run (current state)
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py     # loads sample data, prints shape report + stage status
pytest                             # 4 smoke tests, all green
```

---

## Session log

### 2026-09-28 — Phase 0 scaffolding
- Built the runnable skeleton: `config.py` (+ `config.yaml`), implemented
  `ingest.py` (loader + shape report), and stubs for every downstream stage.
- `scripts/run_pipeline.py` runs end-to-end and reports each stage as
  done/pending — it doubles as a live progress indicator that lights up as
  phases land.
- Added `tests/test_smoke.py` (4 tests, green) and a bundled sample dataset.
- Verified: pipeline runs clean on a fresh checkout; tests pass.
- **Next:** Phase 1 — strict schema validation + real data ingestion.

### 2026-09-01 — Project inception
- Created repo and committed all documentation (overview, README, roadmap,
  methodology skeleton, LICENSE, .gitignore).

<!--
Template for each new entry (copy this, newest at the top):

### YYYY-MM-DD — <short title>
- What I did this session:
- Decisions made:
- Anything broken / to revisit:
- Next action:
-->
