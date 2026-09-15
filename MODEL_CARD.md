# Model Card — Perishable Dynamic Pricing Policy

## Intended use
Category-manager decision support for markdown scheduling. Guardrails
(bounded action set, price-change-frequency logging) are built into the
environment; this is not intended for fully autonomous pricing without
human override capability.

## Metrics (last run)
See `models/offline_eval_report.json`. This repo's honest finding: the
trained policy achieves dramatic waste reduction (~90-95%, far exceeding
the briefing's ≥18% target) but at a real margin cost (~-5% to -11% vs.
the static baseline in this configuration) — a genuine waste/margin
Pareto tradeoff, not a free lunch. The guardrail reward weights
(`env/pricing_gym_env.py`'s `waste_cost` multiplier) can be tuned toward
the margin-preserving end of that frontier; achieving both targets
*simultaneously*, as the aspirational briefing figures suggest, likely
requires a richer policy (PPO with multi-objective reward shaping) than
this repo's tabular Q-learning agent — see `docs/EXTENDING.md`.

## Limitations
Synthetic elasticity-calibrated demand; a single scalar per-category
elasticity likely understates real-world heterogeneity (weather, local
competition, day-of-week). Always run the offline evaluation on fresh
data before promoting any policy update.
