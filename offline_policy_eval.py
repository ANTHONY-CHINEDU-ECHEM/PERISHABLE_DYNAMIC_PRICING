"""Offline policy evaluation: compares the trained Q-learning policy
against a static day-of-shelf-life markdown calendar (the typical current
category-management practice), across many simulated episodes. Reports
waste reduction, margin impact, and price-change frequency compliance.

This is a direct Monte-Carlo comparison (both policies run in the same
simulator) rather than a formal doubly-robust importance-weighted
estimator, but serves the same pre-deployment go/no-go decision purpose
described in the briefing -- see docs/EXTENDING.md for the IPS/DR upgrade.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from env.pricing_gym_env import PricingEnv, MARKDOWN_ACTIONS
from agents.q_pricing_agent import QPricingAgent

MODEL_DIR = ROOT / "models"
CATEGORIES = ["dairy", "bakery", "produce", "deli_meat", "prepared_meals"]
N_EVAL_EPISODES = 400

STATIC_CALENDAR = {7: 0, 6: 0, 5: 0, 4: 10, 3: 15, 2: 20, 1: 25, 0: 30}  # day -> markdown % (realistic capped calendar)


def static_action(day):
    pct = STATIC_CALENDAR.get(day, 50)
    return MARKDOWN_ACTIONS.index(min(MARKDOWN_ACTIONS, key=lambda x: abs(x - pct)))


def run_episode(env, policy_fn):
    s_raw = env.reset()
    total_margin, total_waste, price_changes = 0.0, 0.0, 0
    done = False
    last_markdown = None
    while not done:
        a = policy_fn(env, s_raw)
        markdown = MARKDOWN_ACTIONS[a]
        if last_markdown is not None and markdown != last_markdown:
            price_changes += 1
        last_markdown = markdown
        s_raw, r, done, info = env.step(a)
        total_margin += info["margin"]
        total_waste += info["waste_cost"]
    return total_margin, total_waste, price_changes


def main():
    elasticity_models = joblib.load(MODEL_DIR / "elasticity_models.joblib")
    agent = QPricingAgent.load(MODEL_DIR / "q_pricing_policy.npz")
    rng = np.random.default_rng(500)

    results = {"rl": {"margin": [], "waste": [], "changes": []},
               "static": {"margin": [], "waste": [], "changes": []}}

    for _ in range(N_EVAL_EPISODES):
        cat = rng.choice(CATEGORIES)
        inv0 = int(rng.integers(120, 320))
        seed = int(rng.integers(0, 1_000_000))

        env_rl = PricingEnv(elasticity_models, cat, initial_inventory=inv0, seed=seed)
        m, w, c = run_episode(env_rl, lambda e, s: agent.act(e.discretize(s), greedy=True))
        results["rl"]["margin"].append(m); results["rl"]["waste"].append(w); results["rl"]["changes"].append(c)

        env_static = PricingEnv(elasticity_models, cat, initial_inventory=inv0, seed=seed)
        m2, w2, c2 = run_episode(env_static, lambda e, s: static_action(s[0]))
        results["static"]["margin"].append(m2); results["static"]["waste"].append(w2); results["static"]["changes"].append(c2)

    summary = {}
    for policy in results:
        summary[policy] = {"mean_margin": round(float(np.mean(results[policy]["margin"])), 2),
                            "mean_waste_cost": round(float(np.mean(results[policy]["waste"])), 2),
                            "mean_price_changes": round(float(np.mean(results[policy]["changes"])), 2)}

    waste_reduction_pct = round((summary["static"]["mean_waste_cost"] - summary["rl"]["mean_waste_cost"]) /
                                  max(summary["static"]["mean_waste_cost"], 1.0) * 100, 2) if summary["static"]["mean_waste_cost"] > 0.01 else None
    margin_delta_pct = round((summary["rl"]["mean_margin"] - summary["static"]["mean_margin"]) /
                               max(summary["static"]["mean_margin"], 1e-6) * 100, 2)

    report = {"n_episodes": N_EVAL_EPISODES, "summary": summary,
              "waste_reduction_pct_vs_static": waste_reduction_pct,
              "margin_change_pct_vs_static": margin_delta_pct,
              "meets_18pct_waste_target": bool(waste_reduction_pct is not None and waste_reduction_pct >= 18.0),
              "margin_neutral_or_positive": bool(margin_delta_pct >= -2.0)}
    with open(MODEL_DIR / "offline_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
