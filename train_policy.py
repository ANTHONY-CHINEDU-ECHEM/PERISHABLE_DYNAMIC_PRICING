"""Trains the Q-learning pricing policy across many randomized episodes
spanning all 5 SKU categories, with guardrail constraints (bounded action
set, price-change penalty) baked into the environment reward.
"""
import sys
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from env.pricing_gym_env import PricingEnv
from agents.q_pricing_agent import QPricingAgent

MODEL_DIR = ROOT / "models"
CATEGORIES = ["dairy", "bakery", "produce", "deli_meat", "prepared_meals"]
N_EPISODES = 8000


def main():
    elasticity_models = joblib.load(MODEL_DIR / "elasticity_models.joblib")
    agent = QPricingAgent(seed=42)
    rng = np.random.default_rng(1)
    ep_rewards = []

    for ep in range(N_EPISODES):
        cat = rng.choice(CATEGORIES)
        inv0 = int(rng.integers(120, 320))
        seed = int(rng.integers(0, 1_000_000))
        env = PricingEnv(elasticity_models, cat, initial_inventory=inv0, seed=seed)
        s = env.discretize(env.reset())
        total_r = 0.0
        done = False
        while not done:
            a = agent.act(s)
            s_raw_next, r, done, info = env.step(a)
            s_next = env.discretize(s_raw_next)
            agent.update(s, a, r, s_next, done)
            s = s_next
            total_r += r
        agent.decay_epsilon()
        ep_rewards.append(total_r)
        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1}/{N_EPISODES}  avg_reward(last500)={np.mean(ep_rewards[-500:]):.2f}  eps={agent.epsilon:.3f}")

    agent.save(MODEL_DIR / "q_pricing_policy.npz")
    with open(MODEL_DIR / "training_curve.json", "w") as f:
        json.dump({"episode_rewards": ep_rewards}, f)
    print(f"Training complete. {len(agent.q)} learned states.")


if __name__ == "__main__":
    main()
