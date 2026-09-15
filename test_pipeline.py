import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "q_pricing_policy.npz"


def test_elasticity_signs_are_negative():
    models = joblib.load(ROOT / "models" / "elasticity_models.joblib")
    for cat, m in models.items():
        assert m.coef_[0] < 0, f"{cat} elasticity should be negative (higher price -> lower demand)"


def test_env_never_sells_more_than_inventory():
    from env.pricing_gym_env import PricingEnv
    models = joblib.load(ROOT / "models" / "elasticity_models.joblib")
    env = PricingEnv(models, "dairy", initial_inventory=30, seed=1)
    env.reset()
    for _ in range(7):
        s, r, done, info = env.step(3)
        assert env.inventory >= 0
        if done:
            break


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run agents/train_policy.py first")
def test_policy_reduces_waste_vs_static():
    import json
    report = json.loads((ROOT / "models" / "offline_eval_report.json").read_text())
    assert report["summary"]["rl"]["mean_waste_cost"] < report["summary"]["static"]["mean_waste_cost"]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run agents/train_policy.py first")
def test_api_recommend():
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from serving.api import app
    client = TestClient(app)
    r = client.post("/recommend", json={"days_to_expiry": 1, "inventory_units": 40})
    assert r.status_code == 200
    assert r.json()["recommended_markdown_pct"] in [0, 10, 20, 30, 40, 50]
