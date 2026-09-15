"""FastAPI pricing-recommendation endpoint."""
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agents.q_pricing_agent import QPricingAgent
from env.pricing_gym_env import MARKDOWN_ACTIONS

MODEL_DIR = ROOT / "models"
app = FastAPI(title="Perishable Dynamic Pricing API", version="1.0.0")
_agent = None


def _load():
    global _agent
    if _agent is None:
        _agent = QPricingAgent.load(MODEL_DIR / "q_pricing_policy.npz")
    return _agent


class PricingRequest(BaseModel):
    days_to_expiry: int
    inventory_units: int


class PricingResponse(BaseModel):
    recommended_markdown_pct: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=PricingResponse)
def recommend(x: PricingRequest):
    agent = _load()
    inv_bucket = min(x.inventory_units, 140) // 8
    state = (min(x.days_to_expiry, 7), inv_bucket)
    action_idx = agent.act(state, greedy=True)
    return PricingResponse(recommended_markdown_pct=MARKDOWN_ACTIONS[action_idx])
