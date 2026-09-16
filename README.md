# Perishable Dynamic Pricing Policy

**An AI-driven approach to dynamic pricing for perishable goods, balancing profit maximization with waste reduction through reinforcement learning.**

---

## Overview

This repository implements a production-ready dynamic pricing system for perishable goods (dairy, bakery, produce, deli meats, prepared meals) using a combination of:

- **Demand Elasticity Modeling** — Log-log regression to capture price-sensitivity per product category
- **Tabular Q-Learning** — Reinforcement learning agent optimized over discretized (days-to-expiry, inventory) state space
- **Gym-Style Environment** — Custom pricing simulation with guardrails (bounded markdown actions, price-change penalties, waste tracking)
- **REST API** — Real-time pricing recommendations via FastAPI

The system achieves **dramatic waste reduction (~90–95%)** while maintaining acceptable margins, providing category managers with a data-driven markdown scheduling tool that respects operational constraints.

---

## Key Features

### 🤖 Intelligent Decision Engine
- **Tabular Q-Learning Agent** (`q_pricing_agent.py`) trained over 8,000 episodes across 5 product categories
- **State Space**: Discretized (days_to_expiry, inventory_bucket) pairs enabling fast greedy inference
- **Action Space**: Bounded markdown percentages [0%, 10%, 20%, 30%, 40%, 50%] — guaranteeing practical pricing moves

### 📊 Demand Forecasting
- **Per-Category Elasticity Models** (`demand_model.py`) fit via log-log regression on historical price–demand pairs
- **Dynamic Demand Simulation** that responds to markdown decisions and expiry urgency
- Extensible to include weather, competition, and day-of-week signals

### 🛡️ Built-In Guardrails
- **Price-Change Penalties** — Discourage excessive daily repricing (stability constraint)
- **Bounded Action Set** — Prevents extreme markdowns that harm profitability
- **Waste-Cost Tracking** — Full cost attribution for unsold inventory at expiry
- **Human-in-the-Loop Design** — Recommendations require category manager approval; not fully autonomous

### 🚀 Production-Ready API
- **FastAPI Endpoint** (`/recommend`) — Submit (days_to_expiry, inventory_units) → receive optimal markdown %
- **Health Check** — `/health` for deployment monitoring
- **Lazy Model Loading** — Efficient resource usage via singleton agent pattern

### 📈 Honest Performance Reporting
- **Offline Evaluation** against logged baseline policies
- **Pareto Tradeoff Transparency** — Documents margin cost (~-5% to -11%) alongside waste gains
- **Model Card** with intended use, limitations, and deployment guidance

---

## Repository Structure

```
.
├── api.py                          # FastAPI server for pricing recommendations
├── q_pricing_agent.py              # Tabular Q-learning agent (core RL logic)
├── demand_model.py                 # Demand elasticity fitting (log-log regression)
├── pricing_gym_env.py              # Custom Gym-style environment with guardrails
├── train_policy.py                 # Training script (8,000 episodes across categories)
├── offline_policy_eval.py          # Off-policy evaluation against baseline
├── test_pipeline.py                # Unit & integration tests
├── linucb_bandit.py                # (Optional) LinUCB bandit for exploration
├── eda.ipynb                       # Exploratory data analysis notebook
├── models/
│   ├── q_pricing_policy.npz        # Trained Q-values (compressed NumPy)
│   ├── elasticity_models.joblib    # Per-category demand models
│   ├── elasticity_summary.json     # Elasticity coefficients & R² scores
│   ├── training_curve.json         # Episode rewards during training
│   └── offline_eval_report.json    # Evaluation metrics vs. baseline
├── data/
│   └── price_demand_history.csv    # Historical price–demand dataset (synthetic)
├── MODEL_CARD.md                   # Model documentation (intended use, metrics, limitations)
├── EXTENDING.md                    # Roadmap for enhancements (PPO, doubly-robust eval, real data)
├── Pricing_Guardrail_Portfolio.pdf # Business strategy & technical design document
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- `pip` or `conda`

### Install Dependencies

```bash
pip install numpy pandas scikit-learn joblib fastapi uvicorn pydantic
```

For development/testing:
```bash
pip install jupyter pytest black mypy
```

---

## Quick Start

### 1. Train the Demand Model

```bash
python demand_model.py
```

**Output:**
- `models/elasticity_models.joblib` — Fitted demand elasticity per category
- `models/elasticity_summary.json` — Elasticity coefficients & goodness-of-fit

Example output:
```json
{
  "dairy": {
    "elasticity_coef": -0.85,
    "expiry_urgency_coef": 0.32,
    "intercept": 2.45,
    "r2": 0.78
  },
  ...
}
```

### 2. Train the Pricing Policy

```bash
python train_policy.py
```

**Output:**
- `models/q_pricing_policy.npz` — Trained Q-values for all discovered states
- `models/training_curve.json` — Episode rewards (for convergence analysis)

Training runs 8,000 episodes across 5 categories with episodic greedy-ε exploration. Expected runtime: ~5–10 minutes on modern CPU.

### 3. Run Offline Evaluation

```bash
python offline_policy_eval.py
```

**Output:**
- `models/offline_eval_report.json` — Metrics vs. static baseline policy

Example metrics:
```json
{
  "metric": "value",
  "avg_margin": 156.42,
  "avg_waste_cost": 8.51,
  "waste_reduction_vs_baseline": "92%",
  "margin_change_vs_baseline": "-7.3%"
}
```

### 4. Launch the API

```bash
uvicorn api:app --reload --port 8000
```

**Health Check:**
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

**Get Pricing Recommendation:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"days_to_expiry": 2, "inventory_units": 80}'

# {"recommended_markdown_pct": 30}
```

### 5. Run Tests

```bash
pytest test_pipeline.py -v
```

---

## Usage Examples

### Programmatic Agent Usage

```python
from q_pricing_agent import QPricingAgent
from pricing_gym_env import PricingEnv, MARKDOWN_ACTIONS
import joblib

# Load trained components
elasticity_models = joblib.load("models/elasticity_models.joblib")
agent = QPricingAgent.load("models/q_pricing_policy.npz")

# Create environment for a specific category
env = PricingEnv(elasticity_models, category="dairy", initial_inventory=120)
state = env.reset()

# Get recommended markdown (greedy action)
action_idx = agent.act(state, greedy=True)
markdown_pct = MARKDOWN_ACTIONS[action_idx]

print(f"Recommended markdown: {markdown_pct}%")

# Step the environment
state, reward, done, info = env.step(action_idx)
print(f"Margin: ${info['margin']:.2f}, Waste Cost: ${info['waste_cost']:.2f}")
```

### Interactive Notebook

See `eda.ipynb` for:
- Demand elasticity visualization per category
- Training convergence plots
- Offline evaluation results

---

## Technical Deep Dive

### Demand Elasticity Model

The system fits a **log-log demand elasticity model** per product category:

```
log(units_sold) = intercept + elasticity × log(price / base_price) + expiry_urgency × (4 - days_to_expiry)
```

**Interpretation:**
- `elasticity` (~-0.5 to -1.0): % change in quantity for 1% price change
- `expiry_urgency`: Demand boost factor as expiration approaches
- **Fit via**: Ordinary least squares on historical data

### Reinforcement Learning Agent

**State Space:**
- Days to expiry: [1, 2, ..., 7]
- Inventory bucket: [0, 1, ..., 17] (8-unit buckets up to 140 units)
- **Total discovered states**: typically 100–200 (sparse exploration)

**Action Space:**
- Markdown percentages: [0%, 10%, 20%, 30%, 40%, 50%]
- Bounded to ensure operational feasibility

**Training Algorithm:**
- **Method**: Tabular Q-learning with ε-greedy exploration
- **Hyperparameters**:
  - Learning rate (α): 0.12
  - Discount factor (γ): 0.95
  - Initial ε: 0.25 (decayed by 0.995 per episode, floor 0.03)
- **Episodes**: 8,000 (randomized across categories and initial inventories)

**Reward Function:**
```
reward = margin - waste_cost - price_change_penalty
```

Where:
- `margin` = (price − cost) × units_sold
- `waste_cost` = 1.15 × cost × unsold_units (if inventory > 0 at expiry)
- `price_change_penalty` = 2.0 if markdown % change > 20% else 0

### Why Guardrails Matter

1. **Price-Change Penalty**: Prevents thrashing; category managers prefer stable pricing
2. **Bounded Actions**: Keeps markdowns realistic (≤50%); avoids ultra-low prices that harm brand
3. **Waste Cost Tracking**: Bakes in disposal/handling costs, not just lost purchase price
4. **Human Override**: Recommendations are advisory; managers retain veto power

---

## Performance Metrics

| Metric | Value | vs. Static Baseline |
|--------|-------|-------------------|
| Waste Reduction | 90–95% | ↑↑↑ |
| Avg Margin | $156.42 | ↓ 5–11% |
| Stockout Probability | Low | ≈ |
| Price Changes/Episode | ~2–3 | Controlled |

**Key Finding**: The system presents a **genuine Pareto tradeoff** between waste and margin—not a free lunch. Tuning the `waste_cost` multiplier in `pricing_gym_env.py` shifts the frontier; achieving both targets simultaneously likely requires a richer policy (e.g., PPO with multi-objective reward shaping).

---

## Deployment

### Local Development
```bash
python train_policy.py && python offline_policy_eval.py && uvicorn api:app
```

### Docker (Example Dockerfile)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]
```

### Production Considerations
- **Model Versioning**: Tag `q_pricing_policy.npz` with date/commit hash
- **A/B Testing**: Deploy alongside static baseline for holdout control
- **Retraining Schedule**: Weekly/monthly with fresh demand elasticity fits
- **Monitoring**: Log all recommendations + actual outcomes for offline evaluation
- **Fallback**: Keep static markdown schedules as emergency backup

---

## Roadmap & Extensions

See [`EXTENDING.md`](./EXTENDING.md) for detailed enhancement proposals:

1. **PPO with Multi-Objective Reward** — Smoother waste/margin tradeoff via policy gradient methods
2. **Doubly-Robust Off-Policy Evaluation** — Importance-sampling / DR estimators against real logged data (not just simulator)
3. **Real Elasticity Data** — Fit on actual POS transaction data; add weather, competitor prices, day-of-week
4. **Live Competitor Pricing Feed** — Scrape/API integration for dynamic competitive response
5. **Category-Specific Constraints** — Markdown limits, promotional periods, brand protection rules

---

## Model Card

**Intended Use:**  
Category-manager decision support for markdown scheduling. Not intended for fully autonomous pricing without human override capability.

**Metrics (Last Run):**  
See `models/offline_eval_report.json`.

**Limitations:**
- **Synthetic Data**: Demand based on elasticity calibration, not real POS transactions
- **Scalar Elasticity**: Per-category model may understate heterogeneity (weather, local competition, day-of-week effects)
- **Always Validate**: Run offline evaluation on fresh data before promoting any policy update

**Bias & Fairness:**  
Dynamic pricing can reinforce regional price inequality. Ensure pricing recommendations respect business policies on fairness and customer segments.

For full details, see [`MODEL_CARD.md`](./MODEL_CARD.md).

---

## Testing

```bash
# Run full test suite
pytest test_pipeline.py -v --tb=short

# Test specific module
pytest test_pipeline.py::test_agent_load -v

# Coverage report
pytest test_pipeline.py --cov=. --cov-report=html
```

**Test Coverage:**
- Agent save/load functionality
- Environment reset/step mechanics
- Demand model fitting
- API endpoint health & recommendations
- Integration across the full pipeline

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a pull request with a clear description

**Code Style:**
- Black for formatting: `black *.py`
- Type hints encouraged for clarity
- Docstrings for all public functions

---

## References & Inspiration

- **Sutton & Barto (2018)**: *Reinforcement Learning: An Introduction* — foundational Q-learning theory
- **Gallego & Wang (2014)**: Dynamic pricing under demand uncertainty using RL approaches
- **Markdown Optimization in Retail**: Revenue management and waste minimization balance
- **Gym Environment Design**: OpenAI Gym interface conventions

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Contact & Support

- **Author**: Anthony Chinedu Echem  
- **Email**: anthonychineduechem@gmail.com  
- **GitHub**: [@ANTHONY-CHINEDU-ECHEM](https://github.com/ANTHONY-CHINEDU-ECHEM)

For issues, feature requests, or questions, please open an [issue](https://github.com/ANTHONY-CHINEDU-ECHEM/PERISHABLE_DYNAMIC_PRICING/issues) or reach out directly.

---

## Acknowledgments

- Retail pricing domain expertise from [business stakeholders]
- Elasticity calibration methodology informed by academic pricing literature
- Open-source community: scikit-learn, NumPy, FastAPI

---

**Last Updated**: September 2026  
**Status**: Production-Ready (Evaluation Phase)

---

### 📊 Dashboard Preview

![Dynamic Pricing Dashboard](DYNAMIC%20PRICING%20DASHBOARD.jpeg)

*The dashboard visualizes real-time pricing recommendations, waste reduction metrics, and margin impact across product categories.*
