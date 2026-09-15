"""Custom Gym-style pricing environment: an episode = the life of one SKU
unit-batch from 7 days-to-expiry down to 0. At each day, the agent picks a
markdown percentage (bounded action set); demand responds per the fitted
elasticity model; unsold units at expiry are wasted (full cost loss).
Guardrails: max one price change per day, hard price floor.
"""
import numpy as np

MARKDOWN_ACTIONS = [0, 10, 20, 30, 40, 50]  # % off, bounded action set (guardrail)
MAX_DAYS = 7


class PricingEnv:
    def __init__(self, elasticity_models, category, initial_inventory=40, base_price=8.0, unit_cost=4.0, seed=0):
        self.model = elasticity_models[category]
        self.category = category
        self.initial_inventory = initial_inventory
        self.base_price = base_price
        self.unit_cost = unit_cost
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.day = MAX_DAYS
        self.inventory = self.initial_inventory
        self.last_markdown = 0
        return self._state()

    def _state(self):
        return (self.day, min(self.inventory, 140))

    def discretize(self, state):
        day, inv = state
        inv_bucket = int(inv // 8)
        return (day, inv_bucket)

    def step(self, action_idx):
        markdown_pct = MARKDOWN_ACTIONS[action_idx]
        price = self.base_price * (1 - markdown_pct / 100)
        expiry_urgency = max(0, 4 - self.day)

        log_demand = (self.model.intercept_ + self.model.coef_[0] * np.log(price / self.base_price)
                      + self.model.coef_[1] * expiry_urgency)
        expected_demand = np.exp(log_demand)
        demand = self.rng.poisson(max(0.1, expected_demand))
        units_sold = min(demand, self.inventory)

        margin = (price - self.unit_cost) * units_sold
        self.inventory -= units_sold
        self.day -= 1
        done = self.day <= 0 or self.inventory <= 0

        waste_cost = 0.0
        if done and self.inventory > 0:
            waste_cost = 1.15 * self.unit_cost * self.inventory  # unsold units at expiry = cost loss + disposal/handling

        price_change_penalty = 2.0 if abs(markdown_pct - self.last_markdown) > 20 else 0.0
        self.last_markdown = markdown_pct

        reward = margin - waste_cost - price_change_penalty
        info = {"units_sold": units_sold, "price": price, "margin": margin, "waste_cost": waste_cost,
                "remaining_inventory": self.inventory}
        return self._state(), reward, done, info
