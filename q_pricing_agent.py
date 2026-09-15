"""Tabular Q-learning pricing agent over the discretized (day, inventory-
bucket) state space."""
import numpy as np
from collections import defaultdict

N_ACTIONS = 6  # len(MARKDOWN_ACTIONS)


class QPricingAgent:
    def __init__(self, alpha=0.12, gamma=0.95, epsilon=0.25, seed=0):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(N_ACTIONS))

    def act(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        return int(np.argmax(self.q[state]))

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.q[s_next])
        self.q[s][a] += self.alpha * (target - self.q[s][a])

    def decay_epsilon(self, factor=0.995, min_eps=0.03):
        self.epsilon = max(min_eps, self.epsilon * factor)

    def save(self, path):
        keys = list(self.q.keys())
        values = np.array([self.q[k] for k in keys])
        np.savez_compressed(path, keys=np.array(keys, dtype=object), values=values, allow_pickle=True)

    @classmethod
    def load(cls, path):
        agent = cls()
        data = np.load(path, allow_pickle=True)
        for k, v in zip(data["keys"], data["values"]):
            agent.q[tuple(k)] = v
        return agent
