"""LinUCB contextual bandit: a simpler, provably-safe exploration baseline
for the initial-deployment phase before switching to the full Q-learning
policy, per the briefing's staged rollout approach."""
import numpy as np


class LinUCBBandit:
    def __init__(self, n_actions, n_features, alpha=1.0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.A = [np.eye(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def act(self, context: np.ndarray) -> int:
        p = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = theta @ context
            bound = self.alpha * np.sqrt(context @ A_inv @ context)
            p.append(mean + bound)
        return int(np.argmax(p))

    def update(self, action, context, reward):
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
