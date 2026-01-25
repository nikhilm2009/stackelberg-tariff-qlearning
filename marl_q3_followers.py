# marl_q3_followers.py

import random

class QLCoalitionFollower:
    """
    Independent Q-learning follower with a simple coalition hook.
    State = (leader_action, coalition_hint)
    """
    def __init__(self, epsilon=0.1, gamma=0.9, alpha=0.5):
        self.q = {}  # ((state), action) -> value
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self._coalition_hint = 1

    def _state(self, leader_action):
        return (leader_action, self._coalition_hint)

    def respond_to_tariff(self, leader_action):
        s = self._state(leader_action)
        if random.random() < self.epsilon:
            a = random.choice(["C", "D"])
        else:
            qC = self.q.get((s, "C"), 0.0)
            qD = self.q.get((s, "D"), 0.0)
            a = "C" if qC >= qD else "D"
        self.last_state = s
        self.last_action = a
        return a

    def update(self, state_signal, action, opponent_action, reward):
        s  = self.last_state
        ns = self._state(state_signal)
        max_q_next = max(self.q.get((ns, "C"), 0.0), self.q.get((ns, "D"), 0.0))
        old_q = self.q.get((s, action), 0.0)
        self.q[(s, action)] = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)

    def willing_to_coalesce(self, leader_action):
        s = self._state(leader_action)
        qC = self.q.get((s, "C"), 0.0)
        qD = self.q.get((s, "D"), 0.0)
        margin = abs(qC - qD)
        return (margin < 0.2) or (self.epsilon < 0.2)

    def set_coalition_hint(self, k: int):
        self._coalition_hint = max(1, int(k))
