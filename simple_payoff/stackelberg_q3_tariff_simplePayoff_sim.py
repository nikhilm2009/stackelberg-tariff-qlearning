# stackelberg_q3_tariff_simplePayoff_sim.py


import random
import numpy as np

class StackelbergTariffGame:
    def __init__(self, config):
        self.rounds = config.get("rounds", 50)
        self.leader = config.get("leader")
        self.follower = config.get("follower")
        self.results = {"rounds": []}

    def calculate_payoff(self, leader_action, follower_action):
        payoff_matrix = {
            ("C", "C"): (3, 3),
            ("C", "D"): (0, 5),
            ("D", "C"): (5, 0),
            ("D", "D"): (1, 1)
        }
        return payoff_matrix[(leader_action, follower_action)]

    def run_round(self):
        # Leader observes last follower action (or random start)
        state_leader = getattr(self.follower, "last_action", None) or random.choice(["C", "D"])
        leader_action = self.leader.decide_tariff(state_leader)

        # Follower chooses action (may finalize a deferred update from last round)
        follower_action = self.follower.respond_to_tariff(leader_action)

        # Payoffs
        leader_payoff, follower_payoff = self.calculate_payoff(leader_action, follower_action)

        # Learning updates
        if hasattr(self.leader, "update"):
            self.leader.update(state_leader, leader_action, follower_action, leader_payoff)
        if hasattr(self.follower, "update"):
            self.follower.update(leader_action, follower_action, leader_action, follower_payoff)

        # Log
        self.results["rounds"].append({
            "leader_action": leader_action,
            "follower_action": follower_action,
            "leader_payoff": leader_payoff,
            "follower_payoff": follower_payoff
        })

    def run(self):
        for _ in range(self.rounds):
            self.run_round()
        return self.results

class Q3LearningLeader:
    def __init__(self, epsilon=0.1, gamma=0.9, alpha=0.5):
        self.q = {}
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.last_opponent_actions = ["C"] * 3  # track summary C/D
        self.last_action = None

    def get_state(self):
        return tuple(self.last_opponent_actions)

    def _reduce_opponent(self, opponent_action):
        """Map opponent_action (str or iterable[str]) to single 'C'/'D' by majority."""
        if isinstance(opponent_action, str):
            return opponent_action
        try:
            xs = list(opponent_action)
            cD = sum(1 for x in xs if x == "D")
            cC = len(xs) - cD
            return "D" if cD >= cC else "C"
        except Exception:
            return "C"

    def decide_tariff(self, state=None):
        s = self.get_state()
        if random.random() < self.epsilon:
            a = random.choice(["C", "D"])
        else:
            q_values = {a_: self.q.get((s, a_), 0.0) for a_ in ["C", "D"]}
            a = max(q_values, key=q_values.get)
        self.last_action = a
        return a

    def update(self, state, action, opponent_action, reward):
        # Use pre-transition state for the TD update
        pre_state = tuple(self.last_opponent_actions)
        opp_sym = self._reduce_opponent(opponent_action)
        next_hist = self.last_opponent_actions[1:] + [opp_sym]
        next_state = tuple(next_hist)
        max_q_next = max(self.q.get((next_state, a), 0.0) for a in ["C", "D"])
        old_q = self.q.get((pre_state, action), 0.0)
        self.q[(pre_state, action)] = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        # commit transition last
        self.last_opponent_actions = list(next_hist)

# ---------------- Q-Learning Leader (1-step state) ----------------
class QLearningLeader:
    """
    State = follower's last action (or coalition summary) in {'C','D'}.
    Uses pre-state (self.last_state) and next_state = opponent_action.
    """
    def __init__(self, epsilon=0.1, gamma=0.9, alpha=0.5):
        self.q = {}
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.last_state = None
        self.last_action = None

    def decide_tariff(self, state):
        # state is a single 'C'/'D' symbol
        if random.random() < self.epsilon:
            action = random.choice(["C", "D"])
        else:
            q_values = {a: self.q.get((state, a), 0.0) for a in ["C", "D"]}
            action = max(q_values, key=q_values.get)
        self.last_state = state
        self.last_action = action
        return action

    def update(self, state, action, opponent_action, reward):
        pre_state = self.last_state
        next_state = opponent_action  # follower response (or coalition summary)
        max_q_next = max(self.q.get((next_state, a), 0.0) for a in ["C", "D"])
        old_q = self.q.get((pre_state, action), 0.0)
        self.q[(pre_state, action)] = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)

# ---------------- Q-Learning Followers ----------------
class Q3LearningFollower:
    """
    3-step memory of leader actions. Deferred update to use s_{t+1} correctly:
    - On respond_to_tariff(leader_action_t): first finalize the pending update for t-1
      using s_{t+1} built by shifting pre_state with leader_action_t.
    - Then choose action for round t from the updated state that includes leader_action_t.
    - On update(... reward): store (pre_state, action, reward) and defer TD step to next round.
    """
    def __init__(self, epsilon=0.1, gamma=0.9, alpha=0.5):
        self.q = {}
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.last_leader_actions = ["C"] * 3
        self.last_state = tuple(self.last_leader_actions)
        self.last_action = None
        self.pending = None  # (pre_state, action, reward)

    def get_state(self):
        return tuple(self.last_leader_actions)

    def _finalize_pending(self, next_leader_action):
        if not self.pending:
            return
        pre_state, action, reward = self.pending
        # build s_{t+1} by shifting pre_state with the new leader action
        next_state = tuple(list(pre_state)[1:] + [next_leader_action])
        max_q_next = max(self.q.get((next_state, a), 0.0) for a in ["C", "D"])
        old_q = self.q.get((pre_state, action), 0.0)
        self.q[(pre_state, action)] = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        self.pending = None

    def respond_to_tariff(self, leader_action):
        # finalize previous round now that we know s_{t+1}
        self._finalize_pending(leader_action)
        # incorporate current leader action into history to form s_t
        self.last_leader_actions.pop(0)
        self.last_leader_actions.append(leader_action)
        s = self.get_state()
        if random.random() < self.epsilon:
            a = random.choice(["C", "D"])
        else:
            q_values = {x: self.q.get((s, x), 0.0) for x in ["C", "D"]}
            a = max(q_values, key=q_values.get)
        self.last_state = s
        self.last_action = a
        return a

    def update(self, state, action, opponent_action, reward):
        # Defer the TD step to the next round when s_{t+1} is available
        self.pending = (self.last_state, self.last_action, reward)

class QLearningFollower:
    """
    1-step state = current leader action in {'C','D'}. Deferred update like above,
    but next_state is exactly the next leader action provided at the next call.
    """
    def __init__(self, epsilon=0.1, gamma=0.9, alpha=0.5):
        self.q = {}
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.last_state = None
        self.last_action = None
        self.pending = None  # (pre_state, action, reward)

    def _finalize_pending(self, next_leader_action):
        if not self.pending:
            return
        pre_state, action, reward = self.pending
        next_state = next_leader_action
        max_q_next = max(self.q.get((next_state, a), 0.0) for a in ["C", "D"])
        old_q = self.q.get((pre_state, action), 0.0)
        self.q[(pre_state, action)] = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        self.pending = None

    def respond_to_tariff(self, leader_action):
        # finalize previous update using the new state s_{t+1}
        self._finalize_pending(leader_action)
        s = leader_action
        if random.random() < self.epsilon:
            a = random.choice(["C", "D"])
        else:
            q_values = {x: self.q.get((s, x), 0.0) for x in ["C", "D"]}
            a = max(q_values, key=q_values.get)
        self.last_state = s
        self.last_action = a
        return a

    def update(self, state, action, opponent_action, reward):
        self.pending = (self.last_state, self.last_action, reward)

# ---------------- Simple baselines ----------------
class GreedyLeader:
    def decide_tariff(self, state=None):
        return "D"

class BluffingLeader:
    def __init__(self, bluff_prob=0.2):
        self.bluff_prob = bluff_prob
    def decide_tariff(self, state=None):
        return "D" if random.random() < self.bluff_prob else "C"

class BestResponseFollower:
    def respond_to_tariff(self, state):
        return "D" if state == "C" else "C"

class BluffingFollower:
    def __init__(self, bluff_prob=0.2):
        self.bluff_prob = bluff_prob
    def respond_to_tariff(self, state):
        if random.random() < self.bluff_prob:
            return random.choice(["C", "D"])
        return "D" if state == "C" else "C"

class PiecewiseRuleBasedFollower:
    def __init__(self):
        self.history = []
    def respond_to_tariff(self, state):
        self.history.append(state)
        if len(self.history) >= 3 and self.history[-3:] == ["D", "D", "D"]:
            return "D"
        return "C"

