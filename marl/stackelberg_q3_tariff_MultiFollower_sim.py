# stackelberg_q3_tariff_MultiFollowers_sim.py
# Uses a stable summary signal for the leader (majority of follower actions)
# and passes that summary to the leader update, reducing state explosion.

import random

class StackelbergMultiFollowerGame:
    def __init__(self, config):
        self.rounds   = config.get("rounds", 200)
        self.leader   = config["leader"]
        self.followers = config["followers"]
        self.coal_cfg = config.get("coalition", {"enabled": True, "min_gain": 0.0, "protocol": "greedy-pairwise"})
        self.PM       = config.get("payoff_matrix", {
            ("C","C"): (3,3), ("C","D"): (0,5),
            ("D","C"): (5,0), ("D","D"): (1,1)
        })
        self.results = {"rounds": []}
        self._last_followers_summary = random.choice(["C", "D"])  # seed summary

    def leader_payoff(self, leader_a, follower_a_list):
        total = 0.0
        for fa in follower_a_list:
            lp, _ = self.PM[(leader_a, fa)]
            total += lp
        return total / len(follower_a_list)

    def follower_payoffs(self, leader_a, follower_a_list):
        return [self.PM[(leader_a, fa)][1] for fa in follower_a_list]

    def _coalition_step(self, leader_action, proposed_actions):
        if not self.coal_cfg.get("enabled", True) or len(self.followers) <= 1:
            return proposed_actions
        # simple majority alignment among willing participants
        opt_in = []
        for f in self.followers:
            if hasattr(f, "willing_to_coalesce"):
                opt_in.append(bool(f.willing_to_coalesce(leader_action)))
            else:
                opt_in.append(getattr(f, "epsilon", 0.2) < 0.5)
        coalition_idx = [i for i, ok in enumerate(opt_in) if ok]
        if len(coalition_idx) <= 1:
            return proposed_actions
        votes = {"C": 0, "D": 0}
        for i in coalition_idx:
            votes[proposed_actions[i]] += 1
        coalition_action = "D" if votes["D"] >= votes["C"] else "C"
        final_actions = proposed_actions[:]
        for i in coalition_idx:
            # adopt majority if (weakly) not worse
            curr = proposed_actions[i]
            base = self.PM[(leader_action, curr)][1]
            coal = self.PM[(leader_action, coalition_action)][1]
            if coal >= base:
                final_actions[i] = coalition_action
        return final_actions

    def _summarize_followers(self, follower_actions):
        cD = sum(1 for a in follower_actions if a == "D")
        cC = len(follower_actions) - cD
        return "D" if cD >= cC else "C"

    def run_round(self):
        # Leader acts based on summary of previous followers' majority
        leader_state = self._last_followers_summary
        leader_action = self.leader.decide_tariff(leader_state)

        # Followers respond independently
        proposed_actions = [f.respond_to_tariff(leader_action) for f in self.followers]
        follower_actions = self._coalition_step(leader_action, proposed_actions)

        # Payoffs
        Lp = self.leader_payoff(leader_action, follower_actions)
        Fp_list = self.follower_payoffs(leader_action, follower_actions)

        # Learning updates
        if hasattr(self.leader, "update"):
            # Pass the majority summary as the opponent signal
            opp_summary = self._summarize_followers(follower_actions)
            self.leader.update(leader_state, leader_action, opp_summary, Lp)
        for f, fa, fp in zip(self.followers, follower_actions, Fp_list):
            if hasattr(f, "update"):
                f.update(leader_action, fa, leader_action, fp)

        # Log and carry state
        self.results["rounds"].append({
            "leader_action": leader_action,
            "follower_actions": follower_actions[:],
            "leader_payoff": Lp,
            "follower_payoffs": Fp_list[:]
        })
        self._last_followers_summary = self._summarize_followers(follower_actions)

    def run(self):
        for _ in range(self.rounds):
            self.run_round()
        return self.results

