# test_stackelberg_q3_tariff_simplePayoff.py

import matplotlib.pyplot as plt
from stackelberg_q3_tariff_simplePayoff_sim import *
import random

random.seed(42)

def test_stackelberg_tariff():
    # Available agent classes:
    # Leaders: QLearningLeader, GreedyLeader, BluffingLeader
    # Followers: QLearningFollower, BestResponseFollower, PiecewiseFollower, UnpredictableFollower

    config = {
        "rounds": 500,
        #"leader": GreedyLeader(),
        #"leader": BluffingLeader(0.2),
        "leader": Q3LearningLeader(epsilon=0.5, gamma=0.9, alpha=0.5),
        #"follower": BestResponseFollower()
        "follower": Q3LearningFollower(epsilon=0.5, gamma=0.5, alpha=0.5)
    }

    game = StackelbergTariffGame(config)

    leader_q_tracking = {} if hasattr(config["leader"], "q") else None
    follower_q_tracking = {} if hasattr(config["follower"], "q") else None

    for _ in range(config["rounds"]):
        game.run_round()
        if leader_q_tracking is not None:
            # Carry-forward for all existing keys
            for k in list(leader_q_tracking.keys()):
                last = leader_q_tracking[k][-1] if leader_q_tracking[k] else 0.0
                leader_q_tracking[k].append(last)
            # Overwrite with current values and backfill unseen keys
            for k, val in config["leader"].q.items():
                if k not in leader_q_tracking:
                    leader_q_tracking[k] = [0.0] * (len(game.results["rounds"]))
                    leader_q_tracking[k].append(val)
                else:
                    leader_q_tracking[k][-1] = val
        if follower_q_tracking is not None:
            # Carry-forward for all existing keys
            for k in list(follower_q_tracking.keys()):
                last = follower_q_tracking[k][-1] if follower_q_tracking[k] else 0.0
                follower_q_tracking[k].append(last)
            # Overwrite with current values and backfill unseen keys
            for k, val in config["follower"].q.items():
                if k not in follower_q_tracking:
                    follower_q_tracking[k] = [0.0] * (len(game.results["rounds"]))
                    follower_q_tracking[k].append(val)
                else:
                    follower_q_tracking[k][-1] = val

    results = game.results
    rounds = list(range(len(results["rounds"])))
    leader_actions = [r["leader_action"] for r in results["rounds"]]
    follower_actions = [r["follower_action"] for r in results["rounds"]]
    leader_payoffs = [r["leader_payoff"] for r in results["rounds"]]
    follower_payoffs = [r["follower_payoff"] for r in results["rounds"]]
    cumulative_leader = [sum(leader_payoffs[:i+1]) for i in range(len(leader_payoffs))]
    cumulative_follower = [sum(follower_payoffs[:i+1]) for i in range(len(follower_payoffs))]

    num_panels = 3
    if leader_q_tracking: num_panels += 1
    if follower_q_tracking: num_panels += 1

    fig, axs = plt.subplots(num_panels, 1, figsize=(12, 4 * num_panels))
    panel = 0

    # Panel: Leader actions & payoffs
    axs[panel].plot(rounds, leader_payoffs, label="Leader Payoff", marker='o')
    axs[panel].plot(rounds, [1 if a == "D" else 0 for a in leader_actions], label="Leader Action (1=D)", linestyle='--')
    axs[panel].set_title(f"Leader ({type(config['leader']).__name__}): Payoffs and Actions")
    axs[panel].legend()
    panel += 1

    # Panel: Leader Q-learning
    if leader_q_tracking:
        for (s, a), values in leader_q_tracking.items():
            axs[panel].plot(range(len(values)), values, label=f"Q({a}|{s})")
        axs[panel].set_title(f"Leader ({type(config['leader']).__name__}) Q-Values (ε={config['leader'].epsilon}, γ={config['leader'].gamma}, α={config['leader'].alpha})")
        axs[panel].legend()
        panel += 1

    # Panel: Follower actions & payoffs
    axs[panel].plot(rounds, follower_payoffs, label="Follower Payoff", marker='x')
    axs[panel].plot(rounds, [1 if a == "D" else 0 for a in follower_actions], label="Follower Action (1=D)", linestyle='--')
    axs[panel].set_title(f"Follower ({type(config['follower']).__name__}): Payoffs and Actions")
    axs[panel].legend()
    panel += 1

    # Panel: Follower Q-learning
    if follower_q_tracking:
        for (s, a), values in follower_q_tracking.items():
            axs[panel].plot(range(len(values)), values, label=f"Q({a}|{s})")
        axs[panel].set_title(f"Follower ({type(config['follower']).__name__}) Q-Values (ε={config['follower'].epsilon}, γ={config['follower'].gamma}, α={config['follower'].alpha})")
        axs[panel].legend()
        panel += 1

    # Panel: Cumulative payoffs
    axs[panel].plot(rounds, cumulative_leader, label="Cumulative Leader Payoff")
    axs[panel].plot(rounds, cumulative_follower, label="Cumulative Follower Payoff")
    axs[panel].set_title("Cumulative Payoffs")
    axs[panel].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Stackelberg Tariff Sim: Adaptive and Static Agent Analysis")
    #plt.tight_layout()
    plt.savefig("plots/stackelberg_q3_tariff_simplePayoff_diagnostics.png")
    plt.close()




