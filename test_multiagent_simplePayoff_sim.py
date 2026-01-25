# exampltest)multiagent_simplePayoff.py
# Uses the V2 classes to stabilize leader Q-learning and plots both dynamics and Q metrics.
# Updated to draw RAW series in light color and MOVING AVERAGES in darker/thicker same color.

from stackelberg_q3_tariff_simplePayoff_sim import Q3LearningLeader
from stackelberg_q3_tariff_MultiFollower_sim import StackelbergMultiFollowerGame
from marl_q3_followers import QLCoalitionFollower

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ----------------------- Helpers -----------------------

def rolling_mean_full(x, w=251):
    """Trailing moving average aligned to x (pads with NaN so lengths match)."""
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < w:
        return x.astype(float)
    core = np.convolve(x, np.ones(w)/w, mode="valid")
    pad = np.full(w-1, np.nan)
    return np.concatenate([pad, core])

def plot_raw_and_ma(ax, x, y, y_ma, label_raw, label_ma,
                    lw_raw=1.0, lw_ma=2.6, alpha_raw=0.22, alpha_ma=0.98,
                    ls_raw='-', ls_ma='-'):
    """Plot raw (faint) + moving average (bold) in the SAME color."""
    (raw_line,) = ax.plot(x, y, ls_raw, linewidth=lw_raw, alpha=alpha_raw, label=label_raw)
    ax.plot(x, y_ma, ls_ma, linewidth=lw_ma, alpha=alpha_ma,
            color=raw_line.get_color(), label=label_ma)
    return raw_line

# Long runs (e.g., 15k) look good with 251–501 window
ROLL_W = 500

# ----------------------- Agents & Game -----------------------
leader = Q3LearningLeader(epsilon=0.1, gamma=0.9, alpha=0.1)
followers = [QLCoalitionFollower(epsilon=0.1, gamma=0.95, alpha=0.1) for _ in range(5)]

config = {
    "rounds": 15000,
    "leader": leader,
    "followers": followers,
    "coalition": {"enabled": True, "min_gain": 0.0, "protocol": "greedy-pairwise"}
}

game = StackelbergMultiFollowerGame(config)

# ----------------------- Tracking -----------------------
leader_q_tracking = {}
follower_q_tracking_avg = {}
coalition_maxq_series = []
coalition_margin_series = []
leader_q_size_series = []
follower_q_size_series = []

for _ in range(config["rounds"]):
    game.run_round()
    # Leader Q
    for k, v in leader.q.items():
        leader_q_tracking.setdefault(k, []).append(v)
    # Followers Q (avg per key)
    vals = {}
    cnts = {}
    maxq_list, margin_list = [], []
    for f in followers:
        q = f.q
        last_state = getattr(f, "last_state", None)
        if last_state is not None:
            qC = q.get((last_state, "C"), 0.0)
            qD = q.get((last_state, "D"), 0.0)
            maxq_list.append(max(qC, qD))
            margin_list.append(abs(qC - qD))
        for k, v in q.items():
            vals[k] = vals.get(k, 0.0) + v
            cnts[k] = cnts.get(k, 0) + 1
    for k, tot in vals.items():
        follower_q_tracking_avg.setdefault(k, []).append(tot / cnts[k])
    coalition_maxq_series.append(float(np.mean(maxq_list)) if maxq_list else 0.0)
    coalition_margin_series.append(float(np.mean(margin_list)) if margin_list else 0.0)

    # Q-table sizes per round
    leader_q_size_series.append(len(leader.q))
    follower_q_size_series.append(np.mean([len(f.q) for f in followers]))

# ----------------------- Assemble results -----------------------
results = game.results
rounds = list(range(len(results["rounds"])))
leader_actions = [r["leader_action"] for r in results["rounds"]]
followers_actions = [r["follower_actions"] for r in results["rounds"]]
leader_payoffs = np.array([r["leader_payoff"] for r in results["rounds"]], dtype=float)
follower_payoffs_list = [r["follower_payoffs"] for r in results["rounds"]]
avg_follower_payoffs = np.array([np.mean(fp) for fp in follower_payoffs_list], dtype=float)

leader_action_bin = np.array([1 if a == "D" else 0 for a in leader_actions])
majority_follower_action = np.array([1 if sum(1 for x in acts if x == 'D') >= len(acts)/2 else 0 for acts in followers_actions])
frac_D_followers = np.array([sum(1 for x in acts if x == 'D')/len(acts) for acts in followers_actions], dtype=float)

# ----------------------- Moving averages -----------------------
leader_action_bin_rm        = rolling_mean_full(leader_action_bin,        w=ROLL_W)
majority_follower_action_rm = rolling_mean_full(majority_follower_action, w=ROLL_W)
frac_D_followers_rm         = rolling_mean_full(frac_D_followers,         w=ROLL_W)

leader_payoffs_rm           = rolling_mean_full(leader_payoffs,           w=ROLL_W)
avg_follower_payoffs_rm     = rolling_mean_full(avg_follower_payoffs,     w=ROLL_W)

coalition_maxq_rm           = rolling_mean_full(coalition_maxq_series,    w=ROLL_W)
coalition_margin_rm         = rolling_mean_full(coalition_margin_series,  w=ROLL_W)

# ----------------------- Summary panels -----------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Actions Over Time
axs[0,0].set_title("Actions Over Time")
plot_raw_and_ma(axs[0,0], rounds, leader_action_bin,        leader_action_bin_rm,
                "Leader (1=D) raw",        f"Leader (1=D) MA({ROLL_W})")
plot_raw_and_ma(axs[0,0], rounds, majority_follower_action, majority_follower_action_rm,
                "Follower Maj (1=D) raw", f"Follower Maj MA({ROLL_W})")
plot_raw_and_ma(axs[0,0], rounds, frac_D_followers,         frac_D_followers_rm,
                "Frac D (followers) raw", f"Frac D MA({ROLL_W})")
axs[0,0].legend()

# Per-Round Payoffs
axs[0,1].set_title("Per-Round Payoffs")
plot_raw_and_ma(axs[0,1], rounds, leader_payoffs,       leader_payoffs_rm,
                "Leader payoff raw",       f"Leader payoff MA({ROLL_W})")
plot_raw_and_ma(axs[0,1], rounds, avg_follower_payoffs, avg_follower_payoffs_rm,
                "Follower avg payoff raw", f"Follower avg payoff MA({ROLL_W})")
axs[0,1].legend()

# Cumulative Payoffs
axs[1,0].plot(rounds, np.cumsum(leader_payoffs), label="Leader Cum")
axs[1,0].plot(rounds, np.cumsum(avg_follower_payoffs), label="Follower Avg Cum")
axs[1,0].set_title("Cumulative Payoffs"); axs[1,0].legend()

# Histogram: # Followers choosing D
axs[1,1].hist([sum(1 for x in acts if x == 'D') for acts in followers_actions],
              bins=range(0, len(followers)+2), align='left', rwidth=0.8)
axs[1,1].set_title("# Followers choosing D")

plt.tight_layout(); plt.savefig("plots/multiAgent_simplePayoff_summary_panels.png"); plt.close()

# ----------------------- Q panels -----------------------

def _top_keys(tracking, topn=6):
    scored = []
    for k, series in tracking.items():
        if not series:
            continue
        scored.append(((len(series), abs(series[-1])), k))
    scored.sort(reverse=True)
    return [k for _, k in scored[:topn]]

fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10))

# Leader Q-values (top)
keys_L = _top_keys(leader_q_tracking, 6)
for k in keys_L:
    s, a = k
    axs2[1,0].plot(leader_q_tracking[k], label=f"L Q({a}|{s})")
axs2[1,0].set_title("Leader Q-values (top)"); axs2[1,0].legend()
if keys_L:
    k_star = max(keys_L, key=lambda kk: leader_q_tracking[kk][-1] if leader_q_tracking[kk] else -1e9)
    s = leader_q_tracking[k_star]
    if s:
        x = len(s) - 1
        y = s[-1]
        axs2[1,0].annotate("higher Q → preferred", xy=(x, y), xytext=(max(0, int(x*0.6)), y),
                           arrowprops=dict(arrowstyle='->'))

# Follower Avg Q-values (top)
keys_F = _top_keys(follower_q_tracking_avg, 6)
for k in keys_F:
    s, a = k
    axs2[0,1].plot(follower_q_tracking_avg[k], label=f"Favg Q({a}|{s})")
axs2[0,1].set_title("Follower Avg Q-values (top)"); axs2[0,1].legend()
if keys_F:
    k_star_f = max(keys_F, key=lambda kk: follower_q_tracking_avg[kk][-1] if follower_q_tracking_avg[kk] else -1e9)
    s = follower_q_tracking_avg[k_star_f]
    if s:
        x = len(s) - 1
        y = s[-1]
        axs2[0,1].annotate("↑ indicates coalition leaning to that action",
                           xy=(x, y), xytext=(max(0, int(x*0.5)), y),
                           arrowprops=dict(arrowstyle='->'))

# Coalition Learning Signals (raw faint + MA bold)
axs2[0,0].set_title("Coalition Learning Signals (Followers)")
plot_raw_and_ma(axs2[0,0], list(range(len(coalition_maxq_series))), coalition_maxq_series, coalition_maxq_rm,
                "Avg maxQ raw",        f"Avg maxQ MA({ROLL_W})")
plot_raw_and_ma(axs2[0,0], list(range(len(coalition_margin_series))), coalition_margin_series, coalition_margin_rm,
                "|Q(C)-Q(D)| raw",     f"|Q(C)-Q(D)| MA({ROLL_W})")
axs2[0,0].legend()
if coalition_maxq_series:
    xm = len(coalition_maxq_series) - 1
    ym = coalition_maxq_series[-1]
    axs2[0,0].annotate("Confidence ↑", xy=(xm, ym), xytext=(max(0, int(xm*0.6)), ym),
                       arrowprops=dict(arrowstyle='->'))
if coalition_margin_series:
    xg = len(coalition_margin_series) - 1
    yg = coalition_margin_series[-1]
    axs2[0,0].annotate("Decisiveness ↑", xy=(xg, yg), xytext=(max(0, int(xg*0.4)), yg),
                       arrowprops=dict(arrowstyle='->'))

# Q-table Growth
axs2[1,1].plot(leader_q_size_series, label="|Q| size (leader)")
axs2[1,1].plot(follower_q_size_series, label="|Q| size (followers avg)")
axs2[1,1].set_title("Q-table Growth"); axs2[1,1].legend()






plt.tight_layout(); plt.savefig("plots/multiAgent_simplePayoff_q_panels.png"); plt.close()

print("Saved: multiAgent_simplePayoff_summary_panels.png, multiAgent_simplePayoff_q_panels.png")

