# test_stackelberg_q3_tariff_econ.py
import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------------
# Set deterministic seed
# -------------------------
SEED = 42  # change this number to explore different deterministic runs
random.seed(SEED)
np.random.seed(SEED)

import stackelberg_q3_tariff_econ_sim as sim
print("SIM FILE:", sim.__file__)
print("Has follower extras:",
      hasattr(sim.Q3BinnedFollower, "_x_momentum"),
      hasattr(sim.Q3BinnedFollower, "last_d_str"))

from stackelberg_q3_tariff_econ_sim import (
    EconomicParams,
    EconomicEnvironment,
    make_bins,
    Q3BinnedLeader,
    Q3BinnedFollower,
    BestResponseFollowerEconomic,
    StackelbergTariffGameEconomic,
)

# -----------------------------
# Basic helpers
# -----------------------------
def _rolling_mean(x, win=25):
    if win <= 1:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[win:] - c[:-win]) / float(win)
    if out.size == 0:
        return x.copy()
    pad_val = out[0]
    pad_len = len(x) - len(out)
    return np.concatenate([np.full(pad_len, pad_val), out])

def _to_same_len(arr, n):
    arr = np.asarray(arr, float)
    if len(arr) == n:
        return arr
    if len(arr) == 0:
        return np.zeros(n, dtype=float)
    if len(arr) > n:
        return arr[:n]
    pad = np.full(n - len(arr), arr[-1], dtype=float)
    return np.concatenate([arr, pad])

def tail_present_value(rewards, delta):
    """Tail PV from round t onward: sum_{k=t}^{T-1} delta^{k-t} r_k."""
    r = np.asarray(rewards, dtype=float)
    T = len(r)
    tail = np.zeros(T, dtype=float)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = r[t] + delta * acc
        tail[t] = acc
    return tail

def cumulative_discounted(rewards, delta):
    """Running sum: sum_{k=0}^{t} delta^k r_k."""
    r = np.asarray(rewards, dtype=float)
    out = np.zeros_like(r)
    acc = 0.0
    pow_ = 1.0
    for t in range(len(r)):
        acc += pow_ * r[t]
        out[t] = acc
        pow_ *= delta
    return out

# -----------------------------
# Robust plotting helpers
# -----------------------------
def safe_imshow(ax, mat, title="", cmap="viridis", percent_clip=(5, 95)):
    """
    Robust imshow that gracefully handles empty/degenerate matrices and uses
    percentile clipping to avoid washed-out color scales.
    """
    mat = np.asarray(mat, float)

    # Handle empty
    if mat.size == 0 or mat.shape[0] == 0 or mat.shape[1] == 0:
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
        ax.set_axis_off()
        return None

    vmin, vmax = None, None
    if percent_clip is not None:
        lo, hi = np.percentile(mat, [percent_clip[0], percent_clip[1]])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            vmin, vmax = lo, hi

    im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    return im

def prepare_q_heatmap(agent, actions_sorted=None, top_k=50):
    """
    Build a Q(s,a) matrix using the agent's most-visited states.
    Returns (mat, state_labels, actions_sorted).
    """
    if not hasattr(agent, "q") or agent.q is None or len(agent.q) == 0:
        return np.zeros((0, 0)), [], []

    # Default actions from agent
    if actions_sorted is None:
        actions_sorted = getattr(agent, "tau_bins", None) or getattr(agent, "d_bins", None)
    actions_sorted = list(actions_sorted) if actions_sorted is not None else []

    if len(actions_sorted) == 0:
        return np.zeros((0, 0)), [], []

    # Rank states by visits (fallback to keys in Q if state_visits empty)
    if hasattr(agent, "state_visits") and len(agent.state_visits) > 0:
        state_counts = sorted(agent.state_visits.items(), key=lambda kv: kv[1], reverse=True)
        states_sorted = [s for s, _ in state_counts[:top_k]]
    else:
        states_sorted, seen = [], set()
        for (s, _a) in agent.q.keys():
            if s not in seen:
                seen.add(s)
                states_sorted.append(s)
        states_sorted = states_sorted[:top_k]

    # Build matrix
    mat = np.zeros((len(states_sorted), len(actions_sorted)), dtype=float)
    for r, s in enumerate(states_sorted):
        for c, a in enumerate(actions_sorted):
            mat[r, c] = agent.q.get((s, a), 0.0)

    # Compact labels like "0.10|0.10|0.10|0.35|L|H"
    def _lab(s):
        try:
            return "|".join(s)
        except Exception:
            return str(s)

    labels = [_lab(s) for s in states_sorted]
    return mat, labels, actions_sorted

# -----------------------------
# Configurable experiment block
# -----------------------------
# Leader-favoring scenario (U.S. advantage)
#def build_params(tau_max=0.35, d_max=0.25, trade_form="linear"):
#    return EconomicParams(
#        trade_form="power",
#        M0=450.0,
#        X0=500.0,
#        demand_elast=1.10,   # ↓ sensitivity → τ hurts M less → stronger tariff revenue
#        supply_elast=0.90,   # ↓ export response to d → follower gains less
#        kappa=0.35,          # ↓ M adjusts slower → revenue persists
#        lam=0.40,            # ↓ X adjusts slower → follower gains slower
#        delta=0.98,
#        phi_inflation=0.35,  # ↑ imported-inflation penalty → discourages d
#        leader_cost_w=0.008, # ↓ softer tariff cost
#        follower_cost_w=0.010,# ↑ heavier depreciation/admin cost
#        tau_max=0.35,        # moderate tariff headroom
#        d_max=0.18           # ↓ caps follower’s depreciation
#    )

# Follower-favoring scenario (China/India advantage)
def build_params(tau_max=0.35, d_max=0.25, trade_form="linear"):
    return EconomicParams(
        trade_form="power",
        M0=450.0,
        X0=500.0,
        demand_elast=1.80,   # ↑ sensitivity → τ cuts M quickly → weakens tariff revenue
        supply_elast=1.90,   # ↑ strong export response to d
        kappa=0.60,          # ↑ M adjusts faster → revenue erodes quickly after τ
        lam=0.55,            # ↑ X adjusts faster → follower gains arrive quickly
        delta=0.98,
        phi_inflation=0.20,  # ↓ lower imported-inflation bite from d
        leader_cost_w=0.015, # ↑ higher political/admin cost to τ
        follower_cost_w=0.004,# ↓ cheaper to maintain depreciation
        tau_max=0.25,        # ↓ political/feasible cap on τ
        d_max=0.35           # ↑ more room to depreciate
    )

# def build_params(tau_max=0.35, d_max=0.25, trade_form="linear"):
#     return EconomicParams(
#         M0=450.0,
#         X0=500.0,
#         demand_elast=1.50,
#         supply_elast=1.60,
#         kappa=0.50,
#         lam=0.45,
#         delta=0.98,
#         phi_inflation=0.5,
#         leader_cost_w=0.010,
#         follower_cost_w=0.006,
#         tau_max=tau_max,
#         d_max=d_max,
#     )

def make_uniform_bins(vmin, vmax, n=6):
    # inclusive endpoints
    return make_bins(n, vmin, vmax)

def test_stackelberg_q3_tariff_econ(
    trade_form="linear",
    steps=20000,
    tau_max=0.35,
    d_max=0.25,
    n_bins=6,
    leader_q_init=1800.0,
    follower_q_init=3000.0,
    epsilon_start=0.15,
    alpha=0.18,
    follower_double_q=True,
    smooth_win=25,
):
    # --- Params & bins
    params = build_params(tau_max=tau_max, d_max=d_max, trade_form=trade_form)
    suffix = f"_{params.trade_form}"
    tau_bins = make_uniform_bins(0.0, params.tau_max, n=n_bins)
    d_bins = make_uniform_bins(0.0, params.d_max, n=n_bins)

    # --- Environment
    env = EconomicEnvironment(params)

    # --- Agents
    leader = Q3BinnedLeader(
        tau_bins=tau_bins,
        epsilon=epsilon_start,
        gamma=params.delta,
        alpha=alpha,
        q_init=leader_q_init,  # optimistic init
    )
    follower = Q3BinnedFollower(
        d_bins=d_bins,
        epsilon=epsilon_start,
        gamma=params.delta,alpha=alpha,
        double_q=follower_double_q,
        q_init_f=follower_q_init,  
    )
    # Alt: rule-based follower
    #follower = BestResponseFollowerEconomic(d_bins)

    # --- Orchestrate & run
    game = StackelbergTariffGameEconomic(env, leader, follower, track=True)
    game.run(rounds=steps)

    # --- Extract traces
    rounds = [r["round"] for r in game.results["rounds"]]
    R = np.asarray(rounds, dtype=int)
    tau_trace = np.asarray([r["tau"] for r in game.results["rounds"]], float)
    d_trace = np.asarray([r["d"] for r in game.results["rounds"]], float)
    Lpay = np.asarray([r["leader_pay"] for r in game.results["rounds"]], float)
    Fpay = np.asarray([r["follower_pay"] for r in game.results["rounds"]], float)
    M_trace = np.asarray([r["M"] for r in game.results["rounds"]], float)
    X_trace = np.asarray([r["X"] for r in game.results["rounds"]], float)

    # Decomposition pieces
    rev_trace        = np.asarray([r["rev"] for r in game.results["rounds"]], float)
    cons_loss_trace  = np.asarray([r["cons_loss"] for r in game.results["rounds"]], float)
    aCostL_trace     = np.asarray([r["action_cost_L"] for r in game.results["rounds"]], float)
    export_gain_tr   = np.asarray([r["export_gain"] for r in game.results["rounds"]], float)
    infl_cost_tr     = np.asarray([r["infl_cost"] for r in game.results["rounds"]], float)
    aCostF_trace     = np.asarray([r["action_cost_F"] for r in game.results["rounds"]], float)

    # --- Tail PV & cumulative discounted
    L_tailPV = tail_present_value(Lpay, params.delta)
    F_tailPV = tail_present_value(Fpay, params.delta)
    L_cumdisc = cumulative_discounted(Lpay, params.delta)
    F_cumdisc = cumulative_discounted(Fpay, params.delta)

    # --- Leader/Follower Q traces
    V_leader_raw = _to_same_len(getattr(leader, "q_max_trace", []), len(R))
    V_follower_raw = _to_same_len(getattr(follower, "q_max_trace", []), len(R)) if isinstance(follower, Q3BinnedFollower) else None

    # Q-gap traces
    gap_leader_raw = _to_same_len(getattr(leader, "qgap_trace", []), len(R))
    gap_follower_raw = _to_same_len(getattr(follower, "qgap_trace", []), len(R)) if isinstance(follower, Q3BinnedFollower) else None

    roll = lambda x: _to_same_len(_rolling_mean(x, win=smooth_win), len(R))
    # ==========================================================
    # FIGURE 1: Dynamics overview
    #   1a Actions (τ, d) + rolling averages
    #   1b Trade flows (M, X) + rolling averages
    #   1c Cumulative payoffs: discounted vs undiscounted (+ PV limit)
    # ==========================================================
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))

    # rolling window
    win = 500

    tau_roll = _rolling_mean(tau_trace, win)
    d_roll   = _rolling_mean(d_trace, win)
    axs1[0].plot(R, tau_trace, label="Leader τ", linewidth=1.2, alpha=0.6)
    axs1[0].plot(R, d_trace, label="Follower d", linewidth=1.2, alpha=0.6)
    axs1[0].plot(R, tau_roll, "r", label=f"τ (roll mean, {win})", linewidth=2)
    axs1[0].plot(R, d_roll, "g", label=f"d (roll mean, {win})", linewidth=2)
    axs1[0].set_title("Fig 1a — Actions with rolling averages")
    axs1[0].set_xlabel("Round"); axs1[0].set_ylabel("Action level")
    axs1[0].grid(True); axs1[0].legend()

    # 1b Trade flows
    M_roll = _rolling_mean(M_trace, win)
    X_roll = _rolling_mean(X_trace, win)
    axs1[1].plot(R, M_trace, label="Imports M", linewidth=1.2, alpha=0.6)
    axs1[1].plot(R, X_trace, label="Exports X", linewidth=1.2, alpha=0.6)
    axs1[1].plot(R, M_roll, "r", label=f"M (roll mean, {win})", linewidth=2)
    axs1[1].plot(R, X_roll, "g", label=f"X (roll mean, {win})", linewidth=2)
    axs1[1].set_title("Fig 1b — Trade flows with rolling averages")
    axs1[1].set_xlabel("Round"); axs1[1].set_ylabel("Level")
    axs1[1].grid(True); axs1[1].legend()

    # 1c Cumulative payoffs with twin y-axis
    L_undisc = np.cumsum(Lpay)
    F_undisc = np.cumsum(Fpay)

    ax = axs1[2]

    # discounted curves on left axis
    l1 = ax.plot(R, L_cumdisc, label="Leader cumulative (discounted)", linewidth=2)
    l2 = ax.plot(R, F_cumdisc, label="Follower cumulative (discounted)", linewidth=2, alpha=0.85)

    # PV-limit overlays
    #mean_L = float(np.mean(Lpay)); mean_F = float(np.mean(Fpay))
    #pv_L = mean_L / (1.0 - params.delta); pv_F = mean_F / (1.0 - params.delta)
    #ax.axhline(pv_L, color="k", linestyle=":", linewidth=1, alpha=0.6,
    #        label=f"Leader PV limit ≈ {pv_L:.0f}")
    #ax.axhline(pv_F, color="k", linestyle="--", linewidth=1, alpha=0.6,
    #        label=f"Follower PV limit ≈ {pv_F:.0f}")

    ax.set_ylabel("Discounted cumulative payoff")
    ax.set_title("Fig 1c — Cumulative payoffs (disc. vs undisc.)")
    ax.set_xlabel("Round")
    ax.grid(True)

    # undiscounted curves on right axis
    ax2 = ax.twinx()
    l3 = ax2.plot(R, L_undisc, "--", color="tab:red",
                label="Leader cumulative (undisc.)", linewidth=1.6)
    l4 = ax2.plot(R, F_undisc, "--", color="tab:green",
                label="Follower cumulative (undisc.)", linewidth=1.6, alpha=0.9)
    ax2.set_ylabel("Undiscounted cumulative payoff")

    # combine legends from both axes
    lines = l1 + l2 + l3 + l4
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"plots/stackelberg_q3_econ_fig1{suffix}.png")
    plt.close(fig1)

    # ==========================================================
    # FIGURE 2: Payoff decompositions with rolling averages
    #   2a Leader: revenue, consumer loss, action cost (+ rolling)
    #   2b Follower: export gain, inflation cost, action cost (+ rolling)
    # ==========================================================
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))

    win_dec = 500  # rolling window length

    # --- Leader decomposition ---
    rev_roll       = _rolling_mean(rev_trace,        win_dec)
    cons_loss_roll = _rolling_mean(cons_loss_trace,  win_dec)
    aCostL_roll    = _rolling_mean(aCostL_trace,     win_dec)

    # raw (lighter)
    axs2[0].plot(R, rev_trace,       alpha=0.35, linewidth=1.0, label="Revenue τ·M (raw)")
    axs2[0].plot(R, cons_loss_trace, alpha=0.35, linewidth=1.0, label="Consumer loss (raw)")
    axs2[0].plot(R, aCostL_trace,    alpha=0.35, linewidth=1.0, label="Action cost (τ²) (raw)")
    # rolling (emphasized)
    axs2[0].plot(R, rev_roll,       linewidth=2.0, label=f"Revenue τ·M (roll {win_dec})")
    axs2[0].plot(R, cons_loss_roll, linewidth=2.0, label=f"Consumer loss (roll {win_dec})")
    axs2[0].plot(R, aCostL_roll,    linewidth=2.0, label=f"Action cost (τ²) (roll {win_dec})")

    axs2[0].set_title("Fig 2a — Leader payoff decomposition (raw + rolling)")
    axs2[0].set_xlabel("Round"); axs2[0].set_ylabel("Per-step component")
    axs2[0].grid(True); axs2[0].legend(ncol=2, fontsize=9)

    # --- Follower decomposition ---
    export_gain_roll = _rolling_mean(export_gain_tr, win_dec)
    infl_cost_roll   = _rolling_mean(infl_cost_tr,   win_dec)
    aCostF_roll      = _rolling_mean(aCostF_trace,   win_dec)

    # raw (lighter)
    axs2[1].plot(R, export_gain_tr, alpha=0.35, linewidth=1.0, label="Export gain (X − X0) (raw)")
    axs2[1].plot(R, infl_cost_tr,   alpha=0.35, linewidth=1.0, label="Inflation cost (φ·d·M0) (raw)")
    axs2[1].plot(R, aCostF_trace,   alpha=0.35, linewidth=1.0, label="Action cost (d²) (raw)")
    # rolling (emphasized)
    axs2[1].plot(R, export_gain_roll, linewidth=2.0, label=f"Export gain (roll {win_dec})")
    axs2[1].plot(R, infl_cost_roll,   linewidth=2.0, label=f"Inflation cost (roll {win_dec})")
    axs2[1].plot(R, aCostF_roll,      linewidth=2.0, label=f"Action cost (d²) (roll {win_dec})")

    axs2[1].set_title("Fig 2b — Follower payoff decomposition (raw + rolling)")
    axs2[1].set_xlabel("Round"); axs2[1].set_ylabel("Per-step component")
    axs2[1].grid(True); axs2[1].legend(ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"plots/stackelberg_q3_econ_fig2_decompositions{suffix}.png")
    plt.close(fig2)



    # ==========================================================
    # FIGURE 3: Q tables & Q-gaps  (kept, with follower panels added)
    #   3a Leader Q(s,a) heatmap
    #   3b Leader Q-gap
    #   3c Follower Q(s,a) heatmap
    #   3d Follower Q-gap
    # ==========================================================
    fig3, axs3 = plt.subplots(2, 2, figsize=(14, 10))

    # 3a Leader Q heatmap
    leader_Q_mat, leader_states, leader_actions = prepare_q_heatmap(
        leader, actions_sorted=getattr(leader, "tau_bins", None), top_k=40
    )
    im_a = safe_imshow(axs3[0, 0], leader_Q_mat, title="Fig 3a — Leader Q(s,a) by most-visited states")
    if im_a is not None:
        plt.colorbar(im_a, ax=axs3[0, 0], fraction=0.046, pad=0.04)
        axs3[0, 0].set_yticks(range(0, len(leader_states), max(1, len(leader_states)//20 or 1)))
        axs3[0, 0].set_yticklabels([leader_states[i] for i in axs3[0, 0].get_yticks().astype(int)])
        axs3[0, 0].set_xticks(range(len(leader_actions)))
        axs3[0, 0].set_xticklabels([f"{a:.2f}" for a in leader_actions], rotation=45, ha="right")

    # 3b Leader Q-gap
    axs3[0, 1].plot(R, gap_leader_raw, "--", alpha=0.55, label="Leader Q-gap (raw)")
    axs3[0, 1].plot(R, roll(gap_leader_raw), ":", linewidth=2, label=f"Leader Q-gap (roll {smooth_win})")
    axs3[0, 1].set_title("Fig 3b — Leader Q-gap over time")
    axs3[0, 1].set_xlabel("Round"); axs3[0, 1].set_ylabel("Top-2 Q difference")
    axs3[0, 1].grid(True); axs3[0, 1].legend()

    # 3c Follower Q heatmap
    if isinstance(follower, Q3BinnedFollower):
        foll_Q_mat, foll_states, foll_actions = prepare_q_heatmap(
            follower, actions_sorted=getattr(follower, "d_bins", None), top_k=40
        )
        im_c = safe_imshow(axs3[1, 0], foll_Q_mat, title="Fig 3c — Follower Q(s,a) by most-visited states")
        if im_c is not None:
            plt.colorbar(im_c, ax=axs3[1, 0], fraction=0.046, pad=0.04)
            axs3[1, 0].set_yticks(range(0, len(foll_states), max(1, len(foll_states)//20 or 1)))
            axs3[1, 0].set_yticklabels([foll_states[i] for i in axs3[1, 0].get_yticks().astype(int)])
            axs3[1, 0].set_xticks(range(len(foll_actions)))
            axs3[1, 0].set_xticklabels([f"{a:.2f}" for a in foll_actions], rotation=45, ha="right")
    else:
        axs3[1, 0].text(0.5, 0.5, "Follower is rule-based (no Q-table)",
                        ha="center", va="center")
        axs3[1, 0].set_axis_off()

    # 3d Follower Q-gap
    if isinstance(follower, Q3BinnedFollower) and gap_follower_raw is not None:
        axs3[1, 1].plot(R, gap_follower_raw, "--", alpha=0.55, label="Follower Q-gap (raw)")
        axs3[1, 1].plot(R, roll(gap_follower_raw), ":", linewidth=2, label=f"Follower Q-gap (roll {smooth_win})")
        axs3[1, 1].set_title("Fig 3d — Follower Q-gap over time")
        axs3[1, 1].set_xlabel("Round"); axs3[1, 1].set_ylabel("Top-2 Q difference")
        axs3[1, 1].grid(True); axs3[1, 1].legend()
    else:
        axs3[1, 1].text(0.5, 0.5, "Follower is rule-based (no Q-gap trace)",
                        ha="center", va="center")
        axs3[1, 1].set_axis_off()

    plt.tight_layout()
    plt.savefig(f"plots/stackelberg_q3_econ_fig3_qtables_qgaps{suffix}.png")
    plt.close(fig3)
    # ==========================================================
    # FIGURE 4: Policies & BR snapshot (robust)
    #   4a Empirical policy histograms (τ, d)
    #   4b Follower BR snapshot d*(τ) at current env (+ optional scatter)
    # ==========================================================
    fig4, axs4 = plt.subplots(1, 2, figsize=(14, 5))

    # 4a Histograms
    axs4[0].hist(tau_trace, bins=20, alpha=0.8, label="τ", density=True)
    axs4[0].hist(d_trace,   bins=20, alpha=0.5, label="d", density=True)
    axs4[0].set_title("Fig 4a — Empirical policy histograms")
    axs4[0].set_xlabel("Action level"); axs4[0].set_ylabel("Density")
    axs4[0].grid(True); axs4[0].legend()

    # 4b BR curve (robust)
    taus = np.asarray(tau_bins, dtype=float)
    if taus.size == 0:
        axs4[1].text(0.5, 0.5, "No τ bins", ha="center", va="center"); axs4[1].set_axis_off()
    else:
        # Freeze a clean snapshot of env state (M,X) for evaluating payoffs
        env_snap = EconomicEnvironment(params)
        env_snap.M, env_snap.X = float(env.M), float(env.X)
        if hasattr(env_snap, "prev_X"):
            env_snap.prev_X = env_snap.X  # neutral momentum during evaluation

        def _safe_br_curve(env_snap_, d_bins_, taus_grid):
            br = []
            # back up once (we'll restore every eval)
            M0, X0, prev0 = env_snap_.M, env_snap_.X, getattr(env_snap_, "prev_X", env_snap_.X)
            for tau in taus_grid:
                best_d, best_v = d_bins_[0], -1e30
                for d in d_bins_:
                    # Hard reset env each evaluation to frozen snapshot
                    env_snap_.M, env_snap_.X = M0, X0
                    env_snap_.prev_X = prev0
                    v = env_snap_.evaluate_follower_payoff(tau, d, timing="post")
                    if not np.isfinite(v):
                        continue
                    if v > best_v:
                        best_v, best_d = v, d
                br.append(best_d)
            # final restore (not strictly needed)
            env_snap_.M, env_snap_.X = M0, X0
            env_snap_.prev_X = prev0
            return np.array(br, dtype=float)

        d_bins_arr = np.asarray(d_bins, dtype=float)
        if d_bins_arr.size == 0:
            axs4[1].text(0.5, 0.5, "No d bins", ha="center", va="center"); axs4[1].set_axis_off()
        else:
            br_vals = _safe_br_curve(env_snap, d_bins_arr, taus)

            if br_vals.size == 0 or np.all(~np.isfinite(br_vals)):
                axs4[1].text(0.5, 0.5, "BR curve not computable", ha="center", va="center")
                axs4[1].set_axis_off()
            else:
                # Plot BR curve with markers so a flat line is visible
                axs4[1].plot(taus, br_vals, linewidth=2, marker="o", ms=4, label="Follower BR(d|τ)")
                axs4[1].set_title("Fig 4b — Follower BR snapshot at current env")
                axs4[1].set_xlabel("τ"); axs4[1].set_ylabel("d*")
                # Safe limits
                x_min, x_max = float(np.nanmin(taus)), float(np.nanmax(taus))
                y_min, y_max = float(np.nanmin(d_bins_arr)), float(np.nanmax(d_bins_arr))
                if x_min == x_max: x_min, x_max = x_min - 1e-6, x_max + 1e-6
                if y_min == y_max: y_min, y_max = y_min - 1e-6, y_max + 1e-6
                axs4[1].set_xlim(x_min, x_max)
                axs4[1].set_ylim(y_min, y_max+0.1)
                axs4[1].grid(True); axs4[1].legend()

                # OPTIONAL: overlay empirical scatter of (τ_t, d_t) for context
                # axs4[1].scatter(tau_trace, d_trace, s=4, alpha=0.1, label="Empirical (τ,d)")
                # axs4[1].legend()

    plt.tight_layout()
    plt.savefig(f"plots/stackelberg_q3_econ_fig4_policies_br{suffix}.png")
    plt.close(fig4)


    # ==========================================================
    # FIGURE 5: Tail PV vs Q-value (raw + rolling overlay)
    # ==========================================================
    fig5, axs5 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Leader
    axs5[0].plot(R, L_tailPV, label="Leader Tail PV (t→end)", linewidth=2)
    axs5[0].plot(R, V_leader_raw, "--", alpha=0.55, label="Leader max Q (raw)")
    axs5[0].plot(R, roll(V_leader_raw), ":", linewidth=2, label=f"Leader max Q (roll {smooth_win})")
    axs5[0].set_title("Fig 5a — Leader: Tail PV vs Learned Value (max Q)")
    axs5[0].set_ylabel("Value")
    axs5[0].grid(True); axs5[0].legend()

    # Follower
    if isinstance(follower, Q3BinnedFollower):
        axs5[1].plot(R, F_tailPV, label="Follower Tail PV (t→end)", linewidth=2)
        axs5[1].plot(R, V_follower_raw, "--", alpha=0.55, label="Follower max Q (raw)")
        axs5[1].plot(R, roll(V_follower_raw), ":", linewidth=2, label=f"Follower max Q (roll {smooth_win})")
        axs5[1].set_title("Fig 5b — Follower: Tail PV vs Learned Value (max Q)")
        axs5[1].set_xlabel("Round"); axs5[1].set_ylabel("Value")
        axs5[1].grid(True); axs5[1].legend()
    else:
        axs5[1].text(0.5, 0.5, "Follower is rule-based (no Q-values)",
                     ha="center", va="center")
        axs5[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(f"plots/stackelberg_q3_econ_fig5_tailpv_vs_q{suffix}.png")
    plt.close(fig5)

    # ==========================================================
    # FIGURE 6: Most-visited state heatmaps (moved out of Fig 2)
    #   6a Leader top state visits
    #   6b Follower top state visits
    # ==========================================================
    # Build state visit tables
    def _state_counts(agent, top_k=50):
        items = list(agent.state_visits.items()) if hasattr(agent, "state_visits") else []
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[:top_k]

    leader_sc = _state_counts(leader, top_k=60)
    foll_sc = _state_counts(follower, top_k=60) if isinstance(follower, Q3BinnedFollower) else []

    L_counts = np.array([[c] for (_, c) in leader_sc], dtype=float) if leader_sc else np.zeros((0, 0))
    F_counts = np.array([[c] for (_, c) in foll_sc], dtype=float) if foll_sc else np.zeros((0, 0))

    fig6, axs6 = plt.subplots(1, 2, figsize=(12, 6))
    imL = safe_imshow(axs6[0], L_counts, title="Fig 6a — Leader: top state visits", cmap="magma")
    if imL is not None:
        plt.colorbar(imL, ax=axs6[0], fraction=0.046, pad=0.04)
        yidx = range(0, len(leader_sc), max(1, len(leader_sc)//25 or 1))
        axs6[0].set_yticks(yidx)
        axs6[0].set_yticklabels(["|".join(s) for (s, _) in [leader_sc[i] for i in yidx]])
        axs6[0].set_xticks([])

    imF = safe_imshow(axs6[1], F_counts, title="Fig 6b — Follower: top state visits", cmap="magma")
    if imF is not None:
        plt.colorbar(imF, ax=axs6[1], fraction=0.046, pad=0.04)
        yidx = range(0, len(foll_sc), max(1, len(foll_sc)//25 or 1))
        axs6[1].set_yticks(yidx)
        axs6[1].set_yticklabels(["|".join(s) for (s, _) in [foll_sc[i] for i in yidx]])
        axs6[1].set_xticks([])

    plt.tight_layout()
    plt.savefig(f"plots/stackelberg_q3_econ_fig6_state_visits{suffix}.png")
    plt.close(fig6)

    # -----------------------------
    # Sanity checks
    # -----------------------------
    mean_L = float(np.mean(Lpay))
    mean_F = float(np.mean(Fpay))
    print("\n--- Sanity Checks ---")
    print(f"Mean per-step Leader payoff ≈ {mean_L:.2f} → expected V scale ~ {mean_L/(1-params.delta):.2f}")
    print(f"Mean per-step Follower payoff ≈ {mean_F:.2f} → expected V scale ~ {mean_F/(1-params.delta):.2f}")
    if len(V_leader_raw) > 0:
        print(f"Final Leader max Q ≈ {V_leader_raw[-1]:.2f}")
    if V_follower_raw is not None and len(V_follower_raw) > 0:
        print(f"Final Follower max Q ≈ {V_follower_raw[-1]:.2f}")

    # -----------------------------
    # Coverage & comparability diagnostics (Leader)
    # -----------------------------
    def top_states(agent, k=10):
        items = list(agent.state_visits.items()) if hasattr(agent, "state_visits") else []
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[:k]

    def avg_V_over_top_states(agent, actions, use_max=True):
        items = top_states(agent, k=10)
        vals = []
        for s, _cnt in items:
            qvals = [agent.q.get((s, a), 0.0) for a in actions]
            if not qvals:
                continue
            vals.append(max(qvals) if use_max else sum(qvals) / len(qvals))
        return float(np.mean(vals)) if vals else float("nan")

    print("\n--- Coverage Diagnostics ---")
    print("Top 5 leader states (state → visits):")
    for s, c in top_states(leader, k=5):
        print(f"  {s} → {c}")
    V_leader_top = avg_V_over_top_states(leader, leader.tau_bins, use_max=True)
    mid = len(R) // 2
    print(f"Avg Leader V over top-10 states (max Q): {V_leader_top:.2f}")
    print(f"Leader TailPV @ mid-run (t={mid}): {float(L_tailPV[mid]):.2f}")
    print("--- End Coverage Diagnostics ---\n")

    # -----------------------------
    # Coverage & comparability diagnostics (Follower)
    # -----------------------------
    if isinstance(follower, Q3BinnedFollower):
        print("\n--- Follower Coverage Diagnostics ---")
        print("Top 5 follower states (state → visits):")
        for s, c in top_states(follower, k=5):
            print(f"  {s} → {c}")
        V_follower_top = avg_V_over_top_states(follower, follower.d_bins, use_max=True)
        print(f"Avg Follower V over top-10 states (max Q): {V_follower_top:.2f}")
        print(f"Follower TailPV @ mid-run (t={mid}): {float(F_tailPV[mid]):.2f}")
        print("--- End Follower Coverage Diagnostics ---")

    # Return everything for interactive use if desired
    return {
        "R": R,
        "tau": tau_trace,
        "d": d_trace,
        "M": M_trace,
        "X": X_trace,
        "Lpay": Lpay,
        "Fpay": Fpay,
        "L_tailPV": L_tailPV,
        "F_tailPV": F_tailPV,
        "L_cumdisc": L_cumdisc,
        "F_cumdisc": F_cumdisc,
        "V_leader_raw": V_leader_raw,
        "V_follower_raw": V_follower_raw,
        "gap_leader_raw": gap_leader_raw,
        "gap_follower_raw": gap_follower_raw,
        "leader": leader,
        "follower": follower,
        "env": env,
        "params": params,
    }

# -----------------------------
# Run directly
# -----------------------------
if __name__ == "__main__":

    # # Case B: wider ranges (uncomment to try)
     out = test_stackelberg_q3_tariff_econ(
         steps=500000,
         tau_max=0.5,
         d_max=0.15,
         n_bins=6,
         leader_q_init=3000.0,
         follower_q_init=3000.0,
         epsilon_start=0.15,
         alpha=0.18,
         follower_double_q=True,
         smooth_win=500,
     )




