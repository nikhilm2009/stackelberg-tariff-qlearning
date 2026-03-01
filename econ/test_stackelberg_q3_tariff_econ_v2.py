# test_stackelberg_q3_tariff_econ_v2.py
#
# V2 CHANGES vs test_stackelberg_q3_tariff_econ.py
# =================================================
# - Imports from stackelberg_q3_tariff_econ_sim_v2
# - build_params(): adds E0, export_elast, eta, psi_E, rho_max, follower_cost_rho, beta_E
# - make_uniform_bins(): also used to create rho_bins (5 bins, uniform [0, rho_max])
# - test_stackelberg_q3_tariff_econ_v2():
#     * Agents now receive rho_bins (follower) — 5x5=25 joint actions
#     * Extracts rho_trace, E_trace, export_loss_trace, rho_cost_trace from results
#     * avg_V_over_top_states diagnostic: uses joint_actions for follower
#     * FIGURE 1: old panels PRESERVED; new panel 1d appended (rho, E dynamics)
#     * FIGURE 2: old panels PRESERVED; new panel 2c appended (export_loss, rho_cost)
#     * FIGURE 3: old panels PRESERVED; Q-heatmap uses d_bins (marginal display)
#     * FIGURE 4: old panels PRESERVED; new panel 4c appended (rho histogram + E BR)
#     * FIGURE 5: unchanged (tail PV vs Q)
#     * FIGURE 6: unchanged (state visit heatmaps)
#     * FIGURE 7 (NEW): E flow dynamics + psi_E / rho_max sensitivity sweep
#     * All output filenames have _v2 suffix
# - suffix variable includes "_v2" so files don't overwrite v1 plots
# - __main__ block updated to call v2 function with v2 defaults
# =================================================

import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------------
# Set deterministic seed
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# V2: import from v2 sim module
import stackelberg_q3_tariff_econ_sim_v2 as sim
print("SIM FILE:", sim.__file__)
print("Has follower extras (v2 checks):",
      hasattr(sim.Q3BinnedFollower, "_x_momentum"),
      hasattr(sim.Q3BinnedFollower, "last_d_str"),
      hasattr(sim.Q3BinnedFollower, "last_rho_str"),   # V2
      hasattr(sim.Q3BinnedFollower, "joint_actions"))  # V2

from stackelberg_q3_tariff_econ_sim_v2 import (
    EconomicParams,
    EconomicEnvironment,
    make_bins,
    Q3BinnedLeader,
    Q3BinnedFollower,
    BestResponseFollowerEconomic,
    StackelbergTariffGameEconomic,
)

# ============================================================
# Basic helpers (unchanged from v1)
# ============================================================
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

# ============================================================
# Robust plotting helpers (unchanged from v1)
# ============================================================
def safe_imshow(ax, mat, title="", cmap="viridis", percent_clip=(5, 95)):
    """Robust imshow: handles empty/degenerate matrices with percentile clipping."""
    mat = np.asarray(mat, float)
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
    """Build Q(s,a) matrix using most-visited states.
    V2 note: for Q3BinnedFollower, pass d_bins explicitly (marginal display);
    joint actions are too numerous for readable heatmap axes.
    """
    if not hasattr(agent, "q") or agent.q is None or len(agent.q) == 0:
        return np.zeros((0, 0)), [], []

    if actions_sorted is None:
        actions_sorted = getattr(agent, "tau_bins", None) or getattr(agent, "d_bins", None)
    actions_sorted = list(actions_sorted) if actions_sorted is not None else []
    if len(actions_sorted) == 0:
        return np.zeros((0, 0)), [], []

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

    # V2: for follower with joint actions, marginalise over rho — show max Q per (s, d)
    mat = np.zeros((len(states_sorted), len(actions_sorted)), dtype=float)
    for r_idx, s in enumerate(states_sorted):
        for c_idx, a in enumerate(actions_sorted):
            # Try scalar action first (leader), then marginalise joint (follower)
            direct = agent.q.get((s, a), None)
            if direct is not None:
                mat[r_idx, c_idx] = direct
            else:
                # V2: marginalise over rho dimension — take max Q across rho values
                rho_bins = getattr(agent, "rho_bins", [0.0])
                vals = [agent.q.get((s, (a, rho)), agent.q_init_f if hasattr(agent, "q_init_f") else 0.0)
                        for rho in rho_bins]
                mat[r_idx, c_idx] = max(vals)

    def _lab(s):
        try:
            return "|".join(s)
        except Exception:
            return str(s)

    labels = [_lab(s) for s in states_sorted]
    return mat, labels, actions_sorted


# ============================================================
# Configurable experiment block
# V2: build_params adds retaliatory tariff parameters
# ============================================================

# Leader-favoring scenario (U.S. advantage)
def build_params(tau_max=0.35, d_max=0.25, trade_form="linear",
                 psi_E=1.0, rho_max=0.20, follower_cost_rho=0.05):
    return EconomicParams(
        trade_form="power",
        M0=450.0,
        X0=500.0,
        demand_elast=1.10,    # ↓ sensitivity → τ hurts M less → stronger tariff revenue
        supply_elast=0.90,    # ↓ export response to d → follower gains less
        kappa=0.35,           # ↓ M adjusts slower → revenue persists
        lam=0.40,             # ↓ X adjusts slower → follower gains slower
        delta=0.98,
        phi_inflation=0.35,   # ↑ imported-inflation penalty → discourages d
        leader_cost_w=0.008,  # ↓ softer tariff cost
        follower_cost_w=0.010,# ↑ heavier depreciation/admin cost
        tau_max=0.35,         # moderate tariff headroom
        d_max=0.18,           # ↓ caps follower's depreciation
        # --- V2: leader-favoring retaliation parameters ---
        E0=500.0,             # ↑ large leader export base → retaliation damage is greater in abs. terms
        export_elast=0.80,    # ↓ leader exports are inelastic → ρ suppresses E less → retaliation is weaker
        eta=0.30,             # ↓ E adjusts slowly → retaliation takes longer to bite leader
        psi_E=psi_E,          # default 1.0; raise to amplify export-loss signal in leader payoff
        rho_max=rho_max,      # ↓ 0.20 vs 0.35 — leader's political leverage limits follower retaliation room
        follower_cost_rho=follower_cost_rho,  # ↑ 0.05 → retaliating is costly (WTO exposure, supply chain risk)
        beta_E=0.0,           # spite term off
    )
# Follower-favoring scenario (China/India advantage)
# def build_params(tau_max=0.35, d_max=0.25, trade_form="linear",
#                  psi_E=1.0, rho_max=0.35, follower_cost_rho=0.0):
#     return EconomicParams(
#         trade_form="power",
#         M0=450.0,
#         X0=500.0,
#         demand_elast=1.80,    # ↑ sensitivity → τ cuts M quickly → weakens tariff revenue
#         supply_elast=1.90,    # ↑ strong export response to d
#         kappa=0.60,           # ↑ M adjusts faster → revenue erodes quickly after τ
#         lam=0.55,             # ↑ X adjusts faster → follower gains arrive quickly
#         delta=0.98,
#         phi_inflation=0.20,   # ↓ lower imported-inflation bite from d
#         leader_cost_w=0.015,  # ↑ higher political/admin cost to τ
#         follower_cost_w=0.004,# ↓ cheaper to maintain depreciation
#         tau_max=0.25,         # ↓ political/feasible cap on τ
#         d_max=0.35,           # ↑ more room to depreciate
#         # --- V2: follower-favoring retaliation parameters ---
#         E0=450.0,             # match M0 → symmetric export base; follower retaliation hits a moderately sized target
#         export_elast=1.50,    # ↑ leader exports are elastic → ρ suppresses E strongly → retaliation bites hard
#         eta=0.55,             # ↑ E adjusts fast → retaliation transmits quickly, mirrors lam for X
#         psi_E=psi_E,          # default 1.0; leader feels export loss at full weight
#         rho_max=rho_max,      # ↑ 0.35 → full retaliatory range available; follower has political room to escalate
#         follower_cost_rho=follower_cost_rho,  # ↓ 0.0 → retaliation is cheap (large domestic market, WTO leverage)
#         beta_E=0.0,           # spite term off by default
#     )
# Follower-favoring scenario (China/India advantage) — same base as v1
# def build_params(tau_max=0.35, d_max=0.25, trade_form="linear",
#                  psi_E=1.0, rho_max=0.35, follower_cost_rho=0.0):
#     """V2: psi_E, rho_max, follower_cost_rho added with defaults matching framework doc."""
#     return EconomicParams(
#         trade_form="power",
#         M0=450.0,
#         X0=500.0,
#         demand_elast=1.80,
#         supply_elast=1.90,
#         kappa=0.60,
#         lam=0.55,
#         delta=0.98,
#         phi_inflation=0.20,
#         leader_cost_w=0.015,
#         follower_cost_w=0.004,
#         tau_max=tau_max,
#         d_max=d_max,
#         # V2: new retaliation channel parameters
#         E0=450.0,               # V2: match M0 for symmetric baseline
#         export_elast=1.2,       # V2: moderate export elasticity
#         eta=0.40,               # V2: E adjusts at same speed as kappa
#         psi_E=psi_E,            # V2: export loss weight in leader payoff (default 1.0)
#         rho_max=rho_max,        # V2: max retaliatory tariff (default 0.35)
#         follower_cost_rho=follower_cost_rho,  # V2: quadratic rho cost
#         beta_E=0.0,             # V2: spite term coded but OFF by default
#     )

def make_uniform_bins(vmin, vmax, n=6):
    """Inclusive endpoints, uniform spacing."""
    return make_bins(n, vmin, vmax)


# ============================================================
# V2: main test/plot function
# ============================================================
import os

def test_stackelberg_q3_tariff_econ_v2(
    trade_form="linear",
    steps=20000,
    tau_max=0.35,
    d_max=0.25,
    n_bins=6,
    n_rho_bins=5,             # V2: 5 bins for rho → 5x5=25 joint actions
    leader_q_init=1800.0,
    follower_q_init=3000.0,
    epsilon_start=0.15,
    alpha=0.18,
    follower_double_q=True,
    smooth_win=25,
    psi_E=1.0,                # V2: passed to build_params
    rho_max=0.35,             # V2: passed to build_params
    follower_cost_rho=0.0,    # V2: passed to build_params
):
    os.makedirs("plots_v2", exist_ok=True)

    # --- Params & bins
    params = build_params(tau_max=tau_max, d_max=d_max, trade_form=trade_form,
                          psi_E=psi_E, rho_max=rho_max,
                          follower_cost_rho=follower_cost_rho)

    # V2: suffix now includes _v2 so files never overwrite v1 plots
    suffix = f"_{params.trade_form}_v2"

    tau_bins = make_uniform_bins(0.0, params.tau_max, n=n_bins)
    d_bins   = make_uniform_bins(0.0, params.d_max,   n=n_bins)
    # V2: rho_bins — uniform [0, rho_max], 5 bins (inclusive endpoints)
    rho_bins = make_uniform_bins(0.0, params.rho_max, n=n_rho_bins)

    # --- Environment
    env = EconomicEnvironment(params)

    # --- Agents
    leader = Q3BinnedLeader(
        tau_bins=tau_bins,
        epsilon=epsilon_start,
        gamma=params.delta,
        alpha=alpha,
        q_init=leader_q_init,
    )
    follower = Q3BinnedFollower(
        d_bins=d_bins,
        rho_bins=rho_bins,           # V2: joint action space
        epsilon=epsilon_start,
        gamma=params.delta,
        alpha=alpha,
        double_q=follower_double_q,
        q_init_f=follower_q_init,
    )

    # --- Orchestrate & run
    game = StackelbergTariffGameEconomic(env, leader, follower, track=True)
    game.run(rounds=steps)

    # --- Extract traces (v1 fields unchanged)
    rounds    = [r["round"] for r in game.results["rounds"]]
    R         = np.asarray(rounds, dtype=int)
    tau_trace = np.asarray([r["tau"]          for r in game.results["rounds"]], float)
    d_trace   = np.asarray([r["d"]            for r in game.results["rounds"]], float)
    Lpay      = np.asarray([r["leader_pay"]   for r in game.results["rounds"]], float)
    Fpay      = np.asarray([r["follower_pay"] for r in game.results["rounds"]], float)
    M_trace   = np.asarray([r["M"]            for r in game.results["rounds"]], float)
    X_trace   = np.asarray([r["X"]            for r in game.results["rounds"]], float)

    # V2: new traces
    rho_trace         = np.asarray([r["rho"]          for r in game.results["rounds"]], float)
    E_trace           = np.asarray([r["E"]             for r in game.results["rounds"]], float)
    export_loss_trace = np.asarray([r["export_loss"]   for r in game.results["rounds"]], float)
    rho_cost_trace    = np.asarray([r["rho_cost"]      for r in game.results["rounds"]], float)

    # Decomposition pieces (v1)
    rev_trace       = np.asarray([r["rev"]          for r in game.results["rounds"]], float)
    cons_loss_trace = np.asarray([r["cons_loss"]    for r in game.results["rounds"]], float)
    aCostL_trace    = np.asarray([r["action_cost_L"] for r in game.results["rounds"]], float)
    export_gain_tr  = np.asarray([r["export_gain"]  for r in game.results["rounds"]], float)
    infl_cost_tr    = np.asarray([r["infl_cost"]    for r in game.results["rounds"]], float)
    aCostF_trace    = np.asarray([r["action_cost_F"] for r in game.results["rounds"]], float)

    # --- Tail PV & cumulative discounted (unchanged)
    L_tailPV  = tail_present_value(Lpay, params.delta)
    F_tailPV  = tail_present_value(Fpay, params.delta)
    L_cumdisc = cumulative_discounted(Lpay, params.delta)
    F_cumdisc = cumulative_discounted(Fpay, params.delta)

    # --- Q traces (unchanged)
    V_leader_raw    = _to_same_len(getattr(leader,   "q_max_trace", []), len(R))
    V_follower_raw  = _to_same_len(getattr(follower, "q_max_trace", []), len(R)) \
                      if isinstance(follower, Q3BinnedFollower) else None
    gap_leader_raw  = _to_same_len(getattr(leader,   "qgap_trace",  []), len(R))
    gap_follower_raw= _to_same_len(getattr(follower, "qgap_trace",  []), len(R)) \
                      if isinstance(follower, Q3BinnedFollower) else None

    roll = lambda x: _to_same_len(_rolling_mean(x, win=smooth_win), len(R))

    # ==========================================================
    # FIGURE 1: Dynamics overview
    #   1a Actions (τ, d) + rolling averages          [PRESERVED from v1]
    #   1b Trade flows (M, X) + rolling averages      [PRESERVED from v1]
    #   1c Cumulative payoffs disc. vs undisc.         [PRESERVED from v1]
    #   1d V2 NEW: rho trace + E flow                 [APPENDED]
    # ==========================================================
    win = 500
    fig1, axs1 = plt.subplots(1, 4, figsize=(24, 5))   # V2: 4 panels (was 3)

    # 1a — Actions (unchanged)
    tau_roll = _rolling_mean(tau_trace, win)
    d_roll   = _rolling_mean(d_trace,   win)
    axs1[0].plot(R, tau_trace, label="Leader τ",   linewidth=1.2, alpha=0.6)
    axs1[0].plot(R, d_trace,   label="Follower d",  linewidth=1.2, alpha=0.6)
    axs1[0].plot(R, tau_roll, "r", label=f"τ (roll {win})", linewidth=2)
    axs1[0].plot(R, d_roll,   "g", label=f"d (roll {win})", linewidth=2)
    axs1[0].set_title("Fig 1a — Actions with rolling averages")
    axs1[0].set_xlabel("Round"); axs1[0].set_ylabel("Action level")
    axs1[0].grid(True); axs1[0].legend()

    # 1b — Trade flows M, X (unchanged)
    M_roll = _rolling_mean(M_trace, win)
    X_roll = _rolling_mean(X_trace, win)
    axs1[1].plot(R, M_trace, label="Imports M",   linewidth=1.2, alpha=0.6)
    axs1[1].plot(R, X_trace, label="Exports X",   linewidth=1.2, alpha=0.6)
    axs1[1].plot(R, M_roll, "r", label=f"M (roll {win})", linewidth=2)
    axs1[1].plot(R, X_roll, "g", label=f"X (roll {win})", linewidth=2)
    axs1[1].set_title("Fig 1b — Trade flows with rolling averages")
    axs1[1].set_xlabel("Round"); axs1[1].set_ylabel("Level")
    axs1[1].grid(True); axs1[1].legend()

    # 1c — Cumulative payoffs (unchanged)
    L_undisc = np.cumsum(Lpay)
    F_undisc = np.cumsum(Fpay)
    ax = axs1[2]
    l1 = ax.plot(R, L_cumdisc, label="Leader cumulative (discounted)", linewidth=2)
    l2 = ax.plot(R, F_cumdisc, label="Follower cumulative (discounted)", linewidth=2, alpha=0.85)
    ax.set_ylabel("Discounted cumulative payoff")
    ax.set_title("Fig 1c — Cumulative payoffs (disc. vs undisc.)")
    ax.set_xlabel("Round"); ax.grid(True)
    ax2 = ax.twinx()
    l3 = ax2.plot(R, L_undisc, "--", color="tab:red",   label="Leader cumulative (undisc.)", linewidth=1.6)
    l4 = ax2.plot(R, F_undisc, "--", color="tab:green", label="Follower cumulative (undisc.)", linewidth=1.6, alpha=0.9)
    ax2.set_ylabel("Undiscounted cumulative payoff")
    lines = l1 + l2 + l3 + l4
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=9)

    # 1d — V2 NEW: rho + E dynamics
    rho_roll = _rolling_mean(rho_trace, win)
    E_roll   = _rolling_mean(E_trace,   win)
    ax1d = axs1[3]
    ax1d.plot(R, rho_trace, label="Follower ρ (retaliation)", linewidth=1.2, alpha=0.6, color="tab:orange")
    ax1d.plot(R, rho_roll,  label=f"ρ (roll {win})", linewidth=2, color="tab:orange", linestyle="--")
    ax1d.set_xlabel("Round"); ax1d.set_ylabel("ρ level", color="tab:orange")
    ax1d.tick_params(axis="y", labelcolor="tab:orange")
    ax1d.set_title("Fig 1d (V2) — Retaliation ρ + Leader Export E")
    ax1d.grid(True)
    ax1d_r = ax1d.twinx()
    ax1d_r.plot(R, E_trace, label="Leader exports E", linewidth=1.2, alpha=0.6, color="tab:purple")
    ax1d_r.plot(R, E_roll,  label=f"E (roll {win})",  linewidth=2,   color="tab:purple", linestyle="--")
    ax1d_r.axhline(params.E0, color="tab:purple", linestyle=":", linewidth=1.2, alpha=0.7, label="E baseline")
    ax1d_r.set_ylabel("E level", color="tab:purple")
    ax1d_r.tick_params(axis="y", labelcolor="tab:purple")
    lines_d = (ax1d.get_lines() + ax1d_r.get_lines())
    ax1d.legend(lines_d, [l.get_label() for l in lines_d], fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"plots_v2/stackelberg_q3_econ_fig1{suffix}.png")
    plt.close(fig1)

    # ==========================================================
    # FIGURE 2: Payoff decompositions
    #   2a Leader: revenue, consumer loss, action cost  [PRESERVED from v1]
    #   2b Follower: export gain, infl cost, action cost [PRESERVED from v1]
    #   2c V2 NEW: export_loss (leader) + rho_cost (follower)  [APPENDED]
    # ==========================================================
    win_dec = 500
    fig2, axs2 = plt.subplots(1, 3, figsize=(21, 5))   # V2: 3 panels (was 2)

    # 2a — Leader decomposition (unchanged)
    rev_roll       = _rolling_mean(rev_trace,       win_dec)
    cons_loss_roll = _rolling_mean(cons_loss_trace, win_dec)
    aCostL_roll    = _rolling_mean(aCostL_trace,    win_dec)
    axs2[0].plot(R, rev_trace,       alpha=0.35, linewidth=1.0, label="Revenue τ·M (raw)")
    axs2[0].plot(R, cons_loss_trace, alpha=0.35, linewidth=1.0, label="Consumer loss (raw)")
    axs2[0].plot(R, aCostL_trace,    alpha=0.35, linewidth=1.0, label="Action cost (τ²) (raw)")
    axs2[0].plot(R, rev_roll,       linewidth=2.0, label=f"Revenue τ·M (roll {win_dec})")
    axs2[0].plot(R, cons_loss_roll, linewidth=2.0, label=f"Consumer loss (roll {win_dec})")
    axs2[0].plot(R, aCostL_roll,    linewidth=2.0, label=f"Action cost (τ²) (roll {win_dec})")
    axs2[0].set_title("Fig 2a — Leader payoff decomposition (raw + rolling)")
    axs2[0].set_xlabel("Round"); axs2[0].set_ylabel("Per-step component")
    axs2[0].grid(True); axs2[0].legend(ncol=2, fontsize=9)

    # 2b — Follower decomposition (unchanged)
    export_gain_roll = _rolling_mean(export_gain_tr, win_dec)
    infl_cost_roll   = _rolling_mean(infl_cost_tr,   win_dec)
    aCostF_roll      = _rolling_mean(aCostF_trace,   win_dec)
    axs2[1].plot(R, export_gain_tr, alpha=0.35, linewidth=1.0, label="Export gain (X−X0) (raw)")
    axs2[1].plot(R, infl_cost_tr,   alpha=0.35, linewidth=1.0, label="Inflation cost φ·d·M0 (raw)")
    axs2[1].plot(R, aCostF_trace,   alpha=0.35, linewidth=1.0, label="Action cost (d²) (raw)")
    axs2[1].plot(R, export_gain_roll, linewidth=2.0, label=f"Export gain (roll {win_dec})")
    axs2[1].plot(R, infl_cost_roll,   linewidth=2.0, label=f"Inflation cost (roll {win_dec})")
    axs2[1].plot(R, aCostF_roll,      linewidth=2.0, label=f"Action cost (d²) (roll {win_dec})")
    axs2[1].set_title("Fig 2b — Follower payoff decomposition (raw + rolling)")
    axs2[1].set_xlabel("Round"); axs2[1].set_ylabel("Per-step component")
    axs2[1].grid(True); axs2[1].legend(ncol=2, fontsize=9)

    # 2c — V2 NEW: retaliation channel costs
    exp_loss_roll = _rolling_mean(export_loss_trace, win_dec)
    rho_cost_roll = _rolling_mean(rho_cost_trace,    win_dec)
    axs2[2].plot(R, export_loss_trace, alpha=0.35, linewidth=1.0, color="tab:red",
                 label="Leader export loss ψ_E·(E0−E) (raw)")
    axs2[2].plot(R, rho_cost_trace,    alpha=0.35, linewidth=1.0, color="tab:blue",
                 label="Follower ρ cost c_ρ·ρ² (raw)")
    axs2[2].plot(R, exp_loss_roll, linewidth=2.0, color="tab:red",
                 label=f"Export loss (roll {win_dec})")
    axs2[2].plot(R, rho_cost_roll, linewidth=2.0, color="tab:blue",
                 label=f"ρ cost (roll {win_dec})")
    axs2[2].set_title("Fig 2c (V2) — Retaliation channel: export loss + ρ cost")
    axs2[2].set_xlabel("Round"); axs2[2].set_ylabel("Per-step component")
    axs2[2].grid(True); axs2[2].legend(ncol=1, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"plots_v2/stackelberg_q3_econ_fig2_decompositions{suffix}.png")
    plt.close(fig2)

    # ==========================================================
    # FIGURE 3: Q tables & Q-gaps (PRESERVED from v1, with marginalised follower heatmap)
    #   3a Leader Q(s,a) heatmap
    #   3b Leader Q-gap
    #   3c Follower Q(s,a) heatmap — V2: marginalised over rho (max Q per d)
    #   3d Follower Q-gap
    # ==========================================================
    fig3, axs3 = plt.subplots(2, 2, figsize=(14, 10))

    # 3a Leader Q heatmap (unchanged)
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

    # 3b Leader Q-gap (unchanged)
    axs3[0, 1].plot(R, gap_leader_raw, "--", alpha=0.55, label="Leader Q-gap (raw)")
    axs3[0, 1].plot(R, roll(gap_leader_raw), ":", linewidth=2, label=f"Leader Q-gap (roll {smooth_win})")
    axs3[0, 1].set_title("Fig 3b — Leader Q-gap over time")
    axs3[0, 1].set_xlabel("Round"); axs3[0, 1].set_ylabel("Top-2 Q difference")
    axs3[0, 1].grid(True); axs3[0, 1].legend()

    # 3c Follower Q heatmap — V2: uses d_bins for x-axis; prepare_q_heatmap marginalises rho
    if isinstance(follower, Q3BinnedFollower):
        foll_Q_mat, foll_states, foll_actions = prepare_q_heatmap(
            follower, actions_sorted=getattr(follower, "d_bins", None), top_k=40
        )
        title_3c = "Fig 3c — Follower Q(s,d) — max over ρ [V2]"
        im_c = safe_imshow(axs3[1, 0], foll_Q_mat, title=title_3c)
        if im_c is not None:
            plt.colorbar(im_c, ax=axs3[1, 0], fraction=0.046, pad=0.04)
            axs3[1, 0].set_yticks(range(0, len(foll_states), max(1, len(foll_states)//20 or 1)))
            axs3[1, 0].set_yticklabels([foll_states[i] for i in axs3[1, 0].get_yticks().astype(int)])
            axs3[1, 0].set_xticks(range(len(foll_actions)))
            axs3[1, 0].set_xticklabels([f"{a:.2f}" for a in foll_actions], rotation=45, ha="right")
    else:
        axs3[1, 0].text(0.5, 0.5, "Follower is rule-based (no Q-table)", ha="center", va="center")
        axs3[1, 0].set_axis_off()

    # 3d Follower Q-gap (unchanged)
    if isinstance(follower, Q3BinnedFollower) and gap_follower_raw is not None:
        axs3[1, 1].plot(R, gap_follower_raw, "--", alpha=0.55, label="Follower Q-gap (raw)")
        axs3[1, 1].plot(R, roll(gap_follower_raw), ":", linewidth=2, label=f"Follower Q-gap (roll {smooth_win})")
        axs3[1, 1].set_title("Fig 3d — Follower Q-gap over time")
        axs3[1, 1].set_xlabel("Round"); axs3[1, 1].set_ylabel("Top-2 Q difference")
        axs3[1, 1].grid(True); axs3[1, 1].legend()
    else:
        axs3[1, 1].text(0.5, 0.5, "Follower is rule-based (no Q-gap trace)", ha="center", va="center")
        axs3[1, 1].set_axis_off()

    plt.tight_layout()
    plt.savefig(f"plots_v2/stackelberg_q3_econ_fig3_qtables_qgaps{suffix}.png")
    plt.close(fig3)

    # ==========================================================
    # FIGURE 4: Policies & BR snapshot
    #   4a Empirical policy histograms (τ, d)         [PRESERVED from v1]
    #   4b Follower BR snapshot d*(τ) at current env  [PRESERVED from v1]
    #   4c V2 NEW: rho histogram + rho*(τ) BR curve   [APPENDED]
    # ==========================================================
    fig4, axs4 = plt.subplots(1, 3, figsize=(21, 5))   # V2: 3 panels (was 2)

    # 4a Histograms — tau and d (unchanged)
    axs4[0].hist(tau_trace, bins=20, alpha=0.8, label="τ",        density=True)
    axs4[0].hist(d_trace,   bins=20, alpha=0.5, label="d",        density=True)
    axs4[0].hist(rho_trace, bins=20, alpha=0.5, label="ρ (V2)",   density=True, color="tab:orange")
    axs4[0].set_title("Fig 4a — Empirical policy histograms (τ, d, ρ)")
    axs4[0].set_xlabel("Action level"); axs4[0].set_ylabel("Density")
    axs4[0].grid(True); axs4[0].legend()

    # 4b BR curve for d (unchanged)
    taus = np.asarray(tau_bins, dtype=float)
    if taus.size == 0:
        axs4[1].text(0.5, 0.5, "No τ bins", ha="center", va="center"); axs4[1].set_axis_off()
    else:
        env_snap = EconomicEnvironment(params)
        env_snap.M, env_snap.X, env_snap.E = float(env.M), float(env.X), float(env.E)   # V2: E
        if hasattr(env_snap, "prev_X"):
            env_snap.prev_X = env_snap.X

        def _safe_br_curve_d(env_snap_, d_bins_, rho_fixed, taus_grid):
            """BR over d holding rho fixed at rho_fixed (0 for baseline comparison)."""
            br = []
            M0_, X0_, E0_ = env_snap_.M, env_snap_.X, env_snap_.E
            prev0 = getattr(env_snap_, "prev_X", env_snap_.X)
            for tau in taus_grid:
                best_d, best_v = d_bins_[0], -1e30
                for d in d_bins_:
                    env_snap_.M, env_snap_.X, env_snap_.E = M0_, X0_, E0_
                    env_snap_.prev_X = prev0
                    v = env_snap_.evaluate_follower_payoff(tau, d, rho_fixed, timing="post")
                    if np.isfinite(v) and v > best_v:
                        best_v, best_d = v, d
                br.append(best_d)
            env_snap_.M, env_snap_.X, env_snap_.E = M0_, X0_, E0_
            env_snap_.prev_X = prev0
            return np.array(br, dtype=float)

        d_bins_arr = np.asarray(d_bins, dtype=float)
        if d_bins_arr.size > 0:
            br_vals = _safe_br_curve_d(env_snap, d_bins_arr, 0.0, taus)
            if br_vals.size > 0 and np.any(np.isfinite(br_vals)):
                axs4[1].plot(taus, br_vals, linewidth=2, marker="o", ms=4, label="Follower BR d*(τ)|ρ=0")
                axs4[1].set_title("Fig 4b — Follower BR d*(τ) at current env")
                axs4[1].set_xlabel("τ"); axs4[1].set_ylabel("d*")
                x_min, x_max = float(np.nanmin(taus)), float(np.nanmax(taus))
                y_min, y_max = float(np.nanmin(d_bins_arr)), float(np.nanmax(d_bins_arr))
                if x_min == x_max: x_min, x_max = x_min - 1e-6, x_max + 1e-6
                if y_min == y_max: y_min, y_max = y_min - 1e-6, y_max + 1e-6
                axs4[1].set_xlim(x_min, x_max); axs4[1].set_ylim(y_min, y_max + 0.1)
                axs4[1].grid(True); axs4[1].legend()
            else:
                axs4[1].text(0.5, 0.5, "BR d curve not computable", ha="center", va="center")
                axs4[1].set_axis_off()
        else:
            axs4[1].text(0.5, 0.5, "No d bins", ha="center", va="center"); axs4[1].set_axis_off()

    # 4c — V2 NEW: rho BR curve — best rho*(tau) holding d at its mean
    d_mean_val = float(np.mean(d_trace))
    d_mean_snap = min(d_bins, key=lambda x: abs(x - d_mean_val))  # snap to nearest bin
    rho_bins_arr = np.asarray(rho_bins, dtype=float)

    def _safe_br_curve_rho(env_snap_, rho_bins_, d_fixed, taus_grid):
        """BR over rho holding d fixed at d_fixed."""
        br = []
        M0_, X0_, E0_ = env_snap_.M, env_snap_.X, env_snap_.E
        prev0 = getattr(env_snap_, "prev_X", env_snap_.X)
        for tau in taus_grid:
            best_rho, best_v = rho_bins_[0], -1e30
            for rho in rho_bins_:
                env_snap_.M, env_snap_.X, env_snap_.E = M0_, X0_, E0_
                env_snap_.prev_X = prev0
                v = env_snap_.evaluate_follower_payoff(tau, d_fixed, rho, timing="post")
                if np.isfinite(v) and v > best_v:
                    best_v, best_rho = v, rho
            br.append(best_rho)
        env_snap_.M, env_snap_.X, env_snap_.E = M0_, X0_, E0_
        env_snap_.prev_X = prev0
        return np.array(br, dtype=float)

    if taus.size > 0 and rho_bins_arr.size > 0:
        br_rho_vals = _safe_br_curve_rho(env_snap, list(rho_bins_arr), d_mean_snap, taus)
        if br_rho_vals.size > 0 and np.any(np.isfinite(br_rho_vals)):
            axs4[2].plot(taus, br_rho_vals, linewidth=2, marker="s", ms=4,
                         color="tab:orange", label=f"Follower BR ρ*(τ)|d≈{d_mean_snap:.2f}")
            axs4[2].set_title(f"Fig 4c (V2) — Follower BR ρ*(τ) at current env")
            axs4[2].set_xlabel("τ"); axs4[2].set_ylabel("ρ*")
            axs4[2].set_xlim(float(np.nanmin(taus)), float(np.nanmax(taus)))
            axs4[2].set_ylim(0, float(np.nanmax(rho_bins_arr)) + 0.05)
            axs4[2].grid(True); axs4[2].legend()
        else:
            axs4[2].text(0.5, 0.5, "BR ρ curve not computable", ha="center", va="center")
            axs4[2].set_axis_off()
    else:
        axs4[2].text(0.5, 0.5, "No ρ bins", ha="center", va="center")
        axs4[2].set_axis_off()

    plt.tight_layout()
    plt.savefig(f"plots_v2/stackelberg_q3_econ_fig4_policies_br{suffix}.png")
    plt.close(fig4)

    # ==========================================================
    # FIGURE 5: Tail PV vs Q-value (PRESERVED from v1)
    # ==========================================================
    fig5, axs5 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axs5[0].plot(R, L_tailPV,      label="Leader Tail PV (t→end)", linewidth=2)
    axs5[0].plot(R, V_leader_raw,  "--", alpha=0.55, label="Leader max Q (raw)")
    axs5[0].plot(R, roll(V_leader_raw), ":", linewidth=2, label=f"Leader max Q (roll {smooth_win})")
    axs5[0].set_title("Fig 5a — Leader: Tail PV vs Learned Value (max Q)")
    axs5[0].set_ylabel("Value"); axs5[0].grid(True); axs5[0].legend()

    if isinstance(follower, Q3BinnedFollower):
        axs5[1].plot(R, F_tailPV,      label="Follower Tail PV (t→end)", linewidth=2)
        axs5[1].plot(R, V_follower_raw, "--", alpha=0.55, label="Follower max Q (raw)")
        axs5[1].plot(R, roll(V_follower_raw), ":", linewidth=2, label=f"Follower max Q (roll {smooth_win})")
        axs5[1].set_title("Fig 5b — Follower: Tail PV vs Learned Value (max Q)")
        axs5[1].set_xlabel("Round"); axs5[1].set_ylabel("Value")
        axs5[1].grid(True); axs5[1].legend()
    else:
        axs5[1].text(0.5, 0.5, "Follower is rule-based (no Q-values)", ha="center", va="center")
        axs5[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(f"plots_v2/stackelberg_q3_econ_fig5_tailpv_vs_q{suffix}.png")
    plt.close(fig5)

    # ==========================================================
    # FIGURE 6: Most-visited state heatmaps (PRESERVED from v1)
    # ==========================================================
    def _state_counts(agent, top_k=50):
        items = list(agent.state_visits.items()) if hasattr(agent, "state_visits") else []
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[:top_k]

    leader_sc = _state_counts(leader, top_k=60)
    foll_sc   = _state_counts(follower, top_k=60) if isinstance(follower, Q3BinnedFollower) else []

    L_counts = np.array([[c] for (_, c) in leader_sc], dtype=float) if leader_sc else np.zeros((0, 0))
    F_counts = np.array([[c] for (_, c) in foll_sc],  dtype=float) if foll_sc  else np.zeros((0, 0))

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
    plt.savefig(f"plots_v2/stackelberg_q3_econ_fig6_state_visits{suffix}.png")
    plt.close(fig6)

    # ==========================================================
    # FIGURE 7 (V2 NEW): E-channel deep-dive + psi_E sensitivity sweep
    #   7a E flow over time (raw + rolling) vs baseline E0
    #   7b rho action over time + rolling (separate axis from 1d for clarity)
    #   7c psi_E sensitivity: final mean leader payoff vs psi_E
    #   7d rho_max sensitivity: final mean follower payoff vs rho_max
    # ==========================================================
    fig7, axs7 = plt.subplots(2, 2, figsize=(16, 10))

    # 7a — E flow dynamics
    E_roll7 = _rolling_mean(E_trace, win)
    axs7[0, 0].plot(R, E_trace, alpha=0.5, linewidth=1.0, color="tab:purple", label="E (raw)")
    axs7[0, 0].plot(R, E_roll7, linewidth=2.0, color="tab:purple", label=f"E (roll {win})")
    axs7[0, 0].axhline(params.E0, color="k", linestyle=":", linewidth=1.5, label=f"E baseline = {params.E0:.0f}")
    axs7[0, 0].set_title("Fig 7a (V2) — Leader Export flow E over time")
    axs7[0, 0].set_xlabel("Round"); axs7[0, 0].set_ylabel("E level")
    axs7[0, 0].grid(True); axs7[0, 0].legend()

    # 7b — rho dynamics (more detailed than 1d)
    rho_roll7 = _rolling_mean(rho_trace, win)
    axs7[0, 1].plot(R, rho_trace, alpha=0.5, linewidth=1.0, color="tab:orange", label="ρ (raw)")
    axs7[0, 1].plot(R, rho_roll7, linewidth=2.0, color="tab:orange", label=f"ρ (roll {win})")
    axs7[0, 1].axhline(params.rho_max, color="k", linestyle=":", linewidth=1.2, label=f"ρ_max = {params.rho_max:.2f}")
    axs7[0, 1].set_title("Fig 7b (V2) — Follower retaliation ρ over time")
    axs7[0, 1].set_xlabel("Round"); axs7[0, 1].set_ylabel("ρ level")
    axs7[0, 1].grid(True); axs7[0, 1].legend()

    # 7c — psi_E sensitivity sweep (short runs, fixed seed for reproducibility)
    psi_E_vals  = [0.0, 0.5, 1.0, 2.0, 3.0]
    sweep_steps = max(1000, steps // 20)   # lightweight sweep
    L_means_psi = []
    F_means_psi = []
    for psi_val in psi_E_vals:
        random.seed(SEED); np.random.seed(SEED)
        p_sw = build_params(tau_max=tau_max, d_max=d_max, psi_E=psi_val, rho_max=rho_max)
        e_sw = EconomicEnvironment(p_sw)
        l_sw = Q3BinnedLeader(tau_bins, epsilon=epsilon_start, gamma=p_sw.delta,
                               alpha=alpha, q_init=leader_q_init)
        f_sw = Q3BinnedFollower(d_bins, rho_bins=rho_bins, epsilon=epsilon_start,
                                gamma=p_sw.delta, alpha=alpha,
                                double_q=follower_double_q, q_init_f=follower_q_init)
        g_sw = StackelbergTariffGameEconomic(e_sw, l_sw, f_sw, track=True)
        g_sw.run(rounds=sweep_steps)
        L_means_psi.append(np.mean([r["leader_pay"]   for r in g_sw.results["rounds"]]))
        F_means_psi.append(np.mean([r["follower_pay"] for r in g_sw.results["rounds"]]))

    axs7[1, 0].plot(psi_E_vals, L_means_psi, "o-", linewidth=2, color="tab:blue",  label="Mean Leader payoff")
    axs7[1, 0].plot(psi_E_vals, F_means_psi, "s-", linewidth=2, color="tab:green", label="Mean Follower payoff")
    axs7[1, 0].set_title(f"Fig 7c (V2) — psi_E sensitivity ({sweep_steps} steps each)")
    axs7[1, 0].set_xlabel("psi_E (export loss weight)"); axs7[1, 0].set_ylabel("Mean per-step payoff")
    axs7[1, 0].grid(True); axs7[1, 0].legend()

    # 7d — rho_max sensitivity sweep
    rho_max_vals = [0.0, 0.10, 0.20, 0.35, 0.50]
    L_means_rho  = []
    F_means_rho  = []
    for rm_val in rho_max_vals:
        random.seed(SEED); np.random.seed(SEED)
        rho_bins_sw = make_uniform_bins(0.0, rm_val, n=n_rho_bins) if rm_val > 0 else [0.0]
        p_sw = build_params(tau_max=tau_max, d_max=d_max, psi_E=psi_E, rho_max=rm_val)
        e_sw = EconomicEnvironment(p_sw)
        l_sw = Q3BinnedLeader(tau_bins, epsilon=epsilon_start, gamma=p_sw.delta,
                               alpha=alpha, q_init=leader_q_init)
        f_sw = Q3BinnedFollower(d_bins, rho_bins=rho_bins_sw, epsilon=epsilon_start,
                                gamma=p_sw.delta, alpha=alpha,
                                double_q=follower_double_q, q_init_f=follower_q_init)
        g_sw = StackelbergTariffGameEconomic(e_sw, l_sw, f_sw, track=True)
        g_sw.run(rounds=sweep_steps)
        L_means_rho.append(np.mean([r["leader_pay"]   for r in g_sw.results["rounds"]]))
        F_means_rho.append(np.mean([r["follower_pay"] for r in g_sw.results["rounds"]]))

    axs7[1, 1].plot(rho_max_vals, L_means_rho, "o-", linewidth=2, color="tab:blue",   label="Mean Leader payoff")
    axs7[1, 1].plot(rho_max_vals, F_means_rho, "s-", linewidth=2, color="tab:orange", label="Mean Follower payoff")
    axs7[1, 1].set_title(f"Fig 7d (V2) — rho_max sensitivity ({sweep_steps} steps each)")
    axs7[1, 1].set_xlabel("rho_max (follower retaliation ceiling)"); axs7[1, 1].set_ylabel("Mean per-step payoff")
    axs7[1, 1].grid(True); axs7[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"plots_v2/stackelberg_q3_econ_fig7_E_channel_v2{suffix}.png")
    plt.close(fig7)

    # ============================================================
    # Sanity checks (v1 unchanged + V2 additions)
    # ============================================================
    mean_L = float(np.mean(Lpay))
    mean_F = float(np.mean(Fpay))
    print("\n--- Sanity Checks ---")
    print(f"Mean per-step Leader payoff  ≈ {mean_L:.2f} → expected V ~ {mean_L/(1-params.delta):.2f}")
    print(f"Mean per-step Follower payoff ≈ {mean_F:.2f} → expected V ~ {mean_F/(1-params.delta):.2f}")
    if len(V_leader_raw)  > 0: print(f"Final Leader max Q  ≈ {V_leader_raw[-1]:.2f}")
    if V_follower_raw is not None and len(V_follower_raw) > 0:
        print(f"Final Follower max Q ≈ {V_follower_raw[-1]:.2f}")
    # V2 additions
    print(f"Mean ρ (retaliation) ≈ {float(np.mean(rho_trace)):.4f}  |  max ρ = {float(np.max(rho_trace)):.4f}")
    print(f"Mean E (leader exp.) ≈ {float(np.mean(E_trace)):.2f}   |  E baseline = {params.E0:.2f}")
    print(f"Mean export_loss     ≈ {float(np.mean(export_loss_trace)):.2f}")
    print(f"Mean rho_cost        ≈ {float(np.mean(rho_cost_trace)):.4f}")

    # Coverage diagnostics (unchanged structure)
    def top_states(agent, k=10):
        items = list(agent.state_visits.items()) if hasattr(agent, "state_visits") else []
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[:k]

    def avg_V_over_top_states(agent, actions, use_max=True):
        """V2: actions may be a list of joint tuples or scalars; handles both."""
        items = top_states(agent, k=10)
        vals = []
        for s, _cnt in items:
            qvals = [agent.q.get((s, a), 0.0) for a in actions]
            if not qvals: continue
            vals.append(max(qvals) if use_max else sum(qvals)/len(qvals))
        return float(np.mean(vals)) if vals else float("nan")

    print("\n--- Coverage Diagnostics ---")
    print("Top 5 leader states:")
    for s, c in top_states(leader, k=5):
        print(f"  {s} → {c}")
    V_leader_top = avg_V_over_top_states(leader, leader.tau_bins, use_max=True)
    mid = len(R) // 2
    print(f"Avg Leader V over top-10 states: {V_leader_top:.2f}")
    print(f"Leader TailPV @ mid-run (t={mid}): {float(L_tailPV[mid]):.2f}")

    if isinstance(follower, Q3BinnedFollower):
        print("\n--- Follower Coverage Diagnostics ---")
        print("Top 5 follower states:")
        for s, c in top_states(follower, k=5):
            print(f"  {s} → {c}")
        # V2: use joint_actions for follower Q lookup
        V_follower_top = avg_V_over_top_states(follower, follower.joint_actions, use_max=True)
        print(f"Avg Follower V over top-10 states (joint actions): {V_follower_top:.2f}")
        print(f"Follower TailPV @ mid-run (t={mid}): {float(F_tailPV[mid]):.2f}")
        print(f"Total joint Q-table entries: {len(follower.q)}")

    print("--- End Diagnostics ---\n")

    # Return dict (v1 keys unchanged; V2 keys added)
    return {
        "R": R,
        "tau": tau_trace, "d": d_trace,
        "rho": rho_trace,           # V2
        "M": M_trace, "X": X_trace,
        "E": E_trace,               # V2
        "Lpay": Lpay, "Fpay": Fpay,
        "export_loss": export_loss_trace,  # V2
        "rho_cost": rho_cost_trace,        # V2
        "L_tailPV": L_tailPV, "F_tailPV": F_tailPV,
        "L_cumdisc": L_cumdisc, "F_cumdisc": F_cumdisc,
        "V_leader_raw": V_leader_raw,
        "V_follower_raw": V_follower_raw,
        "gap_leader_raw": gap_leader_raw,
        "gap_follower_raw": gap_follower_raw,
        "leader": leader, "follower": follower,
        "env": env, "params": params,
    }


# ============================================================
# Run directly
# ============================================================
if __name__ == "__main__":
    out = test_stackelberg_q3_tariff_econ_v2(
        steps=500000,
        tau_max=0.5,
        d_max=0.15,
        n_bins=6,
        n_rho_bins=5,          # V2: 5 rho bins → 5×6=30 joint actions (d_bins=6)
        leader_q_init=3000.0,
        follower_q_init=3000.0,
        epsilon_start=0.15,
        alpha=0.18,
        follower_double_q=True,
        smooth_win=500,
        psi_E=1.0,             # V2: symmetric export loss weight
        rho_max=0.35,          # V2: matches tau_max convention
        follower_cost_rho=0.0, # V2: free retaliation to start
    )
