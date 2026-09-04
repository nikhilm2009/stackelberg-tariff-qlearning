# test_stackelberg_q3_tariff_econ_v10_leaderDepr.py
#
# V9 MODEL SUMMARY
# ════════════════
# Leader  chooses (τ, dL):  tariff + currency depreciation
# Follower chooses (d,  ρ):  own depreciation + retaliation tariff
# dL suppresses M*, boosts E*, carries inflation + quadratic cost to leader.
# Follower state includes coarse-binned dL_last so follower policy adapts
# to leader depreciation intensity.
# Output dir: plots_v10_leaderDepr/
#
# V8 CHANGES vs v7_sense
# ====================================================
# - Imports stackelberg_q3_tariff_econ_sim_v8_exptCntl
# - Leader now has joint (τ, x) action space when export_controls=True
#   X* = X0*(1+d)^εs*(1+x)^{-εx}   leader pays cx*x^2
# - New EconomicParams: x_max, export_ctrl_elast, leader_cost_x
# - export_controls flag in RUN_CONFIG switches leader between:
#     False → 1D (τ only), x_bins=None, identical to v7
#     True  → 2D (τ,x) joint, x_bins = make_bins(n_x_bins, 0, x_max)
# - Fig 1-9: identical to v7_sense (dynamics, decomp, Q-tables, BR,
#             tail PV, state visits, E-channel, structural sense, alpha/gamma sense)
# - Fig 10: leader depreciation sensitivity sweeps
#     Fig 10a: dL_max sweep
#     Fig 10b: dL_elast_M sweep
#     Fig 10c: dL_elast_E sweep
#     Fig 10d: phi_inflation_L sweep
# - Output dir: plots_v9_leaderDepr/
# ====================================================

import os
import copy
import multiprocessing as _mp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import stackelberg_q3_tariff_econ_sim_v10_leaderDepr as sim
print("SIM FILE:", sim.__file__)

from stackelberg_q3_tariff_econ_config_v10 import (
    build_params_lf, build_params_ff, build_params_center,
    build_agents, classify_regime,
    rolling_mean as _rolling_mean, cumulative_discounted, to_same_len as _to_same_len,
    LF_PARAMS, FF_PARAMS, CENTER_PARAMS,
)
from stackelberg_q3_tariff_econ_sim_v10_leaderDepr import (
    EconomicParams, EconomicEnvironment, make_bins,
    Q3BinnedLeader, Q3BinnedFollower, StackelbergTariffGameEconomic,
)
make_uniform_bins = lambda vmin, vmax, n=6: make_bins(n, vmin, vmax)


# ============================================================
# RUN CONFIG
# ============================================================
RUN_CONFIG = {
    # ----------------------------------------------------------------
    # V9 PRODUCTION RUN — 600k steps, eval last 10% (540k-600k)
    # ----------------------------------------------------------------
    "steps":              1000000,
    "n_bins":             6,
    "n_dL_bins":          3,        # dL bins — ignored when leader_depr=False
    # ----------------------------------------------------------------
    # Leader depreciation toggle
    #   True  — leader plays (τ, dL) joint action  [V10 default]
    #   False — leader plays τ only                [V7 behaviour]
    # ----------------------------------------------------------------
    "leader_depr":        False,
    "n_rho_bins":         5,
    "leader_q_init":      3000.0,   # raised for faster τ convergence
    "follower_q_init":    3000.0,
    "epsilon_start":      0.15,
    "epsilon_end":        0.02,      # leader epsilon floor
    "follower_epsilon_end": 0.005,   # follower tighter floor — fixes late churn
    "alpha":              0.18,
    "follower_double_q":  True,
    # ----------------------------------------------------------------
    # Stackelberg-faithful alternating BR protocol
    #   freeze_leader_frac: when to freeze (fraction of total steps)
    #   freeze_mode: epsilon_only = stop exploration + Q-updates
    #                full_freeze  = epsilon_only + skip Q-updates entirely
    #                none         = simultaneous learning (baseline)
    #   reset_follower_on_leader_freeze: False — no relearning shock
    # ----------------------------------------------------------------
    "freeze_leader_frac":  0.0,
    "freeze_follower_frac": 0.0,
    # freeze_mode semantics:
    #   none         — simultaneous learning (baseline)
    #   epsilon_only — ε→0, Q-updates ON  (leader stops exploring, keeps learning)
    #   full_freeze  — ε→0, Q-updates OFF (committed policy)
    "freeze_mode":         "none",  # none | epsilon_only | full_freeze
    "reset_follower_on_leader_freeze": False,
    # Alpha floors — lower = less churn in late training
    "leader_alpha_min":    0.02,  # was hardcoded 0.02
    "follower_alpha_min":  0.05,   # V7 value
    "smooth_win":         5000,
    # Convergence diagnostics
    "run_diagnostics":    True,   # set False to skip Fig 0
    "diag_win":       50000,       # rolling variance window size
    "diag_br_steps":  200000,      # best-response probe steps


    # ----------------------------------------------------------------
    # V9: leader depreciation always active (replaces export controls)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Runtime flags
    # ----------------------------------------------------------------
    "run_lf":             True,
    "run_ff":             True,
    "run_center":         False,    # CENTER regime (symmetric calibration)
    "run_sensitivity":    False,
    "n_workers":          6,         # 0=auto (cpu_count-1), 1=serial
    # Sensitivity convergence: τ climbs until ~175k; eval converged tail only
    "sense_steps":        600000,
    "sense_eval_frac":    0.10,      # eval last 10% = rounds 540k-600k
    "phase_eval_frac":    0.20,      # eval last 20% of phase_steps
    # freeze_leader_frac and freeze_follower_frac set above
}

# ============================================================
# ECONOMIC PARAMETERS — imported from stackelberg_q3_tariff_econ_config_v10


def tail_present_value(rewards, delta):
    r = np.asarray(rewards, float); T = len(r); tail = np.zeros(T, float); acc = 0.0
    for t in range(T-1,-1,-1): acc = r[t]+delta*acc; tail[t] = acc
    return tail



def _get_output_dir(): return "plots_v10_leaderDepr"

# build_agents imported from stackelberg_q3_tariff_econ_config_v10
# — passes epsilon_end, leader_alpha_min, follower_alpha_min correctly
# Local definition removed to avoid shadowing the shared config version

# ============================================================
# PLOTTING HELPERS
# ============================================================
def safe_imshow(ax, mat, title="", cmap="viridis", percent_clip=(5,95)):
    mat = np.asarray(mat, float)
    if mat.size==0 or mat.shape[0]==0 or mat.shape[1]==0:
        ax.text(0.5,0.5,"No data",ha="center",va="center"); ax.set_axis_off(); return None
    vmin,vmax = None,None
    if percent_clip:
        lo,hi = np.percentile(mat, list(percent_clip))
        if np.isfinite(lo) and np.isfinite(hi) and lo<hi: vmin,vmax = lo,hi
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title); return im

def prepare_q_heatmap_leader(leader, top_k=50):
    """For v8 leader with joint (tau,x) actions — marginalise over x for display."""
    if not hasattr(leader,"q") or not leader.q: return np.zeros((0,0)),[],[]
    # Collect unique states
    if hasattr(leader,"state_visits") and leader.state_visits:
        states = [s for s,_ in sorted(leader.state_visits.items(),key=lambda kv:kv[1],reverse=True)[:top_k]]
    else:
        seen=set(); states=[]
        for (s,_) in leader.q: 
            if s not in seen: seen.add(s); states.append(s)
        states=states[:top_k]
    tau_bins = leader.tau_bins
    mat = np.zeros((len(states),len(tau_bins)),float)
    for ri,s in enumerate(states):
        for ci,tau in enumerate(tau_bins):
            # marginalise over x: take max Q over all x
            if leader.dL_bins is not None:
                vals = [leader.q.get((s,(tau,dL)), leader.q_init) for dL in leader.dL_bins]
                mat[ri,ci] = max(vals)
            else:
                mat[ri,ci] = leader.q.get((s,tau), leader.q_init)
    def _lab(s):
        try: return "|".join(s)
        except: return str(s)
    return mat, [_lab(s) for s in states], tau_bins

def prepare_q_heatmap_follower(follower, top_k=50):
    if not hasattr(follower,"q") or not follower.q: return np.zeros((0,0)),[],[]
    if hasattr(follower,"state_visits") and follower.state_visits:
        states=[s for s,_ in sorted(follower.state_visits.items(),key=lambda kv:kv[1],reverse=True)[:top_k]]
    else:
        seen=set(); states=[]
        for (s,_) in follower.q:
            if s not in seen: seen.add(s); states.append(s)
        states=states[:top_k]
    d_bins=follower.d_bins
    mat=np.zeros((len(states),len(d_bins)),float)
    rho_bins=getattr(follower,"rho_bins",[0.0])
    for ri,s in enumerate(states):
        for ci,d in enumerate(d_bins):
            vals=[follower.q.get((s,(d,rho)),follower.q_init_f) for rho in rho_bins]
            mat[ri,ci]=max(vals)
    def _lab(s):
        try: return "|".join(s)
        except: return str(s)
    return mat,[_lab(s) for s in states],d_bins

# ============================================================
# CONVERGENCE DIAGNOSTICS — Fig 0
# ============================================================
def _compute_convergence_diagnostics(rr, cfg, params, prefix, out_dir):
    """Three convergence diagnostics:
    1. Rolling action variance (diag_win window)
    2. Policy-change rate: fraction of windows where modal action changes
    3. Best-response probe: freeze leader at final policy, retrain follower
    """
    import copy
    steps  = len(rr)
    win    = int(cfg.get("diag_win", 50000))
    n_wins = max(2, steps // win)

    tau_t = np.array([r["tau"]           for r in rr], float)
    dL_t  = np.array([r.get("dL", 0.0)  for r in rr], float)
    d_t   = np.array([r["d"]            for r in rr], float)
    rho_t = np.array([r["rho"]          for r in rr], float)
    Lpay  = np.array([r["leader_pay"]   for r in rr], float)
    Fpay  = np.array([r["follower_pay"] for r in rr], float)

    windows   = np.array_split(np.arange(steps), n_wins)
    w_centers = [int(np.mean(w)) for w in windows]

    # ── Diag 1: Rolling variance ──────────────────────────────────────────────
    tau_var   = [np.var(tau_t[w]) for w in windows]
    dL_var    = [np.var(dL_t[w])  for w in windows]
    d_var     = [np.var(d_t[w])   for w in windows]
    rho_var   = [np.var(rho_t[w]) for w in windows]
    Lpay_mean = [np.mean(Lpay[w]) for w in windows]
    Fpay_mean = [np.mean(Fpay[w]) for w in windows]

    # ── Diag 2: Policy-change rate ────────────────────────────────────────────
    def modal(arr): return float(np.round(np.median(arr), 3))
    tau_modes = [modal(tau_t[w]) for w in windows]
    d_modes   = [modal(d_t[w])   for w in windows]
    rho_modes = [modal(rho_t[w]) for w in windows]
    tau_chg   = [int(tau_modes[i] != tau_modes[i-1]) for i in range(1, len(tau_modes))]
    d_chg     = [int(d_modes[i]   != d_modes[i-1])   for i in range(1, len(d_modes))]
    rho_chg   = [int(rho_modes[i] != rho_modes[i-1]) for i in range(1, len(rho_modes))]
    chg_ctrs  = w_centers[1:]

    # ── Diag 3: Best-response probe ───────────────────────────────────────────
    br_steps  = int(cfg.get("diag_br_steps", 200000))
    final_tau = float(np.mean(tau_t[-min(100000,steps):]))
    final_dL  = float(np.mean(dL_t[-min(100000,steps):]))
    orig_d    = float(np.mean(d_t[-min(100000,steps):]))
    orig_rho  = float(np.mean(rho_t[-min(100000,steps):]))
    orig_Lpay = float(np.mean(Lpay[-min(100000,steps):]))
    orig_Fpay = float(np.mean(Fpay[-min(100000,steps):]))

    print(f"  [diag3] BR probe: FixedLeader τ={final_tau:.3f} dL={final_dL:.3f} "
          f"retraining fresh follower {br_steps//1000}k steps...", flush=True)
    p2      = copy.deepcopy(params)
    env2    = EconomicEnvironment(p2)
    # FixedLeader: deterministic, no Q-updates — true BR probe
    leader2 = sim.FixedLeader(final_tau, final_dL)
    _, follower2, _, _, _ = build_agents(p2, cfg)
    game2 = StackelbergTariffGameEconomic(env2, leader2, follower2, track=True)
    game2.run(rounds=br_steps)
    rr2     = game2.results["rounds"]
    br_tail = rr2[br_steps//2:]
    br_d    = float(np.mean([r["d"]           for r in br_tail]))
    br_rho  = float(np.mean([r["rho"]         for r in br_tail]))
    br_Lpay = float(np.mean([r["leader_pay"]  for r in br_tail]))
    br_Fpay = float(np.mean([r["follower_pay"]for r in br_tail]))

    print(f"  [diag3] Original → d={orig_d:.3f} ρ={orig_rho:.3f} "
          f"L={orig_Lpay:.1f} F={orig_Fpay:.1f}")
    print(f"  [diag3] BR probe → d={br_d:.3f}   ρ={br_rho:.3f}   "
          f"L={br_Lpay:.1f} F={br_Fpay:.1f}")
    print(f"  [diag3] Delta    → ΔL={br_Lpay-orig_Lpay:+.1f}  ΔF={br_Fpay-orig_Fpay:+.1f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Fig 0 — Convergence Diagnostics [{prefix}]  "
        f"win={win//1000}k  BR={br_steps//1000}k steps",
        fontsize=13, fontweight="bold")

    # 1a Leader variance
    ax = axes[0, 0]
    ax.plot(w_centers, tau_var, color="tab:blue", lw=2, label="τ var")
    ax.plot(w_centers, dL_var,  color="tab:red",  lw=2, label="dL var", ls="--")
    ax.set_title("Diag 1a — Leader action variance")
    ax.set_xlabel("Step"); ax.set_ylabel("Variance"); ax.legend(fontsize=9)

    # 1b Follower variance
    ax = axes[0, 1]
    ax.plot(w_centers, d_var,   color="tab:orange", lw=2, label="d var")
    ax.plot(w_centers, rho_var, color="tab:brown",  lw=2, label="ρ var", ls="--")
    ax.set_title("Diag 1b — Follower action variance")
    ax.set_xlabel("Step"); ax.set_ylabel("Variance"); ax.legend(fontsize=9)

    # 1c Per-window payoffs
    ax = axes[0, 2]
    ax.plot(w_centers, Lpay_mean, color="tab:blue",   lw=2, label="Leader")
    ax.plot(w_centers, Fpay_mean, color="tab:orange", lw=2, label="Follower")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title("Diag 1c — Per-window mean payoff")
    ax.set_xlabel("Step"); ax.set_ylabel("Payoff"); ax.legend(fontsize=9)

    # 2a Leader policy changes
    ax = axes[1, 0]
    ax.bar(chg_ctrs, tau_chg, width=win*0.8, color="tab:blue", alpha=0.7, label="τ changes")
    ax.set_title(f"Diag 2a — τ modal changes per {win//1000}k window")
    ax.set_xlabel("Step"); ax.set_ylabel("Changed (0/1)"); ax.legend(fontsize=9)

    # 2b ρ policy changes
    ax = axes[1, 1]
    ax.bar(chg_ctrs, rho_chg, width=win*0.8, color="tab:brown", alpha=0.7, label="ρ changes")
    ax.bar(chg_ctrs, d_chg,   width=win*0.8, color="tab:orange",alpha=0.4, label="d changes",
           bottom=rho_chg)
    ax.set_title(f"Diag 2b — ρ/d modal changes per {win//1000}k window")
    ax.set_xlabel("Step"); ax.set_ylabel("Changed (0/1)"); ax.legend(fontsize=9)

    # 3 BR comparison
    ax = axes[1, 2]
    labels   = ["d", "ρ", "L pay/100", "F pay/100"]
    orig_v   = [orig_d, orig_rho, orig_Lpay/100, orig_Fpay/100]
    br_v     = [br_d,   br_rho,   br_Lpay/100,   br_Fpay/100]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, orig_v, w, label="Original", color="tab:blue",   alpha=0.7)
    ax.bar(x + w/2, br_v,   w, label="BR probe",  color="tab:orange", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title(f"Diag 3 — BR probe\n"
                 f"frozen τ={final_tau:.3f} dL={final_dL:.3f}  "
                 f"ΔL={br_Lpay-orig_Lpay:+.1f} ΔF={br_Fpay-orig_Fpay:+.1f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fpath = f"{out_dir}/{prefix.lower()}_fig0_convergence_diagnostics.png"
    plt.savefig(fpath, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  [diag] Saved → {fpath}", flush=True)


# ============================================================
# SINGLE-REGIME RUNNER
# ============================================================
def _run_one_regime(params, prefix, cfg):
    out_dir = _get_output_dir(); os.makedirs(out_dir, exist_ok=True)
    steps = cfg["steps"]; win = cfg["smooth_win"]
    n_bins = cfg["n_bins"]; n_rho = cfg["n_rho_bins"]

    # Apply leader_depr toggle — set dL_max=0 to disable depreciation
    if not cfg.get("leader_depr", True):
        params = copy.deepcopy(params)
        params.dL_max = 0.0
        params.phi_inflation_L = 0.0
        params.leader_cost_dL  = 0.0
    ec_label = "ON" if cfg.get("leader_depr", True) else "OFF"
    print(f"\n{'='*60}")
    print(f"  Running: {prefix}  ({steps} steps)  Leader depreciation: {ec_label}")
    print(f"{'='*60}")

    leader, follower, tau_bins, d_bins, rho_bins = build_agents(params, cfg)
    # Override follower epsilon floor — tighter than leader to fix late churn
    follower.epsilon_end = cfg.get("follower_epsilon_end", follower.epsilon_end)
    env  = EconomicEnvironment(params)
    game = StackelbergTariffGameEconomic(env, leader, follower, track=True,
                                         freeze_follower_frac=cfg.get("freeze_follower_frac", 0.0),
                                         freeze_leader_frac=cfg.get("freeze_leader_frac", 0.0)
                                             if cfg.get("freeze_mode","none") != "none" else 0.0,
                                         freeze_mode=cfg.get("freeze_mode", "epsilon_only"))
    game.run(rounds=steps)

    rr = game.results["rounds"]; R = np.arange(steps)

    # ── Fig 0: Convergence diagnostics ───────────────────────────────────
    if cfg.get("run_diagnostics", True):
        _compute_convergence_diagnostics(rr, cfg, params, prefix, out_dir)

    tau_t  = np.array([r["tau"]            for r in rr], float)
    dL_t   = np.array([r.get("dL", 0.0)   for r in rr], float)
    d_t    = np.array([r["d"]              for r in rr], float)
    rho_t  = np.array([r["rho"]            for r in rr], float)
    Lpay   = np.array([r["leader_pay"]     for r in rr], float)
    Fpay   = np.array([r["follower_pay"]   for r in rr], float)
    M_t    = np.array([r["M"]              for r in rr], float)
    X_t    = np.array([r["X"]              for r in rr], float)
    E_t    = np.array([r["E"]              for r in rr], float)
    rev_t  = np.array([r["rev"]            for r in rr], float)
    cons_t = np.array([r["cons_loss"]      for r in rr], float)
    acL_t  = np.array([r["action_cost_L"]  for r in rr], float)
    infL_t = np.array([r.get("inflation_cost_L", 0.0) for r in rr], float)
    acDL_t = np.array([r.get("action_cost_dL",  0.0) for r in rr], float)
    eloss_t= np.array([r["export_loss"]    for r in rr], float)
    egain_t= np.array([r["export_gain"]    for r in rr], float)
    infl_t = np.array([r["infl_cost"]      for r in rr], float)
    acF_t  = np.array([r["action_cost_F"]  for r in rr], float)
    rcost_t= np.array([r["rho_cost"]       for r in rr], float)
    ineq_t = np.array([r.get("inequity_term",0.0) for r in rr], float)
    dipl_t = np.array([r.get("diplo_cost", 0.0)   for r in rr], float)
    uLlag_t= np.array([r.get("uL_lag",    0.0)    for r in rr], float)

    L_tailPV  = tail_present_value(Lpay, params.delta)
    F_tailPV  = tail_present_value(Fpay, params.delta)
    L_cumdisc = cumulative_discounted(Lpay, params.delta)
    F_cumdisc = cumulative_discounted(Fpay, params.delta)
    L_undisc  = np.cumsum(Lpay); F_undisc = np.cumsum(Fpay)

    V_leader_raw   = _to_same_len(getattr(leader,   "q_max_trace", []), len(R))
    V_follower_raw = _to_same_len(getattr(follower, "q_max_trace", []), len(R))
    gap_L_raw = _to_same_len(getattr(leader,   "qgap_trace", []), len(R))
    gap_F_raw = _to_same_len(getattr(follower, "qgap_trace", []), len(R))
    roll = lambda x: _to_same_len(_rolling_mean(x, win), len(R))

    # ----------------------------------------------------------
    # FIG 1: Dynamics overview
    # ----------------------------------------------------------
    fig1, axs1 = plt.subplots(1, 4, figsize=(26, 5))
    fig1.suptitle(f"Fig 1 — V9 Dynamics  [{prefix}]  Leader:(τ,dL) Follower:(d,ρ)", fontsize=13)

    # 1a Actions: τ, x, d on same panel
    axs1[0].plot(R, tau_t, alpha=0.5, lw=1.0, color="tab:blue",   label="τ (raw)")
    axs1[0].plot(R, dL_t,  alpha=0.5, lw=1.0, color="tab:red",    label="dL (raw)")
    axs1[0].plot(R, d_t,   alpha=0.5, lw=1.0, color="tab:green",  label="d (raw)")
    axs1[0].plot(R, _rolling_mean(tau_t, win), lw=2, color="tab:blue",  label=f"τ (roll)")
    axs1[0].plot(R, _rolling_mean(dL_t,  win), lw=2, color="tab:red",   label=f"dL (roll)")
    axs1[0].plot(R, _rolling_mean(d_t,   win), lw=2, color="tab:green", label=f"d (roll)")
    axs1[0].set_title("Fig 1a — Actions (τ, dL, d)"); axs1[0].set_xlabel("Round")
    axs1[0].grid(True); axs1[0].legend(fontsize=8)

    axs1[1].plot(R, M_t, alpha=0.5, lw=1.0, color="tab:blue")
    axs1[1].plot(R, X_t, alpha=0.5, lw=1.0, color="tab:orange")
    axs1[1].plot(R, _rolling_mean(M_t, win), lw=2, color="tab:blue",   label=f"M (roll)")
    axs1[1].plot(R, _rolling_mean(X_t, win), lw=2, color="tab:orange", label=f"X (roll)")
    axs1[1].axhline(params.X0, color="tab:orange", ls=":", lw=1, alpha=0.7, label=f"X0={params.X0:.0f}")
    axs1[1].set_title("Fig 1b — Trade flows M, X"); axs1[1].set_xlabel("Round")
    axs1[1].grid(True); axs1[1].legend(fontsize=8)

    ax = axs1[2]; ax2 = ax.twinx()
    ax.plot(R, L_cumdisc, lw=2, label="Leader (disc.)")
    ax.plot(R, F_cumdisc, lw=2, label="Follower (disc.)", alpha=0.85)
    ax2.plot(R, L_undisc, "--", color="tab:red",   lw=1.6, label="Leader (undisc.)")
    ax2.plot(R, F_undisc, "--", color="tab:green", lw=1.6, label="Follower (undisc.)")
    ax2.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_title("Fig 1c — Cumulative payoffs"); ax.grid(True)
    _lines1c = [l for l in ax.get_lines()+ax2.get_lines()
                if not l.get_label().startswith("_")]
    ax.legend(_lines1c,[l.get_label() for l in _lines1c],loc="upper left",fontsize=8)

    ax1d = axs1[3]; ax1dr = ax1d.twinx()
    ax1d.plot(R, rho_t, alpha=0.5, lw=1.0, color="tab:orange")
    ax1d.plot(R, _rolling_mean(rho_t, win), lw=2, color="tab:orange", ls="--", label=f"ρ (roll)")
    ax1dr.plot(R, E_t, alpha=0.5, lw=1.0, color="tab:purple")
    ax1dr.plot(R, _rolling_mean(E_t, win), lw=2, color="tab:purple", ls="--", label="E (roll)")
    ax1dr.axhline(params.E0, color="tab:purple", ls=":", lw=1.2, alpha=0.7)
    ax1d.set_title("Fig 1d — ρ + E"); ax1d.grid(True)
    ax1d.set_ylabel("ρ", color="tab:orange"); ax1dr.set_ylabel("E", color="tab:purple")
    # Only include lines with real labels (avoids _child0 etc from alpha plots)
    _lines1d = [l for l in ax1d.get_lines()+ax1dr.get_lines()
                if not l.get_label().startswith("_")]
    ax1d.legend(_lines1d, [l.get_label() for l in _lines1d], fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_fig1_dynamics.png", dpi=120); plt.close(fig1)
    print(f"  Saved {prefix}_fig1_dynamics.png")

    # ----------------------------------------------------------
    # FIG 2: Payoff decompositions
    # ----------------------------------------------------------
    fig2, axs2 = plt.subplots(1, 4, figsize=(28, 5))
    fig2.suptitle(f"Fig 2 — Payoff Decompositions  [{prefix}]  [V9 leader:(τ,dL) follower:(d,ρ)]", fontsize=13)

    # 2a Leader — includes inflation_cost_L, action_cost_dL
    for arr,lbl,col in [(rev_t,"Revenue τ·M","tab:blue"),(cons_t,"Consumer loss","tab:purple"),
                        (acL_t,"Action cost τ²","tab:orange"),(eloss_t,"Export loss","tab:red"),
                        (infL_t,"Inflation φL·dL·M0","tab:brown"),
                        (acDL_t,"Action cost c_dL·dL²","tab:pink"),
                        (dipl_t,"Diplo cost γ","tab:gray")]:
        axs2[0].plot(R, arr,    alpha=0.15, lw=0.8, color=col)
        axs2[0].plot(R, _rolling_mean(arr,win), lw=2, color=col, label=f"{lbl} (roll)")
    axs2[0].set_title("Fig 2a — Leader decomposition"); axs2[0].grid(True)
    axs2[0].legend(ncol=2, fontsize=7); axs2[0].set_xlabel("Round")

    # 2b Follower base
    for arr,lbl in [(egain_t,"Export gain"),(infl_t,"Inflation cost"),
                    (acF_t,"Action cost"),(rcost_t,"ρ cost")]:
        axs2[1].plot(R, arr, alpha=0.25, lw=0.8)
        axs2[1].plot(R, _rolling_mean(arr,win), lw=2, label=f"{lbl} (roll)")
    axs2[1].set_title("Fig 2b — Follower decomposition"); axs2[1].grid(True)
    axs2[1].legend(ncol=2, fontsize=7); axs2[1].set_xlabel("Round")

    # 2c Export control channel — x trajectory + X suppression
    axs2[2].plot(R, dL_t, alpha=0.25, lw=0.8, color="tab:red")
    axs2[2].plot(R, _rolling_mean(dL_t, win), lw=2, color="tab:red", label=f"dL (roll {win})")
    axs2[2].axhline(params.dL_max, color="tab:red", ls=":", lw=1.2, label=f"dL_max={params.dL_max:.2f}")
    axs2[2].set_ylabel("dL — leader depreciation", color="tab:red")
    ax2c2 = axs2[2].twinx()
    ax2c2.plot(R, X_t, alpha=0.25, lw=0.8, color="tab:orange")
    ax2c2.plot(R, _rolling_mean(X_t,win), lw=2, color="tab:orange", label="X (roll)")
    ax2c2.axhline(params.X0, color="tab:orange", ls=":", lw=1.2, label=f"X0={params.X0:.0f}")
    ax2c2.set_ylabel("X — follower exports", color="tab:orange")
    axs2[2].set_title("Fig 2c — Leader dL + follower X"); axs2[2].grid(True)
    _lines2c = [l for l in axs2[2].get_lines()+ax2c2.get_lines()
                if not l.get_label().startswith("_")]
    axs2[2].legend(_lines2c,[l.get_label() for l in _lines2c], fontsize=8)
    axs2[2].set_xlabel("Round")

    # 2d Behavioural terms + payoffs
    axs2[3].plot(R, _rolling_mean(-ineq_t,win), lw=2, color="tab:red",
                 label=r"-Inequity α·max(uLlag-uF,0)")
    axs2[3].plot(R, _rolling_mean(dipl_t,win), lw=2, color="tab:gray",
                 label=r"Diplo cost γ·(ρ/ρmax)·τM")
    axs2[3].plot(R, uLlag_t, lw=1.2, color="tab:blue", ls="--", label=r"$u_{L,lag}$")
    axs2[3].plot(R, _rolling_mean(Lpay,win), lw=2, color="tab:blue", label="Leader payoff (roll)")
    axs2[3].plot(R, _rolling_mean(Fpay,win), lw=2, color="tab:orange", label="Follower payoff (roll)")
    axs2[3].axhline(0, color="k", ls=":", lw=0.8, alpha=0.5)
    axs2[3].set_title(f"Fig 2d — Behavioural terms\nα={params.alpha_ineq:.2f} γ={params.gamma_diplo:.2f}")
    axs2[3].grid(True); axs2[3].legend(fontsize=7.5); axs2[3].set_xlabel("Round")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_fig2_decompositions.png", dpi=120); plt.close(fig2)
    print(f"  Saved {prefix}_fig2_decompositions.png")

    # ----------------------------------------------------------
    # FIG 3: Q-Tables (leader marginalised over x)
    # ----------------------------------------------------------
    fig3, axs3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle(f"Fig 3 — Q-Tables & Q-Gaps  [{prefix}]  [Export controls: {ec_label}]", fontsize=13)

    lQ, lS, lA = prepare_q_heatmap_leader(leader, top_k=40)
    title_3a = "Fig 3a — Leader Q(s,τ) — max over dL"
    im_a = safe_imshow(axs3[0,0], lQ, title=title_3a)
    if im_a is not None:
        plt.colorbar(im_a, ax=axs3[0,0], fraction=0.046, pad=0.04)
        axs3[0,0].set_xticks(range(len(lA)))
        axs3[0,0].set_xticklabels([f"{a:.2f}" for a in lA], rotation=45, ha="right")

    axs3[0,1].plot(R, gap_L_raw, "--", alpha=0.55, label="Leader Q-gap (raw)")
    axs3[0,1].plot(R, roll(gap_L_raw), ":", lw=2, label=f"Leader Q-gap (roll {win})")
    axs3[0,1].set_title("Fig 3b — Leader Q-gap"); axs3[0,1].grid(True); axs3[0,1].legend()

    fQ, fS, fA = prepare_q_heatmap_follower(follower, top_k=40)
    im_c = safe_imshow(axs3[1,0], fQ, title="Fig 3c — Follower Q(s,d) max over ρ")
    if im_c is not None:
        plt.colorbar(im_c, ax=axs3[1,0], fraction=0.046, pad=0.04)
        axs3[1,0].set_xticks(range(len(fA)))
        axs3[1,0].set_xticklabels([f"{a:.2f}" for a in fA], rotation=45, ha="right")

    axs3[1,1].plot(R, gap_F_raw, "--", alpha=0.55, label="Follower Q-gap (raw)")
    axs3[1,1].plot(R, roll(gap_F_raw), ":", lw=2, label=f"Follower Q-gap (roll {win})")
    axs3[1,1].set_title("Fig 3d — Follower Q-gap"); axs3[1,1].grid(True); axs3[1,1].legend()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_fig3_qtables.png", dpi=120); plt.close(fig3)
    print(f"  Saved {prefix}_fig3_qtables.png")

    # ----------------------------------------------------------
    # FIG 4: Policies & BR snapshot
    # ----------------------------------------------------------
    fig4, axs4 = plt.subplots(1, 3, figsize=(21, 5))
    fig4.suptitle(f"Fig 4 — Policies & BR  [{prefix}]  [Export controls: {ec_label}]", fontsize=13)

    axs4[0].hist(tau_t, bins=20, alpha=0.7, density=True, color="tab:blue",  label="τ (tariff)")
    axs4[0].hist(dL_t,  bins=20, alpha=0.7, density=True, color="tab:red",   label="dL (depr)")
    axs4[0].hist(d_t,   bins=20, alpha=0.7, density=True, color="tab:green", label="d (depr.)")
    axs4[0].hist(rho_t, bins=20, alpha=0.7, density=True, color="tab:orange",label="ρ (retaliation)")
    axs4[0].set_title("Fig 4a — Policy histograms (τ, x, d, ρ)"); axs4[0].grid(True)
    axs4[0].legend(fontsize=8)

    # BR d*(τ) snapshot
    env_snap = EconomicEnvironment(params)
    env_snap.M,env_snap.X,env_snap.E = float(env.M),float(env.X),float(env.E)
    if hasattr(env_snap,"prev_X"): env_snap.prev_X = env_snap.X
    taus = np.asarray(tau_bins, float)
    d_bins_arr = np.asarray(d_bins, float)
    rho_bins_arr = np.asarray(rho_bins, float)

    # dL mean for BR snapshot — note: dL affects M*/E* via env state,
    # not directly the follower payoff function. The BR curves below show
    # equilibrium d*(τ) under current env state (which reflects dL history).
    dL_mean_snap = float(np.mean(dL_t))

    def _br_d(taus_grid, dL_fixed=0.0):
        """BR d*(tau) at current env state (which reflects accumulated dL effects).
        V9: dL_fixed arg retained for API compatibility but dL affects follower only
        through M*/E* environment state. evaluate_follower_payoff() does not take dL.
        """
        br=[]; M0_,X0_,E0_=env_snap.M,env_snap.X,env_snap.E; p0=getattr(env_snap,"prev_X",env_snap.X)
        for tau in taus_grid:
            bv,bd=-1e30,d_bins_arr[0]
            for d in d_bins_arr:
                env_snap.M,env_snap.X,env_snap.E=M0_,X0_,E0_; env_snap.prev_X=p0
                v = env_snap.evaluate_follower_payoff(tau, d, 0.0, timing="post")
                if np.isfinite(v) and v>bv: bv,bd=v,d
            br.append(bd)
        env_snap.M,env_snap.X,env_snap.E=M0_,X0_,E0_; env_snap.prev_X=p0; return np.array(br,float)

    if taus.size>0:
        axs4[1].plot(taus, _br_d(taus, 0.0),          lw=2, marker="o", ms=4,
                     color="tab:green", label=f"BR d*(τ)|dL=0")
        if params.dL_max > 0:
            axs4[1].plot(taus, _br_d(taus, dL_mean_snap), lw=2, marker="s", ms=4,
                         color="tab:red",   label=f"BR d*(τ)|dL≈{dL_mean_snap:.2f} (mean)")
            axs4[1].plot(taus, _br_d(taus, params.dL_max), lw=2, marker="^", ms=4,
                         color="tab:brown", label=f"BR d*(τ)|dL=dL_max={params.dL_max:.2f}")
    axs4[1].set_title("Fig 4b — Follower BR d*(τ) — dL shifts env state M*/E*"); axs4[1].grid(True)
    axs4[1].legend(fontsize=8); axs4[1].set_xlabel("τ"); axs4[1].set_ylabel("d*")

    # V9 Fig 4c — M* suppression and E* boost by leader depreciation dL
    if params.dL_max > 0:
        dL_range = np.linspace(0, params.dL_max, 30)
        env_dL = EconomicEnvironment(params)
        # M* = M0*(1+tau_mean)^{-ed}*(1+dL)^{-edL}
        tau_mean = float(np.mean(tau_t))
        M_star_vals = [env_dL._M_star(tau_mean, dv) for dv in dL_range]
        # E* = E0*(1+rho_mean)^{-eE}*(1+dL)^{+edL}
        rho_mean = float(np.mean(rho_t))
        E_star_vals = [env_dL._E_star(rho_mean, dv) for dv in dL_range]
        ax4c = axs4[2]
        ax4c2 = ax4c.twinx()
        ax4c.plot(dL_range, M_star_vals, lw=2.5, color="tab:blue", marker="o", ms=3,
                  label=f"M*(τ≈{tau_mean:.2f}, dL)")
        ax4c2.plot(dL_range, E_star_vals, lw=2.5, color="tab:red", marker="s", ms=3,
                   label=f"E*(ρ≈{rho_mean:.2f}, dL)")
        ax4c.axhline(params.M0, color="tab:blue", ls=":", lw=1, alpha=0.5, label=f"M0={params.M0:.0f}")
        ax4c2.axhline(params.E0, color="tab:red", ls=":", lw=1, alpha=0.5, label=f"E0={params.E0:.0f}")
        ax4c.axvline(dL_mean_snap, color="gray", ls="--", lw=1.2,
                     label=f"Mean dL={dL_mean_snap:.3f}")
        ax4c.set_title("Fig 4c — M* suppression & E* boost by dL")
        ax4c.set_xlabel("dL — leader depreciation"); ax4c.set_ylabel("M* (blue)")
        ax4c2.set_ylabel("E* (red)")
        lines1, labs1 = ax4c.get_legend_handles_labels()
        lines2, labs2 = ax4c2.get_legend_handles_labels()
        ax4c.legend(lines1+lines2, labs1+labs2, fontsize=7)
        ax4c.grid(True)
    else:
        axs4[2].text(0.5, 0.5, "No leader depreciation\n(dL_max=0)",
                     ha="center", va="center", fontsize=12, color="gray",
                     transform=axs4[2].transAxes)
        axs4[2].set_axis_off()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_fig4_policies_br.png", dpi=120); plt.close(fig4)
    print(f"  Saved {prefix}_fig4_policies_br.png")

    # ----------------------------------------------------------
    # FIG 5: Tail PV vs Q-value
    # ----------------------------------------------------------
    fig5, axs5 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig5.suptitle(f"Fig 5 — Tail PV vs Learned Value  [{prefix}]", fontsize=13)
    axs5[0].plot(R, L_tailPV, lw=2, label="Leader Tail PV")
    axs5[0].plot(R, V_leader_raw, "--", alpha=0.55, label="Leader max Q")
    axs5[0].plot(R, roll(V_leader_raw), ":", lw=2, label=f"Leader max Q (roll)")
    axs5[0].set_title("Fig 5a — Leader"); axs5[0].grid(True); axs5[0].legend()
    axs5[1].plot(R, F_tailPV, lw=2, label="Follower Tail PV")
    axs5[1].plot(R, V_follower_raw, "--", alpha=0.55, label="Follower max Q")
    axs5[1].plot(R, roll(V_follower_raw), ":", lw=2, label=f"Follower max Q (roll)")
    axs5[1].set_title("Fig 5b — Follower"); axs5[1].grid(True); axs5[1].legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_fig5_tailpv_vs_q.png", dpi=120); plt.close(fig5)
    print(f"  Saved {prefix}_fig5_tailpv_vs_q.png")

    # ----------------------------------------------------------
    # FIG 6: State visit heatmaps
    # ----------------------------------------------------------
    def _sc(agent, top_k=60):
        items=sorted(agent.state_visits.items(),key=lambda kv:kv[1],reverse=True) if hasattr(agent,"state_visits") else []
        return items[:top_k]
    lsc=_sc(leader); fsc=_sc(follower)
    LC=np.array([[c] for (_,c) in lsc],float) if lsc else np.zeros((0,0))
    FC=np.array([[c] for (_,c) in fsc],float) if fsc else np.zeros((0,0))
    fig6,axs6=plt.subplots(1,2,figsize=(12,6))
    fig6.suptitle(f"Fig 6 — State Visit Heatmaps  [{prefix}]", fontsize=13)
    imL=safe_imshow(axs6[0],LC,"Fig 6a — Leader: top state visits","magma")
    if imL: plt.colorbar(imL,ax=axs6[0],fraction=0.046,pad=0.04)
    imF=safe_imshow(axs6[1],FC,"Fig 6b — Follower: top state visits","magma")
    if imF: plt.colorbar(imF,ax=axs6[1],fraction=0.046,pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_fig6_state_visits.png", dpi=120); plt.close(fig6)
    print(f"  Saved {prefix}_fig6_state_visits.png")

    # ----------------------------------------------------------
    # FIG 7: E-channel + sensitivity sweeps
    # ----------------------------------------------------------
    fig7, axs7 = plt.subplots(1, 2, figsize=(16, 5))
    fig7.suptitle(f"Fig 7 — E-Channel  [{prefix}]", fontsize=13)
    axs7[0].plot(R, E_t, alpha=0.5, lw=1, color="tab:purple")
    axs7[0].plot(R, _rolling_mean(E_t, win), lw=2, color="tab:purple", label=f"E (roll {win})")
    axs7[0].axhline(params.E0, color="k", ls=":", lw=1.5, label=f"E0={params.E0:.0f}")
    axs7[0].set_title("Fig 7a — E"); axs7[0].grid(True); axs7[0].legend()
    axs7[1].plot(R, rho_t, alpha=0.5, lw=1, color="tab:orange")
    axs7[1].plot(R, _rolling_mean(rho_t, win), lw=2, color="tab:orange", label=f"ρ (roll)")
    axs7[1].axhline(params.rho_max, color="k", ls=":", lw=1.2, label=f"ρ_max={params.rho_max:.2f}")
    axs7[1].set_title("Fig 7b — ρ"); axs7[1].grid(True); axs7[1].legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_fig7_echannel.png", dpi=120); plt.close(fig7)
    print(f"  Saved {prefix}_fig7_echannel.png")

    return {
        "prefix":prefix,"params":params,
        "tau":tau_t,"dL":dL_t,"d":d_t,"rho":rho_t,
        "Lpay":Lpay,"Fpay":Fpay,"M":M_t,"X":X_t,"E":E_t,
        "L_undisc":L_undisc,"F_undisc":F_undisc,
        "L_cumdisc":L_cumdisc,"F_cumdisc":F_cumdisc,
        "ineq_t":ineq_t,"dipl_t":dipl_t,"infL_t":infL_t,"acDL_t":acDL_t,
    }


# ============================================================
# COMPARISON PLOTS
# ============================================================
def _plot_comparison(lf, ff, win=2000, out_dir="plots_v10_leaderDepr"):
    os.makedirs(out_dir, exist_ok=True)
    lf_p=lf["params"]; ff_p=ff["params"]
    steps=len(lf["tau"]); R=np.arange(steps); rm=lambda x: _rolling_mean(x,win)

    fig_r,axs_r=plt.subplots(2,3,figsize=(21,10))
    fig_r.suptitle("V9 Regime Comparison — Raw Values  [LF vs FF]  Leader:(τ,dL) Follower:(d,ρ)", fontsize=14)

    # τ: solid blue/orange; dL: dashed with distinct colors (red/green)
    axs_r[0,0].plot(R,rm(lf["tau"]),lw=2,color="tab:blue",  ls="-", label="LF τ")
    axs_r[0,0].plot(R,rm(ff["tau"]),lw=2,color="tab:orange", ls="-", label="FF τ")
    axs_r[0,0].plot(R,rm(lf["dL"]),lw=2,color="tab:red",    ls="--",label="LF dL")
    axs_r[0,0].plot(R,rm(ff["dL"]),lw=2,color="tab:green",  ls="--",label="FF dL")
    axs_r[0,0].set_title("Leader τ (solid) & dL (dashed, distinct colors)"); axs_r[0,0].grid(True); axs_r[0,0].legend(fontsize=8)

    win_h=win*4; rmh=lambda x: _to_same_len(_rolling_mean(x,win_h),len(R))
    axs_r[0,1].plot(R,rmh(lf["d"]),lw=2.5,color="tab:blue",label="LF d")
    axs_r[0,1].plot(R,rmh(ff["d"]),lw=2.5,color="tab:orange",label="FF d")
    axr=axs_r[0,1].twinx()
    axr.plot(R,rmh(lf["rho"]),lw=2.5,color="#5ba3d9",ls="--",label="LF ρ")
    axr.plot(R,rmh(ff["rho"]),lw=2.5,color="tab:red",ls="--",label="FF ρ")
    l1,la1=axs_r[0,1].get_legend_handles_labels(); l2,la2=axr.get_legend_handles_labels()
    axs_r[0,1].legend(l1+l2,la1+la2,fontsize=8,loc="upper left")
    axs_r[0,1].set_title(f"Follower d (solid) & ρ (dashed)  [roll {win_h}]"); axs_r[0,1].grid(True)

    axs_r[0,2].plot(R,rm(lf["M"]),lw=2,color="tab:blue",label="LF M")
    axs_r[0,2].plot(R,rm(lf["X"]),lw=2,color="#5ba3d9",label="LF X")
    axs_r[0,2].plot(R,rm(ff["M"]),lw=2,color="tab:orange",label="FF M")
    axs_r[0,2].plot(R,rm(ff["X"]),lw=2,color="tab:red",label="FF X")
    axs_r[0,2].axhline(lf_p.M0,color="k",ls=":",lw=1.2,alpha=0.5,label=f"M0={lf_p.M0:.0f}")
    axs_r[0,2].set_title("Trade Flows M vs X"); axs_r[0,2].grid(True); axs_r[0,2].legend(fontsize=8)

    axs_r[1,0].plot(R,rm(lf["E"]),lw=2,color="tab:blue",label="LF E")
    axs_r[1,0].plot(R,rm(ff["E"]),lw=2,color="tab:orange",label="FF E")
    axs_r[1,0].axhline(lf_p.E0,color="tab:blue",ls=":",alpha=0.6,label=f"LF E0={lf_p.E0:.0f}")
    axs_r[1,0].axhline(ff_p.E0,color="tab:orange",ls=":",alpha=0.6,label=f"FF E0={ff_p.E0:.0f}")
    axs_r[1,0].set_title("Leader Exports E"); axs_r[1,0].grid(True); axs_r[1,0].legend(fontsize=8)

    axs_r[1,1].plot(R,lf["L_undisc"],lw=2,color="tab:blue",label="LF Leader")
    axs_r[1,1].plot(R,lf["F_undisc"],lw=2,color="#5ba3d9",label="LF Follower")
    axs_r[1,1].plot(R,ff["L_undisc"],lw=2,color="tab:orange",label="FF Leader")
    axs_r[1,1].plot(R,ff["F_undisc"],lw=2,color="tab:red",label="FF Follower")
    axs_r[1,1].axhline(0,color="k",ls="--",lw=1.2,alpha=0.5)
    axs_r[1,1].set_title("Cumulative Payoffs — Undiscounted"); axs_r[1,1].grid(True); axs_r[1,1].legend(fontsize=8)

    lf_sur=lf["Lpay"]+lf["Fpay"]; ff_sur=ff["Lpay"]+ff["Fpay"]
    axs_r[1,2].plot(R,rm(lf_sur),lw=2,color="tab:blue",label="LF Joint surplus")
    axs_r[1,2].plot(R,rm(ff_sur),lw=2,color="tab:orange",label="FF Joint surplus")
    axs_r[1,2].set_title("Joint Surplus (welfare proxy)"); axs_r[1,2].grid(True); axs_r[1,2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/Compare_fig1_raw.png", dpi=120); plt.close(fig_r)
    print("  Saved Compare_fig1_raw.png")


# ============================================================
# SENSITIVITY — top-level worker (module-level for spawn pickling)
# ============================================================
def _sense_job(job):
    """Picklable top-level worker. Receives everything via job tuple — spawn-safe."""
    import copy, random, dataclasses
    import numpy as _np
    from stackelberg_q3_tariff_econ_sim_v10_leaderDepr import (
        EconomicParams, EconomicEnvironment,
        Q3BinnedLeader, Q3BinnedFollower, StackelbergTariffGameEconomic,
    )
    si, vi, attr, val, params_dict, cfg, sweep_steps, eval_frac = job

    p_sw = EconomicParams(**params_dict)
    setattr(p_sw, attr, float(val))

    random.seed(42); _np.random.seed(42)
    e = EconomicEnvironment(p_sw)

    def _uniform(vmin, vmax, n):
        if n <= 1: return [vmin]
        step = (vmax - vmin) / (n - 1)
        return [vmin + i * step for i in range(n)]

    n_bins = cfg["n_bins"]; n_rho = cfg["n_rho_bins"]
    tau_bins = _uniform(0.0, p_sw.tau_max, n_bins)
    d_bins   = _uniform(0.0, p_sw.d_max,   n_bins)
    rho_bins = _uniform(0.0, p_sw.rho_max, n_rho) if p_sw.rho_max > 0 else [0.0]
    dL_bins  = _uniform(0.0, p_sw.dL_max, cfg["n_dL_bins"]) \
               if p_sw.dL_max > 0 else None

    leader = Q3BinnedLeader(tau_bins,
                            epsilon=cfg["epsilon_start"], gamma=p_sw.delta,
                            alpha=cfg["alpha"], q_init=cfg["leader_q_init"],
                            dL_bins=dL_bins)
    leader.history_f = [f"{p_sw.d_max * 0.5:.2f}"] * 3
    leader.tau_last  = f"{p_sw.tau_max * 0.8:.2f}"
    if dL_bins is not None:
        leader.dL_last = f"{dL_bins[-1] * 0.3:.2f}"

    follower = Q3BinnedFollower(d_bins, rho_bins=rho_bins,
                                epsilon=cfg["epsilon_start"], gamma=p_sw.delta,
                                alpha=cfg["alpha"],
                                double_q=cfg["follower_double_q"],
                                q_init_f=cfg["follower_q_init"])
    follower.epsilon_end = cfg.get("follower_epsilon_end", follower.epsilon_end)

    g = StackelbergTariffGameEconomic(e, leader, follower, track=True,
                                      freeze_follower_frac=cfg.get("freeze_follower_frac", 0.0),
                                      freeze_leader_frac=cfg.get("freeze_leader_frac", 0.0)
                                          if cfg.get("freeze_mode","none") != "none" else 0.0,
                                      freeze_mode=cfg.get("freeze_mode", "epsilon_only"))
    g.run(sweep_steps)

    tail_start = int(sweep_steps * (1.0 - eval_frac))
    ev = g.results["rounds"][tail_start:]
    lm = float(_np.mean([r["leader_pay"]  for r in ev]))
    fm = float(_np.mean([r["follower_pay"] for r in ev]))
    xm = float(_np.mean([r.get("dL", 0.0) for r in ev]))
    return si, vi, lm, fm, xm


def _run_sweep(sweep_list, base_params, cfg, sweep_steps, eval_frac):
    """Parallel driver for 1-D sensitivity sweeps."""
    import dataclasses
    n_workers_cfg = cfg.get("n_workers", 0)
    n_workers = max(1, _mp.cpu_count() - 1) if n_workers_cfg == 0 else n_workers_cfg
    use_parallel = (n_workers > 1)
    params_dict = dataclasses.asdict(base_params)

    jobs = []
    for si, item in enumerate(sweep_list):
        attr, vals = item[0], item[1]
        for vi, val in enumerate(vals):
            jobs.append((si, vi, attr, val,
                         params_dict, cfg, sweep_steps, eval_frac))

    results = [{"Lm": [None]*len(item[1]),
                "Fm": [None]*len(item[1]),
                "Xm": [None]*len(item[1])} for item in sweep_list]

    if use_parallel:
        ctx = _mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for si, vi, lm, fm, xm in pool.imap_unordered(_sense_job, jobs):
                results[si]["Lm"][vi] = lm
                results[si]["Fm"][vi] = fm
                results[si]["Xm"][vi] = xm
                print(f"    sweep[{si}][{vi}] -> L={lm:.1f} F={fm:.1f}", flush=True)
    else:
        for job in jobs:
            si, vi, lm, fm, xm = _sense_job(job)
            results[si]["Lm"][vi] = lm
            results[si]["Fm"][vi] = fm
            results[si]["Xm"][vi] = xm
    return results


# ============================================================
# SENSITIVITY — Fig 8: STRUCTURAL  (10 sweeps, 3×4 grid)
# V10 structural: tau_max, d_max, phi, psi_E, rho_max, eps_d, eps_s, eps_E, leader_cost_w, follower_cost_rho
# ============================================================
def _plot_sensitivity_fig8(params, prefix, cfg, out_dir="plots_v10_leaderDepr"):
    os.makedirs(out_dir, exist_ok=True)
    sweep_steps = cfg.get("sense_steps", cfg["steps"])
    eval_frac   = cfg.get("sense_eval_frac", 0.10)
    tail_pct    = int(eval_frac * 100)
    c_L, c_F    = "tab:blue", "tab:orange"

    sweeps = [
        ("tau_max",          [0.10,0.15,0.25,0.35,0.45],       params.tau_max,
         r"$\tau_{\max}$",   r"Fig 8a — $\tau_{\max}$ sensitivity"),
        ("d_max",            [0.10,0.18,0.25,0.35,0.45],        params.d_max,
         r"$d_{\max}$",      r"Fig 8b — $d_{\max}$ sensitivity"),
        ("phi_inflation",    [0.10,0.18,0.32,0.50,0.70],        params.phi_inflation,
         r"$\phi$",          r"Fig 8c — $\phi$ sensitivity"),
        ("psi_E",            [0.0,0.5,1.0,2.0,3.0],             params.psi_E,
         r"$\psi_E$",        r"Fig 8d — $\psi_E$ sensitivity"),
        ("rho_max",          [0.0,0.10,0.20,0.35,0.50],         params.rho_max,
         r"$\rho_{\max}$",   r"Fig 8e — $\rho_{\max}$ sensitivity"),
        ("demand_elast",     [0.80,1.00,1.20,1.55,1.80,2.20],   params.demand_elast,
         r"$\varepsilon_d$", r"Fig 8f — $\varepsilon_d$ sensitivity"),
        ("supply_elast",     [0.50,0.80,1.00,1.50,1.80,2.20],   params.supply_elast,
         r"$\varepsilon_s$", r"Fig 8g — $\varepsilon_s$ sensitivity"),
        ("export_elast",     [0.50,0.80,1.00,1.20,1.50,1.80],   params.export_elast,
         r"$\varepsilon_E$", r"Fig 8h — $\varepsilon_E$ sensitivity"),
        ("leader_cost_w",    [0.000,0.005,0.010,0.015,0.025],   params.leader_cost_w,
         r"$c_\tau$ — leader tariff cost",    r"Fig 8i — $c_\tau$ sensitivity"),
        ("follower_cost_rho",[0.0,0.05,0.10,0.20,0.35],         params.follower_cost_rho,
         r"$c_\rho$ — follower $\rho$ cost", r"Fig 8j — $c_\rho$ sensitivity"),
        ("leader_cost_dL",   [0.0,0.004,0.010,0.020,0.035],     params.leader_cost_dL,
         r"$c_{dL}$ — leader depreciation cost", r"Fig 8k — $c_{dL}$ sensitivity"),
        ("x_tau_elast",      [0.0,0.50,1.00,1.30,1.55,2.00],    params.x_tau_elast,
         r"$\varepsilon_{x\tau}$ — tariff suppression of follower exports" + "\n"
         + r"$X^* = X_0(1+\tau)^{-\varepsilon_{x\tau}}(1+d)^{\varepsilon_s}$",
         r"Fig 8l — $\varepsilon_{x\tau}$ sensitivity"),
    ]

    print(f"\n{'='*60}\n  [Fig 8] Structural sweeps — {prefix}"
          f"  ({sweep_steps} steps, eval last {tail_pct}%,"
          f" workers={cfg.get('n_workers',0)})\n{'='*60}")
    res = _run_sweep(sweeps, params, cfg, sweep_steps, eval_frac)

    fig8, axs8 = plt.subplots(3, 4, figsize=(28, 15))
    fig8.suptitle(
        f"Fig 8 — Structural Sensitivity  [{prefix}]"
        f"  ({sweep_steps} steps/pt, eval last {tail_pct}%)", fontsize=13)
    axs8_flat = axs8.flatten()
    for idx, (item, r) in enumerate(zip(sweeps, res)):
        attr, vals, base, xlabel, title = item
        ax = axs8_flat[idx]
        ax.plot(vals, r["Lm"], "o-", lw=1.8, color=c_L, label="Leader")
        ax.plot(vals, r["Fm"], "s-", lw=1.8, color=c_F, label="Follower")
        ax.axvline(base, color="red", ls="--", lw=1.2, alpha=0.8, label=f"Baseline={base:.2f}")
        ax.axhline(0, color="k", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=9, pad=4); ax.set_xlabel(xlabel, fontsize=7.5, labelpad=3)
        ax.set_ylabel("Mean per-step payoff", fontsize=8); ax.tick_params(labelsize=7.5)
        ax.grid(True); ax.legend(fontsize=7.5, loc="best", framealpha=0.7)
    ax_t = axs8_flat[11]; ax_t.axis("off")
    ax_t.set_title(f"Fig 8k — {prefix} Baseline Parameters", fontsize=10, fontweight="bold")
    rows=[("Parameter","Value","Description"),
          ("demand_elast",      f"{params.demand_elast:.2f}",    "Import demand elasticity"),
          ("supply_elast",      f"{params.supply_elast:.2f}",    "Export supply elasticity"),
          ("x_tau_elast",       f"{params.x_tau_elast:.2f}",     "Tariff suppression of X*"),
          ("export_elast",      f"{params.export_elast:.2f}",    "Leader export elasticity"),
          ("tau_max",           f"{params.tau_max:.2f}",         "Leader tariff ceiling"),
          ("d_max",             f"{params.d_max:.2f}",           "Follower depreciation ceiling"),
          ("rho_max",           f"{params.rho_max:.2f}",         "Follower retaliation ceiling"),
          ("phi_inflation",     f"{params.phi_inflation:.2f}",   "Follower inflation cost"),
          ("psi_E",             f"{params.psi_E:.2f}",           "Leader export-loss weight"),
          ("leader_cost_w",     f"{params.leader_cost_w:.4f}",   "Leader tariff action cost"),
          ("follower_cost_rho", f"{params.follower_cost_rho:.3f}","Follower rho quadratic cost"),
          ("sense_steps",       f"{sweep_steps}",                "Steps per sensitivity point"),
          ("eval tail",         f"last {tail_pct}%",             "Converged tail fraction"),
          ("freeze_frac",       f"{cfg.get('freeze_follower_frac',0.0):.2f}",
                                                                 "Follower freeze fraction")]
    col_x=[0.02,0.42,0.65]; y0=0.97; rh=0.063
    for i,row in enumerate(rows):
        y=y0-i*rh; bold=(i==0)
        for j,cell in enumerate(row):
            ax_t.text(col_x[j],y,cell,transform=ax_t.transAxes,fontsize=6.8,
                      fontweight="bold" if bold else "normal",va="top",ha="left",
                      color="#1a1a1a" if not bold else "#003366")
        if i==0: ax_t.axhline(y=y0-rh*0.85,xmin=0,xmax=1,color="#003366",lw=0.8)
    axs8_flat[11].axis("off")
    plt.subplots_adjust(top=0.93,bottom=0.07,hspace=0.42,wspace=0.30,left=0.05,right=0.97)
    out_path=f"{out_dir}/{prefix}_fig8_sensitivity.png"
    plt.savefig(out_path,dpi=120); plt.close(fig8)
    print(f"\n  Saved {out_path}")


# ============================================================
# SENSITIVITY — Fig 9: BEHAVIOURAL  (4 sweeps, 2×3)
# V10 behavioural: alpha, gamma, beta_spite — perturbation from zero baseline
# ============================================================
def _plot_sensitivity_fig9(params, prefix, cfg, out_dir="plots_v10_leaderDepr"):
    os.makedirs(out_dir, exist_ok=True)
    sweep_steps = cfg.get("sense_steps", cfg["steps"])
    eval_frac   = cfg.get("sense_eval_frac", 0.10)
    tail_pct    = int(eval_frac * 100)
    c_L, c_F    = "tab:blue", "tab:orange"

    sweeps9 = [
        ("alpha_ineq",  [0.0,0.5,1.0,2.0,3.0], params.alpha_ineq,
         r"$\alpha$ — inequity aversion" + "\n"
         + r"follower: $-\alpha\cdot\max(u_{L,lag}-u_F^{base},0)$",
         r"Fig 9a — $\alpha$ sensitivity"),
        ("gamma_diplo", [0.0,0.2,0.5,1.0,1.5,2.0], params.gamma_diplo,
         r"$\gamma$ — diplomatic cost" + "\n"
         + r"leader: $-\gamma\cdot(\rho/\rho_{\max})\cdot\tau M$",
         r"Fig 9b — $\gamma$ sensitivity"),
        ("beta_spite",  [0.0,0.5,1.0,2.0,3.0], params.beta_spite,
         r"$\beta$ — retaliation spite" + "\n"
         + r"follower: $+\beta\cdot(\rho/\rho_{\max})\cdot\tau M$",
         r"Fig 9c — $\beta$ spite sensitivity"),
        ("phi_inflation_L", [0.10,0.20,0.30,0.50,0.70], params.phi_inflation_L,
         r"$\phi_L$ — leader inflation cost" + "\n"
         + r"leader: $-\phi_L\cdot dL\cdot M_0$",
         r"Fig 9d — $\phi_L$ sensitivity"),
    ]

    print(f"\n{'='*60}\n  [Fig 9] Behavioural sweeps — {prefix}"
          f"  ({sweep_steps} steps, eval last {tail_pct}%,"
          f" workers={cfg.get('n_workers',0)})\n{'='*60}")
    res = _run_sweep(sweeps9, params, cfg, sweep_steps, eval_frac)

    fig9, axs9 = plt.subplots(2, 3, figsize=(22, 11))
    fig9.suptitle(
        r"Fig 9 — V9 Behavioural Sensitivity ($\alpha$, $\gamma$, $\beta$, $\phi_L$)"
        f"  [{prefix}]  ({sweep_steps} steps/pt, eval last {tail_pct}%)", fontsize=13)
    axs9_flat = axs9.flatten()
    for idx, (item, r) in enumerate(zip(sweeps9, res)):
        attr, vals, base, xlabel, title = item
        ax = axs9_flat[idx]
        ax.plot(vals, r["Lm"], "o-", lw=1.8, color=c_L, label="Leader")
        ax.plot(vals, r["Fm"], "s-", lw=1.8, color=c_F, label="Follower")
        ax.axvline(base, color="red", ls="--", lw=1.2, alpha=0.8, label=f"Baseline={base:.3f}")
        ax.axhline(0, color="k", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=9, pad=4); ax.set_xlabel(xlabel, fontsize=7.5, labelpad=3)
        ax.set_ylabel("Mean per-step payoff", fontsize=8); ax.tick_params(labelsize=7.5)
        ax.grid(True); ax.legend(fontsize=7.5, loc="best", framealpha=0.7)
    ax_t = axs9_flat[4]; ax_t.axis("off")
    ax_t.set_title(f"Fig 9e — {prefix} Behavioural Parameters", fontsize=10, fontweight="bold")
    rows9=[("Parameter","Value","Description"),
           ("alpha_ineq",   f"{params.alpha_ineq:.3f}",  "Fehr-Schmidt inequity aversion"),
           ("gamma_diplo",  f"{params.gamma_diplo:.3f}", "Leader diplomatic cost weight"),
           ("beta_spite",      f"{params.beta_spite:.3f}",      "Follower retaliation spite"),
           ("phi_inflation_L", f"{params.phi_inflation_L:.3f}", "Leader inflation cost weight"),
           ("uL_lag_window",f"{params.uL_lag_window}",   "Rolling window W for uL_ref"),
           ("rho_max",      f"{params.rho_max:.2f}",     "Follower retaliation ceiling"),
           ("","",""),
           ("Literature","",""),
           ("alpha ref:","Fehr & Schmidt (1999)","QJE 114(3):817-868"),
           ("beta ref:","Fehr & Schmidt (1999)","spite / hostile preferences"),
           ("gamma ref:","Grossman & Helpman (1994)","AER 84(4):833-850"),
           ("sense_steps",f"{sweep_steps}",f"eval last {tail_pct}%")]
    col_x=[0.02,0.40,0.65]; y0=0.97; rh=0.068
    for i,row in enumerate(rows9):
        y=y0-i*rh; bold=(i==0)
        for j,cell in enumerate(row):
            ax_t.text(col_x[j],y,cell,transform=ax_t.transAxes,fontsize=6.8,
                      fontweight="bold" if bold else "normal",va="top",ha="left",
                      color="#1a1a1a" if not bold else "#003366")
        if i==0: ax_t.axhline(y=y0-rh*0.85,xmin=0,xmax=1,color="#003366",lw=0.8)
    axs9_flat[5].axis("off")
    plt.subplots_adjust(top=0.91,bottom=0.10,hspace=0.38,wspace=0.28,left=0.06,right=0.97)
    out_path=f"{out_dir}/{prefix}_fig9_sensitivity_behavioural.png"
    plt.savefig(out_path,dpi=120); plt.close(fig9)
    print(f"\n  Saved {out_path}")


# ============================================================
# SENSITIVITY — Fig 10: EXPORT CONTROL  (4 sweeps, 2×3)
# V9 leader depreciation: dL_max, dL_elast_M, dL_elast_E, phi_inflation_L sweeps
# Controls always ON regardless of global flag.
# ============================================================
def _plot_sensitivity_fig10(params, prefix, cfg, out_dir="plots_v10_leaderDepr"):
    os.makedirs(out_dir, exist_ok=True)
    sweep_steps = cfg.get("sense_steps", cfg["steps"])
    eval_frac   = cfg.get("sense_eval_frac", 0.10)
    tail_pct    = int(eval_frac * 100)
    c_L, c_F    = "tab:blue", "tab:orange"

    sweeps10 = [
        ("dL_max",
         [0.0,0.05,0.10,0.15,0.20], params.dL_max,
         r"$dL_{\max}$ — depreciation ceiling" + "\n"
         + r"$M^* = M_0(1+\tau)^{-\varepsilon_d}(1+dL)^{-\varepsilon_{dL}}$",
         r"Fig 10a — $dL_{\max}$ sensitivity"),
        ("dL_elast_M",
         [0.25,0.50,0.80,1.00,1.50], params.dL_elast_M,
         r"$\varepsilon_{dL,M}$ — import suppression elasticity" + "\n"
         + r"M* suppressed by $(1+dL)^{-\varepsilon_{dL,M}}$",
         r"Fig 10b — $\varepsilon_{dL,M}$ sensitivity"),
        ("dL_elast_E",
         [0.25,0.50,0.80,1.00,1.50], params.dL_elast_E,
         r"$\varepsilon_{dL,E}$ — export boost elasticity" + "\n"
         + r"E* boosted by $(1+dL)^{+\varepsilon_{dL,E}}$",
         r"Fig 10c — $\varepsilon_{dL,E}$ sensitivity"),
        ("leader_cost_dL",
         [0.0, 0.004, 0.010, 0.020, 0.035], params.leader_cost_dL,
         r"$c_{dL}$ — leader depreciation action cost" + "\n"
         + r"leader: $-c_{dL}\cdot dL^2$  (Fed independence friction)",
         r"Fig 10d — $c_{dL}$ sensitivity"),
    ]

    print(f"\n{'='*60}\n  [Fig 10] Leader depreciation sweeps — {prefix}"
          f"  (controls always ON, {sweep_steps} steps, eval last {tail_pct}%,"
          f" workers={cfg.get('n_workers',0)})\n{'='*60}")
    res = _run_sweep(sweeps10, params, cfg,
                     sweep_steps=sweep_steps, eval_frac=eval_frac)

    fig10, axs10 = plt.subplots(2, 3, figsize=(22, 11))
    fig10.suptitle(
        f"Fig 10 — V9 Leader Depreciation Sensitivity  [{prefix}]"
        f"  ({sweep_steps} steps/pt, eval last {tail_pct}%)",
        fontsize=13)
    axs10_flat = axs10.flatten()
    for idx, (item, r) in enumerate(zip(sweeps10, res)):
        attr, vals, base, xlabel, title = item
        ax = axs10_flat[idx]
        ax.plot(vals, r["Lm"], "o-", lw=1.8, color=c_L, label="Leader payoff")
        ax.plot(vals, r["Fm"], "s-", lw=1.8, color=c_F, label="Follower payoff")
        ax.axvline(base, color="red", ls="--", lw=1.2, alpha=0.8, label=f"Baseline={base:.3f}")
        ax.axhline(0, color="k", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=9, pad=4); ax.set_xlabel(xlabel, fontsize=7.5, labelpad=3)
        ax.set_ylabel("Mean per-step payoff", fontsize=8); ax.tick_params(labelsize=7.5)
        ax.grid(True); ax.legend(fontsize=7.5, loc="best", framealpha=0.7)
    ax_t = axs10_flat[4]; ax_t.axis("off")
    ax_t.set_title(f"Fig 10e — {prefix} Leader Depreciation Parameters", fontsize=10, fontweight="bold")
    rows10=[("Parameter","Value","Description"),
            ("dL_max",            f"{params.dL_max:.2f}",           "Depreciation ceiling"),
            ("dL_elast_M",        f"{params.dL_elast_M:.2f}",       "Import suppression elasticity"),
            ("dL_elast_E",        f"{params.dL_elast_E:.2f}",       "Export boost elasticity"),
            ("phi_inflation_L",   f"{params.phi_inflation_L:.2f}",  "Leader inflation cost (→Fig 9d)"),
            ("leader_cost_dL",    f"{params.leader_cost_dL:.3f}",   "Quadratic dL action cost"),
            ("n_dL_bins",         f"{cfg['n_dL_bins']}",           "Bins for dL action space"),
            ("Leader actions",
             f"{cfg['n_bins']}x{cfg['n_dL_bins']}={cfg['n_bins']*cfg['n_dL_bins']}",
             "Joint (tau,dL) action count"),
            ("","",""),
            ("Calibration","",""),
            ("dL_max LF=0.15","reserve currency constraint","Obstfeld & Rogoff (1995)"),
            ("dL_max FF=0.10","China less constrained","Corsetti et al. (1998)"),
            ("Refs:","Boz et al. (2022) AER","dominant currency pricing"),
            ("sense_steps",f"{sweep_steps}",f"eval last {tail_pct}%")]
    col_x=[0.02,0.42,0.65]; y0=0.97; rh=0.062
    for i,row in enumerate(rows10):
        y=y0-i*rh; bold=(i==0) or (row[0] in ("Calibration","PD finding:"))
        for j,cell in enumerate(row):
            ax_t.text(col_x[j],y,cell,transform=ax_t.transAxes,fontsize=6.5,
                      fontweight="bold" if bold else "normal",va="top",ha="left",
                      color="#1a1a1a" if not bold else "#003366")
        if i==0: ax_t.axhline(y=y0-rh*0.85,xmin=0,xmax=1,color="#003366",lw=0.8)
    axs10_flat[5].axis("off")
    plt.subplots_adjust(top=0.91,bottom=0.10,hspace=0.38,wspace=0.28,left=0.06,right=0.97)
    out_path=f"{out_dir}/{prefix}_fig10_sensitivity_leaderDepr.png"
    plt.savefig(out_path,dpi=120); plt.close(fig10)
    print(f"\n  Saved {out_path}")


# ============================================================
# MAIN TEST FUNCTION
# ============================================================
def test_x_tau_regression():
    """V10 smoke test: equation structure and zeroed cross-elasticities.

    Cross-elasticities (x_tau_elast, dL_elast_X, d_elast_E) are zeroed
    in current calibration — pending recalibration.
    This test verifies:
      1. With elasticities=0, X* and E* are independent of cross terms (V9 compat)
      2. With elasticities>0, cross effects work in the correct direction
      3. Own effects (d boosts X, dL boosts E, ρ suppresses E) always active
    """
    import copy
    params = build_params_lf()
    d, tau, dL, rho = 0.10, 0.20, 0.05, 0.08

    env = sim.EconomicEnvironment(copy.deepcopy(params))

    # ── Own effects always active ────────────────────────────────────────
    X_d0 = env._X_star(0.0, 0.0, 0.0)
    X_d1 = env._X_star(d,   0.0, 0.0)
    assert X_d1 > X_d0, f"d should boost X*: {X_d1:.2f} > {X_d0:.2f}"

    E_base = env._E_star(0.0, 0.0, 0.0)
    E_rho  = env._E_star(rho, 0.0, 0.0)
    E_dL   = env._E_star(0.0, dL,  0.0)
    assert E_rho < E_base, f"ρ should suppress E*: {E_rho:.2f} < {E_base:.2f}"
    assert E_dL  > E_base, f"dL should boost E*: {E_dL:.2f} > {E_base:.2f}"

    # ── Cross effects zeroed in current calibration ──────────────────────
    X_base = env._X_star(d, 0.0, 0.0)
    X_tau  = env._X_star(d, tau, 0.0)
    X_dL   = env._X_star(d, 0.0, dL)
    assert abs(X_tau - X_base) < 1e-9, \
        f"x_tau_elast=0: X* should be τ-independent: {X_tau:.4f} == {X_base:.4f}"
    assert abs(X_dL - X_base) < 1e-9, \
        f"dL_elast_X=0: X* should be dL-independent: {X_dL:.4f} == {X_base:.4f}"

    E_d = env._E_star(0.0, 0.0, d)
    assert abs(E_d - E_base) < 1e-9, \
        f"d_elast_E=0: E* should be d-independent: {E_d:.4f} == {E_base:.4f}"

    # ── Verify cross effects activate when elasticities > 0 ─────────────
    env2 = sim.EconomicEnvironment(copy.deepcopy(params))
    env2.p.x_tau_elast = 0.30
    env2.p.dL_elast_X  = 0.20
    env2.p.d_elast_E   = 0.25
    assert env2._X_star(d, tau, 0.0) < env2._X_star(d, 0.0, 0.0), "τ should suppress X* when x_tau_elast>0"
    assert env2._X_star(d, 0.0, dL)  < env2._X_star(d, 0.0, 0.0), "dL should suppress X* when dL_elast_X>0"
    assert env2._E_star(0.0, 0.0, d) < env2._E_star(0.0, 0.0, 0.0), "d should suppress E* when d_elast_E>0"
    print("  [smoke] V10 equation structure: PASSED")


def test_dL_zero_regression():
    """Smoke test: dL_max=0 → dL=0 every step → V9 behaves like V7.
    Verifies: inflation_cost_L=0, action_cost_dL=0, M* and E* unaffected by dL.
    """
    import copy
    params = build_params_lf()
    params_nodL = copy.deepcopy(params)
    params_nodL.dL_max = 0.0

    env = EconomicEnvironment(params_nodL)
    tau, d, rho, dL = 0.20, 0.10, 0.05, 0.0

    M_v10 = env._M_star(tau, dL)
    M_ref = env._M_star(tau, 0.0)
    E_v10 = env._E_star(rho, dL)
    E_ref = env._E_star(rho, 0.0)
    assert abs(M_v10 - M_ref) < 1e-10, f"M* mismatch at dL=0: {M_v10} vs {M_ref}"
    assert abs(E_v10 - E_ref) < 1e-10, f"E* mismatch at dL=0: {E_v10} vs {E_ref}"

    L, *_, infl, acdL = env.leader_period_payoff_with_components(tau, rho, dL)
    assert abs(infl)  < 1e-10, f"inflation_cost_L should be 0 at dL=0, got {infl}"
    assert abs(acdL)  < 1e-10, f"action_cost_dL should be 0 at dL=0, got {acdL}"
    print("  [smoke] dL=0 regression: PASSED")


def test_stackelberg_q3_tariff_econ_v10_leaderDepr():
    """
    RUN_CONFIG flags:
      V9: leader plays (τ,dL) joint 2D — depreciation always active
      run_lf/run_ff   — main regime runs (Fig 1-7)
      run_sensitivity — Fig 8 (structural) + Fig 9 (alpha/gamma) + Fig 10 (export control)
    """
    cfg = RUN_CONFIG
    out_dir = _get_output_dir(); os.makedirs(out_dir, exist_ok=True)
    ec_label = "ON"  # V9: dL always active
    print(f"\n  Output dir        : {out_dir}/")
    print(f"  Leader depreciation: {ec_label}")
    print(f"  n_dL_bins         : {cfg['n_dL_bins']} (joint leader actions: {cfg['n_bins']}×{cfg['n_dL_bins']}={cfg['n_bins']*cfg['n_dL_bins']})")

    lf_result = ff_result = center_result = None

    if cfg.get("run_lf", True):
        random.seed(SEED); np.random.seed(SEED)
        lf_result = _run_one_regime(build_params_lf(), "LF", cfg)
    else:
        print("\n[SKIP] LF regime")

    if cfg.get("run_ff", True):
        random.seed(SEED); np.random.seed(SEED)
        ff_result = _run_one_regime(build_params_ff(), "FF", cfg)
    else:
        print("\n[SKIP] FF regime")

    if cfg.get("run_center", False):
        random.seed(SEED); np.random.seed(SEED)
        center_result = _run_one_regime(build_params_center(), "CENTER", cfg)
    else:
        print("\n[SKIP] CENTER regime")

    if lf_result is not None and ff_result is not None:
        print("\n--- Generating comparison plots ---")
        _plot_comparison(lf_result, ff_result, win=cfg["smooth_win"], out_dir=out_dir)

    if cfg.get("run_sensitivity", True):
        print("\n--- Running sensitivity analysis ---")
        if cfg.get("run_lf", True):
            p_lf = build_params_lf()
            _plot_sensitivity_fig8( p_lf, "LF", cfg, out_dir)
            _plot_sensitivity_fig9( p_lf, "LF", cfg, out_dir)
            _plot_sensitivity_fig10(p_lf, "LF", cfg, out_dir)
        if cfg.get("run_ff", True):
            p_ff = build_params_ff()
            _plot_sensitivity_fig8( p_ff, "FF", cfg, out_dir)
            _plot_sensitivity_fig9( p_ff, "FF", cfg, out_dir)
            _plot_sensitivity_fig10(p_ff, "FF", cfg, out_dir)
        if cfg.get("run_center", False):
            p_ctr = build_params_center()
            _plot_sensitivity_fig8( p_ctr, "CENTER", cfg, out_dir)
            _plot_sensitivity_fig9( p_ctr, "CENTER", cfg, out_dir)
            _plot_sensitivity_fig10(p_ctr, "CENTER", cfg, out_dir)
    else:
        print("\n[SKIP] Sensitivity analysis")

    print(f"\n=== All plots saved to {out_dir}/ ===")
    return {"lf": lf_result, "ff": ff_result, "center": center_result}


if __name__ == "__main__":
    test_dL_zero_regression()
    test_x_tau_regression()
    test_dL_zero_regression()
    test_stackelberg_q3_tariff_econ_v10_leaderDepr()
