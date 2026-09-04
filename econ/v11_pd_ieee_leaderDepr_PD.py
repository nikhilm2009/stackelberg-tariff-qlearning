# test_stackelberg_q3_tariff_econ_v10_leaderDepr_PD.py
#
# V10 MODEL SUMMARY — Full Strategic Symmetry
# ════════════════
# Leader  chooses (τ, dL):  tariff + currency depreciation
# Follower chooses (d,  ρ):  own depreciation + retaliation tariff
# dL suppresses M*, boosts E*, carries inflation + quadratic cost to leader.
#
# Phase diagram structure:
#   Fig 8  — Structural    PD1–PD4   (unchanged from V8)
#   Fig 9  — Behavioural   PD5–PD10  (PD5–PD9 kept; PD10 replaced by φ_L vs ρ_max)
#   Fig 10 — V10 Depreciation PD11–PD14 (dL_max, ε_dL,M, ε_dL,E, ε_dL,X params)
#
# V10 Leader Depreciation Phase Diagrams
# ================================
# - Imports stackelberg_q3_tariff_econ_sim_v10_leaderDepr (via config_v10)
# - Leader plays joint (τ, dL) action space (depreciation always ON in phase sweeps)
# - Fig 8: Structural    PD1–PD4   unchanged
# - Fig 9: Behavioural   PD5–PD10  PD10 replaced: α vs d_max → φ_L vs ρ_max
# - Fig 10: V10 Depreciation PD11–PD14 — dL instrument + cross-elasticities
#     PD11: dL_max vs ρ_max  — depreciation ceiling vs retaliation ceiling
#     PD12: dL_max vs τ_max  — depreciation ceiling vs tariff ceiling
#     PD13: ε_dL,M vs ρ_max  — import suppression elasticity vs retaliation ceiling
#     PD14: ε_dL,E vs ρ_max  — export boost elasticity vs retaliation ceiling
# - run_pds in RUN_CONFIG selects which PDs to run
# - phase_grid_default: 6 (fast 6×6) or 10 (fine 10×10)
# - Output dir: phase_plots_v11_leaderDepr/
#
# FIX LOG (vs previous version):
#   1. _run_single_seed: StackelbergTariffGameEconomic now receives freeze params from cfg
#   2. _run_main_regime: StackelbergTariffGameEconomic now receives freeze params from cfg
#   3. RUN_CONFIG: follower_q_init=1000 (was 3000) — matches test config
#   4. RUN_CONFIG: freeze_mode="full_freeze" (was missing — defaulted silently to epsilon_only)
#   5. RUN_CONFIG: leader_alpha_min=0.02, follower_alpha_min=0.05 — restore V7 floors
#   6. RUN_CONFIG: phase_steps=1000000, freeze_leader_frac=0.30 — freeze at 300k,
#      follower gets 700k to converge to fixed leader policy
# ================================

import csv
import os, copy, time
import multiprocessing as _mp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import random

SEED = 42
random.seed(SEED); np.random.seed(SEED)

import stackelberg_q3_tariff_econ_sim_v10_leaderDepr as sim
print("SIM FILE:", sim.__file__)
from stackelberg_q3_tariff_econ_config_v10 import (
    build_params_lf, build_params_ff, build_params_center,
    build_agents, classify_regime,
    rolling_mean as _rolling_mean, cumulative_discounted,
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
# RUNTIME GUIDE (at ~3000 steps/sec pure Python):
#   6×6 grid, 1M steps/pt, 5 seeds → 180 runs × 1M = 180M steps
#   freeze at 30% (300k) → leader committed, follower gets 700k to converge
#   eval last 20% (800k–1M) = 200k rounds per seed
#   V10: leader (τ,dL) 5×3=15 joint actions; follower state 8-tuple with dL_bin.
#
# Quick check:  run_pds = [11]  or  [12]
# Full Fig 10:  run_pds = [11, 12, 13, 14]
# Full suite:   run_pds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# ============================================================
RUN_CONFIG = {
    # ================================================================
    # SHARED SETTINGS — apply to both Fig 8/9 and Fig 10
    # ================================================================
    "steps":               800000,   # main regime run (LF/FF/CENTER Fig1-2)
    "n_dL_bins":           3,        # dL bins (only used when leader_depr=True)
    "alpha":               0.18,
    "follower_double_q":   True,
    "leader_alpha_min":    0.02,     # V7 value — lower caused limit cycles
    "follower_alpha_min":  0.05,     # V7 value
    "freeze_leader_frac":  0.0,      # no freeze — validated for production
    "freeze_follower_frac":0.0,
    "freeze_mode":         "none",
    "smooth_win":          5000,
    "phase_eval_frac":     0.20,     # eval last 20% of phase_steps
    "run_lf":              False,
    "run_ff":              False,
    "run_phase_plots":     True,
    "n_workers":           6,        # 0=auto (cpu_count-1), 1=serial

    # ----------------------------------------------------------------
    # Epsilon — V7 values used for BOTH Fig 8/9 and Fig 10
    # Higher epsilon_start (0.15 vs 0.08) and higher floor (0.02 vs 0.002)
    # give faster convergence and cleaner PD boundaries in both modes.
    # Validated: PD2 clean at 100k/pt, PD11 clean at 1M/pt with these.
    # ----------------------------------------------------------------
    "epsilon_start":       0.15,     # V7: was 0.08 in V10
    "epsilon_end":         0.02,     # V7: was 0.002 in V10

    # ================================================================
    # MODE SELECTION — uncomment ONE block below
    # ================================================================

    # ── Fig 8/9 MODE — Structural & Behavioural PDs ─────────────────
    # leader_depr=False: τ-only leader (V7), 7-tuple follower state,
    # V7 bin counts and q_init. Validated: clean PD boundaries at 100k/pt.
    # ----------------------------------------------------------------
     "leader_depr":         False,    # τ-only — V7 action space
     "n_bins":              6,        # V7 bin count
     "n_rho_bins":          5,        # V7 bin count
    # ----------------------------------------------------------------
    # follower_q_init controls initialization sensitivity:
    #   3000 → optimistic, fast convergence, clean single-regime cells (PRODUCTION)
    #   1000 → cautious, surfaces Multiple cells at boundary   (ROBUSTNESS CHECK)
    # Main paper figures use 3000. Robustness paragraph uses 1000 result.
    # Do NOT mix across PDs in the same figure.
    # ----------------------------------------------------------------
     #"follower_q_init":     3000.0,   # PRODUCTION — change to 1000 for robustness run
     "follower_q_init":   1000.0,   # ROBUSTNESS — reveals coordination-sensitive cells
     "leader_q_init":       3000.0,
     "ctr_regime":          "LF",     # LF anchor for Fig 8/9
     "phase_steps":         1500000,   # 600k sufficient with V7 settings
     "phase_grid_default":  8,        # 8×8 production grid
     "n_phase_seeds":       10,
     #"run_pds":             [1, 2, 6, 7],   # USE LF
     "run_pds":              [2],   # USE LF

     #"run_pds":             [6],   # USE CENTER

    # ── Fig 10/PD11 MODE — Depreciation PDs ──────────────────────────────
    # leader_depr=True: joint (τ,dL) leader, 8-tuple follower state,
    # V10 bin counts. CENTER anchor shows dL regime-shifting effect.
    # Validated: PD11 clean at 1M/pt, 5-seed, CENTER anchor.
    # ----------------------------------------------------------------
    #   "leader_depr":         True,     # joint (τ,dL) — V10 action space
    #   "n_bins":              5,        # V10 bin count
    #   "n_rho_bins":          3,        # V10 bin count
    #   "follower_q_init":     1000.0,   # V10 asymmetric init
    #   "leader_q_init":       3000.0,
    #   "ctr_regime":          "CENTER", # CENTER anchor for Fig 10
    #   "phase_steps":         1000000,  # 1M needed for dL convergence
    #   "phase_grid_default":  8,        # 8×8 production grid
    #   "n_phase_seeds":       5,
    #   "run_pds":             [11], # Fig 10 depreciation PDs

    # ----------------------------------------------------------------
    # PD trajectory inset (separate PNG, avoids axis space mismatch)
    #   show_trajectory=True  → generate pd2_trajectory_inset_{regime}.png
    #   trajectory_pd         → which PD triggers the inset (default "PD2")
    #   trajectory_window     → rolling window size in steps
    #   trajectory_regime     → "LF"|"FF"|"CENTER" — must match run_lf/run_ff
    # NOTE: run_lf/run_ff must be True so the sidecar .npy exists first.
    # ----------------------------------------------------------------
    "show_trajectory":     False,   # True to generate inset
    "trajectory_pd":       "PD2",
    "trajectory_window":   50000,
    "trajectory_regime":   "LF",
}

# ============================================================
# HELPERS
# ============================================================
def _get_output_dir(): return "phase_plots_v11_leaderDepr"

def _apply_leader_depr(params, cfg):
    """Apply leader_depr toggle from cfg to a params object.

    When leader_depr=False (V7 mode / Fig 8-9):
      - dL_max=0          → build_agents gives leader τ-only action space (5 actions)
      - phi_inflation_L=0 → no inflation cost in leader payoff
      - leader_cost_dL=0  → no quadratic dL cost in leader payoff
    Always returns a deepcopy — never mutates the original params.
    """
    if cfg.get("leader_depr", True):
        return params  # dL on — use params as-is
    p = copy.deepcopy(params)
    p.dL_max          = 0.0
    p.phi_inflation_L = 0.0
    p.leader_cost_dL  = 0.0
    return p

def _pgrid(pd_name, cfg):
    key = f"phase_grid_{pd_name.lower()}"
    if key in cfg: return int(cfg[key])
    return int(cfg.get("phase_grid_default", 6))

# ============================================================
# GRID RUNNER
# ============================================================
_SEED_POOL = [42, 123, 777, 999, 2024, 314, 1066, 53, 160, 1031]

def _get_seeds(cfg):
    n = int(cfg.get("n_phase_seeds", 5))
    return _SEED_POOL[:n]

PHASE_SEEDS = _SEED_POOL[:5]  # default alias


def _run_single_seed(params, cfg, seed):
    """Run one seed and classify regime using simple eval tail.
    Evaluates last phase_eval_frac of steps (default 20% = 200k rounds at 1M steps).
    Returns: regime, mean_L, mean_F, tau_frac, d_frac, rho_frac, mean_dL

    FIX: StackelbergTariffGameEconomic now receives freeze params from cfg.
    Previously instantiated with track=True only — freeze was silently ignored.
    """
    ps        = cfg["phase_steps"]
    eval_frac = float(cfg.get("phase_eval_frac", 0.20))
    es        = int(ps * (1.0 - eval_frac))
    random.seed(seed); np.random.seed(seed)
    params    = _apply_leader_depr(params, cfg)   # leader_depr toggle
    e = EconomicEnvironment(params)
    l, f, _, _, _ = build_agents(params, cfg, pd_mode=True)
    # FIX: pass freeze parameters from cfg
    g = StackelbergTariffGameEconomic(
        e, l, f, track=True,
        freeze_leader_frac=float(cfg.get("freeze_leader_frac", 0.0)),
        freeze_follower_frac=float(cfg.get("freeze_follower_frac", 0.0)),
        freeze_mode=cfg.get("freeze_mode", "full_freeze"),
    )
    g.run(rounds=ps)
    ev = g.results["rounds"][es:]
    mt = float(np.mean([r["tau"]          for r in ev]))
    md = float(np.mean([r["d"]            for r in ev]))
    mr = float(np.mean([r["rho"]          for r in ev]))
    ml = float(np.mean([r["leader_pay"]   for r in ev]))
    mf = float(np.mean([r["follower_pay"] for r in ev]))
    mx = float(np.mean([r.get("dL", 0.0) for r in ev]))
    regime, df, rf, fs = classify_regime(mt, md, mr,
        params.tau_max, params.d_max, params.rho_max, mean_leader_pay=ml)
    return (regime, ml, mf,
            mt/params.tau_max if params.tau_max > 0 else 0,
            df, rf, mx)


def _run_grid_point(params, cfg):
    """Run all seeds for one grid point. Final regime: majority vote."""
    from collections import Counter
    regimes, Lps, Fps, xms = [], [], [], []
    phase_seeds = _get_seeds(cfg)
    for seed in phase_seeds:
        t0 = time.time()
        regime, lp, fp, tf, df, rf, mx = _run_single_seed(params, cfg, seed)
        elapsed = time.time() - t0
        print(f"      seed={seed}  → {regime:10s}  L={lp:7.1f}  F={fp:7.1f}  "
              f"dL={mx:.3f}  [{elapsed:.1f}s]", flush=True)
        regimes.append(regime); Lps.append(lp); Fps.append(fp); xms.append(mx)

    votes = Counter(regimes)
    top_regime, top_count = votes.most_common(1)[0]
    # Strict majority required (>50%); ties → "Multiple" (genuine coordination ambiguity)
    if top_count <= len(regimes) / 2:
        maj = "Multiple"
    else:
        maj = top_regime
    return (maj, float(np.mean(Lps)), float(np.mean(Fps)),
            float(np.mean(xms)), list(zip(phase_seeds, Lps)))

# ============================================================
# PLOTTER
# ============================================================
REGIME_COLORS = {"Escalation":0, "Transition":1, "Deterrence":2, "Multiple":3}
CMAP = ListedColormap(["#e74c3c","#f39c12","#27ae60","#7f8c8d"])

def _plot_phase_diagram(ax, x_vals, y_vals, regime_grid, Lpay_grid,
                        xlabel, ylabel, title,
                        lf_x=None, lf_y=None, ff_x=None, ff_y=None,
                        center_x=None, center_y=None,
                        font_size=11, smooth_contour=False):
    nx, ny = len(x_vals), len(y_vals)
    dx = (x_vals[-1]-x_vals[0])/(nx-1)/2 if nx>1 else 0.05
    dy = (y_vals[-1]-y_vals[0])/(ny-1)/2 if ny>1 else 0.05
    xe = np.concatenate([[x_vals[0]-dx],(x_vals[:-1]+x_vals[1:])/2,[x_vals[-1]+dx]])
    ye = np.concatenate([[y_vals[0]-dy],(y_vals[:-1]+y_vals[1:])/2,[y_vals[-1]+dy]])
    ax.set_facecolor("white")
    ax.pcolormesh(xe, ye, regime_grid, cmap=CMAP, vmin=0, vmax=3, alpha=0.75, shading="flat")
    try:
        X, Y = np.meshgrid(x_vals, y_vals)
        _lpg_plot = Lpay_grid
        if smooth_contour:
            from scipy.ndimage import gaussian_filter
            _lpg_plot = gaussian_filter(Lpay_grid.astype(float), sigma=0.8)
        cs = ax.contour(X, Y, _lpg_plot, levels=[0], colors=["black"], linewidths=2.5)
        # clabel removed per IEEE spec — contour line preserved, label removed
        # Keep only the longest path — drop disjointed fragments (version-safe)
        try:
            all_paths = (cs.get_paths() if hasattr(cs, "get_paths")
                         else cs.collections[0].get_paths())
            if len(all_paths) > 1:
                longest = max(all_paths, key=lambda p: len(p.vertices))
                # Remove original contour and redraw with longest path only
                for col in cs.collections:
                    col.remove()
                import matplotlib.patches as _mp2
                from matplotlib.path import Path as _Path
                ax.add_patch(_mp2.PathPatch(longest, fill=False,
                             edgecolor="black", linewidth=2.5, zorder=4))
        except Exception:
            pass
        # zero payoff label removed per IEEE spec
    except Exception: pass
    # Annotate Multiple cells with mean leader payoff (numeric only)
    for yi in range(ny):
        for xi in range(nx):
            if regime_grid[yi, xi] == REGIME_COLORS["Multiple"]:
                mean_L = Lpay_grid[yi, xi]
                ax.text(x_vals[xi], y_vals[yi], f"{mean_L:+.1f}",
                        ha="center", va="center", fontsize=7,
                        color="black", fontweight="bold", zorder=8)
    # Baseline star markers — black edge on all, text labels in black (color-neutral)
    _dx = (x_vals[-1] - x_vals[0]) * 0.02
    _dy = (y_vals[-1] - y_vals[0]) * 0.015
    if lf_x is not None:
        ax.scatter([lf_x], [lf_y], marker="*", s=180, color="blue",
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.text(lf_x + _dx, lf_y + _dy, "LF", fontsize=8,
                color="black", ha="left", va="bottom", zorder=6)
    if ff_x is not None:
        ax.scatter([ff_x], [ff_y], marker="*", s=180, color="darkorange",
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.text(ff_x + _dx, ff_y + _dy, "FF", fontsize=8,
                color="black", ha="left", va="bottom", zorder=6)
    if center_x is not None:
        ax.scatter([center_x], [center_y], marker="*", s=180, color="white",
                   edgecolors="black", linewidths=1.0, zorder=5)
        ax.text(center_x + _dx, center_y + _dy, "CENTER", fontsize=8,
                color="black", ha="left", va="bottom", zorder=6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=2)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25)

def _plot_skipped(ax, pd_name):
    ax.set_facecolor("#f5f5f5")
    ax.text(0.5, 0.5, f"{pd_name}\n(not run)", ha="center", va="center",
            fontsize=13, color="#999999", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{pd_name} — skipped", fontsize=10, color="#999999")

# ============================================================
# PD REGISTRY — all 14 phase diagrams
# ============================================================
def _build_pd_registry_raw():
    registry = [
        # ── Fig 8: structural (PD1–PD4) ──────────────────────────
        dict(name="PD1",  fig_group="fig8",
             title="Phase Diagram: Export Vulnerability vs Depreciation Capacity",
             x_attr="d_max",           x_range=(0.10, 0.45),
             y_attr="psi_E",           y_range=(0.0,  3.0),
             xlabel="Follower depreciation ceiling ($d_{\\max}$)",
             ylabel="Leader export sensitivity ($\\psi_E$)",
             lf_x=LF_PARAMS["d_max"],           lf_y=LF_PARAMS["psi_E"],
             ff_x=FF_PARAMS["d_max"],           ff_y=FF_PARAMS["psi_E"]),
        dict(name="PD2",  fig_group="fig8",
             title="Phase Diagram: Retaliation Capacity vs Tariff Capacity",
             x_attr="rho_max",         x_range=(0.0,  0.50),
             y_attr="tau_max",         y_range=(0.10, 0.45),
             xlabel="Follower retaliation ceiling ($\\rho_{\\max}$)",
             ylabel="Leader tariff ceiling ($\\tau_{\\max}$)",
             lf_x=LF_PARAMS["rho_max"],         lf_y=LF_PARAMS["tau_max"],
             ff_x=FF_PARAMS["rho_max"],         ff_y=FF_PARAMS["tau_max"]),
        dict(name="PD3",  fig_group="fig8",
             title=r"PD3 — $\varepsilon_E$ vs $\varepsilon_d$",
             x_attr="export_elast",    x_range=(0.5,  1.8),
             y_attr="demand_elast",    y_range=(0.8,  2.2),
             xlabel=r"$\varepsilon_E$ — leader export elasticity",
             ylabel=r"$\varepsilon_d$ — import demand elasticity",
             lf_x=LF_PARAMS["export_elast"],    lf_y=LF_PARAMS["demand_elast"],
             ff_x=FF_PARAMS["export_elast"],    ff_y=FF_PARAMS["demand_elast"]),
        dict(name="PD4",  fig_group="fig8",
             title="Phase Diagram: Depreciation Capacity vs Retaliation Capacity",
             x_attr="d_max",           x_range=(0.10, 0.45),
             y_attr="rho_max",         y_range=(0.0,  0.50),
             xlabel="Follower depreciation ceiling ($d_{\\max}$)",
             ylabel="Follower retaliation ceiling ($\\rho_{\\max}$)",
             lf_x=LF_PARAMS["d_max"],           lf_y=LF_PARAMS["rho_max"],
             ff_x=FF_PARAMS["d_max"],           ff_y=FF_PARAMS["rho_max"]),
        # ── Fig 9: behavioural (PD5–PD10) ────────────────────────
        dict(name="PD5",  fig_group="fig9",
             title=r"PD5 — $\alpha$ vs $\rho_{\max}$",
             x_attr="alpha_ineq",      x_range=(0.0,  3.0),
             y_attr="rho_max",         y_range=(0.0,  0.50),
             xlabel=r"$\alpha$ — inequity aversion weight",
             ylabel=r"$\rho_{\max}$ — follower retaliation ceiling",
             lf_x=LF_PARAMS["alpha_ineq"],      lf_y=LF_PARAMS["rho_max"],
             ff_x=FF_PARAMS["alpha_ineq"],      ff_y=FF_PARAMS["rho_max"]),
        dict(name="PD6",  fig_group="fig9",
             title="Phase Diagram: Diplomatic Cost vs Retaliation Capacity",
             x_attr="gamma_diplo",     x_range=(0.0,  3.0),
             y_attr="rho_max",         y_range=(0.0,  0.50),
             xlabel="Leader diplomatic cost ($\\xi$)",
             ylabel="Follower retaliation ceiling ($\\rho_{\\max}$)",
             lf_x=LF_PARAMS["gamma_diplo"],     lf_y=LF_PARAMS["rho_max"],
             ff_x=FF_PARAMS["gamma_diplo"],     ff_y=FF_PARAMS["rho_max"]),
        dict(name="PD7",  fig_group="fig9",
             title="Phase Diagram: Diplomatic Cost vs Export Vulnerability",
             x_attr="gamma_diplo",     x_range=(0.0,  3.0),
             y_attr="psi_E",           y_range=(0.0,  3.0),
             xlabel="Leader diplomatic cost ($\\xi$)",
             ylabel="Leader export sensitivity ($\\psi_E$)",
             lf_x=LF_PARAMS["gamma_diplo"],     lf_y=LF_PARAMS["psi_E"],
             ff_x=FF_PARAMS["gamma_diplo"],     ff_y=FF_PARAMS["psi_E"]),
        dict(name="PD8",  fig_group="fig9",
             title=r"PD8 — $\gamma$ vs $\tau_{\max}$",
             x_attr="gamma_diplo",     x_range=(0.0,  3.0),
             y_attr="tau_max",         y_range=(0.10, 0.45),
             xlabel=r"$\gamma$ — leader diplomatic cost",
             ylabel=r"$\tau_{\max}$ — leader tariff ceiling",
             lf_x=LF_PARAMS["gamma_diplo"],     lf_y=LF_PARAMS["tau_max"],
             ff_x=FF_PARAMS["gamma_diplo"],     ff_y=FF_PARAMS["tau_max"]),
        dict(name="PD9",  fig_group="fig9",
             title=r"PD9 — $\gamma$ vs $\varepsilon_E$",
             x_attr="gamma_diplo",     x_range=(0.0,  3.0),
             y_attr="export_elast",    y_range=(0.5,  1.8),
             xlabel=r"$\gamma$ — leader diplomatic cost",
             ylabel=r"$\varepsilon_E$ — leader export elasticity",
             lf_x=LF_PARAMS["gamma_diplo"],     lf_y=LF_PARAMS["export_elast"],
             ff_x=FF_PARAMS["gamma_diplo"],     ff_y=FF_PARAMS["export_elast"]),
        dict(name="PD10", fig_group="fig9",
             title=r"PD10 — $\phi_L$ vs $\rho_{\max}$",
             x_attr="phi_inflation_L", x_range=(0.10, 0.70),
             y_attr="rho_max",         y_range=(0.0,  0.50),
             xlabel=r"$\phi_L$ — leader inflation cost of depreciation",
             ylabel=r"$\rho_{\max}$ — follower retaliation ceiling",
             lf_x=LF_PARAMS["phi_inflation_L"], lf_y=LF_PARAMS["rho_max"],
             ff_x=FF_PARAMS["phi_inflation_L"], ff_y=FF_PARAMS["rho_max"]),
        # ── Fig 10: V10 leader depreciation (PD11–PD14) ──────────
        dict(name="PD11", fig_group="fig10",
             title="Phase Diagram: Leader Depreciation Capacity vs Retaliation Capacity",
             x_attr="dL_max",  x_range=(0.0,  0.25),
             y_attr="rho_max", y_range=(0.0,  0.50),
             xlabel="Leader depreciation ceiling ($dL_{\\max}$)",
             ylabel="Follower retaliation ceiling ($\\rho_{\\max}$)",
             lf_x=LF_PARAMS["dL_max"], lf_y=LF_PARAMS["rho_max"],
             ff_x=FF_PARAMS["dL_max"], ff_y=FF_PARAMS["rho_max"]),
        dict(name="PD12", fig_group="fig10",
             title=r"PD12 — $dL_{\max}$ vs $\tau_{\max}$",
             x_attr="dL_max",  x_range=(0.0,  0.25),
             y_attr="tau_max", y_range=(0.10, 0.45),
             xlabel=r"$dL_{\max}$ — leader depreciation ceiling",
             ylabel=r"$\tau_{\max}$ — leader tariff ceiling",
             lf_x=LF_PARAMS["dL_max"], lf_y=LF_PARAMS["tau_max"],
             ff_x=FF_PARAMS["dL_max"], ff_y=FF_PARAMS["tau_max"]),
        dict(name="PD13", fig_group="fig10",
             title=r"PD13 — $\varepsilon_{dL,M}$ vs $\rho_{\max}$",
             x_attr="dL_elast_M", x_range=(0.25, 1.50),
             y_attr="rho_max",    y_range=(0.0,  0.50),
             xlabel=r"$\varepsilon_{dL,M}$ — import suppression elasticity",
             ylabel=r"$\rho_{\max}$ — follower retaliation ceiling",
             lf_x=LF_PARAMS["dL_elast_M"], lf_y=LF_PARAMS["rho_max"],
             ff_x=FF_PARAMS["dL_elast_M"], ff_y=FF_PARAMS["rho_max"]),
        dict(name="PD14", fig_group="fig10",
             title=r"PD14 — $\varepsilon_{dL,E}$ vs $\rho_{\max}$",
             x_attr="dL_elast_E", x_range=(0.25, 1.50),
             y_attr="rho_max",    y_range=(0.0,  0.50),
             xlabel=r"$\varepsilon_{dL,E}$ — export boost elasticity",
             ylabel=r"$\rho_{\max}$ — follower retaliation ceiling",
             lf_x=LF_PARAMS["dL_elast_E"], lf_y=LF_PARAMS["rho_max"],
             ff_x=FF_PARAMS["dL_elast_E"], ff_y=FF_PARAMS["rho_max"]),
    ]

    # Add center_x/center_y to every PD from CENTER_PARAMS
    for pd in registry:
        xa = pd["x_attr"]
        ya = pd["y_attr"]
        pd["center_x"] = CENTER_PARAMS.get(xa)
        pd["center_y"] = CENTER_PARAMS.get(ya)

    return registry


def _build_pd_registry():
    return _build_pd_registry_raw()

_build_pd_registry_orig = _build_pd_registry

# ============================================================
# PARALLEL WORKER — must be top-level for multiprocessing spawn
# ============================================================
def _grid_point_worker(args):
    """Top-level wrapper for Pool.map — unpacks a grid-point job and runs it."""
    pd_spec, xv, yv, cfg = args
    p = copy.deepcopy(pd_spec["base_params"])
    setattr(p, pd_spec["x_attr"], float(xv))
    setattr(p, pd_spec["y_attr"], float(yv))
    # Edge cases: zero ceilings need explicit assignment
    if pd_spec["x_attr"] == "rho_max"      and xv == 0.0: p.rho_max      = 0.0
    if pd_spec["y_attr"] == "rho_max"      and yv == 0.0: p.rho_max      = 0.0
    if pd_spec["x_attr"] == "dL_max"       and xv == 0.0: p.dL_max       = 0.0
    if pd_spec["y_attr"] == "dL_max"       and yv == 0.0: p.dL_max       = 0.0
    if pd_spec["x_attr"] == "x_tau_elast"  and xv == 0.0: p.x_tau_elast  = 0.0
    if pd_spec["x_attr"] == "dL_elast_X"   and xv == 0.0: p.dL_elast_X   = 0.0
    if pd_spec["x_attr"] == "d_elast_E"    and xv == 0.0: p.d_elast_E    = 0.0
    # leader_depr toggle — applied after grid overrides so dL_max sweep
    # in Fig 10 is not clobbered when leader_depr=True
    p = _apply_leader_depr(p, cfg)
    regime, lp, fp, mx, seed_pays = _run_grid_point(p, cfg)
    return (pd_spec["xi"], pd_spec["yi"], regime, lp, fp, mx, seed_pays,
            float(xv), float(yv))


# ============================================================
# INDIVIDUAL PD SAVER
# ============================================================
def _save_single_pd(pd_name, result_tuple, cfg, out_dir, patches, center_handle, anchor_label, steps):
    """Save a single PD as its own PNG immediately after grid completion."""
    if result_tuple is None:
        return
    xv, yv, rg, lpg, p_ = result_tuple
    fig_s, ax_s = plt.subplots(1, 1, figsize=(8, 7))
    fig_s.patch.set_facecolor("white")
    # IEEE title: human-readable name only, no run metadata
    fig_s.suptitle(p_["title"], fontsize=11, fontweight="bold")
    _plot_phase_diagram(ax_s, xv, yv, rg, lpg,
        xlabel=p_["xlabel"], ylabel=p_["ylabel"], title="",
        lf_x=p_["lf_x"], lf_y=p_["lf_y"],
        ff_x=p_["ff_x"], ff_y=p_["ff_y"],
        center_x=p_.get("center_x"), center_y=p_.get("center_y"),
        font_size=11, smooth_contour=(pd_name == "PD4"))
    # Regime legend only — baselines labelled directly on plot
    fig_s.legend(handles=patches, loc="lower center", ncol=4,
                 fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.00),
                 borderpad=0.4, handlelength=1.2, columnspacing=0.8)
    # Use subplots_adjust for precise control — tight_layout ignores suptitle
    # top=0.93: plot sits just below title with minimal gap
    # bottom=0.13: enough room for x-axis label + legend fully below it
    plt.subplots_adjust(top=0.93, bottom=0.13, left=0.10, right=0.97)
    # IEEE output filename per spec
    _ieee_names = {
        "PD1":  "fig_pd1_export_vulnerability.png",
        "PD2":  "fig_pd2_retaliation_vs_tariff.png",
        "PD4":  "fig_pd4_follower_instruments.png",
        "PD6":  "fig_pd6_diplomatic_cost_retaliation.png",
        "PD7":  "fig_pd7_diplomatic_export_vulnerability.png",
        "PD11": "fig_pd11_leader_depreciation.png",
    }
    out_name = _ieee_names.get(pd_name, f"{pd_name.lower()}_phase_diagram.png")
    out_path = f"{out_dir}/{out_name}"
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig_s)
    print(f"  → Saved {out_path}", flush=True)

    # ── Save CSV alongside PNG ────────────────────────────────────────
    _ieee_csv = {k: v.replace(".png", ".csv") for k, v in _ieee_names.items()}
    csv_name = _ieee_csv.get(pd_name, f"{pd_name.lower()}_phase_diagram.csv")
    csv_path = f"{out_dir}/{csv_name}"
    n_seeds  = cfg.get("n_phase_seeds", 5)
    with open(csv_path, "w", newline="") as _f:
        w = csv.writer(_f)
        w.writerow(["x_val", "y_val", "regime", "mean_lpay",
                    "x_attr", "y_attr", "n_seeds"])
        x_attr = p_.get("x_attr", "x")
        y_attr = p_.get("y_attr", "y")
        _regime_names = {v: k for k, v in
                         {"Escalation":0,"Transition":1,"Deterrence":2,"Multiple":3}.items()}
        for yi, yv_ in enumerate(yv):
            for xi, xv_ in enumerate(xv):
                rname = _regime_names.get(int(round(rg[yi, xi])), "Deterrence")
                w.writerow([f"{xv_:.4f}", f"{yv_:.4f}", rname,
                            f"{lpg[yi, xi]:.4f}", x_attr, y_attr, n_seeds])
    print(f"  → Saved {csv_path}", flush=True)

    # ── Per-seed payoff CSV (for threshold sensitivity analysis) ──────────
    per_seed_csv_path = csv_path.replace(".csv", "_per_seed.csv")
    if hasattr(_save_single_pd, "_per_seed_store"):
        with open(per_seed_csv_path, "w", newline="") as _f2:
            w2 = csv.writer(_f2)
            w2.writerow(["x_val", "y_val", "seed", "mean_lpay"])
            for (xv_, yv_), sp_list in _save_single_pd._per_seed_store.items():
                for seed_, lp_ in sp_list:
                    w2.writerow([f"{xv_:.4f}", f"{yv_:.4f}", seed_, f"{lp_:.4f}"])
        print(f"  → Saved {per_seed_csv_path}", flush=True)

    # ── Trajectory inset (separate PNG) ──────────────────────────────
    # PD2 axes are parameter space (rho_max, tau_max); trajectory is
    # action space (actual rho(t), tau(t)) — different dimensions.
    # Generate a zoomed inset PNG rather than overlaying on PD2 directly.
    if cfg.get("show_trajectory", False) and pd_name == cfg.get("trajectory_pd", "PD2"):
        traj_regime = cfg.get("trajectory_regime", "LF")
        traj_path   = f"{out_dir}/{traj_regime}_trajectory.npy"
        if os.path.exists(traj_path):
            traj = np.load(traj_path)
            rho_roll, tau_roll = traj[0].tolist(), traj[1].tolist()
            fig_i, ax_i = plt.subplots(1, 1, figsize=(6, 5))
            # Zoom to trajectory region with padding
            rho_pad = max(0.02, (max(rho_roll)-min(rho_roll)) * 0.3)
            tau_pad = max(0.02, (max(tau_roll)-min(tau_roll)) * 0.3)
            x_lo = max(0.0,   min(rho_roll) - rho_pad)
            x_hi = min(xv[-1], max(rho_roll) + rho_pad)
            y_lo = max(0.0,   min(tau_roll) - tau_pad)
            y_hi = min(yv[-1], max(tau_roll) + tau_pad)
            # Replot PD background zoomed
            xi_m = (xv >= x_lo) & (xv <= x_hi)
            yi_m = (yv >= y_lo) & (yv <= y_hi)
            xv_z, yv_z = xv[xi_m], yv[yi_m]
            rg_z  = rg[np.ix_(yi_m, xi_m)]
            lpg_z = lpg[np.ix_(yi_m, xi_m)]
            if len(xv_z) >= 2 and len(yv_z) >= 2:
                dxz = (xv_z[-1]-xv_z[0])/(len(xv_z)-1)/2
                dyz = (yv_z[-1]-yv_z[0])/(len(yv_z)-1)/2
                ax_i.pcolormesh(
                    np.concatenate([[xv_z[0]-dxz],(xv_z[:-1]+xv_z[1:])/2,[xv_z[-1]+dxz]]),
                    np.concatenate([[yv_z[0]-dyz],(yv_z[:-1]+yv_z[1:])/2,[yv_z[-1]+dyz]]),
                    rg_z, cmap=CMAP, vmin=0, vmax=3, alpha=0.65, shading="flat")
                try:
                    Xz, Yz = np.meshgrid(xv_z, yv_z)
                    ax_i.contour(Xz, Yz, lpg_z, levels=[0],
                                        colors=["black"], linewidths=1.5)
                    # clabel removed per IEEE spec
                except Exception: pass
            # Trajectory path — thin line, larger endpoint markers
            ax_i.plot(rho_roll, tau_roll, color="black", lw=1.0, alpha=0.75, linestyle="--", zorder=7)
            ax_i.scatter(rho_roll[0], tau_roll[0], marker="o", s=130,
                         color="white", edgecolors="black", linewidths=2.0,
                         zorder=8, label="t=0 (early policy)")
            ax_i.scatter(rho_roll[-1], tau_roll[-1], marker="X", s=160,
                         color="black", zorder=8, label="t=T (converged)")
            ax_i.annotate("t=0", xy=(rho_roll[0], tau_roll[0]),
                           xytext=(8, 8), textcoords="offset points", fontsize=9, zorder=9)
            ax_i.annotate("t=T", xy=(rho_roll[-1], tau_roll[-1]),
                           xytext=(5, -11), textcoords="offset points", fontsize=9, zorder=9)
            # Every 3rd arrow only — direction visible, no noodle soup
            arrow_step = max(1, len(rho_roll) // 7)
            for i in range(0, len(rho_roll)-1, arrow_step):
                ax_i.annotate("", xy=(rho_roll[i+1], tau_roll[i+1]),
                               xytext=(rho_roll[i], tau_roll[i]),
                               arrowprops=dict(arrowstyle="->", lw=1.0,
                                               color="black", alpha=0.55), zorder=7)
            ax_i.set_xlim(x_lo, x_hi); ax_i.set_ylim(y_lo, y_hi)
            ax_i.set_xlabel("Follower retaliation action ($\\rho$)", fontsize=10)
            ax_i.set_ylabel("Leader tariff action ($\\tau$)", fontsize=10)
            ax_i.set_title("Policy Convergence within Deterrence Region",
                           fontsize=11, fontweight="bold")
            ax_i.legend(fontsize=8, loc="upper left", framealpha=0.8)
            ax_i.grid(True, alpha=0.25)
            fig_i.patch.set_facecolor("white")
            plt.tight_layout()
            inset_path = f"{out_dir}/fig_pd2_convergence_inset.png"
            plt.savefig(inset_path, dpi=300, bbox_inches="tight"); plt.close(fig_i)
            print(f"  → Saved trajectory inset → {inset_path}", flush=True)
        else:
            print(f"  → [WARN] No trajectory sidecar at {traj_path} — "
                  f"set run_lf=True and show_trajectory=True first", flush=True)


# ============================================================
# PHASE DIAGRAM RUNNER
# ============================================================
def _run_phase_diagrams(base_params, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    run_pds  = set(cfg.get("run_pds", list(range(1, 15))))
    steps    = cfg["phase_steps"]
    registry = _build_pd_registry()

    patches = [mpatches.Patch(color=c, label=l) for c,l in
               [("#e74c3c","Escalation"),("#f39c12","Transition"),
                ("#27ae60","Deterrence"),("#7f8c8d","Multiple")]]
    import matplotlib.lines as mlines
    center_handle = mlines.Line2D([],[], marker="*", color="white",
                                  markeredgecolor="black", markeredgewidth=1.2,
                                  markersize=12, linestyle="None", label="CENTER baseline")
    run_count = 0
    anchor_label = cfg.get("ctr_regime","LF")
    results = {}

    for pd in registry:
        pd_num  = int(pd["name"][2:])
        pd_name = pd["name"]
        _n      = _pgrid(pd_name, cfg)
        x_vals  = np.linspace(pd["x_range"][0], pd["x_range"][1], _n)
        y_vals  = np.linspace(pd["y_range"][0], pd["y_range"][1], _n)
        nx, ny  = len(x_vals), len(y_vals)

        if pd_num not in run_pds:
            print(f"\n[SKIP] {pd_name}", flush=True)
            results[pd_name] = None
            continue
        # Guard: Fig 10 PDs (11-14) require leader_depr=True
        if pd_num >= 11 and not cfg.get("leader_depr", True):
            print(f"\n[SKIP] {pd_name} — requires leader_depr=True (dL axis meaningless when dL=0)",
                  flush=True)
            results[pd_name] = None
            continue
        if pd.get("_text_panel"):
            print(f"\n[TEXT PANEL] {pd_name} — no grid run", flush=True)
            results[pd_name] = None
            continue

        regime_grid = np.zeros((ny, nx), float)
        Lpay_grid   = np.zeros((ny, nx), float)

        jobs = []
        for yi, yv in enumerate(y_vals):
            for xi, xv in enumerate(x_vals):
                jobs.append(({
                    "base_params": base_params,
                    "x_attr": pd["x_attr"], "y_attr": pd["y_attr"],
                    "xi": xi, "yi": yi,
                }, xv, yv, cfg))

        n_jobs = len(jobs)
        n_workers_cfg = cfg.get("n_workers", 0)
        n_workers = max(1, _mp.cpu_count() - 1) if n_workers_cfg == 0 else n_workers_cfg
        use_parallel = (n_workers > 1)

        freeze_pct = int(cfg.get("freeze_leader_frac", 0.0) * 100)
        print(f"\n{'='*60}")
        print(f"  {pd_name} — {pd['x_attr']} × {pd['y_attr']}  "
              f"({nx}×{ny}={nx*ny} pts, {nx*ny*len(_get_seeds(cfg))} seed runs)  "
              f"workers={n_workers} ({'parallel' if use_parallel else 'serial'})  "
              f"freeze@{freeze_pct}%  mode={cfg.get('freeze_mode','full_freeze')}")
        print(f"{'='*60}", flush=True)
        pd_t0 = time.time()
        _per_seed_store = {}   # (x_val, y_val) -> [(seed, lp), ...]

        if use_parallel:
            ctx = _mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                for done_i, res in enumerate(
                        pool.imap_unordered(_grid_point_worker, jobs), 1):
                    xi_r, yi_r, regime, lp, fp, mx, seed_pays, xv_r, yv_r = res
                    regime_grid[yi_r, xi_r] = REGIME_COLORS.get(regime, 2)
                    Lpay_grid[yi_r, xi_r]   = lp
                    _per_seed_store[(xv_r, yv_r)] = seed_pays
                    run_count += len(_get_seeds(cfg))
                    elapsed = time.time() - pd_t0
                    eta = (elapsed / done_i) * (n_jobs - done_i)
                    print(f"  ✓ {pd_name} ({done_i}/{n_jobs}) "
                          f"{pd['x_attr']}={xv_r:.3f} {pd['y_attr']}={yv_r:.3f} "
                          f"→ {regime:10s}  L={lp:.1f}  "
                          f"ETA={int(eta//60)}m{int(eta%60):02d}s", flush=True)
        else:
            for done_i, job in enumerate(jobs, 1):
                pt_t0 = time.time()
                xi_r, yi_r, regime, lp, fp, mx, seed_pays, xv_r, yv_r = _grid_point_worker(job)
                regime_grid[yi_r, xi_r] = REGIME_COLORS.get(regime, 2)
                Lpay_grid[yi_r, xi_r]   = lp
                _per_seed_store[(xv_r, yv_r)] = seed_pays
                run_count += len(_get_seeds(cfg))
                pt_e = time.time() - pt_t0
                elapsed = time.time() - pd_t0
                eta = (elapsed / done_i) * (n_jobs - done_i)
                print(f"  ✓ {pd_name} ({done_i}/{n_jobs}) "
                      f"{pd['x_attr']}={xv_r:.3f} {pd['y_attr']}={yv_r:.3f} "
                      f"→ {regime:10s}  L={lp:.1f}  "
                      f"pt={pt_e:.1f}s  ETA={int(eta//60)}m{int(eta%60):02d}s", flush=True)

        results[pd_name] = (x_vals, y_vals, regime_grid, Lpay_grid, pd)
        _save_single_pd._per_seed_store = _per_seed_store
        _save_single_pd(pd_name, results[pd_name], cfg, out_dir,
                        patches, center_handle, anchor_label, steps)

    # ── Fig 8: PD1–PD4 (2×2) ──────────────────────────────────────────
    if any(n in run_pds for n in [1,2,3,4]):
        fig8, axs8 = plt.subplots(2, 2, figsize=(16, 13))
        gs = "  ".join(f"PD{n}={_pgrid(f'PD{n}',cfg)}×{_pgrid(f'PD{n}',cfg)}"
                       for n in [1,2,3,4])
        fig8.suptitle("Fig 8 — Phase Diagrams: Structural Parameters",
                      fontsize=13, fontweight="bold", y=0.99)
        for ax, n in zip(axs8.flatten(), [1,2,3,4]):
            nm = f"PD{n}"
            if results.get(nm) is None: _plot_skipped(ax, nm)
            else:
                xv,yv,rg,lpg,p_ = results[nm]
                _plot_phase_diagram(ax, xv, yv, rg, lpg,
                    xlabel=p_["xlabel"], ylabel=p_["ylabel"], title=p_["title"],
                    lf_x=p_["lf_x"], lf_y=p_["lf_y"],
                    ff_x=p_["ff_x"], ff_y=p_["ff_y"],
                    center_x=p_.get("center_x"), center_y=p_.get("center_y"),
                    font_size=11)
        fig8.legend(handles=patches, loc="lower center", ncol=4,
                    fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.0),
                    borderpad=0.4, handlelength=1.2, columnspacing=0.8)
        plt.subplots_adjust(top=0.96, bottom=0.04, hspace=0.18, wspace=0.20,
                            left=0.07, right=0.97)
        out8 = f"{out_dir}/fig8_phase_diagrams_structural.png"
        plt.savefig(out8, dpi=300, bbox_inches="tight"); plt.close(fig8)
        print(f"\n  Saved {out8}")

    # ── Fig 9: PD5–PD10 (2×3) ─────────────────────────────────────────
    if any(n in run_pds for n in [5,6,7,8,9,10]):
        fig9, axs9 = plt.subplots(2, 3, figsize=(22, 13))
        gs = "  ".join(f"PD{n}={_pgrid(f'PD{n}',cfg)}×{_pgrid(f'PD{n}',cfg)}"
                       for n in [5,6,7,8,9,10])
        fig9.suptitle("Fig 9 — Phase Diagrams: Behavioural Parameters",
                      fontsize=13, fontweight="bold", y=0.99)
        for ax, n in zip(axs9.flatten(), [5,6,7,8,9,10]):
            nm = f"PD{n}"
            if results.get(nm) is None: _plot_skipped(ax, nm)
            else:
                xv,yv,rg,lpg,p_ = results[nm]
                _plot_phase_diagram(ax, xv, yv, rg, lpg,
                    xlabel=p_["xlabel"], ylabel=p_["ylabel"], title=p_["title"],
                    lf_x=p_["lf_x"], lf_y=p_["lf_y"],
                    ff_x=p_["ff_x"], ff_y=p_["ff_y"],
                    center_x=p_.get("center_x"), center_y=p_.get("center_y"),
                    font_size=11)
        fig9.legend(handles=patches, loc="lower center", ncol=4,
                    fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.0),
                    borderpad=0.4, handlelength=1.2, columnspacing=0.8)
        plt.subplots_adjust(top=0.96, bottom=0.04, hspace=0.18, wspace=0.20,
                            left=0.06, right=0.97)
        out9 = f"{out_dir}/fig9_phase_diagrams_behavioural.png"
        plt.savefig(out9, dpi=300, bbox_inches="tight"); plt.close(fig9)
        print(f"\n  Saved {out9}")

    # ── Fig 10: PD11–PD14 (2×2) leader depreciation ───────────────────
    if any(n in run_pds for n in [11,12,13,14]):
        fig10, axs10 = plt.subplots(2, 2, figsize=(16, 13))
        gs = "  ".join(f"PD{n}={_pgrid(f'PD{n}',cfg)}×{_pgrid(f'PD{n}',cfg)}"
                       for n in [11,12,13,14])
        fig10.suptitle("Fig 10 — Phase Diagrams: Leader Depreciation Parameters",
            fontsize=13, fontweight="bold", y=0.99)
        registry = _build_pd_registry()
        for ax, n in zip(axs10.flatten(), [11,12,13,14]):
            nm = f"PD{n}"
            if results.get(nm) is None:
                _plot_skipped(ax, nm)
            else:
                xv,yv,rg,lpg,p_ = results[nm]
                _plot_phase_diagram(ax, xv, yv, rg, lpg,
                    xlabel=p_["xlabel"], ylabel=p_["ylabel"], title=p_["title"],
                    lf_x=p_["lf_x"], lf_y=p_["lf_y"],
                    ff_x=p_["ff_x"], ff_y=p_["ff_y"],
                    center_x=p_.get("center_x"), center_y=p_.get("center_y"),
                    font_size=11)
        fig10.legend(handles=patches, loc="lower center", ncol=4,
                     fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.0),
                     borderpad=0.4, handlelength=1.2, columnspacing=0.8)
        plt.subplots_adjust(top=0.96, bottom=0.04, hspace=0.18, wspace=0.20,
                            left=0.07, right=0.97)
        out10 = f"{out_dir}/fig10_phase_diagrams_leaderDepr.png"
        plt.savefig(out10, dpi=300, bbox_inches="tight"); plt.close(fig10)
        print(f"\n  Saved {out10}")


# ============================================================
# MAIN REGIME RUN — Fig 1–2
# FIX: StackelbergTariffGameEconomic now receives freeze params from cfg
# ============================================================
def _run_main_regime(params, prefix, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    steps = cfg["steps"]; win = cfg["smooth_win"]
    n_rho = cfg["n_rho_bins"]
    params = _apply_leader_depr(params, cfg)   # leader_depr toggle
    ec_label = "ON" if cfg.get("leader_depr", True) else "OFF"
    print(f"\n{'='*60}\n  Running: {prefix}  ({steps} steps)  "
          f"Leader depreciation: {ec_label}\n{'='*60}")
    env    = EconomicEnvironment(params)
    leader, follower, _, _, _ = build_agents(params, cfg, pd_mode=True)
    # FIX: pass freeze parameters from cfg
    game = StackelbergTariffGameEconomic(
        env, leader, follower, track=True,
        freeze_leader_frac=float(cfg.get("freeze_leader_frac", 0.0)),
        freeze_follower_frac=float(cfg.get("freeze_follower_frac", 0.0)),
        freeze_mode=cfg.get("freeze_mode", "full_freeze"),
    )
    game.run(rounds=steps)
    rr = game.results["rounds"]; R = np.arange(steps); rm = lambda x: _rolling_mean(x, win)

    tau_t  = np.array([r["tau"]            for r in rr], float)
    dL_t   = np.array([r.get("dL", 0.0)   for r in rr], float)
    d_t    = np.array([r["d"]              for r in rr], float)
    rho_t  = np.array([r["rho"]            for r in rr], float)
    Lpay   = np.array([r["leader_pay"]     for r in rr], float)
    Fpay   = np.array([r["follower_pay"]   for r in rr], float)
    M_t    = np.array([r["M"]              for r in rr], float)
    X_t    = np.array([r["X"]              for r in rr], float)
    E_t    = np.array([r["E"]              for r in rr], float)
    ineq_t = np.array([r.get("inequity_term",0.0) for r in rr], float)
    diplo_t= np.array([r.get("diplo_cost",0.0)    for r in rr], float)
    infL_t = np.array([r.get("inflation_cost_L",0.0) for r in rr], float)
    acDL_t = np.array([r.get("action_cost_dL",0.0)   for r in rr], float)
    uLlag_t= np.array([r.get("uL_lag",0.0)        for r in rr], float)
    rev_t  = np.array([r["rev"]            for r in rr], float)
    cons_t = np.array([r["cons_loss"]      for r in rr], float)
    eloss_t= np.array([r["export_loss"]    for r in rr], float)
    egain_t= np.array([r["export_gain"]    for r in rr], float)
    infl_t = np.array([r["infl_cost"]      for r in rr], float)
    acf_t  = np.array([r["action_cost_F"]  for r in rr], float)
    rc_t   = np.array([r["rho_cost"]       for r in rr], float)
    Lu = np.cumsum(Lpay); Fu = np.cumsum(Fpay)
    Ld = cumulative_discounted(Lpay, params.delta)
    Fd = cumulative_discounted(Fpay, params.delta)

    freeze_at = int(steps * cfg.get("freeze_leader_frac", 0.0))

    # Fig 1
    fig1, axs1 = plt.subplots(1, 4, figsize=(26, 5))
    fig1.suptitle(f"Fig 1 — V9 Dynamics  [{prefix}]  Leader:(τ,dL) Follower:(d,ρ)", fontsize=13)
    axs1[0].plot(R, tau_t, alpha=0.25, lw=0.8, color="tab:blue")
    axs1[0].plot(R, dL_t,  alpha=0.25, lw=0.8, color="tab:red")
    axs1[0].plot(R, d_t,   alpha=0.25, lw=0.8, color="tab:green")
    axs1[0].plot(R, rm(tau_t), lw=2, color="tab:blue",   label="τ (roll)")
    axs1[0].plot(R, rm(dL_t),  lw=2, color="tab:red",    label="dL (roll)")
    axs1[0].plot(R, rm(d_t),   lw=2, color="tab:green",  label="d (roll)")
    if freeze_at > 0:
        axs1[0].axvspan(freeze_at, steps, alpha=0.08, color="gray", label=f"post-freeze")
    axs1[0].set_title("Fig 1a — Actions (τ, dL, d)"); axs1[0].grid(True); axs1[0].legend(fontsize=8)
    axs1[1].plot(R, M_t, alpha=0.25, lw=0.8, color="tab:blue")
    axs1[1].plot(R, X_t, alpha=0.25, lw=0.8, color="tab:orange")
    axs1[1].plot(R, rm(M_t), lw=2, color="tab:blue",   label="M (roll)")
    axs1[1].plot(R, rm(X_t), lw=2, color="tab:orange", label="X (roll)")
    axs1[1].axhline(params.X0, color="tab:orange", ls=":", lw=1, alpha=0.7, label=f"X0={params.X0:.0f}")
    axs1[1].set_title("Fig 1b — Trade flows M, X"); axs1[1].grid(True); axs1[1].legend(fontsize=8)
    ax1c = axs1[2]; ax1c2 = ax1c.twinx()
    ax1c.plot(R, Ld, lw=2, color="tab:blue",  label="Leader (disc.)")
    ax1c.plot(R, Fd, lw=2, color="tab:orange",label="Follower (disc.)")
    ax1c2.plot(R, Lu, lw=2, color="tab:blue",   ls="--", label="Leader (undisc.)")
    ax1c2.plot(R, Fu, lw=2, color="tab:green",  ls="--", label="Follower (undisc.)")
    ax1c2.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax1c.set_title("Fig 1c — Cumulative payoffs"); ax1c.grid(True)
    lines = [l for l in ax1c.get_lines()+ax1c2.get_lines()
             if not l.get_label().startswith("_")]
    ax1c.legend(lines,[l.get_label() for l in lines], fontsize=7)
    ax1d = axs1[3]; ax1d2 = ax1d.twinx()
    ax1d.plot(R, rho_t, alpha=0.25, lw=0.8, color="tab:orange")
    ax1d.plot(R, rm(rho_t), lw=2, color="tab:orange", label="ρ (roll)")
    ax1d2.plot(R, E_t, alpha=0.25, lw=0.8, color="tab:purple")
    ax1d2.plot(R, rm(E_t), lw=2, color="tab:purple", label="E (roll)")
    ax1d2.axhline(params.E0, color="tab:purple", ls=":", alpha=0.5)
    ax1d.set_title("Fig 1d — ρ + E"); ax1d.grid(True)
    lines2 = [l for l in ax1d.get_lines()+ax1d2.get_lines()
              if not l.get_label().startswith("_")]
    ax1d.legend(lines2,[l.get_label() for l in lines2], fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_fig1_dynamics.png", dpi=120); plt.close(fig1)
    print(f"  Saved {prefix}_fig1_dynamics.png")

    # ── Trajectory sidecar for PD inset ──────────────────────────────
    if cfg.get("show_trajectory", False):
        win = int(cfg.get("trajectory_window", 50000))
        rho_roll, tau_roll = [], []
        for start in range(0, len(rr), win):
            chunk = rr[start:start+win]
            if chunk:
                rho_roll.append(float(np.mean([r["rho"] for r in chunk])))
                tau_roll.append(float(np.mean([r["tau"] for r in chunk])))
        traj_path = f"{out_dir}/{prefix}_trajectory.npy"
        np.save(traj_path, np.array([rho_roll, tau_roll]))
        print(f"  Saved trajectory sidecar → {traj_path}  ({len(rho_roll)} windows)")

    # Fig 2
    fig2, axs2 = plt.subplots(1, 4, figsize=(28, 5))
    fig2.suptitle(f"Fig 2 — Payoff Decompositions  [{prefix}]", fontsize=13)
    axs2[0].plot(R, rm(rev_t),   lw=2, color="tab:blue",   label="Revenue")
    axs2[0].plot(R, rm(cons_t),  lw=2, color="tab:purple", label="Consumer loss")
    axs2[0].plot(R, rm(eloss_t), lw=2, color="tab:red",    label="Export loss")
    axs2[0].plot(R, rm(infL_t),  lw=2, color="tab:brown",  label="Inflation φL·dL·M0")
    axs2[0].plot(R, rm(acDL_t),  lw=2, color="tab:pink",   label="Action cost c_dL·dL²")
    axs2[0].plot(R, rm(diplo_t), lw=2, color="tab:gray",   label="Diplo cost γ")
    axs2[0].set_title("Fig 2a — Leader"); axs2[0].legend(fontsize=7); axs2[0].grid(True)
    axs2[1].plot(R, rm(egain_t), lw=2, label="Export gain")
    axs2[1].plot(R, rm(infl_t),  lw=2, label="Inflation cost")
    axs2[1].plot(R, rm(acf_t),   lw=2, label="Action cost")
    axs2[1].plot(R, rm(rc_t),    lw=2, color="tab:blue", label="ρ cost")
    axs2[1].set_title("Fig 2b — Follower base"); axs2[1].legend(fontsize=8); axs2[1].grid(True)
    axs2[2].plot(R, dL_t,  alpha=0.25, lw=0.8, color="tab:red")
    axs2[2].plot(R, rm(dL_t),  lw=2, color="tab:red",    label=f"dL (roll {win})")
    axs2[2].axhline(params.dL_max, color="tab:red", ls=":", lw=1.2, label=f"dL_max={params.dL_max:.2f}")
    axs2[2].set_ylabel("dL — leader depreciation", color="tab:red")
    ax2c2 = axs2[2].twinx()
    ax2c2.plot(R, X_t, alpha=0.25, lw=0.8, color="tab:orange")
    ax2c2.plot(R, rm(X_t), lw=2, color="tab:orange", label="X (roll)")
    ax2c2.axhline(params.X0, color="tab:orange", ls=":", lw=1.2, label=f"X0={params.X0:.0f}")
    ax2c2.set_ylabel("X — follower exports", color="tab:orange")
    axs2[2].set_title("Fig 2c — Leader dL + follower X"); axs2[2].grid(True)
    lines_c = [l for l in axs2[2].get_lines()+ax2c2.get_lines()
               if not l.get_label().startswith("_")]
    axs2[2].legend(lines_c,[l.get_label() for l in lines_c], fontsize=8)
    axs2[3].plot(R, rm(-ineq_t), lw=2, color="tab:red",  label=r"-Inequity α·max(uLlag-uF,0)")
    axs2[3].plot(R, rm(diplo_t), lw=2, color="tab:gray", label=r"Diplo cost γ·(ρ/ρmax)·τM")
    axs2[3].plot(R, uLlag_t, lw=1.5, color="tab:blue", ls="--", label=r"$u_{L,lag}$")
    axs2[3].plot(R, rm(Lpay), lw=2, color="tab:blue",   label="Leader payoff")
    axs2[3].plot(R, rm(Fpay), lw=2, color="tab:orange", label="Follower payoff")
    axs2[3].axhline(0, color="k", ls=":", lw=0.8, alpha=0.5)
    axs2[3].set_title("Fig 2d — Behavioural terms + payoffs"); axs2[3].legend(fontsize=7); axs2[3].grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_fig2_decompositions.png", dpi=120); plt.close(fig2)
    print(f"  Saved {prefix}_fig2_decompositions.png")

    half = steps//2
    mt=float(np.mean(tau_t[half:])); md=float(np.mean(d_t[half:]))
    mr=float(np.mean(rho_t[half:])); mx=float(np.mean(dL_t[half:]))
    regime,df,rf,fs = classify_regime(mt,md,mr,params.tau_max,params.d_max,
                                       params.rho_max,float(np.mean(Lpay[half:])))
    print(f"\n--- [{prefix}] Sanity ---")
    print(f"  Leader={float(np.mean(Lpay)):.2f}  Follower={float(np.mean(Fpay)):.2f}")
    print(f"  Mean dL={mx:.4f}  Mean τ={mt:.4f}  Classifier: {regime}")


# ============================================================
# MAIN
# ============================================================
def test_stackelberg_q3_tariff_econ_v10_leaderDepr_PD():
    """
    RUN_CONFIG controls:
      run_pds            — list of PDs to run (1–14); subset for quick whatif
      phase_grid_default — 6 (fast) or 10 (fine); applies to all PDs
      phase_grid_pdN     — per-PD override
      run_lf/run_ff      — main regime Fig 1-2
      ctr_regime         — "LF" or "FF" anchor for phase diagrams
      freeze_leader_frac — fraction of phase_steps at which leader is frozen
      freeze_mode        — full_freeze | epsilon_only | none
    """
    cfg = RUN_CONFIG; out_dir = _get_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    run_pds = cfg.get("run_pds", list(range(1, 15)))
    freeze_pct = int(cfg.get("freeze_leader_frac", 0.0) * 100)
    print(f"\n  Output dir      : {out_dir}/")
    print(f"  PDs to run      : {sorted(run_pds)}")
    print(f"  Grid size       : {cfg.get('phase_grid_default',6)}×{cfg.get('phase_grid_default',6)}")
    print(f"  n_dL_bins       : {cfg['n_dL_bins']}  (leader joint actions: "
          f"{cfg['n_bins']}×{cfg['n_dL_bins']}={cfg['n_bins']*cfg['n_dL_bins']})")
    print(f"  freeze_leader   : {freeze_pct}%  mode={cfg.get('freeze_mode','full_freeze')}")
    print(f"  leader_q_init   : {cfg['leader_q_init']}")
    print(f"  follower_q_init : {cfg['follower_q_init']}")
    print(f"  leader_alpha_min: {cfg.get('leader_alpha_min', 0.005)}")
    print(f"  follower_alpha_min: {cfg.get('follower_alpha_min', 0.010)}")
    print(f"  leader_depr     : {cfg.get('leader_depr', True)}  "
          f"({'dL ON — Fig10' if cfg.get('leader_depr', True) else 'dL OFF — Fig8/9 V7-mode'})")

    if cfg.get("run_lf", True):
        random.seed(SEED); np.random.seed(SEED)
        _run_main_regime(build_params_lf(), "LF", cfg, out_dir)
    if cfg.get("run_ff", True):
        random.seed(SEED); np.random.seed(SEED)
        _run_main_regime(build_params_ff(), "FF", cfg, out_dir)
    if cfg.get("ctr_regime","LF") == "CENTER":
        random.seed(SEED); np.random.seed(SEED)
        _run_main_regime(build_params_center(), "CENTER", cfg, out_dir)

    if cfg.get("run_phase_plots", True):
        ctr = cfg.get("ctr_regime","LF")
        anchor = build_params_lf() if ctr=="LF" else (
                 build_params_center() if ctr=="CENTER" else build_params_ff())
        print(f"\n--- Phase diagrams [{ctr} anchor]  PDs={sorted(run_pds)} ---")
        random.seed(SEED); np.random.seed(SEED)
        _run_phase_diagrams(anchor, cfg, out_dir)

    print(f"\n=== All saved to {out_dir}/ ===")


if __name__ == "__main__":
    test_stackelberg_q3_tariff_econ_v10_leaderDepr_PD()
