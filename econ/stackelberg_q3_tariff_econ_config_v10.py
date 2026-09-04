# stackelberg_q3_tariff_econ_config_v10.py
#
# V10 SHARED CONFIGURATION
# ════════════════════════
# Single source of truth for parameters, agent builders, and classifier.
# Imported by both the sensitivity test file and the phase diagram file.
#
# V10 CHANGES FROM V9
# ═══════════════════
# 1. X* now responds to τ:  X* = X0·(1+τ)^{-εd}·(1+d)^{εs}
#    — bilateral consistency: leader tariff suppresses follower exports
#    — new param: x_tau_elast (elasticity of X* to τ, defaults to demand_elast)
# 2. Behavioural params zeroed as default: alpha_ineq=0, beta_spite=0, gamma_diplo=0
#    — equilibrium driven by structural params only
#    — Fig 9 sensitivity = perturbation analysis around zero baseline
# 3. Shared config extracted here — test and PD files import from this module
# 4. All v9/v8 artifacts removed
#
# CLASSIFIER FIX (current session):
# — Payoff-primary with fs tiebreaker only in neutral band (-10, +10)
# — Outside band: payoff alone decides (no fs veto)
# — Inside band: fs thresholds 0.45 / 0.30 resolve ambiguous cases
# — Consistent fallback (no payoff): same fs thresholds
#
# FOLLOWER_COST_RHO (current):
# — rho_cost E0-anchored in sim: follower_cost_rho * rho^2 * E0
# — Values set so effective max rho_cost ≈ 0.10 across calibrations:
#   LF:     0.005 × 0.04 × 500 = 0.10
#   FF:     0.003 × 0.1225 × 450 = 0.165
#   CENTER: 0.004 × 0.073 × 475 = 0.139

import copy
import stackelberg_q3_tariff_econ_sim_v10_leaderDepr as sim
from stackelberg_q3_tariff_econ_sim_v10_leaderDepr import (
    EconomicParams, Q3BinnedLeader, Q3BinnedFollower,
    StackelbergTariffGameEconomic, EconomicEnvironment,
)

# ============================================================
# UTILITIES
# ============================================================
def make_uniform_bins(vmin: float, vmax: float, n: int = 6):
    from stackelberg_q3_tariff_econ_sim_v10_leaderDepr import make_bins
    return make_bins(n, vmin, vmax)

# ============================================================
# CALIBRATIONS
# ============================================================
def build_params_lf():
    """
    Leader-Favoring (LF) — V10 calibration.
    Structural baseline: α=β=γ=0 (pure self-interest).
    Leader has tariff + depreciation advantage.
    follower_cost_rho=0.005 (E0-anchored: effective max = 0.005*0.04*500 = 0.10)
    """
    return EconomicParams(
        trade_form="power",
        M0=450.0, X0=500.0, E0=500.0,
        demand_elast=1.30, supply_elast=0.85, export_elast=1.00,
        kappa=0.20, lam=0.40, eta=0.30, delta=0.98,
        phi_inflation=0.50, leader_cost_w=0.008, follower_cost_w=0.015,
        tau_max=0.35, d_max=0.18, psi_E=0.90, rho_max=0.20,
        follower_cost_rho=0.10,          # E0-anchored: effective max = 0.005*0.04*500 = 0.10
        # V10: behavioural params zeroed — pure structural equilibrium
        alpha_ineq=0.0, beta_spite=0.5, gamma_diplo=0.0,
        uL_lag_window=50,
        # V10: cross-elasticities — zeroed pending recalibration
        x_tau_elast=0.0,    # τ→X* (activate in future calibration)
        dL_elast_X=0.0,     # dL→X* (activate in future calibration)
        d_elast_E=0.0,      # d→E*  (activate in future calibration)
        # leader depreciation
        dL_max=0.15,
        dL_elast_M=0.0,      # zeroed — M* cost captured by phi_inflation_L in payoff
        dL_elast_E=0.60,
        phi_inflation_L=0.30,
        leader_cost_dL=0.010,
    )


def build_params_ff():
    """
    Follower-Favoring (FF) — V10 calibration.
    Structural baseline: α=β=γ=0.
    Follower has depreciation + retaliation advantage.
    follower_cost_rho=0.003 (E0-anchored: effective max = 0.003*0.1225*450 = 0.165)
    """
    return EconomicParams(
        trade_form="power",
        M0=450.0, X0=500.0, E0=450.0,
        demand_elast=1.55, supply_elast=1.20, export_elast=1.50,
        kappa=0.60, lam=0.55, eta=0.55, delta=0.98,
        phi_inflation=0.20, leader_cost_w=0.015, follower_cost_w=0.020,
        tau_max=0.25, d_max=0.25, psi_E=1.4, rho_max=0.35,
        follower_cost_rho=0.12,          # E0-anchored: effective max = 0.003*0.1225*450 = 0.165
        # V10: behavioural params zeroed
        alpha_ineq=0.0, beta_spite=0.5, gamma_diplo=0.0,
        uL_lag_window=50,
        # V10: cross-elasticities zeroed
        x_tau_elast=0.0,
        dL_elast_X=0.0,
        d_elast_E=0.0,
        # leader depreciation
        dL_max=0.10,
        dL_elast_M=0.0,      # zeroed — M* cost captured by phi_inflation_L in payoff
        dL_elast_E=0.60,
        phi_inflation_L=0.30,
        leader_cost_dL=0.010,
    )


def build_params_center():
    """
    CENTER calibration — V10.
    Midpoint between LF and FF, targeting Deterrence/Transition boundary.
    Designed so dL PDs show regime-shifting power of depreciation.
    follower_cost_rho=0.004 (E0-anchored: effective max = 0.004*0.073*475 = 0.139)
    """
    return EconomicParams(
        trade_form="power",
        M0=450.0, X0=500.0, E0=475.0,
        demand_elast=1.42, supply_elast=1.00, export_elast=1.25,
        kappa=0.60, lam=0.55, eta=0.55, delta=0.98,
        phi_inflation=0.28, leader_cost_w=0.012, follower_cost_w=0.012,
        tau_max=0.28, d_max=0.22, psi_E=1.15, rho_max=0.27,
        follower_cost_rho=0.06,          # E0-anchored: effective max = 0.004*0.073*475 = 0.139
        # V10: behavioural params zeroed
        alpha_ineq=0.0, beta_spite=0.5, gamma_diplo=0.0,
        uL_lag_window=50,
        # V10: cross-elasticities zeroed
        x_tau_elast=0.0,
        dL_elast_X=0.0,
        d_elast_E=0.0,
        # leader depreciation
        dL_max=0.12,
        dL_elast_M=0.0,      # zeroed — M* cost captured by phi_inflation_L in payoff
        dL_elast_E=0.60,
        phi_inflation_L=0.30,
        leader_cost_dL=0.010,
    )


# ============================================================
# REFERENCE PARAM DICTS (for phase diagram star placement)
# ============================================================
LF_PARAMS = dict(
    demand_elast=1.30, supply_elast=0.85, export_elast=1.00,
    tau_max=0.35, d_max=0.18, rho_max=0.20, psi_E=0.90,
    alpha_ineq=0.0, beta_spite=0.5, gamma_diplo=0.0,
    dL_max=0.15, dL_elast_M=0.0, dL_elast_E=0.60,
    phi_inflation_L=0.30, leader_cost_dL=0.010,
    x_tau_elast=0.0,
    dL_elast_X=0.0,
    d_elast_E=0.0,
)
FF_PARAMS = dict(
    demand_elast=1.55, supply_elast=1.20, export_elast=1.50,
    tau_max=0.25, d_max=0.25, rho_max=0.35, psi_E=1.4,
    alpha_ineq=0.0, beta_spite=0.5, gamma_diplo=0.0,
    dL_max=0.10, dL_elast_M=0.0, dL_elast_E=0.60,
    phi_inflation_L=0.30, leader_cost_dL=0.010,
    x_tau_elast=0.0,
    dL_elast_X=0.0,
    d_elast_E=0.0,
)
CENTER_PARAMS = dict(
    demand_elast=1.42, supply_elast=1.00, export_elast=1.25,
    tau_max=0.28, d_max=0.22, rho_max=0.27, psi_E=1.15,
    alpha_ineq=0.0, beta_spite=0.5, gamma_diplo=0.0,
    dL_max=0.12, dL_elast_M=0.0, dL_elast_E=0.60,
    phi_inflation_L=0.30, leader_cost_dL=0.010,
    x_tau_elast=0.0,
    dL_elast_X=0.0,
    d_elast_E=0.0,
)


# ============================================================
# AGENT BUILDER — single source for both test and PD files
# ============================================================
def build_agents(params, cfg, pd_mode=False):
    """Build leader and follower Q-learning agents.

    pd_mode=False (sensitivity/test): dL_bins=None when dL_max=0
    pd_mode=True  (phase diagrams):   dL_bins=[0.0] when dL_max=0
      — keeps action space consistent across all PD grid points

    Returns: leader, follower, tau_bins, d_bins, rho_bins
    """
    n_bins = cfg["n_bins"]
    n_rho  = cfg["n_rho_bins"]
    tau_bins = make_uniform_bins(0.0, params.tau_max, n_bins)
    d_bins   = make_uniform_bins(0.0, params.d_max,   n_bins)
    rho_bins = make_uniform_bins(0.0, params.rho_max, n_rho) \
               if params.rho_max > 0 else [0.0]

    if params.dL_max > 0:
        dL_bins = make_uniform_bins(0.0, params.dL_max, cfg["n_dL_bins"])
    else:
        #dL_bins = [0.0] if pd_mode else None
        dL_bins =  None


    eps_end = float(cfg.get("epsilon_end", 0.002))  # V10: configurable floor
    l_alpha_min = float(cfg.get("leader_alpha_min", 0.02))   # V7 default restored
    f_alpha_min = float(cfg.get("follower_alpha_min", 0.05)) # V7 default restored
    leader = Q3BinnedLeader(
        tau_bins, epsilon=cfg["epsilon_start"], gamma=params.delta,
        alpha=cfg["alpha"], q_init=cfg["leader_q_init"], dL_bins=dL_bins,
        epsilon_end=eps_end, alpha_min=l_alpha_min)
    leader.history_f = [f"{params.d_max * 0.5:.2f}"] * 3
    leader.tau_last  = f"{params.tau_max * 0.8:.2f}"
    if dL_bins is not None:
        leader.dL_last = f"{(dL_bins[-1] * 0.3):.2f}"

    follower = Q3BinnedFollower(
        d_bins, rho_bins=rho_bins,
        epsilon=cfg["epsilon_start"], gamma=params.delta,
        alpha=cfg["alpha"], double_q=cfg["follower_double_q"],
        q_init_f=cfg["follower_q_init"],
        epsilon_end=eps_end, alpha_min=f_alpha_min)

    return leader, follower, tau_bins, d_bins, rho_bins


# ============================================================
# REGIME CLASSIFIER — single source
# ============================================================
def classify_regime(mean_tau, mean_d, mean_rho, tau_max, d_max, rho_max,
                    mean_leader_pay=None):
    if mean_leader_pay is not None:
        if mean_leader_pay < -15:
            regime = "Escalation"
        elif mean_leader_pay > 15:
            regime = "Deterrence"
        else:
            regime = "Transition"
        return regime, 0.0, 0.0, 0.0

    # Fallback — no payoff available (not used in PD runs)
    d_frac   = mean_d   / d_max   if d_max   > 0 else 0.0
    rho_frac = mean_rho / rho_max if rho_max > 0 else 0.0
    fs = 0.5 * d_frac + 0.5 * rho_frac
    if fs >= 0.45:
        regime = "Escalation"
    elif fs <= 0.30:
        regime = "Deterrence"
    else:
        regime = "Transition"
    return regime, d_frac, rho_frac, fs



# ============================================================
# SHARED PLOTTING HELPERS
# ============================================================
import numpy as np

def rolling_mean(x, win=25):
    if win <= 1: return np.asarray(x, float)
    x = np.asarray(x, float)
    if x.size == 0: return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[win:] - c[:-win]) / float(win)
    if out.size == 0: return x.copy()
    return np.concatenate([np.full(len(x) - len(out), out[0]), out])


def cumulative_discounted(rewards, delta):
    r = np.asarray(rewards, dtype=float)
    out = np.zeros_like(r)
    acc, pw = 0.0, 1.0
    for t in range(len(r)):
        acc += pw * r[t]; out[t] = acc; pw *= delta
    return out


def to_same_len(arr, n):
    arr = np.asarray(arr, float)
    if len(arr) == n: return arr
    if len(arr) == 0: return np.zeros(n)
    return np.interp(np.linspace(0, len(arr)-1, n),
                     np.arange(len(arr)), arr)
