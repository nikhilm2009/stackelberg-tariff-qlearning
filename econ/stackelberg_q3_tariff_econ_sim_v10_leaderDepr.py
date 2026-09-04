# stackelberg_q3_tariff_econ_sim_v10_leaderDepr.py
#
# V10 MODEL SUMMARY — Full Strategic Symmetry
# ════════════════
# Leader  chooses (τ, dL):  tariff + currency depreciation
# Follower chooses (d,  ρ):  own depreciation + retaliation tariff
#
# dL (leader depreciation) effects:
#   M* = M0·(1+τ)^{-εd}·(1+dL)^{-εdL}   — suppresses imports
#   E* = E0·(1+ρ)^{-εE}·(1+dL)^{+εdL}   — boosts leader exports
#   X* = X0·(1+d)^{εs}                    — follower exports (unaffected by dL)
#   leader cost: -φL·dL·M0  (inflation) - c_dL·dL²  (action cost)
#
# dL observability:
#   dL is a macro instrument (exchange rate) that shifts M* and E*;
#   the follower observes its effects through the environment state (Xbin,
#   X_momentum) and also directly via a coarse-binned dL_last in its state
#   tuple. This allows the follower to condition its (d,ρ) policy on whether
#   the leader is actively depreciating.
#
# Regression invariant: dL_max=0  →  dL=0 every step → V10 ≡ V7 exactly.
# Smoke test: set dL_max=0, x_tau_elast=0; verify M*=M0*(1+τ)^{-εd},
#             X*=X0*(1+d)^εs, E*=E0*(1+ρ)^{-εE}.
#
# V8 BUG FIX — evaluate_follower_payoff() timing correction
# ============================================================
# Bug: "post" branch ran _transition BEFORE follower_period_payoff,
#      so disruption_cost = ψx·x·X_{t+1} (post-transition X).
#      But step() runs follower_period_payoff BEFORE _transition,
#      so disruption_cost = ψx·x·X_t (pre-transition X).
#      Mismatch caused BR grid search to undervalue export controls
#      (X_{t+1} < X_t when x > 0, so disruption appeared smaller).
# Fix: swap order in "post" branch — evaluate payoff first, then
#      transition. Both "pre" and "post" now use X_t consistently.
#      The finally-block still restores M/X/E so the env is clean.
#      Note: "pre" and "post" now differ only in whether the state
#      is advanced AFTER the return — useful for chained calls.
# ============================================================
#
# V9 CHANGES — leader depreciation replaces export controls
# ============================================================
# Leader action space: (tau, x) -> (tau, dL)
# REMOVED: x_max, export_ctrl_elast, leader_cost_x, psi_x
#          _X_star x factor, action_cost_x, disruption_cost
# ADDED:   dL_max, dL_elast_M, dL_elast_E, phi_inflation_L, leader_cost_dL
#          _M_star dL suppression, _E_star dL boost
#          inflation_cost_L, action_cost_dL in leader payoff
# Follower unchanged — responds to tau as before.
# References: Obstfeld & Rogoff (1995) JPE; Corsetti et al. (1998);
#             Boz et al. (2022) AER dominant currency pricing
# ============================================================
# V4 CHANGES (Fix C — gap redefinition for gap_endogenous spite)
# ============================================================
# Problem (diagnosed from Fig 10c / Fig 11):
#   Under FF the follower consistently outperforms the leader, so
#   leader_welfare − follower_welfare < 0 always → gap_recent ≤ 0 always
#   → max(0, gap_recent) = 0 always → beta_t = beta_0 = 0 always.
#   The gap_endogenous spite model was structurally inactive in FF.
#
# Fix C — redefine gap as leader absolute welfare level:
#   gap_t = max(0, rolling_mean(leader_payoff)) / gap_scale
#
#   Economic interpretation: the follower escalates spite in proportion to
#   how well the leader is doing in absolute terms, not relative terms.
#   Spite rises when the leader is welfare-positive (winning the conflict)
#   and is dormant when the leader is welfare-negative (already losing).
#
#   Regime behaviour:
#     LF: leader payoff > 0 most of the time  → gap_t > 0 → beta_t rises ✓
#     FF: leader payoff < 0 most of the time  → gap_t = 0 → beta_t = beta_0
#         (reduced-severity FF; spite is dormant but not distorting) ✓
#
# Changes in this file:
#   * _update_gap(): gap formula changed from (lmean - fmean)/scale
#                    to max(0, lmean)/scale; max(0,...) outer clip removed
#                    (gap_recent now always >= 0; inner clip preserved)
#   * follower_period_payoff_with_components(): docstring updated
#   * SIM_VERSION, file header updated
#   * All other logic unchanged — backward compatible with spite_model="none"
# ============================================================
#
# V3 CHANGES (adds endogenous spite — Model 2)
# ============================================================
# New in V3:
#   + EconomicParams: spite_model ("none"/"tau_scaled"/"gap_endogenous")
#     beta_0, gamma_beta, beta_max, gap_window, gap_scale
#   + EconomicEnvironment: rolling payoff buffers, gap_recent, beta_t
#   + follower_period_payoff: three-way spite branch
#   + Logging: beta_t, gap_recent added to results
#   + Backward compatible: spite_model="none" → identical to v2
# ============================================================
#
# V2 CHANGES (ref: Stackelberg_Retaliation_Extension_v1.docx)
# ============================================================
# Phase 1A — EconomicParams:
#   + E0, export_elast, eta, psi_E, rho_max, follower_cost_rho, beta_E (spite, default=0)
#
# Phase 1B — EconomicEnvironment:
#   + self.E state variable (reset, transitions)
#   + _E_star(rho)  — power-form export target
#   + _transition() now accepts (tau, d, rho) and mean-reverts E
#   + leader_period_payoff_with_components() returns export_loss as 5th component
#   + follower_period_payoff_with_components(d, rho) returns rho_cost as 5th component
#   + Optional spite term: beta_E * (tau/tau_max) * (E0 - E) in follower payoff (normalised; default beta_E=0.0)
#   + evaluate_follower_payoff(tau, d, rho) updated signature
#   + step(tau, d, rho) updated signature
#
# Phase 2A — BestResponseFollowerEconomic:
#   + rho_bins parameter; joint (d, rho) grid search (5x5=25 evals)
#   + respond() returns Tuple[float, float] — (best_d, best_rho)
#   + last_rho_action attribute
#
# Phase 2B — Q3BinnedFollower:
#   + rho_bins parameter; joint_actions list of (d, rho) tuples
#   + State extended to 7-tuple: (tau3, tau2, tau1, Xbin, d_last, rho_last, X_momentum)
#   + Q-table keyed by (7-tuple-state, (d, rho)) joint action
#   + respond() returns Tuple[float, float]
#   + update() handles joint action keys
#
# Phase 3A — Q3BinnedLeader:
#   + rho_last tracked in state (7-tuple: d1,d2,d3,tau_last,Mbin,Xbin,rho_last)
#   + decide_tariff() accepts follower_last_rho
#   + update() accepts follower_new_rho
#
# Phase 3B — StackelbergTariffGameEconomic:
#   + step() unpacks (d_t, rho_t) from follower
#   + Passes rho_t to env.step(), leader.update()
#   + Logs 'rho', 'E', 'export_loss', 'rho_cost', 'spite_term' in results
#
# Unchanged from v1:
#   - Inflation cost form: phi * d * M0  (kept as-is, v1 convention)
#   - Consumer loss form:  0.5 * tau * (M0 - M)  (kept as-is, v1 convention)
#   - All epsilon-decay, double-Q, adaptive alpha logic
#   - make_bins utility
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import random
import statistics as _stats
from collections import Counter, deque

SIM_VERSION = "econ-sim v10_leaderDepr — X* bilateral fix: X*=X0·(1+τ)^{-εx_τ}·(1+d)^{εs}; α=β=γ=0 default"

# ============================================================
# Phase 1A: EconomicParams — new fields annotated with # V2
# ============================================================
@dataclass
class EconomicParams:
    trade_form: str = "linear"   # "linear" or "power"
    M0: float = 300.0
    X0: float = 300.0
    demand_elast: float = 1.5
    supply_elast: float = 1.2
    kappa: float = 0.4
    lam: float = 0.35
    delta: float = 0.97
    horizon: int = 1
    phi_inflation: float = 0.20
    leader_cost_w: float = 0.0
    follower_cost_w: float = 0.0
    tau_max: float = 0.35
    d_max: float = 0.35
    # --- V2: new parameters for leader export channel and retaliation ---
    E0: float = 300.0            # V2: baseline leader export volume (symmetric with M0, X0)
    export_elast: float = 1.2    # V2: elasticity of E to retaliatory tariff rho (mirrors demand_elast)
    eta: float = 0.40            # V2: mean-reversion speed for E (mirrors kappa)
    psi_E: float = 1.0           # V2: weight on export-loss term in leader payoff
    rho_max: float = 0.35        # V2: max retaliatory tariff follower can impose (mirrors tau_max)
    follower_cost_rho: float = 0.0  # V2: quadratic cost c_rho for follower using retaliation
    # --- V7: behavioural payoff parameters ---
    # Fehr-Schmidt inequity aversion: follower suffers when leader welfare > follower welfare
    alpha_ineq:    float = 0.0    # V7: inequity aversion weight α — Fehr & Schmidt (1999)
    # Retaliation spite: follower gains from beta_spite * tau * (E0-E) — aligned with DRQN V5
    beta_spite:    float = 0.0    # V7: follower retaliation spite weight β
    # Diplomatic cost: leader pays gamma * (rho/rho_max) * (tau*M) when follower retaliates
    gamma_diplo:   float = 0.0    # V7: leader diplomatic/political cost weight γ
    uL_lag_window: int   = 50     # V7: rolling window W for uL_ref (50=smoother reference payoff)
    # --- V8: leader export control parameters ---
    # --- V10: X* bilateral parameter ---
    x_tau_elast:        float = 0.0   # V10: εx,τ — τ suppresses X*
                                      #     X*=X0·(1+τ)^{-εx,τ}·(1+d)^{εs}·(1+dL)^{-εdL,X}
    dL_elast_X:         float = 0.0   # V10: εdL,X — leader dL suppresses follower X*
                                      #     captures expenditure-switching: dL makes
                                      #     follower exports more expensive in leader market
    d_elast_E:          float = 0.0   # V10: εd,X — follower d suppresses leader E*
                                      #     follower depreciation makes leader exports
                                      #     less competitive in follower market
    # --- V9/V10: leader depreciation parameters ---
    dL_max:             float = 0.15  # V9: max leader depreciation intensity
    dL_elast_M:         float = 0.80  # V9: εdL — M* suppression: M*=M0·(1+τ)^{-εd}·(1+dL)^{-εdL}
    dL_elast_E:         float = 0.60  # V9: E* boost:  E*=E0·(1+ρ)^{-εE}·(1+dL)^{+εdL}
    phi_inflation_L:    float = 0.30  # V9: leader inflation cost: −φL·dL·M0
    leader_cost_dL:     float = 0.010 # V9: quadratic action cost: −c_dL·dL²


# ============================================================
# Phase 1B: EconomicEnvironment — surgical additions marked V2
# ============================================================
class EconomicEnvironment:
    def __init__(self, params: EconomicParams):
        self.p = params
        self.reset()

    def reset(self):
        self.M = self.p.M0
        self.X = self.p.X0
        self.E = self.p.E0   # V2: initialise leader export flow
        # for momentum (set/updated by orchestrator)
        self.prev_X = self.X
        # V7: lagged leader payoff buffer for inequity aversion (avoids simultaneity)
        _w = max(1, int(getattr(self.p, "uL_lag_window", 1)))
        self._uL_buf = deque(maxlen=_w)
        self.uL_lag: float = 0.0       # V7: lagged/rolling-mean leader payoff (previous period)
        self._prev_uL_lag: float = 0.0  # V7: value of uL_lag BEFORE current update (used in F)

    def _update_uL_lag(self, L_pay: float):
        """V7: update lagged leader payoff buffer.
        self._prev_uL_lag = value used by follower this period (uL_{t-1})
        self.uL_lag       = updated value available next period  (uL_t)
        Genuine one-step lag: follower at t uses L_{t-1}, never L_t.
        """
        self._prev_uL_lag = self.uL_lag          # save what F just used
        self._uL_buf.append(L_pay)
        self.uL_lag = _stats.mean(self._uL_buf)  # now stores L_t for t+1

    def _M_star(self, tau: float, dL: float = 0.0) -> float:
        """V10: M* = M0·(1+τ)^{-εm}
        dL_elast_M zeroed — dL import cost captured by phi_inflation_L in payoff.
        dL arg kept for API compatibility; has no effect when dL_elast_M=0.
        """
        if getattr(self.p, "trade_form", "linear") == "power":
            base = self.p.M0 * (1.0 + tau) ** (-self.p.demand_elast)
        else:
            base = self.p.M0 * (1.0 - self.p.demand_elast * tau)
        dL_elast_M = getattr(self.p, "dL_elast_M", 0.0)
        return max(0.0, base * (1.0 + max(0.0, dL)) ** (-dL_elast_M))

    def _X_star(self, d: float, tau: float = 0.0, dL: float = 0.0) -> float:
        """V10: X* = X0·(1+τ)^{-εx,τ}·(1+d)^{εs}·(1+dL)^{-εdL,X}
        τ  suppresses X* (bilateral: leader tariff hits follower exports)
        d  boosts    X* (follower depreciation makes exports competitive)
        dL suppresses X* (leader depreciation makes follower goods more expensive)
        All elasticities default to 0.0 → V9 behaviour exactly.
        """
        x_tau_elast = getattr(self.p, "x_tau_elast", 0.0)
        dL_elast_X  = getattr(self.p, "dL_elast_X",  0.0)
        if getattr(self.p, "trade_form", "linear") == "power":
            base = self.p.X0 * (1.0 + d) ** self.p.supply_elast
        else:
            base = self.p.X0 * (1.0 + self.p.supply_elast * d)
        return max(0.0,
                   base
                   * (1.0 + max(0.0, tau)) ** (-x_tau_elast)
                   * (1.0 + max(0.0, dL))  ** (-dL_elast_X))

    # V2: new method — leader export target suppressed by rho (power-form only, per framework doc)
    def _E_star(self, rho: float, dL: float = 0.0, d: float = 0.0) -> float:
        """V10: E* = E0·(1+ρ)^{-εE}·(1+dL)^{+εdL,E}·(1+d)^{-εd,X}
        ρ  suppresses E (retaliation tariff hits leader exports)
        dL boosts    E (leader depreciation makes exports competitive)
        d  suppresses E (follower depreciation makes leader exports more expensive)
        All elasticities default to 0.0 → V9 behaviour exactly.
        """
        dL_elast_E = getattr(self.p, "dL_elast_E", 0.0)
        d_elast_E  = getattr(self.p, "d_elast_E",  0.0)
        return max(0.0,
                   self.p.E0
                   * (1.0 + rho)          ** (-self.p.export_elast)
                   * (1.0 + max(0.0, dL)) ** ( dL_elast_E)
                   * (1.0 + max(0.0, d))  ** (-d_elast_E))

    # V9: _transition
    def _transition(self, tau: float, d: float, rho: float = 0.0, dL: float = 0.0):
        """V9: M* suppressed by tau+dL; E* boosted by dL; X* no x factor."""
        M_star = self._M_star(tau, dL)
        X_star = self._X_star(d, tau, dL)   # V10: τ and dL suppress X*; d boosts
        E_star = self._E_star(rho, dL, d)   # V10: ρ and d suppress E*; dL boosts
        self.M = (1 - self.p.kappa) * self.M + self.p.kappa * M_star
        self.X = (1 - self.p.lam)   * self.X + self.p.lam   * X_star
        self.E = (1 - self.p.eta)   * self.E + self.p.eta   * E_star

    # V9: leader payoff
    def leader_period_payoff_with_components(self, tau: float, rho: float = 0.0,
                                             dL: float = 0.0):
        """Leader per-period payoff decomposition.
        Components (v1/v2 unchanged):
          rev           = tau * M
          cons_loss     = 0.5 * tau * (M0 - M)
          action_cost_L = leader_cost_w * tau^2
          export_loss   = psi_E * max(0, E0 - E)
        V7:
          diplo_cost    = gamma_diplo * (rho/rho_max) * (tau*M)
        V9 new (replaces V8 export control costs):
        V9: inflation_cost_L = phi_inflation_L*dL*M0, action_cost_dL = c_dL*dL^2
        Total: L = rev - cons_loss - action_cost_L - export_loss
                     - diplo_cost - inflation_cost_L - action_cost_dL
        """
        rev              = tau * self.M
        cons_loss        = 0.5 * tau * (self.p.M0 - self.M)
        action_cost_L    = self.p.leader_cost_w * (tau ** 2)
        export_loss      = self.p.psi_E * max(0.0, self.p.E0 - self.E)
        _rho_frac_L      = rho / self.p.rho_max if self.p.rho_max > 0 else 0.0
        diplo_cost       = self.p.gamma_diplo * _rho_frac_L * (tau * self.M)
        inflation_cost_L = getattr(self.p, "phi_inflation_L", 0.0) * dL * self.p.M0
        action_cost_dL   = getattr(self.p, "leader_cost_dL", 0.0) * (dL ** 2)
        L = (rev - cons_loss - action_cost_L - export_loss
             - diplo_cost - inflation_cost_L - action_cost_dL)
        return L, rev, cons_loss, action_cost_L, export_loss, diplo_cost, \
               inflation_cost_L, action_cost_dL

    # V9: follower payoff (unchanged from v7 — no x, no disruption_cost)
    def follower_period_payoff_with_components(self, d: float, rho: float = 0.0,
                                               tau: float = 0.0):
        """Follower per-period payoff decomposition.
        Components (unchanged from v1):
          export_gain  = X - X0
          infl_cost    = phi_inflation * d * M0   [linear M-anchored form, kept as v1]
          action_cost_F= follower_cost_w * d^2
        New in V2:
          rho_cost     = follower_cost_rho * rho^2   [quadratic WTO/diplomatic friction]
          inequity_term = alpha_ineq * max(0, uL_lag - uF_base)  [V7: Fehr-Schmidt]
            uL_lag = rolling mean of leader payoff over uL_lag_window periods (default 1)
            uF_base = export_gain - infl_cost - action_cost_F - rho_cost (pre-spite)
          spite_term     = beta_spite * tau * max(0, E0-E)          [V10-V5 aligned]
        Total: F = uF_base - inequity_term + spite_term
        V9: disruption_cost removed — no export controls.
        """
        export_gain   = (self.X - self.p.X0)
        infl_cost     = self.p.phi_inflation * d * self.p.M0
        action_cost_F = self.p.follower_cost_w * (d ** 2)
        #rho_cost      = self.p.follower_cost_rho * (rho ** 2) * self.p.E0  # V10: E0-anchored symmetric with phi_L*dL*M0
        rho_cost      = self.p.follower_cost_rho * (rho ** 2)  # V10: E0-anchored symmetric with phi_L*dL*M0
        # V7: Fehr-Schmidt inequity aversion + τ-triggered retaliation spite
        # inequity_term: follower suffers when rolling-mean leader welfare > follower base
        # Uses uL_lag (rolling mean of past W periods) — smoother than one-step lag
        # uF_base computed pre-spite to avoid circularity
        _uF_base      = export_gain - infl_cost - action_cost_F - rho_cost
        inequity_term = self.p.alpha_ineq * max(0.0, self.uL_lag - _uF_base)
        # spite_term: β * tau * max(0, E0-E) — aligned with DRQN V5 formulation
        spite_term    = self.p.beta_spite * tau * max(0.0, self.p.E0 - self.E)
        # V9: no disruption_cost
        F = _uF_base - inequity_term + spite_term
        return F, export_gain, infl_cost, action_cost_F, rho_cost, spite_term

    # V9: step
    def step(self, tau: float, d: float, rho: float = 0.0, dL: float = 0.0):
        """V9: dL replaces x. dL=0 recovers v7 behaviour exactly."""
        L, rev, cons_loss, action_cost_L, export_loss, diplo_cost, \
            inflation_cost_L, action_cost_dL = \
            self.leader_period_payoff_with_components(tau, rho, dL)      # V9: dL
        # V7 timing: F reads self.uL_lag from PREVIOUS period.
        F, export_gain, infl_cost, action_cost_F, rho_cost, spite_term = \
            self.follower_period_payoff_with_components(d, rho, tau)     # V9: no x
        self._update_uL_lag(L)
        self._transition(tau, d, rho, dL)                                # V9: dL
        _uF_base_log  = export_gain - infl_cost - action_cost_F - rho_cost
        inequity_term = self.p.alpha_ineq * max(
            0.0, self._prev_uL_lag - _uF_base_log)
        diag = {
            "M": self.M, "X": self.X,
            "E": self.E,
            "rev": rev, "cons_loss": cons_loss, "action_cost_L": action_cost_L,
            "export_loss":      export_loss,
            "inflation_cost_L": inflation_cost_L,  # V9
            "action_cost_dL":   action_cost_dL,    # V9
            "export_gain":    export_gain, "infl_cost": infl_cost,
            "action_cost_F":  action_cost_F,
            "rho_cost":       rho_cost,
            "spite_term":     spite_term,
            "diplo_cost":     diplo_cost,
            "inequity_term":  inequity_term,
            "uL_lag":         self._prev_uL_lag,
            "uL_current":     self.uL_lag,
        }
        return L, F, diag

    # V9: evaluate_follower_payoff
    def evaluate_follower_payoff(self, tau: float, d: float, rho: float = 0.0,
                                 timing: str = "post") -> float:
        """V9: snapshot evaluation for BR. x removed (no export controls)."""
        M_bak, X_bak, E_bak = self.M, self.X, self.E
        prev_bak = getattr(self, "prev_X", self.X)
        try:
            if timing == "pre":
                F, *_ = self.follower_period_payoff_with_components(d, rho, tau)
                return F
            elif timing == "post":
                F, *_ = self.follower_period_payoff_with_components(d, rho, tau)
                self._transition(tau, d, rho, 0.0)   # V10: tau passed through _transition→_X_star
                return F
            else:
                raise ValueError("timing must be 'pre' or 'post'")
        finally:
            self.M, self.X, self.E = M_bak, X_bak, E_bak
            self.prev_X = prev_bak


# ============================================================
# Utility: uniform bins (unchanged from v1)
# ============================================================
def make_bins(n: int, v_min: float, v_max: float) -> List[float]:
    if n <= 1:
        return [v_min]
    step = (v_max - v_min) / (n - 1)
    return [v_min + i * step for i in range(n)]


# ============================================================
# Phase 2A: BestResponseFollowerEconomic
# V2 changes: joint (d, rho) grid search; returns Tuple[float, float]
# ============================================================
class BestResponseFollowerEconomic:
    # V2: added rho_bins parameter; fallback [] means no retaliation (rho=0 only)
    def __init__(self, d_bins: List[float], rho_bins: Optional[List[float]] = None):
        self.d_bins   = d_bins
        self.rho_bins = rho_bins if rho_bins is not None else [0.0]  # V2
        self.last_action     = 0.0
        self.last_rho_action = 0.0                                   # V2

    # V2: returns (best_d, best_rho); env.evaluate_follower_payoff signature updated
    def respond(self, tau: float, env: "EconomicEnvironment") -> Tuple[float, float]:
        """Joint best-response over (d, rho) grid.
        Grid size: |d_bins| x |rho_bins| — kept at 5x5=25 for tractability.
        """
        best_d, best_rho = self.d_bins[0], self.rho_bins[0]
        best_val = float("-inf")
        for d in self.d_bins:
            for rho in self.rho_bins:                                # V2 inner loop over rho
                F = env.evaluate_follower_payoff(tau, d, rho, timing="post")
                if F > best_val:
                    best_val = F
                    best_d, best_rho = d, rho
        self.last_action     = best_d
        self.last_rho_action = best_rho                              # V2
        return best_d, best_rho                                      # V2 returns tuple


# ============================================================
# Phase 3A: Q3BinnedLeader
# V2 changes: rho_last added to 7-tuple state;
#             decide_tariff/update accept follower rho
# ============================================================

# ============================================================
# FIXED LEADER — for BR diagnostic probe
# ============================================================
class FixedLeader:
    """Deterministic leader that always plays (tau, dL).
    Used in BR diagnostic probe — not part of main training run.
    """
    def __init__(self, tau: float, dL: float = 0.0):
        self.tau         = tau
        self.dL          = dL
        self.tau_last    = f"{tau:.2f}"
        self.dL_last     = f"{dL:.2f}"
        self.history_f   = ["0.00"] * 3
        self.rho_last    = "0.00"
        self.total_payoff = 0.0

    def decide_tariff(self, follower_last_d=None, follower_last_rho=None):
        return (self.tau, self.dL)

    def update(self, *args, **kwargs):
        pass  # Fixed policy — no Q-updates


class Q3BinnedLeader:
    # V9: joint action (tau, dL); state 8-tuple with dL_last
    def __init__(self, tau_bins: List[float], epsilon=0.15, gamma=0.95, alpha=0.2,
                 q_init: float = 0.0, dL_bins: Optional[List[float]] = None,
                 epsilon_end: float = 0.002, alpha_min: float = 0.005):
        self.tau_bins    = tau_bins
        self.dL_bins     = dL_bins  # None or list of dL values
        self.epsilon     = epsilon
        self.epsilon_end = epsilon_end  # V10: configurable floor
        self.alpha_min   = alpha_min    # V10: configurable floor (was 0.02)
        self.gamma       = gamma
        self.alpha       = alpha
        self.history_f = [f"{0.00:.2f}"] * 3
        self.tau_last  = f"{self.tau_bins[0]:.2f}"
        self.rho_last  = "0.00"
        self.dL_last   = "0.00"   # V9: last leader depreciation level (string, for state)
        self.q_init    = q_init

        # V9: joint_actions = [(tau, dL)] if dL_bins provided, else [tau] (1D)
        if self.dL_bins is not None and len(self.dL_bins) > 0:
            self.joint_actions_L: List = [(tau, dL) for tau in self.tau_bins for dL in self.dL_bins]
        else:
            self.joint_actions_L: List = list(self.tau_bins)  # 1D fallback

        # Q-table keyed by (state, action) — action is float (v7) or (tau,x) tuple (v8)
        self.q: Dict[Tuple, float] = {}
        self.last_state:  Optional[Tuple] = None
        self.last_action = None   # float (v7) or (tau, x) tuple (v8)
        self.total_payoff = 0.0
        self.env = None

        self.state_visits  = Counter()
        self.greedy_flags: List[bool]  = []
        self.epsilon_trace: List[float] = []
        self.qgap_trace:   List[float]  = []
        self.q_max_trace:  List[float]  = []
        self.q_avg_trace:  List[float]  = []
        self.state_trace:  List[Tuple]  = []
        self.sa_visits:    Dict[Tuple, int] = {}

    @staticmethod
    def _coarse_bin(val: float, base: float) -> str:
        ratio = 0.0 if base == 0 else val / base
        if ratio < 0.8: return "L"
        if ratio < 1.2: return "M"
        return "H"

    def set_env(self, env) -> None:
        self.env = env

    # V2: 7-tuple state; V8: 8-tuple when export controls active
    def _state(self) -> Tuple:
        if self.env is None:
            Mbin, Xbin = "?", "?"
        else:
            Mbin = self._coarse_bin(self.env.M, self.env.p.M0)
            Xbin = self._coarse_bin(self.env.X, self.env.p.X0)
        base = tuple(self.history_f) + (self.tau_last, Mbin, Xbin, self.rho_last)
        # V9: append dL_last if depreciation active
        if self.dL_bins is not None:
            return base + (self.dL_last,)
        return base

    def _q_vals_for(self, s: Tuple) -> Dict:
        """Return {action: Q-value} for all actions. Action is float (v7) or (tau,x) tuple (v8)."""
        default = self.q_init
        return {a: self.q.get((s, a), default) for a in self.joint_actions_L}

    # V2/V8: decide_tariff returns float (v7) or (tau, x) tuple (v8)
    def decide_tariff(self, follower_last_d: Optional[float] = None,
                      follower_last_rho: Optional[float] = None):
        """Leader chooses action: tau (v7) or (tau, x) (v8 with export controls).
        Returns float in v7 mode, (float, float) tuple in v8 mode.
        """
        if follower_last_d is not None:
            self.history_f.pop(0)
            self.history_f.append(f"{follower_last_d:.2f}")
        if follower_last_rho is not None:
            self.rho_last = f"{follower_last_rho:.2f}"

        s = self._state()
        self.state_visits[s] += 1
        vals = self._q_vals_for(s)

        if vals:
            max_q_snapshot = max(vals.values())
            avg_q_snapshot = sum(vals.values()) / len(vals)
        else:
            max_q_snapshot = avg_q_snapshot = 0.0
        self.state_trace.append(s)
        self.q_max_trace.append(max_q_snapshot)
        self.q_avg_trace.append(avg_q_snapshot)

        # ε-greedy over joint_actions_L
        if vals:
            max_q = max(vals.values())
            argmax_actions = [a for a, v in vals.items() if v == max_q]
        else:
            argmax_actions = self.joint_actions_L[:]

        explore = (random.random() < self.epsilon)
        a = random.choice(self.joint_actions_L) if explore else random.choice(argmax_actions)

        self.epsilon_trace.append(self.epsilon)
        self.greedy_flags.append(a in argmax_actions)
        if len(vals) >= 2:
            sorted_q = sorted(vals.values(), reverse=True)
            self.qgap_trace.append(sorted_q[0] - sorted_q[1])
        else:
            self.qgap_trace.append(0.0)

        # V9: unpack action — tuple (tau,dL) or float tau
        if isinstance(a, tuple):
            tau_chosen, dL_chosen = a
            self.tau_last = f"{tau_chosen:.2f}"
            self.dL_last  = f"{dL_chosen:.2f}"  # V9
        else:
            tau_chosen = a
            self.tau_last = f"{tau_chosen:.2f}"

        self.epsilon     = max(self.epsilon_end, self.epsilon * 0.995)
        self.last_state  = s
        self.last_action = a
        return a  # float or (tau, x) tuple — orchestrator unpacks

    # V2/V8: update() supports both 1D and 2D action spaces
    def update(self, follower_new_d: float, reward: float,
               follower_new_rho: float = 0.0):
        """Q-learning update. Works for both v7 (tau only) and v8 (tau, x) action spaces.
        Skipped when frozen (epsilon=0 and epsilon_end=0) — policy is committed.
        """
        # Gate is now external (in StackelbergTariffGameEconomic.step)
        # No internal skip needed — update() always runs when called
        s = self.last_state
        a = self.last_action

        self.rho_last = f"{follower_new_rho:.2f}"
        self.history_f.pop(0)
        self.history_f.append(f"{follower_new_d:.2f}")
        s_next = self._state()

        old_q    = self.q.get((s, a), self.q_init)
        max_next = max(self.q.get((s_next, ap), self.q_init) for ap in self.joint_actions_L)

        key = (s, a)
        self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
        alpha = max(self.alpha_min, 1.0 / (self.sa_visits[key] ** 0.5))

        new_q = old_q + alpha * (reward + self.gamma * max_next - old_q)
        self.q[(s, a)] = new_q
        self.total_payoff += reward


# ============================================================
# Phase 2B: Q3BinnedFollower
# V2 changes:
#   - rho_bins parameter; joint_actions = [(d,r) for d in d_bins for r in rho_bins]
#   V9 STATE: 8-tuple (tau3, tau2, tau1, Xbin, d_last, rho_last, dL_bin, X_momentum)
#   dL_bin: coarse bin of leader depreciation {"LOW","MED","HIGH"}
#   respond(leader_tau, leader_dL) — dL binned and stored in state
#   Q-table keyed by (8-tuple-state, (d, rho)) — joint action tuple
# ============================================================
class Q3BinnedFollower:
    # V2: rho_bins added; builds joint_actions list
    def __init__(self, d_bins: List[float], rho_bins: Optional[List[float]] = None,
                 epsilon=0.15, gamma=0.95, alpha=0.2,
                 double_q: bool = False, q_init_f: float = 0.0,
                 epsilon_end: float = 0.002, alpha_min: float = 0.010):
        self.d_bins      = d_bins
        # V2: rho_bins defaults to [0.0] (no retaliation) for backward compat
        self.rho_bins    = rho_bins if rho_bins is not None else [0.0]
        self.epsilon     = epsilon
        self.epsilon_end = epsilon_end
        self.alpha_min   = alpha_min    # V10: configurable floor (was 0.05)
        self.gamma       = gamma
        self.alpha       = alpha
        self.q_init_f    = q_init_f
        self.history_l = [f"{0.00:.2f}"] * 3   # last 3 leader tau-actions
        self.last_state:  Optional[Tuple] = None
        self.last_action: Optional[Tuple] = None  # V2: now a (d, rho) tuple
        self.total_payoff = 0.0
        self.last_d_str   = f"{self.d_bins[0]:.2f}"
        self.last_rho_str = "0.00"                 # V2: track last rho for state
        self.last_dL_str  = "LOW"                  # V9: coarse-binned leader dL for follower state

        # V2: enumerate all joint actions (d, rho); 5×5=25 with recommended bin counts
        self.joint_actions: List[Tuple[float, float]] = [
            (d, rho) for d in self.d_bins for rho in self.rho_bins
        ]

        self.double_q = double_q
        if self.double_q:
            self.qA: Dict[Tuple, float] = {}
            self.qB: Dict[Tuple, float] = {}
            self.q:  Dict[Tuple, float] = {}   # public average for diagnostics
        else:
            self.q: Dict[Tuple, float] = {}

        # Diagnostics (unchanged structure)
        self.state_visits  = Counter()
        self.greedy_flags: List[bool]  = []
        self.epsilon_trace: List[float] = []
        self.qgap_trace:   List[float]  = []
        self.q_max_trace:  List[float]  = []
        self.q_avg_trace:  List[float]  = []
        self.state_trace:  List[Tuple]  = []
        self.sa_visits:    Dict[Tuple, int] = {}
        self.env = None

    @staticmethod
    def _coarse_bin(val: float, base: float) -> str:
        # V2: slightly wider M band consistent with v1 follower
        ratio = 0.0 if base == 0 else val / base
        if ratio < 0.90: return "L"
        if ratio < 1.20: return "M"
        return "H"

    def set_env(self, env) -> None:
        self.env = env

    def _x_momentum(self) -> str:
        if self.env is None or not hasattr(self.env, "prev_X"):
            return "UNK"
        dx = self.env.X - self.env.prev_X
        if dx >  1e-6: return "UP"
        if dx < -1e-6: return "DN"
        return "FLAT"

    # V9:  state is 8-tuple (tau3, tau2, tau1, Xbin, d_last, rho_last, dL_bin, X_momentum)
    #      when leader depreciation is active (dL_max > 0).
    # V10: revert to V7 7-tuple when dL_max=0 (leader_depr=False).
    #      Constant "LOW" dL_bin inflated state space ~3x with zero information,
    #      preventing V7-quality convergence for Fig 8/9 structural PDs.
    def _state(self) -> Tuple:
        if self.env is None:
            Xbin, mom = "?", "UNK"
        else:
            Xbin = self._coarse_bin(self.env.X, self.env.p.X0)
            mom  = self._x_momentum()
        # Include dL_bin only when leader depreciation is active
        if self.env is not None and self.env.p.dL_max > 0:
            return tuple(self.history_l) + (Xbin, self.last_d_str,
                                            self.last_rho_str, self.last_dL_str, mom)
        # V7-equivalent 7-tuple when dL_max=0
        return tuple(self.history_l) + (Xbin, self.last_d_str, self.last_rho_str, mom)

    def _q_val(self, s, a: Tuple[float, float]) -> float:
        if not self.double_q:
            return self.q.get((s, a), self.q_init_f)
        return 0.5 * (self.qA.get((s, a), self.q_init_f) +
                      self.qB.get((s, a), self.q_init_f))

    def _q_vals_for(self, s) -> Dict[Tuple[float, float], float]:
        return {a: self._q_val(s, a) for a in self.joint_actions}   # V2: over joint actions

    # V9: respond() accepts leader_dL so follower state includes dL bin
    def respond(self, leader_tau: float, leader_dL: float = 0.0) -> Tuple[float, float]:
        """Follower ε-greedy over joint action space (d, rho).
        leader_dL: current leader depreciation (V9). Binned and stored in state.
        Returns (chosen_d, chosen_rho).
        """
        # V9: coarse-bin leader dL for state tuple
        if   leader_dL < 0.05: self.last_dL_str = "LOW"
        elif leader_dL < 0.10: self.last_dL_str = "MED"
        else:                  self.last_dL_str = "HIGH"
        self.history_l.pop(0)
        self.history_l.append(f"{leader_tau:.2f}")
        s = self._state()
        self.state_visits[s] += 1

        vals = self._q_vals_for(s)

        if vals:
            max_q_snapshot = max(vals.values())
            avg_q_snapshot = sum(vals.values()) / len(vals)
        else:
            max_q_snapshot = avg_q_snapshot = 0.0
        self.state_trace.append(s)
        self.q_max_trace.append(max_q_snapshot)
        self.q_avg_trace.append(avg_q_snapshot)

        # ε-greedy over joint actions
        if vals:
            max_q = max(vals.values())
            argmax_actions = [a for a, v in vals.items() if v == max_q]
        else:
            argmax_actions = self.joint_actions[:]

        explore = (random.random() < self.epsilon)
        a = random.choice(self.joint_actions) if explore else random.choice(argmax_actions)  # V2: joint

        self.epsilon_trace.append(self.epsilon)
        self.greedy_flags.append(a in argmax_actions)
        if len(vals) >= 2:
            sorted_q = sorted(vals.values(), reverse=True)
            self.qgap_trace.append(sorted_q[0] - sorted_q[1])
        else:
            self.qgap_trace.append(0.0)

        # V2: unpack joint action; track both last_d_str and last_rho_str for state
        chosen_d, chosen_rho = a
        self.last_d_str   = f"{chosen_d:.2f}"
        self.last_rho_str = f"{chosen_rho:.2f}"                # V2
        self.epsilon      = max(self.epsilon_end, self.epsilon * 0.995)
        self.last_state   = s
        self.last_action  = a   # V2: now a (d, rho) tuple
        return chosen_d, chosen_rho                             # V2: returns tuple

    # V2: update handles joint action tuple; s_next includes rho dimension
    def update(self, leader_next_tau: float, reward: float):
        """Deferred Q-update called by orchestrator one step later.
        leader_next_tau is tau_t (the next leader action) needed to complete s'.
        Joint action a = (d, rho) tuple — Q-table keys use this directly.
        """
        s = self.last_state
        a = self.last_action   # V2: (d, rho) tuple

        if self.env is None:
            Xbin, mom = "?", "UNK"
        else:
            Xbin = self._coarse_bin(self.env.X, self.env.p.X0)
            mom  = self._x_momentum()

        # V9: s_next is 8-tuple matching _state() structure (includes dL_last)
        # V10: s_next must match _state() structure — 7-tuple when dL_max=0, 8-tuple when dL active
        if self.env is not None and self.env.p.dL_max > 0:
            s_next = (self.history_l[1], self.history_l[2], f"{leader_next_tau:.2f}",
                      Xbin, self.last_d_str, self.last_rho_str, self.last_dL_str, mom)
        else:
            s_next = (self.history_l[1], self.history_l[2], f"{leader_next_tau:.2f}",
                      Xbin, self.last_d_str, self.last_rho_str, mom)

        if not self.double_q:
            old_q    = self.q.get((s, a), self.q_init_f)
            max_next = max(self.q.get((s_next, ap), self.q_init_f) for ap in self.joint_actions)

            key = (s, a)
            self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
            alpha = max(self.alpha_min, 1.0 / (self.sa_visits[key] ** 0.5))

            new_q = old_q + alpha * (reward + self.gamma * max_next - old_q)
            self.q[(s, a)] = new_q
        else:
            # Double Q-learning (unchanged logic, V2: uses joint action tuples)
            if random.random() < 0.5:
                argmax_a = max(self.joint_actions,
                               key=lambda ap: self.qA.get((s_next, ap), self.q_init_f))
                target   = reward + self.gamma * self.qB.get((s_next, argmax_a), self.q_init_f)
                old_q    = self.qA.get((s, a), self.q_init_f)
                key = (s, a)
                self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
                alpha = max(self.alpha_min, 1.0 / (self.sa_visits[key] ** 0.5))
                self.qA[(s, a)] = old_q + alpha * (target - old_q)
            else:
                argmax_a = max(self.joint_actions,
                               key=lambda ap: self.qB.get((s_next, ap), self.q_init_f))
                target   = reward + self.gamma * self.qA.get((s_next, argmax_a), self.q_init_f)
                old_q    = self.qB.get((s, a), self.q_init_f)
                key = (s, a)
                self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
                alpha = max(self.alpha_min, 1.0 / (self.sa_visits[key] ** 0.5))
                self.qB[(s, a)] = old_q + alpha * (target - old_q)

            # Keep public average in sync (diagnostic / heatmap)
            self.q[(s, a)] = 0.5 * (self.qA.get((s, a), self.q_init_f) +
                                    self.qB.get((s, a), self.q_init_f))

        self.total_payoff += reward


# ============================================================
# Phase 3B: StackelbergTariffGameEconomic (Orchestrator)
# V2 changes:
#   - step() unpacks (d_t, rho_t) from follower.respond()
#   - Passes rho_t to env.step() and leader.update()
#   - Handles deferred follower update with rho propagation
#   - Logs 'rho', 'E', 'export_loss', 'rho_cost', 'spite_term'
# ============================================================
class StackelbergTariffGameEconomic:
    def __init__(self, env: EconomicEnvironment, leader, follower, track: bool = True,
                 freeze_follower_frac: float = 0.0,
                 freeze_leader_frac: float = 0.0,
                 freeze_mode: str = "epsilon_only"):
        """
        freeze_mode:
          none         — simultaneous learning (baseline, shows co-adaptation)
          epsilon_only — ε→0 (stop exploration), Q-updates continue [DEFAULT]
          full_freeze  — ε→0 AND skip Q-updates (committed policy)
freeze_follower_frac: fraction of rounds after which follower learning freezes.
            follower epsilon -> 0, Q-updates suppressed.
            Use to stabilise rho instability in FF regime.
        freeze_leader_frac: fraction of rounds after which leader learning freezes.
            leader epsilon -> 0, Q-updates suppressed.
            Use when leader tau has not converged and keeps perturbing follower.
            The later of freeze_leader_frac and freeze_follower_frac determines
            when both agents are fully frozen (eval-only tail).
        Both default to 0.0 (never freeze) — fully backward compatible.
        """
        self.env      = env
        self.leader   = leader
        self.follower = follower
        if hasattr(self.leader, "set_env"):
            self.leader.set_env(self.env)
        if hasattr(self.follower, "set_env"):
            self.follower.set_env(self.env)
        self.track   = track
        self.results = {"rounds": []} if track else None
        self.t       = 0
        self._pending_f_reward = None
        self._pending_rho      = 0.0    # V2: stash rho for deferred follower update logging
        self.env.prev_X        = self.env.X
        # V8 freeze
        self.freeze_follower_frac = freeze_follower_frac
        self._follower_frozen     = False
        self._freeze_follower_at  = 0   # set in run()
        self.freeze_leader_frac   = freeze_leader_frac
        self.freeze_mode          = freeze_mode
        self._leader_frozen       = False
        self._freeze_leader_at    = 0   # set in run()

    def run(self, rounds: int = 300):
        self._freeze_follower_at = int(rounds * self.freeze_follower_frac) \
                                   if self.freeze_follower_frac > 0.0 else rounds + 1
        self._freeze_leader_at   = int(rounds * self.freeze_leader_frac) \
                                   if self.freeze_leader_frac   > 0.0 else rounds + 1
        for _ in range(rounds):
            self.step()

    def step(self):
        self.t += 1

        # V8 freeze: suppress follower learning
        if (not self._follower_frozen) and (self.t >= self._freeze_follower_at):
            self._follower_frozen = True
            if hasattr(self.follower, "epsilon"):
                self.follower.epsilon = 0.0
            print(f"  [freeze] Follower frozen at round {self.t} "
                  f"(frac={self.freeze_follower_frac:.2f})", flush=True)
        # V10 Stackelberg freeze
        # freeze_mode="none" → complete no-op, _leader_frozen stays False
        mode = getattr(self, "freeze_mode", "epsilon_only")
        if (mode != "none")\
                and (not self._leader_frozen)\
                and (self.t >= self._freeze_leader_at):
            self._leader_frozen = True
            if mode in ("epsilon_only", "full_freeze"):
                if hasattr(self.leader, "epsilon"):
                    self.leader.epsilon     = 0.0
                    self.leader.epsilon_end = 0.0
                self.leader._freeze_mode = mode
            print(f"  [freeze] Leader frozen [{mode}] at round {self.t} "
                  f"(frac={self.freeze_leader_frac:.2f})", flush=True)

        # Read last follower action (d, rho)
        follower_last_d   = getattr(self.follower, "last_action", None)
        follower_last_rho = getattr(self.follower, "last_rho_action", None)
        if isinstance(follower_last_d, tuple):
            follower_last_d, follower_last_rho = follower_last_d

        # Leader moves — returns float (v7) or (tau, x) tuple (v8)
        leader_action = self.leader.decide_tariff(
            follower_last_d=follower_last_d,
            follower_last_rho=follower_last_rho,
        )
        # V9: unpack leader action — (tau, dL) or float
        if isinstance(leader_action, tuple):
            tau_t, dL_t = leader_action
        else:
            tau_t, dL_t = leader_action, 0.0

        # Complete follower's deferred update using tau_t as "next tau"
        if isinstance(self.follower, Q3BinnedFollower) and self._pending_f_reward is not None:
            self.follower.update(leader_next_tau=tau_t, reward=self._pending_f_reward)
            self._pending_f_reward = None

        # Follower responds — always returns (d_t, rho_t)
        # V9: pass dL_t so follower state includes leader depreciation bin
        if isinstance(self.follower, BestResponseFollowerEconomic):
            d_t, rho_t = self.follower.respond(tau_t, self.env)
        else:
            d_t, rho_t = self.follower.respond(tau_t, dL_t)  # V9: dL_t passed

        # Environment step — V9: passes dL_t
        prev_X_for_mom = self.env.X
        L_pay, F_pay, diag = self.env.step(tau_t, d_t, rho_t, dL_t)       # V9
        self.env.prev_X = prev_X_for_mom

        # Leader Q-update (suppressed when leader is frozen)
        # Leader Q-update gate:
        #   none / epsilon_only → always update (exploration off but learning on)
        #   full_freeze         → skip update (committed policy)
        leader_updates_allowed = (
            isinstance(self.leader, Q3BinnedLeader)
            and not (
                self._leader_frozen
                and self.freeze_mode == "full_freeze"
            )
        )
        if leader_updates_allowed:
            self.leader.update(
                follower_new_d=d_t,
                follower_new_rho=rho_t,
                reward=L_pay,
            )

        # Store follower reward only if not frozen (frozen = no Q-updates)
        if isinstance(self.follower, Q3BinnedFollower) and not self._follower_frozen:
            self._pending_f_reward = F_pay
        elif self._follower_frozen:
            self._pending_f_reward = None   # discard — no update will be applied

        # Logging — V9: dL replaces x
        if self.track:
            self.results["rounds"].append({
                "round":         self.t,
                "tau":           tau_t,
                "dL":            dL_t,             # V9: leader depreciation level
                "d":             d_t,
                "rho":           rho_t,
                "M":             diag.get("M"),
                "X":             diag.get("X"),
                "E":             diag.get("E"),
                "leader_pay":    L_pay,
                "follower_pay":  F_pay,
                "rev":           diag.get("rev"),
                "cons_loss":     diag.get("cons_loss"),
                "action_cost_L": diag.get("action_cost_L"),
                "inflation_cost_L": diag.get("inflation_cost_L", 0.0),  # V9
                "action_cost_dL":   diag.get("action_cost_dL",   0.0),  # V9
                "export_loss":   diag.get("export_loss"),
                "export_gain":   diag.get("export_gain"),
                "infl_cost":     diag.get("infl_cost"),
                "action_cost_F": diag.get("action_cost_F"),
                "rho_cost":      diag.get("rho_cost"),
                "spite_term":      diag.get("spite_term",      0.0),
                "diplo_cost":      diag.get("diplo_cost",      0.0),
                "inequity_term": diag.get("inequity_term", 0.0),
                "uL_lag":        diag.get("uL_lag",        0.0),
                "uL_current":    diag.get("uL_current",    0.0),
            })
