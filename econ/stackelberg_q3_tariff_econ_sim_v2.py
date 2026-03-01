# stackelberg_q3_tariff_econ_sim_v2.py
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
#   + Optional spite term: beta_E * (E0 - E) in follower payoff (default beta_E=0.0)
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
#   + Logs 'rho', 'E', 'export_loss', 'rho_cost' in results
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
from collections import Counter

SIM_VERSION = "econ-sim v3-v2 — retaliatory tariff channel + E flow + joint follower action"

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
    beta_E: float = 0.0          # V2: spite term — direct follower utility from (E0-E); default=0 (instrumental only)


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

    def _M_star(self, tau: float) -> float:
        # Import demand target under tariff (unchanged from v1)
        if getattr(self.p, "trade_form", "linear") == "power":
            return max(0.0, self.p.M0 * (1.0 + tau) ** (-self.p.demand_elast))
        else:
            return max(0.0, self.p.M0 * (1.0 - self.p.demand_elast * tau))

    def _X_star(self, d: float) -> float:
        # Export target under depreciation (unchanged from v1)
        if getattr(self.p, "trade_form", "linear") == "power":
            return self.p.X0 * (1.0 + d) ** (self.p.supply_elast)
        else:
            return self.p.X0 * (1.0 + self.p.supply_elast * d)

    # V2: new method — leader export target suppressed by rho (power-form only, per framework doc)
    def _E_star(self, rho: float) -> float:
        """Leader export target under follower retaliatory tariff.
        E* = E0 * (1 + rho)^(-export_elast)
        Mirrors _M_star: higher rho suppresses E below E0.
        Always power-form regardless of trade_form setting.
        """
        return max(0.0, self.p.E0 * (1.0 + rho) ** (-self.p.export_elast))

    # V2: _transition now takes rho and mean-reverts E
    def _transition(self, tau: float, d: float, rho: float = 0.0):
        """Update all three trade flows one period forward.
        M and X: unchanged from v1.
        E: V2 addition — partial adjustment toward E_star(rho).
        """
        M_star = self._M_star(tau)
        X_star = self._X_star(d)
        E_star = self._E_star(rho)                                          # V2
        self.M = (1 - self.p.kappa) * self.M + self.p.kappa * M_star
        self.X = (1 - self.p.lam)   * self.X + self.p.lam   * X_star
        self.E = (1 - self.p.eta)   * self.E + self.p.eta   * E_star       # V2

    # V2: leader payoff now includes export_loss as 5th return value
    def leader_period_payoff_with_components(self, tau: float):
        """Leader per-period payoff decomposition.
        Components (unchanged from v1):
          rev       = tau * M
          cons_loss = 0.5 * tau * (M0 - M)      [triangle DWL approximation, kept as v1]
          action_cost_L = leader_cost_w * tau^2
        New in V2:
          export_loss = psi_E * max(0, E0 - E)   [penalty when retaliation has suppressed E]
        Total: L = rev - cons_loss - action_cost_L - export_loss
        """
        rev           = tau * self.M
        cons_loss     = 0.5 * tau * (self.p.M0 - self.M)
        action_cost_L = self.p.leader_cost_w * (tau ** 2)
        export_loss   = self.p.psi_E * max(0.0, self.p.E0 - self.E)        # V2
        L = rev - cons_loss - action_cost_L - export_loss                   # V2 updated
        return L, rev, cons_loss, action_cost_L, export_loss

    # V2: follower payoff takes rho, computes rho_cost and optional spite term
    def follower_period_payoff_with_components(self, d: float, rho: float = 0.0):
        """Follower per-period payoff decomposition.
        Components (unchanged from v1):
          export_gain  = X - X0
          infl_cost    = phi_inflation * d * M0   [linear M-anchored form, kept as v1]
          action_cost_F= follower_cost_w * d^2
        New in V2:
          rho_cost     = follower_cost_rho * rho^2   [quadratic WTO/diplomatic friction]
          spite_term   = beta_E * (E0 - E)            [optional; default beta_E=0.0]
        Total: F = export_gain - infl_cost - action_cost_F - rho_cost + spite_term
        """
        export_gain   = (self.X - self.p.X0)
        infl_cost     = self.p.phi_inflation * d * self.p.M0
        action_cost_F = self.p.follower_cost_w * (d ** 2)
        rho_cost      = self.p.follower_cost_rho * (rho ** 2)              # V2
        spite_term    = self.p.beta_E * max(0.0, self.p.E0 - self.E)      # V2 (0 by default)
        F = export_gain - infl_cost - action_cost_F - rho_cost + spite_term  # V2 updated
        return F, export_gain, infl_cost, action_cost_F, rho_cost

    # V2: step now takes rho and propagates to transition and payoffs
    def step(self, tau: float, d: float, rho: float = 0.0):
        L, rev, cons_loss, action_cost_L, export_loss = \
            self.leader_period_payoff_with_components(tau)
        F, export_gain, infl_cost, action_cost_F, rho_cost = \
            self.follower_period_payoff_with_components(d, rho)
        self._transition(tau, d, rho)                                       # V2 passes rho
        diag = {
            "M": self.M, "X": self.X,
            "E": self.E,                                                    # V2
            "rev": rev, "cons_loss": cons_loss, "action_cost_L": action_cost_L,
            "export_loss": export_loss,                                     # V2
            "export_gain": export_gain, "infl_cost": infl_cost,
            "action_cost_F": action_cost_F,
            "rho_cost": rho_cost,                                           # V2
        }
        return L, F, diag

    # V2: evaluate_follower_payoff now accepts rho for joint BR evaluation
    def evaluate_follower_payoff(self, tau: float, d: float, rho: float = 0.0,
                                 timing: str = "post") -> float:
        """Snapshot evaluation for BR curve / grid search.
        rho=0.0 default preserves backward compatibility with any legacy callers.
        """
        M_bak, X_bak, E_bak = self.M, self.X, self.E                      # V2 backs up E
        prev_bak = getattr(self, "prev_X", self.X)
        try:
            if timing == "pre":
                F, *_ = self.follower_period_payoff_with_components(d, rho)
                return F
            elif timing == "post":
                self._transition(tau, d, rho)                               # V2 passes rho
                F, *_ = self.follower_period_payoff_with_components(d, rho)
                return F
            else:
                raise ValueError("timing must be 'pre' or 'post'")
        finally:
            self.M, self.X, self.E = M_bak, X_bak, E_bak                  # V2 restores E
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
class Q3BinnedLeader:
    # V2: state is now 7-tuple (d1,d2,d3,tau_last,Mbin,Xbin,rho_last)
    def __init__(self, tau_bins: List[float], epsilon=0.15, gamma=0.95, alpha=0.2, q_init: float = 0.0):
        self.tau_bins = tau_bins
        self.epsilon  = epsilon
        self.gamma    = gamma
        self.alpha    = alpha
        self.history_f = [f"{0.00:.2f}"] * 3   # last 3 follower d-actions
        self.tau_last  = f"{self.tau_bins[0]:.2f}"
        self.rho_last  = "0.00"                  # V2: last observed follower rho (coarse string)
        self.q_init    = q_init

        # Q-table keyed by (7-tuple-state, tau_action)    V2: state width 6→7
        self.q: Dict[Tuple, float] = {}
        self.last_state:  Optional[Tuple] = None
        self.last_action: Optional[float] = None
        self.total_payoff = 0.0
        self.env = None

        # Diagnostics (unchanged)
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

    # V2: state is now a 7-tuple (appends rho_last)
    def _state(self) -> Tuple:
        if self.env is None:
            Mbin, Xbin = "?", "?"
        else:
            Mbin = self._coarse_bin(self.env.M, self.env.p.M0)
            Xbin = self._coarse_bin(self.env.X, self.env.p.X0)
        # V2: rho_last appended as 7th element
        return tuple(self.history_f) + (self.tau_last, Mbin, Xbin, self.rho_last)

    def _q_vals_for(self, s: Tuple) -> Dict[float, float]:
        default = self.q_init
        return {a: self.q.get((s, a), default) for a in self.tau_bins}

    # V2: follower_last_rho parameter added; updates rho_last before computing state
    def decide_tariff(self, follower_last_d: Optional[float] = None,
                      follower_last_rho: Optional[float] = None) -> float:
        """Leader chooses tau. Incorporates observed (d, rho) from previous follower step.
        follower_last_rho is V2 addition; None defaults to no state update for rho.
        """
        if follower_last_d is not None:
            self.history_f.pop(0)
            self.history_f.append(f"{follower_last_d:.2f}")
        # V2: update rho memory
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

        # ε-greedy
        if vals:
            max_q = max(vals.values())
            argmax_actions = [a for a, v in vals.items() if v == max_q]
        else:
            argmax_actions = self.tau_bins[:]

        explore = (random.random() < self.epsilon)
        a = random.choice(self.tau_bins) if explore else random.choice(argmax_actions)

        self.epsilon_trace.append(self.epsilon)
        self.greedy_flags.append(a in argmax_actions)
        if len(vals) >= 2:
            sorted_q = sorted(vals.values(), reverse=True)
            self.qgap_trace.append(sorted_q[0] - sorted_q[1])
        else:
            self.qgap_trace.append(0.0)

        self.tau_last  = f"{a:.2f}"
        self.epsilon   = max(0.02, self.epsilon * 0.995)
        self.last_state  = s
        self.last_action = a
        return a

    # V2: follower_new_rho parameter added; updates rho_last for s_next construction
    def update(self, follower_new_d: float, reward: float,
               follower_new_rho: float = 0.0):
        """Q-learning update. follower_new_rho is V2 addition.
        s_next incorporates observed (d_t, rho_t) from this step.
        """
        s = self.last_state
        a = self.last_action

        # V2: incorporate new rho into next state
        self.rho_last = f"{follower_new_rho:.2f}"
        self.history_f.pop(0)
        self.history_f.append(f"{follower_new_d:.2f}")
        s_next = self._state()

        old_q    = self.q.get((s, a), self.q_init)
        max_next = max(self.q.get((s_next, ap), self.q_init) for ap in self.tau_bins)

        key = (s, a)
        self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
        alpha = max(0.02, 1.0 / (self.sa_visits[key] ** 0.5))

        new_q = old_q + alpha * (reward + self.gamma * max_next - old_q)
        self.q[(s, a)] = new_q
        self.total_payoff += reward


# ============================================================
# Phase 2B: Q3BinnedFollower
# V2 changes:
#   - rho_bins parameter; joint_actions = [(d,r) for d in d_bins for r in rho_bins]
#   - State: 7-tuple (tau3, tau2, tau1, Xbin, d_last, rho_last, X_momentum)
#   - Q-table keyed by (7-tuple-state, (d, rho)) — joint action tuple
#   - respond() returns Tuple[float, float]
#   - update() handles joint action tuples; sa_visits key uses joint action
# ============================================================
class Q3BinnedFollower:
    # V2: rho_bins added; builds joint_actions list
    def __init__(self, d_bins: List[float], rho_bins: Optional[List[float]] = None,
                 epsilon=0.15, gamma=0.95, alpha=0.2,
                 double_q: bool = False, q_init_f: float = 0.0):
        self.d_bins    = d_bins
        # V2: rho_bins defaults to [0.0] (no retaliation) for backward compat
        self.rho_bins  = rho_bins if rho_bins is not None else [0.0]
        self.epsilon   = epsilon
        self.gamma     = gamma
        self.alpha     = alpha
        self.q_init_f  = q_init_f
        self.history_l = [f"{0.00:.2f}"] * 3   # last 3 leader tau-actions
        self.last_state:  Optional[Tuple] = None
        self.last_action: Optional[Tuple] = None  # V2: now a (d, rho) tuple
        self.total_payoff = 0.0
        self.last_d_str   = f"{self.d_bins[0]:.2f}"
        self.last_rho_str = "0.00"                 # V2: track last rho for state

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

    # V2: state is 7-tuple (tau3, tau2, tau1, Xbin, d_last, rho_last, X_momentum)
    def _state(self) -> Tuple:
        if self.env is None:
            Xbin, mom = "?", "UNK"
        else:
            Xbin = self._coarse_bin(self.env.X, self.env.p.X0)
            mom  = self._x_momentum()
        # V2: rho_last_str inserted between d_last and mom
        return tuple(self.history_l) + (Xbin, self.last_d_str, self.last_rho_str, mom)

    def _q_val(self, s, a: Tuple[float, float]) -> float:
        if not self.double_q:
            return self.q.get((s, a), self.q_init_f)
        return 0.5 * (self.qA.get((s, a), self.q_init_f) +
                      self.qB.get((s, a), self.q_init_f))

    def _q_vals_for(self, s) -> Dict[Tuple[float, float], float]:
        return {a: self._q_val(s, a) for a in self.joint_actions}   # V2: over joint actions

    # V2: respond() returns (d, rho) tuple; Q-table indexed by joint action
    def respond(self, leader_tau: float) -> Tuple[float, float]:
        """Follower ε-greedy over joint action space (d, rho).
        Returns (chosen_d, chosen_rho).
        """
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
        self.epsilon      = max(0.02, self.epsilon * 0.995)
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

        # V2: s_next is 7-tuple matching _state() structure
        s_next = (self.history_l[1], self.history_l[2], f"{leader_next_tau:.2f}",
                  Xbin, self.last_d_str, self.last_rho_str, mom)

        if not self.double_q:
            old_q    = self.q.get((s, a), self.q_init_f)
            max_next = max(self.q.get((s_next, ap), self.q_init_f) for ap in self.joint_actions)

            key = (s, a)
            self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
            alpha = max(0.05, 1.0 / (self.sa_visits[key] ** 0.5))

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
                alpha = max(0.05, 1.0 / (self.sa_visits[key] ** 0.5))
                self.qA[(s, a)] = old_q + alpha * (target - old_q)
            else:
                argmax_a = max(self.joint_actions,
                               key=lambda ap: self.qB.get((s_next, ap), self.q_init_f))
                target   = reward + self.gamma * self.qA.get((s_next, argmax_a), self.q_init_f)
                old_q    = self.qB.get((s, a), self.q_init_f)
                key = (s, a)
                self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
                alpha = max(0.05, 1.0 / (self.sa_visits[key] ** 0.5))
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
#   - Logs 'rho', 'E', 'export_loss', 'rho_cost'
# ============================================================
class StackelbergTariffGameEconomic:
    def __init__(self, env: EconomicEnvironment, leader, follower, track: bool = True):
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

    def run(self, rounds: int = 300):
        for _ in range(rounds):
            self.step()

    def step(self):
        self.t += 1
        # V2: read both d and rho from follower's last action
        follower_last_d   = getattr(self.follower, "last_action", None)
        follower_last_rho = getattr(self.follower, "last_rho_action", None)  # BR follower
        # For Q-follower the last_action is a (d,rho) tuple; unpack if needed
        if isinstance(follower_last_d, tuple):
            follower_last_d, follower_last_rho = follower_last_d

        # Leader moves — V2: passes observed rho as well
        tau_t = self.leader.decide_tariff(
            follower_last_d=follower_last_d,
            follower_last_rho=follower_last_rho,       # V2
        )

        # Complete follower's deferred update using tau_t as "next tau"
        if isinstance(self.follower, Q3BinnedFollower) and self._pending_f_reward is not None:
            self.follower.update(leader_next_tau=tau_t, reward=self._pending_f_reward)
            self._pending_f_reward = None

        # Follower responds — V2: always returns (d_t, rho_t)
        if isinstance(self.follower, BestResponseFollowerEconomic):
            d_t, rho_t = self.follower.respond(tau_t, self.env)             # V2 BR returns tuple
        else:
            d_t, rho_t = self.follower.respond(tau_t)                       # V2 Q-follower returns tuple

        # Environment step — V2: passes rho_t
        prev_X_for_mom = self.env.X
        L_pay, F_pay, diag = self.env.step(tau_t, d_t, rho_t)              # V2
        self.env.prev_X = prev_X_for_mom

        # Leader Q-update — V2: passes follower_new_rho
        if isinstance(self.leader, Q3BinnedLeader):
            self.leader.update(
                follower_new_d=d_t,
                follower_new_rho=rho_t,                                     # V2
                reward=L_pay,
            )

        # Defer follower update (unchanged timing logic; rho already stored in follower state)
        if isinstance(self.follower, Q3BinnedFollower):
            self._pending_f_reward = F_pay

        # Logging — V2: adds rho, E, export_loss, rho_cost
        if self.track:
            self.results["rounds"].append({
                "round":        self.t,
                "tau":          tau_t,
                "d":            d_t,
                "rho":          rho_t,                                      # V2
                "M":            diag.get("M"),
                "X":            diag.get("X"),
                "E":            diag.get("E"),                              # V2
                "leader_pay":   L_pay,
                "follower_pay": F_pay,
                "rev":          diag.get("rev"),
                "cons_loss":    diag.get("cons_loss"),
                "action_cost_L":diag.get("action_cost_L"),
                "export_loss":  diag.get("export_loss"),                    # V2
                "export_gain":  diag.get("export_gain"),
                "infl_cost":    diag.get("infl_cost"),
                "action_cost_F":diag.get("action_cost_F"),
                "rho_cost":     diag.get("rho_cost"),                      # V2
            })
