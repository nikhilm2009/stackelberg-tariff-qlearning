# Economic Stackelberg Model — IEEE CIFEr 2026

This directory contains the full implementation for:

> *Regime Structure in Adaptive Tariff Conflicts: A Multi-Agent Reinforcement Learning Analysis*
> Nikhil Muthukumar, Micah Chiang, Jeffrey Chen, and Phil Mui. IEEE CIFEr 2026.

The model studies a two-country tariff conflict as a repeated Stackelberg game
with an explicit economic environment. The central contribution is a
**MARL-based phase-diagram methodology** that maps structural parameter spaces
to stable outcome regimes and coordination-sensitive regions.

---

## What This Models

Two countries interact repeatedly:

- **Leader** (structurally privileged economy) — sets import tariff τ and
  optionally currency depreciation d_L
- **Follower** (responding economy) — sets retaliatory tariff ρ and currency
  depreciation d

The leader moves first each period (Stackelberg timing). Both agents update
policies simultaneously via tabular Q-learning rather than under explicit
commitment. Trade flows (imports M, follower exports X, leader exports E)
evolve through partial adjustment dynamics governed by elasticity parameters.

---

## Economic Model

### Trade Flows

Each variable adjusts gradually toward a target level:

```
M* = M0 · (1 + τ)^{-εd}                          # import demand
X* = X0 · (1 + d)^{εs}                            # follower export supply
E* = E0 · (1 + ρ)^{-εE} · (1 + dL)^{εdL}         # leader exports

Mt = (1 − κ)·M_{t-1} + κ·M*                       # partial adjustment
```

### Payoffs

**Leader:**
```
L = τM − 0.5τ(M0−M) − cL·τ² − ψE·max(0, E0−E)
  − ξ·(ρ/ρmax)·(τM) − φL·dL·IL0 − cdL·dL²
```

**Follower:**
```
F = (X − X0) − φF·d·IF0 − cF·d² − cρ·ρ²
```

Behavioural parameters (inequity aversion α, spite β, diplomatic cost ξ)
are zeroed in the structural baseline so outcomes are driven by capacity
parameters alone. Perturbation analysis introduces ξ > 0 to study diplomatic
cost effects.

### Calibrations

Three parameter configurations are studied:

| Calibration | Interpretation | τ_max | d_max | ρ_max |
|-------------|----------------|-------|-------|-------|
| LF | Leader-favoring | 0.35 | 0.18 | 0.20 |
| FF | Follower-favoring | 0.25 | 0.25 | 0.35 |
| CENTER | Near boundary | 0.28 | 0.22 | 0.27 |

Import demand elasticities (1.30–1.55) and tariff ceilings (25%–35%) are
consistent with empirical estimates from the 2018–2019 US–China trade conflict.

---

## Learning Architecture

### Leader — Standard Q-Learning
- State: 7-tuple encoding follower's recent depreciation history, last tariff,
  binned import/export levels, follower's last retaliation
- Action: tariff bin τ ∈ [0, τ_max] (or joint (τ, d_L) when depreciation active)
- Single Q-table, adaptive learning rate, ε-greedy exploration

### Follower — Double Q-Learning
- State: 7-tuple encoding leader's recent tariff history, export level,
  own last actions, export momentum signal
- Action: joint (d, ρ) bin — up to 30 combinations
- Two Q-tables updated symmetrically to reduce overestimation bias on the
  larger joint action space

### Key Hyperparameters
- Training: 600,000 steps per seed (1,000,000 for depreciation extension)
- Grid: 8×8 parameter grid per phase diagram
- Seeds: 5–10 per grid cell; majority vote determines regime classification
- Regime thresholds: L̄ > +15 → Deterrence, L̄ < −15 → Escalation

---

## Main Results

### Phase Diagrams
Six phase diagrams map structural parameters to three stable regimes:
**Deterrence** (green), **Transition** (orange), **Escalation** (red).
A fourth outcome, **Multiple** (gray), marks coordination-sensitive cells
near regime boundaries where no majority exists across seeds.

### Key Findings

1. **Capacity-ratio boundary** — The Deterrence–Escalation boundary tracks the
   ratio ρ_max/τ_max. Interpolated contours give:
   - D/T boundary: ρ_max = 0.042 + 0.786·τ_max (R² = 0.943)
   - T/E boundary: ρ_max = 0.131 + 1.275·τ_max (R² = 0.982)

2. **Leader depreciation** — Non-monotone inverted-W effect with an interior
   optimum near d_L,max ≈ 0.04 and partial recovery at higher levels.

3. **Coordination sensitivity** — Near boundaries, identical parameters produce
   different outcomes across seeds, revealing multiple stable attractors.
   Lower Q-initialization expands these regions, confirming attractor-selection
   interpretation rather than convergence failure.

4. **Policy stability** — A frozen-leader relearning probe (600,000 rounds)
   preserved majority regime classification in all 64 cells. Follower
   retaliation correlation 0.9946 (seed-level); depreciation correlation
   0.9808 (cell-level). Follower payoff declined in 39/64 cells (mean
   ΔF = −0.21), attributable to loss of co-adaptive benefit rather than
   insufficient probe training.

---

## Files

### Core Simulation
| File | Description |
|------|-------------|
| `stackelberg_q3_tariff_econ_sim_v10_leaderDepr.py` | Simulation engine — environment, agents, game loop |
| `stackelberg_q3_tariff_econ_config_v10.py` | Calibrations, agent builder, regime classifier |

### Experiment Runners
| File | Description |
|------|-------------|
| `v11_pd_ieee_leaderDepr_PD.py` | Phase diagram runner — IEEE figures, per-seed CSV output |
| `test_stackelberg_q3_tariff_econ_v10_leaderDepr.py` | Dynamics and sensitivity analysis |

### Analysis Scripts
| File | Description |
|------|-------------|
| `pd2_br_probe_grid_frozenQ.py` | Frozen-Q leader relearning probe across full grid |
| `pd2_threshold_sensitivity.py` | Seed-level threshold sensitivity (±10, ±15, ±20) |
| `plot_pd2_boundary_fit.py` | Boundary-fit figure generator |
| `analyze_br_probe_follower.py` | Follower payoff analysis from probe CSV |

### Data
| File | Description |
|------|-------------|
| `fig_pd2_retaliation_vs_tariff_3000.csv` | Q₀=3000 baseline phase diagram (10 seeds) |
| `fig_pd2_retaliation_vs_tariff_1000_1000K.csv` | Q₀=1000 robustness run (10 seeds, 1M steps) |
| `pd2_br_probe_frozenQ_results_raw.csv` | 600k relearning probe raw results |
| `pd2_br_probe_frozenQ_results_raw_200k.csv` | 200k relearning probe raw results |

### Output
| Directory | Description |
|-----------|-------------|
| `phase_plots_v11_leaderDepr/` | All phase diagram PNGs and CSVs |

---

## Quick Start

```bash
# 1) Install dependencies
uv sync
source .venv/bin/activate

# 2) Run phase diagrams (PD2 only for a quick test)
#    Edit RUN_CONFIG: run_pds = [2]
python v11_pd_ieee_leaderDepr_PD.py

# 3) Run dynamics and sensitivity analysis
python test_stackelberg_q3_tariff_econ_v10_leaderDepr.py

# 4) Run relearning probe
python pd2_br_probe_grid_frozenQ.py

# 5) Threshold sensitivity (requires per-seed CSV from step 2)
python pd2_threshold_sensitivity.py
```

---

## Requirements

- Python 3.10+
- NumPy, Matplotlib, SciPy
- No deep learning frameworks required

---

## Citation

```bibtex
@inproceedings{tariff_marl_cifer2025,
  title     = {Regime Structure in Adaptive Tariff Conflicts:
               A Multi-Agent Reinforcement Learning Analysis},
  booktitle = {Proceedings of the IEEE Symposium on Computational
               Intelligence for Financial Engineering and Economics (CIFEr)},
  url       = {https://cifer2026.mhirano.jp/},
  year      = {2026}
}
```

---

## License

MIT License
