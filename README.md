# Stackelberg Tariff Games with Reinforcement Learning

This repository documents a research progression from simple strategic games
to a published multi-agent reinforcement learning model of international trade
conflicts. The work spans two years and three modeling stages, each building
directly on the previous one.

**Stage 3 (IEEE CIFEr 2026)** — *Regime Structure in Adaptive Tariff Conflicts:
A Multi-Agent Reinforcement Learning Analysis*
(Nikhil Muthukumar, Micah Chiang, Jeffrey Chen, and Phil Mui) — is the current active research
thread and the most complete work in the repository.

**Stage 3** was accepted to the
[IEEE Symposium on Computational Intelligence for Financial Engineering and Economics (CIFEr) 2026](https://cifer2026.mhirano.jp/).

**Stages 1 and 2** were presented at the
[Southern California Conference for Undergraduate Research (SCCUR) 2025](https://www.sccur.org).

---

## Research Progression

```
Stage 1 — Simple Payoffs
    Single leader–follower, abstract rewards, tabular Q-learning
    ↓
Stage 2 — Multi-Agent Coordination
    One leader, multiple followers, coalition dynamics
    ↓
Stage 3 — Economic Stackelberg Model  ← IEEE CIFEr 2026
    Full trade-flow model, phase diagrams, regime structure
```

Each stage isolates a specific research question while preserving the
interpretability and tabular-RL foundation established at the start.

---

## Stage 1 — Learning in Simple Strategic Games
📁 `simple_payoff/` &nbsp;|&nbsp; 📄 `README_SimplePayoff.md`

The baseline stage studies whether short-horizon memory can support
cooperation in a single leader–follower setting with abstract payoffs.
Key questions: can tabular Q-learning with recent action history produce
punishment and forgiveness? How do learned outcomes compare to static
Nash equilibria?

➡️ Start here for the **simplest possible Stackelberg RL setup**.

---

## Stage 2 — Multi-Agent Coordination and Coalitions
📁 `marl/` &nbsp;|&nbsp; 📄 `README_MARL.md`

Extends Stage 1 to one leader and multiple followers. Followers may form
coalitions, vote on joint strategies, and align against the leader. Studies
when rational coalitions form and whether they erode the leader's strategic
advantage under adaptive learning.

*Presented at SCCUR 2025.*

➡️ Read this for **multi-agent RL, coordination, and coalition dynamics**.

---

## Stage 3 — Economic Stackelberg Model (IEEE CIFEr 2026)
📁 `econ/` &nbsp;|&nbsp; 📄 `README_EconPayoff.md`

The main research contribution. Models a two-country tariff conflict as a
repeated Stackelberg game with an explicit economic environment: import
demand, export supply, currency depreciation, and retaliatory tariffs, all
governed by elasticity relationships and partial adjustment dynamics.

Both agents learn via tabular Q-learning (the follower uses Double Q-learning
to handle its larger joint action space). The central output is a suite of
**phase diagrams** mapping structural parameters to three stable outcome
regimes: Deterrence, Transition, and Escalation.

**Key findings:**
- The Deterrence–Escalation boundary is organized by the ratio of follower
  retaliation capacity to leader tariff capacity
- Leader currency depreciation has a non-monotone inverted-W effect on the
  Deterrence region, with an interior optimum and partial recovery at higher
  levels
- Near regime boundaries, identical parameters produce different outcomes
  across learning runs, revealing coordination-sensitive regions with multiple
  stable attractors

**Published:** *Regime Structure in Adaptive Tariff Conflicts: A Multi-Agent
Reinforcement Learning Analysis*,
Nikhil Muthukumar, Micah Chiang, Jeffrey Chen, and Phil Mui. [IEEE CIFEr 2026](https://cifer2026.mhirano.jp/).

*Also presented at SCCUR 2025.*

➡️ Start here for **economic modeling, phase diagrams, and regime structure**.

---

## Design Philosophy

All three stages share the same core principles:

- **Interpretability over performance** — tabular Q-learning keeps value
  functions directly inspectable
- **Explicit diagnostics** — convergence checks, best-response probes, and
  sensitivity analyses at every stage
- **Incremental complexity** — each stage adds one new modeling layer without
  discarding what came before
- **Comparability with theory** — learned policies are compared against
  analytical benchmarks throughout

Deep RL and continuous action spaces are intentionally excluded to keep
behavior analyzable and results reproducible.

---

## Repository Structure

```
├── README.md                      # This file
│
├── README_SimplePayoff.md         # Stage 1 documentation
├── README_MARL.md                 # Stage 2 documentation
├── README_EconPayoff.md           # Stage 3 documentation (IEEE CIFEr)
│
├── simple_payoff/                 # Stage 1 — SCCUR 2025
│   ├── stackelberg_q3_tariff_simplePayoff_sim.py
│   ├── test_stackelberg_q3_tariff_simplePayoff.py
│   └── plots/
│
├── marl/                          # Stage 2 — SCCUR 2025
│   ├── stackelberg_q3_tariff_MultiFollower_sim.py
│   ├── marl_q3_followers.py
│   ├── test_multiagent_simplePayoff.py
│   └── plots/
│
├── econ/                          # Stage 3 — IEEE CIFEr 2026
│   ├── stackelberg_q3_tariff_econ_sim_v10_leaderDepr.py
│   ├── stackelberg_q3_tariff_econ_config_v10.py
│   ├── test_stackelberg_q3_tariff_econ_v10_leaderDepr.py
│   ├── v11_pd_ieee_leaderDepr_PD.py
│   ├── pd2_br_probe_grid_frozenQ.py
│   ├── pd2_threshold_sensitivity.py
│   ├── plot_pd2_boundary_fit.py
│   └── phase_plots_v11_leaderDepr/
│
├── pyproject.toml
├── .gitignore
└── LICENSE
```

---

## Requirements

```bash
uv sync
source .venv/bin/activate
```

- Python 3.10+
- NumPy, Matplotlib, SciPy
- No deep learning frameworks required

---

## License

MIT License
