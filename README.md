# Stackelberg Tariff Game with Q-Learning

This repository implements a **two-player Stackelberg tariff–currency game** combining a stylized economic payoff model with **tabular Q-learning**. The framework is designed to study **adaptive dynamics in trade conflicts**, where one country sets tariffs and the other responds with currency depreciation.

The project focuses on interpretability, economic intuition, and diagnostic clarity rather than deep neural networks.

---

## Overview

We model a repeated interaction between:

- **Leader (e.g., United States)**  
  Chooses a tariff rate τ on imports.

- **Follower (e.g., China or India)**  
  Responds by choosing a currency depreciation level d.

The leader moves first (Stackelberg timing), and the follower reacts after observing the tariff. Both agents may either:
- Follow a **best-response (BR)** rule derived from the payoff model, or
- Learn adaptively using **tabular Q-learning** with discretized action spaces.

---

## Economic Model

## Trade Dynamics

Imports and exports evolve through partial adjustment:

- **Imports (leader)**  
  M_{t+1} = M_t · (1 − κ · τ_t · demand_elast)

- **Exports (follower)**  
  X_{t+1} = X_t · (1 + λ · d_t · supply_elast)


### Payoffs

**Leader payoff**
- Tariff revenue
- Consumer surplus loss
- Quadratic tariff cost

**Follower payoff**
- Export gain from depreciation
- Inflation cost from more expensive imports
- Quadratic depreciation cost

Discounted present values are computed using a common discount factor δ.

---

## Reinforcement Learning Setup

### Q-Learning Agents

Both leader and follower can use **tabular Q-learning**:

- Discrete action bins (typically 6–8 levels)
- ε-greedy exploration
- Standard Q-update rule:

  Q(s, a) ← (1 − α) · Q(s, a)  
     + α · [ r + γ · maxₐ′ Q(s′, a′) ]


### State Representation
States are intentionally compact and interpretable:
- Last 3 actions (τ or d history)
- Current policy level
- Optional regime labels (e.g., low/high elasticity cases)

This design enables **clear diagnostics** and avoids opaque representations.

---

## Files

### `stackelberg_q3_tariff_econ_sim.py`
Core implementation:
- `EconomicParams`: parameter container
- `EconomicEnvironment`: trade flow and payoff logic
- `Q3BinnedLeader`, `Q3BinnedFollower`: tabular Q-learning agents
- `BestResponseFollowerEconomic`: analytical best-response baseline
- `StackelbergTariffGameEconomic`: simulation driver

### `test_stackelberg_q3_tariff_econ.py`
Experiment and visualization script:
- Runs long-horizon simulations (e.g., 20,000 rounds)
- Generates **Figures 1–5**, including:
  - Policy time series
  - Trade flows
  - Payoff decompositions
  - Q-value diagnostics vs discounted PV
  - State-visit coverage and convergence checks

---

## Figures & Diagnostics

The framework emphasizes **debuggability and validation**:

- Comparison of learned Q-values vs realized discounted payoffs
- Rolling averages to assess convergence
- Most-visited-state heatmaps
- Policy histograms and best-response curves
- Coverage diagnostics to detect under-exploration or overestimation

These tools help identify when learning aligns with theory—and when it does not.

---

## Typical Use Cases

- Studying **leader-favoring vs follower-favoring** trade environments
- Comparing **best-response vs adaptive learning**
- Demonstrating bounded rationality and path dependence
- Educational or research-oriented simulations of trade policy dynamics

---

## Requirements

- Python 3.9+
- NumPy
- Matplotlib

No deep learning frameworks are required.

---

## Design Philosophy

This repository prioritizes:
- Economic transparency over black-box models
- Clear diagnostics over raw performance
- Reproducible experiments with fixed seeds
- A clean baseline for future extensions (e.g., deep RL, multi-country models)

---

## Future Extensions (Out of Scope Here)

- Deep Q-learning or actor–critic methods
- Continuous action spaces
- Recurrent or attention-based state encoding
- Multi-country or networked trade models

These are intentionally excluded to keep the current framework focused and interpretable.

---

## License

MIT License 

---

## Citation

If you use this code in academic work, please cite as:

> *Stackelberg Tariff Game with Q-Learning*,  
> GitHub repository, YYYY.

---

## Contact

For questions or extensions, please contact the repository owner.
