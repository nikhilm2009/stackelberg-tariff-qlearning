# Multi‑Follower Stackelberg Tariff MARL[Simple Payoff]

Simulation framework for tariff wars with **one leader** and **multiple followers** who can coordinate via a simple coalition protocol. Followers are tabular Q‑learning agents with optional **Q3 memory** (3‑step opponent history), uncertainty‑based opt‑in, quorum voting, adoption thresholds, and stickiness.

> Files in this module

* **`stackelberg_q3_tariff_MultiFollower_sim.py`** — Core game engine (Stackelberg leader → followers respond), payoff matrix, coalition step, per‑round logging.
* **`marl_q3_followers.py`** — Follower agent classes:

  * `QLCoalitionFollower` (ε‑greedy, γ, α; optional Q3 state; uncertainty gate `tau_rel`; adoption thresholds `min_gain`, `min_gain_disc`; stickiness).
  * (extendable) Base utilities for state encoding, Q‑updates, ε decay.
* **`test_multiagent_simplePayoff_sim.py`** — Example runner that instantiates a leader and N followers, runs multi‑round simulations, and saves diagnostic plots.

---

## Why this repo

Trade disputes frequently feature **asymmetric leadership** (Stackelberg) and **collective reactions** by blocs. This code lets you:

1. Compare static vs learning leaders against MARL followers.
2. Toggle coalition economics to show when blocs **form** vs **fail to form**.
3. Produce clear plots for posters/papers (actions, payoffs, Q dynamics, coalition diagnostics).

---

## Quick start

```bash
# 1) Create a fresh env (optional)
uv sync 
source .venv/bin/activate

# 2) Run the example
python -s test_multiagent_simplePayoff.py
```

The runner will train for the configured number of rounds and save figures like:

* `multi_summary_panels.png` — Actions/Payoffs/Cumulatives/Histogram
* `multi_q_panels.png` — Q‑values (leader & followers), coalition signals, Q‑table growth
* `multi_coalition_diagnostics.png` — Opt‑ins, adoptions, agreement rate, adoption histogram

*(Filenames may vary slightly depending on your test script flags.)*

---

## Core concepts

### Stackelberg game loop

1. Leader observes follower state summary → chooses action (`C`/`D`).
2. Each follower proposes an action from its Q‑policy.
3. **Coalition step** (optional): willing followers opt‑in, vote, and (subject to thresholds) adopt majority.
4. Payoffs are applied via a 2×2 matrix; Q‑tables are updated; round is logged.

### Follower MARL (tabular Q)

* **Q update:**
  [ Q(s,a) ← Q(s,a) + α(r + γ max_{a′} Q(s′,a′) − Q(s,a)) ]
* **State:** simple last‑leader action *or* Q3 (last 3 actions) depending on class config.
* **Policy:** ε‑greedy; optional ε‑decay.

### Coalition mechanics (high‑level)

* **Opt‑in:** follower calls `willing_to_coalesce(leader_action)`.

  * Uses **relative uncertainty**: ( rel = |Q_C − Q_D| / (|Q_C| + |Q_D| + ε) ).
  * Gate by `tau_rel` (e.g., 0.10 ⇒ ≤10% difference counts as “uncertain”).
  * Optional extra gate: `epsilon ≤ eps_gate`.
* **Vote:** majority over opted‑in members’ *proposed* actions; tie‑break configurable.
* **Adoption:** member switches to majority action only if expected lift clears at least one threshold:

  * Absolute: `min_gain` (stage payoff gap)
  * Discount‑normalized: `min_gain_disc` with factor `(1−γ)`
  * Optional **stickiness**: once aligned, stay for `k` rounds unless loss > `delta_unstick`.
* **Quorum:** require a minimum opt‑in count to execute a coalition.

---

## Configuration knobs

| Parameter       | Where           | Meaning                                                            |
| --------------- | --------------- | ------------------------------------------------------------------ |
| `epsilon`       | agents          | Exploration rate for ε‑greedy policy                               |
| `gamma`         | agents          | Discount factor                                                    |
| `alpha`         | agents          | Learning rate                                                      |
| `tau_rel`       | followers       | Relative uncertainty threshold for opt‑in (≤ triggers willingness) |
| `eps_gate`      | followers       | Opt‑in requires ε ≤ eps_gate (optional)                            |
| `min_gain`      | coalition/agent | Required stage‑payoff lift to adopt vote                           |
| `min_gain_disc` | coalition/agent | Required discount‑normalized lift `(1−γ)*ΔQ`                       |
| `stickiness_k`  | coalition/agent | Rounds to remain aligned once adopted                              |
| `delta_unstick` | coalition/agent | Allow leaving if solo beats coalition by > this                    |
| `quorum`        | coalition       | Minimum opted‑in members to form a coalition                       |
| `tie_break`     | coalition       | Action used on vote ties (`"C"`/`"D"`)                             |

---

## Minimal examples

### A) Coalition forms

```python
from stackelberg_q3_tariff_MultiFollower_sim import StackelbergMultiFollowerGame
from marl_q3_followers import QLCoalitionFollower
from stackelberg_q3_tariff_MultiFollower_sim import Q3LearningLeader

leader = Q3LearningLeader(epsilon=0.10, gamma=0.95, alpha=0.10)
followers = [QLCoalitionFollower(epsilon=0.10, gamma=0.95, alpha=0.10) for _ in range(5)]
for f in followers:
    f.tau_rel = 0.10
    f.eps_gate = 0.10
    f.min_gain = 0.0
    f.stickiness_k = 5

config = {
    "rounds": 15000,
    "leader": leader,
    "followers": followers,
    "coalition": {"enabled": True, "quorum": 3, "tie_break": "C"},
    "save_diagnostics_png": True,
    "diagnostics_png": "multi_coalition_diagnostics.png",
}

Game = StackelbergMultiFollowerGame(config)
Game.run()
```

### B) Coalition does **not** form (rationally blocked)

```python
leader = Q3LearningLeader(epsilon=0.10, gamma=0.95, alpha=0.10)
followers = [QLCoalitionFollower(epsilon=0.10, gamma=0.95, alpha=0.10) for _ in range(5)]
for f in followers:
    f.tau_rel = 0.01     # very strict uncertainty
    f.eps_gate = 0.05
    f.min_gain = 2.1     # PD stage max gain ≤2 → blocks adoption
    f.stickiness_k = 0

config = {
    "rounds": 15000,
    "leader": leader,
    "followers": followers,
    "coalition": {"enabled": True, "quorum": 5, "tie_break": "C", "min_gain": 2.1},
    "save_diagnostics_png": True,
}

Game = StackelbergMultiFollowerGame(config)
Game.run()
```

---

## Plots

* **Summary panels:** leader & follower actions (with moving averages), per‑round payoffs, cumulative payoffs, histogram of #followers choosing `D`.
* **Q panels:** leader top‑Q traces, follower average Q traces, coalition signals (avg maxQ and |ΔQ|), Q‑table growth.
* **Diagnostics:** coalition opt‑ins/adoptions over time (raw + MA), agreement rate, adoption size histogram.

> Tip: For long runs (10–15k), use moving‑average windows of ~`251–501` to depoise raw signals.

---

## Reproducibility

Set seeds at the top of your runner to keep figures stable:

```python
import random, numpy as np
random.seed(42); np.random.seed(42)
```

If your agents decay ε each round, note the decay schedule in the figure caption.

---

## Extending

* Replace payoffs with **economic terms** (export gains, inflation penalties, action costs) by modifying the payoff matrix and reward shaping.
* Add new follower personas by subclassing the follower base and implementing state encoder + `select_action` + `update`.
* Swap the coalition protocol (e.g., weighted voting, leader‑signaled coordination).

---

## License

MIT 

---

## Citation

If you use or extend this code in a report or poster, you can cite the project as:

> *Analyzing Socio-Political Reasoning Using Multi-Agent AI Trained on Societal Dilemmas.*
> SCCUR — Southern California Conference for Undergraduate Research, 2025.