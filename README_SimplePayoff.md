# Stackelberg Q3 Simulation

A compact, reproducible setup to study **leader–follower (Stackelberg)** interactions with **tabular Q-learning** and a short-horizon **Q3 (n=3) temporal-difference** update. The environment models a simple tariff game where the **leader** commits first and **followers** respond after observing the leader’s move. An optional **coalition layer** lets followers coordinate when it increases expected value.

> Files included in this bundle
>
> - `stackelberg_q3_tariff_simplePayoff_sim.py` — core simulation (environment + agents + training loop).
> - `test_stackelberg_q3_tariff_simplePayoff.py` — minimal smoke tests / quick-run examples.

---

## 1) Why this repo?

- **Sequential timing:** captures first-mover commitment and observability.
- **Learning vs. rules:** compare static (rule-based) strategies with learning agents.
- **Short-horizon credit:** Q3 targets handle brief retaliation/forgiveness sequences.
- **Coalitions (opt-in):** followers can coordinate via simple, payoff-screened rules.

---

## 2) Quick start

### Requirements
- Python 3.9+
- No heavy dependencies expected (standard library + `numpy` if used by your code).

### Run a quick simulation
```bash
# Option A: run the main simulation
python stackelberg_q3_tariff_simplePayoff_sim.py

# Option B: run the test / example harness
python test_stackelberg_q3_tariff_simplePayoff.py
```

Both scripts print a short summary to stdout (episode returns, cooperation rates, etc.).

---

## 3) Core concepts

- **State**: encodes the leader’s action and a short memory of recent joint actions; may include a `coalition_hint`.
- **Actions**: tariffs as discrete choices or {C, D}-style actions.
- **Rewards (simple payoff)**: stylized tariff revenue minus retaliation/inflation penalties (see code comments).
- **Agents**:
  - **Q-learning** (1‑step TD).
  - **Q3** (n=3 TD target): spreads credit over 3 steps before bootstrapping.
- **Coalition layer** (followers):
  - **Opt‑in** if uncertain (|Q(C)−Q(D)| small) or confident/low‑epsilon (stability).
  - Majority action among opt‑ins; adopt if expected gain ≥ `min_gain`.
  - Non-participants keep their proposed actions.

---

## 4) Typical CLI options (if supported)

> Adjust to match the argparse options defined in your script(s).

- `--episodes` (int): number of training episodes (default: 500–2000)
- `--horizon` (int): steps per episode (T)
- `--seed` (int): RNG seed
- `--alpha` (float): learning rate (0.1–0.3 common)
- `--gamma` (float): discount factor (0.95–0.99 common)
- `--epsilon-start` / `--epsilon-min` / `--epsilon-decay`
- `--q3` / `--n-step` (int): set to `3` to enable Q3 target
- `--use-coalitions` (flag): enable coalition coordination
- `--coalition-min-gain` (float): minimum expected advantage to adopt majority
- `--coalition-interval` (int): run coalition step every K steps
- `--log-dir` (str): optional output folder for CSV logs/plots

> If your scripts don’t expose CLI flags, edit the parameters near the top of the file(s).

---

## 5) Interpreting results

- **Leader advantage:** compare average returns of leader vs followers with/without coalitions.
- **Cooperation trajectories:** fraction of {C, C} outcomes / low-tariff states over time.
- **Policy stability:** duration of cooperative paths; retaliation/forgiveness cycle length.
- **Coalition metrics:** opt‑in rate, coalition lifetime, violation rate, side‑payment volume (if modeled).

---

## 6) Reproducibility tips

- Set `seed` for all runs and note Python/Numpy versions.
- Keep a fixed `episodes × horizon` budget when comparing configurations.
- When sweeping hyperparameters (α, γ, ε, coalition rules), report **median** and **IQR** across ≥ 20 seeds.

---

## 7) Minimal code sketch (Q3 update)

```python
# Pseudocode; see actual implementation in the simulation file.
Q[s, a] += alpha * (
    r_t + gamma * r_{t+1} + gamma**2 * r_{t+2}
    + gamma**3 * max_a Q[s_{t+3}, a] - Q[s, a]
)
```

Action selection is **epsilon‑greedy** (explore with prob. ε; otherwise take `argmax_a Q[s, a]`).

---

## 8) Troubleshooting

- **No improvement / stuck in D,D**: increase `gamma`, start with higher `epsilon`, try Q3 (`--n-step 3`).
- **Unstable learning**: reduce `alpha`, add an ε floor (e.g., 0.05), increase episodes.
- **Leader dominates**: enable coalitions or raise `coalition_min_gain` so only beneficial blocs form.
- **Followers can’t coordinate**: decrease `coalition_min_gain` or increase coalition interval K.

---

## 9) Citing / attribution

If you use or extend this code in a report or poster, you can cite the project as:

> *Analyzing Socio-Political Reasoning Using Multi-Agent AI Trained on Societal Dilemmas.*
> SCCUR — Southern California Conference for Undergraduate Research, 2025.


## 10) License

MIT (or your preferred license). Update this section if you choose a different license.
