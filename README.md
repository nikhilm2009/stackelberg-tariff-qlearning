# Stackelberg Tariff Games with Reinforcement Learning

This repository explores **leader–follower (Stackelberg) interactions** using reinforcement learning, progressing from **toy strategic games** to a **structured economic model of tariffs and currency responses**.  
The codebase is intentionally modular, with each stage of the project documented in its own README.

The goal is to study how **learning, memory, and coordination** alter strategic outcomes in trade-style conflicts.

---

## How to Navigate This Repository

The project is organized chronologically. Each README corresponds to a distinct modeling stage and research question.

### 1. Baseline: Learning in Simple Strategic Games  
📄 **`README_SimplePayoff.md`**

**What it covers**
- Single leader and single follower  
- Abstract, toy payoffs (no macroeconomics)  
- Tabular Q-learning and Q3 (short-history states)  
- Symmetric action sets (e.g., cooperate vs retaliate)

**Why it exists**
This stage isolates *learning dynamics*:
- Can short-horizon memory support cooperation?
- How does punishment and forgiveness emerge?
- How do outcomes differ from static Nash equilibria?

➡️ Start here if you want the **simplest possible Stackelberg RL setup**.

---

### 2. Multi-Agent Extension: Coalitions and Coordination  
📄 **`README_MARL.md`**

**What it adds**
- One leader, **multiple followers**
- Still toy payoffs, but richer interaction
- Coalition logic: followers may opt in, vote, and align
- Diagnostics for coalition formation and stability

**Why it exists**
This stage studies **collective behavior**:
- When do rational coalitions form?
- How uncertainty and learning affect coordination
- Whether follower coalitions erode the leader’s advantage

➡️ Read this if you’re interested in **multi-agent RL, coordination, and coalition dynamics**.

---

### 3. Economic Model: Tariffs vs Currency Depreciation  
📄 **`README_EconPayoff.md`**

**What it adds**
- Explicit economic environment (imports, exports, elasticities)
- Leader sets tariff rate τ
- Follower responds with currency depreciation d
- Interpretable payoff components (revenue, consumer loss, inflation, policy costs)
- Comparison between learned policies and analytical best responses

**Why it exists**
This stage grounds the framework in **economic realism**:
- Links RL behavior to economic intuition
- Enables policy-level interpretation
- Tests whether learning converges to (or deviates from) theory

➡️ Start here if you care about **economic modeling, interpretability, and diagnostics**.

---

## Conceptual Progression

```
Simple Payoffs
   ↓
Learning Dynamics (Q / Q3)
   ↓
Multi-Follower Coordination
   ↓
Coalitions
   ↓
Economic Stackelberg Model
```

Each stage builds directly on the previous one, increasing realism while preserving transparency.

---

## Design Philosophy

Across all stages, the project prioritizes:
- **Interpretability over black-box performance**
- **Tabular RL** to expose learning mechanisms
- **Clear diagnostics** (Q-traces, policy paths, coalition signals)
- **Comparability** between static theory and adaptive agents

Deep RL and continuous control are intentionally left out to keep behavior analyzable.

---

## Where to Start

- **New to the project?** → `README_SimplePayoff.md`  
- **Interested in MARL & coalitions?** → `README_MARL.md`  
- **Focused on economics & policy interpretation?** → `README_EconPayoff.md`

---

## Repository Structure
```
├── README.md                     # Top-level overview (project roadmap & navigation)
│
├── README_SimplePayoff.md         # Stage 1: Single leader–follower, toy payoffs
├── README_MARL.md                 # Stage 2: Multi-follower MARL with coalitions
├── README_EconPayoff.md           # Stage 3: Economic Stackelberg model
│
├── simple_payoff/
│   ├── stackelberg_q3_tariff_simplePayoff_sim.py
│   ├── test_stackelberg_q3_tariff_simplePayoff.py
│   └── plots/ 
│
├── marl/
│   ├── stackelberg_q3_tariff_MultiFollower_sim.py
│   ├── marl_q3_followers.py
│   ├── test_multiagent_simplePayoff.py
│   └── plots/
│
├── econ/
│   ├── stackelberg_q3_tariff_econ_sim.py
│   ├── test_stackelberg_q3_tariff_econ.py
│   ├── plots/
│
├── pyproject.toml
├── .gitignore
└── LICENSE
```




## Future Directions

Planned or conceptual extensions include:
- Allowing followers to choose *both* depreciation and retaliatory tariffs
- Adding export losses explicitly to the leader payoff
- Multi-country follower blocs
- Deep or recurrent RL for longer-horizon memory

These are intentionally separated from the current codebase to keep each stage focused.

---

## License

MIT License
