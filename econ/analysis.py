"""
Post-processing module for Stackelberg RL trade war simulator.

This module converts raw simulation trajectories into:
1. Regime classification (deterrence / escalation / oscillation / mixed)
2. Regime shift detection
3. Phase diagram generation (parameter sweeps)
4. Robustness analysis (perturbation tests)
5. Parameter sensitivity ranking

Designed to work with:
results["rounds"] from StackelbergTariffGameEconomic
"""

from stackelberg_q3_tariff_econ_sim import EconomicParams, EconomicEnvironment, Q3BinnedLeader, Q3BinnedFollower, StackelbergTariffGameEconomic
import numpy as np
from collections import defaultdict
from copy import deepcopy

def extract_series(results, key):
    return [r[key] for r in results["rounds"]]

def volatility(series):
    # Measures instability / cycling. High variance = oscillatory or unstable regime.
    return np.var(series)


def trend(series):
    #Linear trend (slope) of a time series, Positive slope = escalation (increasing tariffs or retaliation)
    x = np.arange(len(series))
    return np.polyfit(x, series, 1)[0]


def final_level(series):
    return series[-1]


def welfare_avg(results, key):
    return np.mean(extract_series(results, key))


# REGIME CLASSIFICATION
def classify_regime(results, vol_threshold=0.01, escalation_threshold=0.001):
    """
    Classifies system behavior into regimes.

    Regimes:
    - deterrence: stable, low volatility
    - escalation: both τ and d increasing
    - oscillation: high volatility
    - mixed: everything else
    """

    tau = extract_series(results, "tau")
    d   = extract_series(results, "d")

    v_tau = volatility(tau)
    v_d   = volatility(d)

    slope_tau = trend(tau)
    slope_d   = trend(d)

    if v_tau < vol_threshold and v_d < vol_threshold:
        return "deterrence"

    if slope_tau > escalation_threshold and slope_d > escalation_threshold:
        return "escalation"

    if v_tau >= vol_threshold or v_d >= vol_threshold:
        return "oscillation"

    return "mixed"

# 4. REGIME SHIFT DETECTION
def detect_regime_shifts(results, window=20):
    """
    Detects when system changes behavior over time.

    Idea:
    - classify early vs late window
    - compare regimes
    """

    rounds = results["rounds"]
    T = len(rounds)

    if T < 2 * window:
        return "insufficient_data"

    early = {"rounds": rounds[:window]}
    late  = {"rounds": rounds[-window:]}

    return {
        "early_regime": classify_regime(early),
        "late_regime": classify_regime(late),
        "shifted": classify_regime(early) != classify_regime(late)
    }

def run_parameter_sweep(simulator_class, base_env, leader, follower, param_grid, rounds=200):
    """
    Runs simulations across parameter grid.

    param_grid format:
    {
        "demand_elast": [1.0, 1.5, 2.0],
        "leader_cost_w": [0.0, 0.1, 0.2]
    }

    Returns:
        list of (params, regime)
    """

    results_map = []

    keys = list(param_grid.keys())

    def recurse_build(params, idx=0):
        if idx == len(keys):
            env = deepcopy(base_env)

            # apply parameters
            for k, v in params.items():
                setattr(env.p, k, v)

            game = simulator_class(env, deepcopy(leader), deepcopy(follower))
            game.run(rounds)

            regime = classify_regime(game.results)

            results_map.append((deepcopy(params), regime))
            return

        key = keys[idx]
        for val in param_grid[key]:
            params[key] = val
            recurse_build(params, idx + 1)

    recurse_build({})
    return results_map


# robustness analysis
def perturb_param(value, sigma=0.05):
    return value * (1 + np.random.normal(0, sigma))


def robustness_test(simulator_class, base_env, leader, follower, param_name, n_trials=50, noise=0.05, rounds=200):      
    # Measures how stable a regime is under perturbation.
    # Returns: fraction of runs that stay in baseline regime

    # baseline regime
    base_game = simulator_class(base_env, leader, follower)
    base_game.run(rounds)
    base_regime = classify_regime(base_game.results)

    matches = 0

    for _ in range(n_trials):
        env = deepcopy(base_env)

        original = getattr(env.p, param_name)
        setattr(env.p, param_name, perturb_param(original, noise))

        game = simulator_class(env, deepcopy(leader), deepcopy(follower))
        game.run(rounds)

        regime = classify_regime(game.results)

        if regime == base_regime:
            matches += 1

    return matches / n_trials

# parameter sensitivity ranking
def sensitivity_ranking(simulator_class, base_env, leader, follower, param_list, n_trials=30, noise=0.05, rounds=200):
    # Ranks parameters by how much they affect regime changes.
    # Output: list of (param, instability_score)

    base_game = simulator_class(base_env, leader, follower)
    base_game.run(rounds)
    base_regime = classify_regime(base_game.results)

    scores = {}

    for param in param_list:
        flips = 0

        for _ in range(n_trials):
            env = deepcopy(base_env)

            original = getattr(env.p, param)
            setattr(env.p, param, perturb_param(original, noise))

            game = simulator_class(env, deepcopy(leader), deepcopy(follower))
            game.run(rounds)

            regime = classify_regime(game.results)

            if regime != base_regime:
                flips += 1

        scores[param] = flips/n_trials

    # sort descending by sensitivity
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ESTING
params = EconomicParams(
    demand_elast=1.5,
    supply_elast=1.2,
    leader_cost_w=0.05,
    follower_cost_w=0.05
)

env = EconomicEnvironment(params)

leader = Q3BinnedLeader(tau_bins=[0.5, 0.6, 0.3, 0.3])
follower = Q3BinnedFollower(d_bins=[0.5, 0.1, 0.7, 0.6])

game = StackelbergTariffGameEconomic(env, leader, follower)
game.run(rounds=300)

results = game.results

print(classify_regime(game.results, 0.4, 0.01))
print(detect_regime_shifts(game.results))