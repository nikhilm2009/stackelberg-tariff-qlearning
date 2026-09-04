# pd2_br_probe_grid_frozenQ.py
# ============================================================
# PD2 Grid-Level Best-Response Probe — Frozen Q-Table Leader
# ============================================================
# Distinguishes from pd2_br_probe_grid.py (modal tau only):
#   HERE: leader Q-table is fully frozen (epsilon=0, alpha=0)
#         leader acts state-dependently using learned Q-table
#         but makes no further updates
#
#   THERE: leader fixed to single modal tau scalar (FixedLeader)
#
# For each (tau_max, rho_max) cell on the PD2 8x8 grid:
#
#   Phase 1 — MARL training
#     Standard Leader Q vs Follower Double-Q for MARL_STEPS rounds.
#     Record majority regime and learned payoffs in final eval window.
#
#   Phase 2 — Frozen-Q BR probe
#     Copy trained leader Q-table. Set epsilon=0, alpha=0 (greedy, no updates).
#     Run fresh follower Q-agent against frozen-Q leader for BR_STEPS rounds.
#     Record probe regime and payoffs in final eval window.
#
#   Output CSV columns:
#     rho_max, tau_max,
#     orig_regime, orig_lpay, orig_fpay, orig_rho, orig_d, orig_tau,
#     probe_regime, probe_lpay, probe_fpay, probe_rho, probe_d,
#     delta_L, delta_F, regime_flipped, seed
#
# Usage:
#   python pd2_br_probe_grid_frozenQ.py
#
# Requires (same directory):
#   stackelberg_q3_tariff_econ_sim_v10_leaderDepr.py
#   stackelberg_q3_tariff_econ_config_v10.py
#
# Outputs:
#   pd2_br_probe_frozenQ_results.csv
#   pd2_br_probe_frozenQ_results_raw.csv
# ============================================================

import copy, csv, os, random, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, ".")

from stackelberg_q3_tariff_econ_sim_v10_leaderDepr import (
    EconomicEnvironment, StackelbergTariffGameEconomic,
)
from stackelberg_q3_tariff_econ_config_v10 import (
    build_params_lf, build_agents, classify_regime,
)

# ── Configuration ─────────────────────────────────────────────────────────
CFG = {
    # MARL training — match production PD2 settings
    "n_bins":               6,
    "n_rho_bins":           5,
    "n_dL_bins":            3,
    "leader_depr":          False,
    "leader_q_init":        3000.0,
    "follower_q_init":      3000.0,
    "epsilon_start":        0.15,
    "epsilon_end":          0.02,
    "follower_epsilon_end": 0.005,
    "alpha":                0.18,
    "leader_alpha_min":     0.02,
    "follower_alpha_min":   0.05,
    "follower_double_q":    True,
    "freeze_mode":          "none",
    "freeze_leader_frac":   0.0,
    "freeze_follower_frac": 0.0,

    # Steps
    "marl_steps": 600000,   # MARL training rounds per seed
    "br_steps":   600000,   # BR probe rounds (fresh follower)
    "eval_frac":  0.20,     # eval last 20% of each phase

    # Grid
    "n_seeds":    5,         # seeds per cell
    "n_workers":  6,

    # Output
    "out_csv": "pd2_br_probe_frozenQ_results.csv",
    "out_dir": ".",
}

# PD2 grid — matches production CSV
RHO_VALS = [0.0, 0.0714, 0.1429, 0.2143, 0.2857, 0.3571, 0.4286, 0.5000]
TAU_VALS  = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
#TAU_VALS = [0.25, 0.30, 0.35]

REGIME_HI =  15.0
REGIME_LO = -15.0

SEED_POOL = [42, 123, 777, 999, 2024, 314, 1066, 53, 160, 1031]


def classify(lpay):
    if lpay > REGIME_HI:  return "Deterrence"
    if lpay < REGIME_LO:  return "Escalation"
    return "Transition"


def run_cell(args):
    """Run one (tau_max, rho_max, seed) combination. Returns dict of results."""
    tau_max, rho_max, seed, cfg = args

    t_cell = time.time()
    print(f"  [start] tau={tau_max:.3f} rho={rho_max:.4f} seed={seed} | "
          f"phase1 MARL {cfg['marl_steps']//1000}k steps...", flush=True)

    random.seed(seed)
    np.random.seed(seed)

    # ── Build params ──────────────────────────────────────────────────────
    params = build_params_lf()
    params.tau_max = tau_max
    params.rho_max = rho_max
    if not cfg["leader_depr"]:
        params.dL_max = 0.0

    # ── Phase 1: MARL training ────────────────────────────────────────────
    env    = EconomicEnvironment(params)
    leader, follower, tau_bins, d_bins, rho_bins = build_agents(params, cfg)
    follower.epsilon_end = cfg.get("follower_epsilon_end", follower.epsilon_end)

    game = StackelbergTariffGameEconomic(env, leader, follower, track=True)
    game.run(rounds=cfg["marl_steps"])

    rr     = game.results["rounds"]
    n_eval = int(cfg["marl_steps"] * cfg["eval_frac"])
    tail   = rr[-n_eval:]

    orig_lpay = float(np.mean([r["leader_pay"]   for r in tail]))
    orig_fpay = float(np.mean([r["follower_pay"] for r in tail]))
    orig_d    = float(np.mean([r["d"]            for r in tail]))
    orig_rho  = float(np.mean([r["rho"]          for r in tail]))
    orig_tau  = float(np.mean([r["tau"]          for r in tail]))
    orig_regime = classify(orig_lpay)

    # ── Phase 2: Frozen-Q BR probe ────────────────────────────────────────
    print(f"  [probe] tau={tau_max:.3f} rho={rho_max:.4f} seed={seed} | "
          f"phase2 frozen-Q probe {cfg['br_steps']//1000}k steps "
          f"(mean train tau={orig_tau:.3f})...", flush=True)

    # Deep copy trained leader — freeze Q-table (no exploration, no updates)
    leader2 = copy.deepcopy(leader)
    leader2.epsilon     = 0.0    # greedy — always picks argmax Q
    leader2.epsilon_end = 0.0    # floor also zeroed
    leader2.alpha       = 0.0    # no Q-updates — table is frozen

    # Fresh environment — reset trade flows to baseline
    env2     = EconomicEnvironment(copy.deepcopy(params))

    # Fresh follower — same hyperparams, new Q-table
    _, follower2, _, _, _ = build_agents(params, cfg)
    follower2.epsilon_end = cfg.get("follower_epsilon_end", follower2.epsilon_end)

    game2 = StackelbergTariffGameEconomic(env2, leader2, follower2, track=True)
    game2.run(rounds=cfg["br_steps"])

    rr2     = game2.results["rounds"]
    n_eval2 = int(cfg["br_steps"] * cfg["eval_frac"])
    tail2   = rr2[-n_eval2:]

    probe_lpay   = float(np.mean([r["leader_pay"]   for r in tail2]))
    probe_fpay   = float(np.mean([r["follower_pay"] for r in tail2]))
    probe_d      = float(np.mean([r["d"]            for r in tail2]))
    probe_rho    = float(np.mean([r["rho"]          for r in tail2]))
    probe_regime = classify(probe_lpay)

    delta_L = probe_lpay - orig_lpay
    delta_F = probe_fpay - orig_fpay
    flipped = (probe_regime != orig_regime)

    elapsed_cell = time.time() - t_cell
    print(f"  [done]  tau={tau_max:.3f} rho={rho_max:.4f} seed={seed} | "
          f"orig={orig_regime:12} probe={probe_regime:12} "
          f"dL={delta_L:+.1f} dF={delta_F:+.1f} "
          f"flipped={flipped} ({elapsed_cell:.0f}s)", flush=True)

    return {
        "tau_max":        tau_max,
        "rho_max":        rho_max,
        "seed":           seed,
        "orig_regime":    orig_regime,
        "orig_lpay":      round(orig_lpay,  3),
        "orig_fpay":      round(orig_fpay,  3),
        "orig_rho":       round(orig_rho,   4),
        "orig_d":         round(orig_d,     4),
        "orig_tau":       round(orig_tau,   4),
        "probe_regime":   probe_regime,
        "probe_lpay":     round(probe_lpay, 3),
        "probe_fpay":     round(probe_fpay, 3),
        "probe_rho":      round(probe_rho,  4),
        "probe_d":        round(probe_d,    4),
        "delta_L":        round(delta_L, 3),
        "delta_F":        round(delta_F, 3),
        "regime_flipped": int(flipped),
    }


def majority_regime(regimes):
    from collections import Counter
    counts = Counter(regimes)
    top, top_count = counts.most_common(1)[0]
    return top if top_count > len(regimes) / 2 else "Multiple"


def main():
    cfg   = CFG
    seeds = SEED_POOL[:cfg["n_seeds"]]

    jobs = []
    for tau_max in TAU_VALS:
        for rho_max in RHO_VALS:
            for seed in seeds:
                jobs.append((tau_max, rho_max, seed, cfg))

    total    = len(jobs)
    est_min  = total * (cfg["marl_steps"] + cfg["br_steps"]) / 1e6 * 2.5 / cfg["n_workers"]
    sep      = "=" * 70

    print(f"PD2 BR probe (frozen-Q leader): {total} jobs "
          f"({len(TAU_VALS)}x{len(RHO_VALS)} grid, "
          f"{cfg['n_seeds']} seeds, "
          f"{cfg['marl_steps']//1000}k MARL + "
          f"{cfg['br_steps']//1000}k frozen-Q probe steps)")
    print(f"Workers:   {cfg['n_workers']}")
    print(f"Est. time: ~{est_min:.0f} min (rough estimate)")
    print(f"Output:    {cfg['out_csv']}")
    print(f"Key diff:  leader Q-table frozen (epsilon=0, alpha=0) — "
          f"NOT single modal tau\n")

    t0 = time.time()

    print(sep)
    print(f"Starting {total} jobs across {cfg['n_workers']} workers...")
    print(sep)
    print()

    with Pool(cfg["n_workers"]) as pool:
        raw_results = pool.map(run_cell, jobs)

    print()
    print(sep)
    print("All jobs complete. Aggregating results...")
    print(sep)
    print()

    # ── Aggregate per cell ────────────────────────────────────────────────
    from collections import defaultdict
    cell_data = defaultdict(list)
    for r in raw_results:
        cell_data[(r["tau_max"], r["rho_max"])].append(r)

    summary_rows = []
    flip_cells   = []

    n_cells = len(TAU_VALS) * len(RHO_VALS)
    print(f"Aggregating {n_cells} cells (majority vote across {cfg['n_seeds']} seeds)...\n")

    print(f"{'tau_max':>8} {'rho_max':>8} {'orig_reg':>12} {'probe_reg':>12} "
          f"{'mean_dL':>8} {'mean_dF':>8} {'flipped':>8}")
    print("-" * 74)

    for tau_max in TAU_VALS:
        for rho_max in RHO_VALS:
            key        = (tau_max, rho_max)
            seeds_data = cell_data[key]

            orig_regimes  = [r["orig_regime"]  for r in seeds_data]
            probe_regimes = [r["probe_regime"] for r in seeds_data]
            mean_orig_lpay  = np.mean([r["orig_lpay"]  for r in seeds_data])
            mean_probe_lpay = np.mean([r["probe_lpay"] for r in seeds_data])
            mean_dL         = np.mean([r["delta_L"]    for r in seeds_data])
            mean_dF         = np.mean([r["delta_F"]    for r in seeds_data])
            mean_orig_rho   = np.mean([r["orig_rho"]   for r in seeds_data])
            mean_probe_rho  = np.mean([r["probe_rho"]  for r in seeds_data])

            orig_majority  = majority_regime(orig_regimes)
            probe_majority = majority_regime(probe_regimes)
            flipped        = (orig_majority != probe_majority)
            n_flipped      = sum(r["regime_flipped"] for r in seeds_data)

            row = {
                "tau_max":          tau_max,
                "rho_max":          rho_max,
                "orig_regime":      orig_majority,
                "probe_regime":     probe_majority,
                "mean_orig_lpay":   round(mean_orig_lpay,  2),
                "mean_probe_lpay":  round(mean_probe_lpay, 2),
                "mean_delta_L":     round(mean_dL, 2),
                "mean_delta_F":     round(mean_dF, 2),
                "mean_orig_rho":    round(mean_orig_rho,  4),
                "mean_probe_rho":   round(mean_probe_rho, 4),
                "majority_flipped": int(flipped),
                "seeds_flipped":    n_flipped,
                "n_seeds":          len(seeds_data),
            }
            summary_rows.append(row)

            if flipped:
                flip_cells.append((tau_max, rho_max, orig_majority, probe_majority))

            flag = " <-- FLIP" if flipped else ""
            print(f"{tau_max:>8.3f} {rho_max:>8.4f} {orig_majority:>12} "
                  f"{probe_majority:>12} {mean_dL:>+8.2f} {mean_dF:>+8.2f} "
                  f"{str(flipped):>8}{flag}")

    # ── Correlation of orig vs probe rho ──────────────────────────────────
    orig_rhos  = [r["mean_orig_rho"]  for r in summary_rows]
    probe_rhos = [r["mean_probe_rho"] for r in summary_rows]
    rho_corr   = float(np.corrcoef(orig_rhos, probe_rhos)[0, 1])
    rho_mad    = float(np.mean(np.abs(np.array(orig_rhos) - np.array(probe_rhos))))

    # ── Write CSVs ────────────────────────────────────────────────────────
    summary_path = os.path.join(cfg["out_dir"], cfg["out_csv"])
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary: {summary_path}")

    raw_path = os.path.join(cfg["out_dir"], cfg["out_csv"].replace(".csv", "_raw.csv"))
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()))
        writer.writeheader()
        writer.writerows(raw_results)
    print(f"Saved raw:     {raw_path}")

    # ── Summary statistics ────────────────────────────────────────────────
    total_cells  = len(summary_rows)
    n_flips      = len(flip_cells)
    total_seeds  = sum(r["n_seeds"] for r in summary_rows)
    seed_flips   = sum(r["seeds_flipped"] for r in summary_rows)
    pos_dL       = sum(1 for r in summary_rows if r["mean_delta_L"] > 0)
    elapsed      = time.time() - t0

    print(f"\n=== SUMMARY (frozen-Q leader probe) ===")
    print(f"Total cells:             {total_cells}")
    print(f"Majority flips:          {n_flips} ({n_flips/total_cells*100:.1f}%)")
    print(f"Stable cells:            {total_cells - n_flips}")
    print(f"Seed-level flips:        {seed_flips} of {total_seeds} runs")
    print(f"Mean |delta_L|:          {np.mean([abs(r['mean_delta_L']) for r in summary_rows]):.2f}")
    print(f"Max  |delta_L|:          {np.max([abs(r['mean_delta_L'])  for r in summary_rows]):.2f}")
    print(f"Positive delta_L cells:  {pos_dL} of {total_cells}")
    print(f"Rho correlation:         {rho_corr:.4f}")
    print(f"Rho mean abs diff:       {rho_mad:.4f}")
    print(f"Elapsed:                 {elapsed/60:.1f} min")

    if flip_cells:
        print(f"\n=== FLIPPED CELLS ===")
        for tau, rho, orig, probe in flip_cells:
            print(f"  tau={tau:.3f} rho={rho:.4f}  {orig} -> {probe}")

    # ── Comparison with modal-tau probe (if CSV exists) ───────────────────
    modal_csv = "pd2_br_probe_results.csv"
    if os.path.exists(modal_csv):
        print(f"\n=== COMPARISON WITH MODAL-TAU PROBE ===")
        modal = {}
        with open(modal_csv) as f:
            for row in csv.DictReader(f):
                modal[(float(row["tau_max"]), float(row["rho_max"]))] = row
        agree = sum(
            1 for r in summary_rows
            if modal.get((r["tau_max"], r["rho_max"]), {}).get("probe_regime") == r["probe_regime"]
        )
        print(f"Probe regime agreement between frozen-Q and modal-tau: "
              f"{agree}/{total_cells} cells ({agree/total_cells*100:.1f}%)")

    # ── Decision rule ─────────────────────────────────────────────────────
    print(f"\n=== PAPER INCLUSION DECISION RULE ===")
    boundary_regimes = {"Transition", "Multiple"}
    boundary_flips = sum(
        1 for tau, rho, orig, probe in flip_cells
        if orig in boundary_regimes or probe in boundary_regimes
    )
    if n_flips == 0:
        print("CLEAN: 0 majority flips. Strong stability result. Include in IV-H.")
    elif n_flips <= 6 and boundary_flips == n_flips:
        print("ACCEPTABLE: All flips at boundary/Transition cells. "
              "Include sentence in IV-H.")
    else:
        interior_flips = n_flips - boundary_flips
        print(f"CAUTION: {n_flips} flips ({interior_flips} interior). "
              "Review before including in paper.")


if __name__ == "__main__":
    main()
