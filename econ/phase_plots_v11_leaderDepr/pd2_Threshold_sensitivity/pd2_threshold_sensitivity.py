# pd2_threshold_sensitivity.py
# ============================================================
# Threshold Sensitivity Analysis — proper seed-level procedure
# ============================================================
# Implements the correct classification pipeline matching the paper:
#   1. For each (rho_max, tau_max) cell and each seed, classify
#      the seed's mean evaluation payoff at thresholds ±10, ±15, ±20
#   2. Recompute majority vote per cell at each threshold
#   3. Report how many majority classifications differ from ±15 baseline
#   4. Check whether any non-Transition interior cells change
#
# Requires (same directory):
#   fig_pd2_retaliation_vs_tariff_per_seed.csv
#   — produced by v11_pd_ieee_leaderDepr_PD.py alongside the mean CSV
#   — columns: x_val, y_val, seed, mean_lpay
#
# Output:
#   pd2_threshold_sensitivity_results.csv
# ============================================================

import csv
import numpy as np
from collections import Counter, defaultdict

# ── Input filename (matches v11 output naming) ────────────────────────────
INPUT_CSV = "fig_pd2_retaliation_vs_tariff_per_seed.csv"

THRESHOLDS = [
    ("pm10",  "+/-10",  10.0, -10.0),
    ("pm15",  "+/-15",  15.0, -15.0),   # baseline
    ("pm20",  "+/-20",  20.0, -20.0),
]

def classify(lpay, hi, lo):
    if lpay > hi:  return "Deterrence"
    if lpay < lo:  return "Escalation"
    return "Transition"

def majority_regime(regimes):
    counts = Counter(regimes)
    top, top_count = counts.most_common(1)[0]
    return top if top_count > len(regimes) / 2 else "Multiple"

# ── Load per-seed payoffs ─────────────────────────────────────────────────
per_seed = defaultdict(list)
with open(INPUT_CSV) as f:
    for r in csv.DictReader(f):
        # x_val = rho_max, y_val = tau_max (matches PD2 axis convention)
        key = (round(float(r["x_val"]), 4), round(float(r["y_val"]), 4))
        per_seed[key].append(float(r["mean_lpay"]))

n_cells = len(per_seed)
n_seeds = len(next(iter(per_seed.values())))
print(f"Loaded {n_cells} cells, {n_seeds} seeds each ({n_cells*n_seeds} total runs)")

# ── Classify at each threshold ────────────────────────────────────────────
results = {}
for key_label, display_label, hi, lo in THRESHOLDS:
    cell_regimes = {}
    for cell_key, payoffs in per_seed.items():
        seed_regimes = [classify(lp, hi, lo) for lp in payoffs]
        cell_regimes[cell_key] = majority_regime(seed_regimes)
    results[key_label] = cell_regimes

baseline = results["pm15"]

# ── Agreement analysis ────────────────────────────────────────────────────
print("\n=== AGREEMENT WITH +/-15 BASELINE ===")
for key_label, display_label, hi, lo in THRESHOLDS:
    agree = sum(1 for k in baseline if results[key_label][k] == baseline[k])
    print(f"  {display_label}: {agree}/{n_cells} cells agree ({agree/n_cells*100:.1f}%)")

# ── Cells that change ─────────────────────────────────────────────────────
print("\n=== CELLS THAT DIFFER FROM +/-15 BASELINE ===")
for key_label, display_label, hi, lo in THRESHOLDS:
    if key_label == "pm15":
        continue
    changed = [(k, baseline[k], results[key_label][k])
               for k in baseline if results[key_label][k] != baseline[k]]
    print(f"\n  Under {display_label} ({len(changed)} changes):")
    for cell_key, orig, new_r in sorted(changed):
        print(f"    rho={cell_key[0]:.4f}  tau={cell_key[1]:.4f}  "
              f"+/-15={orig:12}  {display_label}={new_r:12}")

# ── Regime counts ─────────────────────────────────────────────────────────
print("\n=== REGIME COUNTS AT EACH THRESHOLD ===")
print(f"{'Regime':>14} {'±10':>8} {'±15':>8} {'±20':>8}")
for regime in ["Deterrence", "Transition", "Escalation", "Multiple"]:
    c10 = sum(1 for v in results["pm10"].values() if v == regime)
    c15 = sum(1 for v in results["pm15"].values() if v == regime)
    c20 = sum(1 for v in results["pm20"].values() if v == regime)
    print(f"  {regime:>12} {c10:>8} {c15:>8} {c20:>8}")

# ── Interior cell stability check ────────────────────────────────────────
print("\n=== INTERIOR CELL STABILITY ===")
print("(interior = not classified as Transition under ±15 baseline)")
transition_keys = {k for k, v in baseline.items() if v == "Transition"}
multiple_keys   = {k for k, v in baseline.items() if v == "Multiple"}
boundary_keys   = transition_keys | multiple_keys

for key_label, display_label, hi, lo in THRESHOLDS:
    if key_label == "pm15":
        continue
    interior_flips = [
        k for k in baseline
        if results[key_label][k] != baseline[k]
        and k not in boundary_keys
    ]
    all_boundary = len(interior_flips) == 0
    print(f"\n  {display_label}: {len(interior_flips)} non-boundary cells changed "
          f"{'— all changes confined to boundary' if all_boundary else ''}")
    for k in interior_flips:
        print(f"    rho={k[0]:.4f}  tau={k[1]:.4f}  "
              f"+/-15={baseline[k]}  {display_label}={results[key_label][k]}")

# ── Write output CSV ──────────────────────────────────────────────────────
rows = []
for cell_key in sorted(per_seed.keys()):
    row = {
        "rho_max":      cell_key[0],
        "tau_max":      cell_key[1],
        "regime_pm10":  results["pm10"][cell_key],
        "regime_pm15":  results["pm15"][cell_key],
        "regime_pm20":  results["pm20"][cell_key],
        "agree_pm10":   int(results["pm10"][cell_key] == baseline[cell_key]),
        "agree_pm20":   int(results["pm20"][cell_key] == baseline[cell_key]),
        "changed_pm10": int(results["pm10"][cell_key] != baseline[cell_key]),
        "changed_pm20": int(results["pm20"][cell_key] != baseline[cell_key]),
        "is_boundary":  int(cell_key in boundary_keys),
    }
    rows.append(row)

with open("pd2_threshold_sensitivity_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("\nSaved: pd2_threshold_sensitivity_results.csv")

# ── Paper sentence — generated from actual data ───────────────────────────
agree_10 = sum(1 for k in baseline if results["pm10"][k] == baseline[k])
agree_20 = sum(1 for k in baseline if results["pm20"][k] == baseline[k])

changed_10 = [k for k in baseline if results["pm10"][k] != baseline[k]]
changed_20 = [k for k in baseline if results["pm20"][k] != baseline[k]]
all_boundary_10 = all(k in boundary_keys for k in changed_10)
all_boundary_20 = all(k in boundary_keys for k in changed_20)
boundary_qualifier = (
    "All cells that changed were in the Transition band adjacent to the regime boundary."
    if (all_boundary_10 and all_boundary_20)
    else "Note: some changes occurred outside the Transition band — review before using this qualifier."
)

print(f"\n=== PAPER SENTENCE ===")
print(
    f"As a robustness check, each seed was reclassified individually at "
    f"thresholds of $\\pm10$ and $\\pm20$, and majority votes were recomputed "
    f"per cell. Under $\\pm10$, {agree_10} of {n_cells} cells retained their "
    f"baseline majority classification; under $\\pm20$, {agree_20} of {n_cells} "
    f"did. {boundary_qualifier}"
)
