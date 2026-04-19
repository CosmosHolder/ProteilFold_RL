"""
csv_to_json.py
ProteinFold-RL — Convert training log CSV to JSON

Run this once after training completes:
    python csv_to_json.py

Input  : logs/training_log.csv
Output : frontend/assets/data/training_log.json

The JSON is fetched by charts.js to populate the dashboard
with real training data instead of synthetic fallback.
"""

import csv
import json
import os
import sys

INPUT_CSV  = os.path.join("logs", "training_log.csv")
OUTPUT_JSON = os.path.join("frontend", "assets", "data", "training_log.json")


def convert():
    # ── Validate input ────────────────────────────────────────
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input file not found: {INPUT_CSV}")
        print("  Run train.py first to generate the training log.")
        sys.exit(1)

    if os.path.getsize(INPUT_CSV) == 0:
        print(f"[ERROR] Input file is empty: {INPUT_CSV}")
        print("  Training may not have completed successfully.")
        sys.exit(1)

    # ── Read CSV ──────────────────────────────────────────────
    rows = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "episode"     : int(row["episode"]),
                "protein"     : row.get("protein", "1L2Y"),
                "stage"       : int(row.get("stage", 1)),
                "total_reward": float(row["total_reward"]),
                "final_energy": float(row["final_energy"]),
                "rmsd"        : float(row["rmsd"]),
                "steps"       : int(row["steps"]),
                "clashes"     : int(row["clashes"]),
                "policy_loss" : float(row.get("policy_loss", 0)),
                "value_loss"  : float(row.get("value_loss",  0)),
                "entropy"     : float(row.get("entropy",     0)),
            })

    if len(rows) == 0:
        print(f"[ERROR] CSV has no data rows.")
        sys.exit(1)

    # ── Write JSON ────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # ── Summary ───────────────────────────────────────────────
    energies = [r["final_energy"] for r in rows]
    rmsds    = [r["rmsd"]         for r in rows]

    print("=" * 55)
    print("ProteinFold-RL — CSV → JSON Conversion Complete")
    print("=" * 55)
    print(f"  Episodes converted : {len(rows)}")
    print(f"  Input              : {INPUT_CSV}")
    print(f"  Output             : {OUTPUT_JSON}")
    print(f"  Energy range       : {min(energies):.2f} → {max(energies):.2f} kcal/mol")
    print(f"  RMSD range         : {min(rmsds):.3f} → {max(rmsds):.3f} Å")
    print("=" * 55)
    print("  Dashboard will now load real training data ✓")


if __name__ == "__main__":
    convert()