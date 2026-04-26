"""
analysis/training_report.py
ProteinFold-RL — Training Report Generator

Reads logs/training_log.csv and produces:
  1. A clean text summary printed to console
  2. logs/training_report.txt  — saved text report
  3. logs/training_curves.png  — 4-panel plot (energy, RMSD, reward, entropy)

One job: turn raw training numbers into proof of learning.

Run : python analysis/training_report.py
this is the running mechanism
"""

import os
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Config ────────────────────────────────────────────────────
LOG_FILE      = "logs/training_log.csv"
REPORT_FILE   = "logs/training_report.txt"
CURVES_FILE   = "logs/training_curves.png"
SMOOTH_WINDOW = 20   # rolling average window for plots

os.makedirs("logs", exist_ok=True)


# ── Data loading ──────────────────────────────────────────────

def load_log(path: str) -> dict:
    """
    Load training_log.csv into a dict of lists.

    Returns
    -------
    dict with keys: episode, protein, stage, total_reward,
    final_energy, rmsd, steps, clashes,
    policy_loss, value_loss, entropy
    """
    if not os.path.exists(path):
        print(f"[ERROR] Log file not found: {path}")
        print("  Run train.py first.")
        sys.exit(1)

    data = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["episode"].append(int(row["episode"]))
            data["protein"].append(row["protein"])
            data["stage"].append(int(row["stage"]))
            data["total_reward"].append(float(row["total_reward"]))
            data["final_energy"].append(float(row["final_energy"]))
            data["rmsd"].append(float(row["rmsd"]))
            data["steps"].append(int(row["steps"]))
            data["clashes"].append(int(row["clashes"]))
            data["policy_loss"].append(float(row["policy_loss"]))
            data["value_loss"].append(float(row["value_loss"]))
            data["entropy"].append(float(row["entropy"]))

    return dict(data)


# ── Statistics ────────────────────────────────────────────────

def rolling_mean(values: list, window: int) -> list:
    """Compute rolling mean with given window size."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start:i + 1])))
    return result


def compute_stats(data: dict) -> dict:
    """Compute all summary statistics from loaded log data."""
    n          = len(data["episode"])
    energies   = data["final_energy"]
    rmsds      = data["rmsd"]
    rewards    = data["total_reward"]
    proteins   = data["protein"]

    # First vs last 10% of training
    tenth = max(1, n // 10)
    early_energy = energies[:tenth]
    late_energy  = energies[-tenth:]
    early_rmsd   = rmsds[:tenth]
    late_rmsd    = rmsds[-tenth:]
    early_reward = rewards[:tenth]
    late_reward  = rewards[-tenth:]

    # Per-protein episode counts
    protein_counts = defaultdict(int)
    for p in proteins:
        protein_counts[p] += 1

    # Best values
    best_energy_ep = int(np.argmin(energies)) + 1
    best_rmsd_ep   = int(np.argmin(rmsds))    + 1

    return {
        "n_episodes"         : n,
        "proteins_trained"   : sorted(set(proteins)),
        "protein_counts"     : dict(protein_counts),

        "best_energy"        : float(min(energies)),
        "best_energy_ep"     : best_energy_ep,
        "best_rmsd"          : float(min(rmsds)),
        "best_rmsd_ep"       : best_rmsd_ep,

        "early_energy_mean"  : float(np.mean(early_energy)),
        "late_energy_mean"   : float(np.mean(late_energy)),
        "energy_improvement" : float(np.mean(early_energy) - np.mean(late_energy)),
        "energy_pct"         : float(100 * (np.mean(early_energy) -
                                            np.mean(late_energy)) /
                                     (abs(np.mean(early_energy)) + 1e-8)),

        "early_rmsd_mean"    : float(np.mean(early_rmsd)),
        "late_rmsd_mean"     : float(np.mean(late_rmsd)),
        "rmsd_improvement"   : float(np.mean(early_rmsd) - np.mean(late_rmsd)),
        "rmsd_pct"           : float(100 * (np.mean(early_rmsd) -
                                            np.mean(late_rmsd)) /
                                     (np.mean(early_rmsd) + 1e-8)),

        "early_reward_mean"  : float(np.mean(early_reward)),
        "late_reward_mean"   : float(np.mean(late_reward)),
        "reward_improvement" : float(np.mean(late_reward) - np.mean(early_reward)),

        "avg_steps"          : float(np.mean(data["steps"])),
        "avg_clashes"        : float(np.mean(data["clashes"])),
        "avg_entropy"        : float(np.mean(
            [e for e in data["entropy"] if e != 0]
        )),
    }


# ── Text report ───────────────────────────────────────────────

def build_report(stats: dict) -> str:
    """Build the full text report string."""
    s = stats
    imp_sign = "✅" if s["energy_improvement"] > 0 else "⚠️"
    rmsd_sign = "✅" if s["rmsd_improvement"] > 0 else "⚠️"
    rew_sign  = "✅" if s["reward_improvement"] > 0 else "⚠️"

    lines = [
        "═" * 62,
        "  ProteinFold-RL — Training Report",
        "═" * 62,
        "",
        f"  Episodes trained   : {s['n_episodes']}",
        f"  Proteins           : {', '.join(s['proteins_trained'])}",
        "",
        "  ── Best Results ─────────────────────────────────────",
        f"  Best RMSD          : {s['best_rmsd']:.3f} Å"
        f"  (episode {s['best_rmsd_ep']})",
        f"  Best Energy        : {s['best_energy']:.3f} kcal/mol"
        f"  (episode {s['best_energy_ep']})",
        "",
        "  ── Learning Progress (early vs late 10%) ────────────",
        f"  Energy  : {s['early_energy_mean']:8.2f} → {s['late_energy_mean']:8.2f}"
        f"  ({s['energy_improvement']:+.2f} kcal/mol,"
        f" {s['energy_pct']:+.1f}%)  {imp_sign}",
        f"  RMSD    : {s['early_rmsd_mean']:8.3f} → {s['late_rmsd_mean']:8.3f} Å"
        f"  ({s['rmsd_improvement']:+.3f} Å,"
        f" {s['rmsd_pct']:+.1f}%)  {rmsd_sign}",
        f"  Reward  : {s['early_reward_mean']:8.2f} → {s['late_reward_mean']:8.2f}"
        f"  ({s['reward_improvement']:+.2f})  {rew_sign}",
        "",
        "  ── Episode Statistics ───────────────────────────────",
        f"  Avg steps/episode  : {s['avg_steps']:.1f}",
        f"  Avg clashes/episode: {s['avg_clashes']:.2f}",
        f"  Avg policy entropy : {s['avg_entropy']:.4f}",
        "",
        "  ── Per-Protein Episodes ─────────────────────────────",
    ]

    for pdb, count in sorted(s["protein_counts"].items()):
        lines.append(f"  {pdb:6} : {count:5d} episodes")

    lines += [
        "",
        "═" * 62,
    ]
    return "\n".join(lines)


# ── Plot ──────────────────────────────────────────────────────

def plot_curves(data: dict, output_path: str):
    """Generate 4-panel training curves plot."""
    episodes = data["episode"]
    n        = len(episodes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("ProteinFold-RL — Training Curves",
                 color="white", fontsize=14, y=1.01)

    panels = [
        ("final_energy", "Energy (kcal/mol)",    "#4fc3f7", "lower = better folding"),
        ("rmsd",         "RMSD (Å)",              "#81c784", "lower = closer to native"),
        ("total_reward", "Total Reward",           "#ffb74d", "higher = better agent"),
        ("entropy",      "Policy Entropy",         "#ce93d8", "exploration over time"),
    ]

    for ax, (key, ylabel, color, subtitle) in zip(axes.flat, panels):
        ax.set_facecolor("#0f0f1a")

        raw    = data[key]
        smooth = rolling_mean(raw, SMOOTH_WINDOW)

        # Raw values (faint)
        ax.plot(episodes, raw,
                color=color, alpha=0.15, linewidth=0.8, zorder=1)

        # Smoothed trend (bold)
        ax.plot(episodes, smooth,
                color=color, linewidth=2.0, zorder=2,
                label=f"Rolling mean (w={SMOOTH_WINDOW})")

        # Trend line
        if n > 10:
            z = np.polyfit(episodes, raw, 1)
            p = np.poly1d(z)
            trend_color = "#00e676" if (
                (key in ("final_energy", "rmsd") and z[0] < 0) or
                (key not in ("final_energy", "rmsd") and z[0] > 0)
            ) else "#ff7043"
            ax.plot(episodes, p(episodes),
                    "--", color=trend_color, linewidth=1.2,
                    alpha=0.7, zorder=3, label="Trend")

        ax.set_xlabel("Episode", color="white", fontsize=10)
        ax.set_ylabel(ylabel,    color="white", fontsize=10)
        ax.set_title(subtitle,   color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="white")
        ax.legend(fontsize=8, facecolor="#1a1a2e",
                  labelcolor="white", edgecolor="#333355")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────

def generate():
    print("=" * 62)
    print("ProteinFold-RL — Training Report Generator")
    print("=" * 62)

    # Load
    print(f"\n  Loading {LOG_FILE}...")
    data  = load_log(LOG_FILE)
    print(f"  {len(data['episode'])} episodes loaded.")

    # Stats
    stats  = compute_stats(data)
    report = build_report(stats)

    # Print to console
    print()
    print(report)

    # Save text report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report → {REPORT_FILE}")

    # Save curves plot
    print(f"\n  Generating training curves...")
    plot_curves(data, CURVES_FILE)

    print("\n" + "=" * 62)
    print("Report complete.")
    print("=" * 62)


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    generate()
