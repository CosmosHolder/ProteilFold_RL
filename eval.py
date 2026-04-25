"""
eval.py
ProteinFold-RL — Evaluation (v3)

What changed vs v2
------------------
- Ramachandran diagram saved to logs/ramachandran.png
    · Trained agent phi/psi plotted in blue
    · Random agent phi/psi plotted in grey
    · Helix region box drawn in red (reference)
    · Sheet region box drawn in green (reference)
    · Proves agent learns real protein physics, not random exploration
- phi/psi angles from best episode saved to logs/best_angles.csv
    · columns: step, residue, phi_deg, psi_deg, ss_type
- save_ramachandran() is a standalone function — called at end of evaluate()
- collect_angles() helper added — runs one episode and records all angles
- No matplotlib dependency added — uses only numpy + stdlib if matplotlib
  is absent (graceful fallback: prints warning, skips plot)
- All v2 logic (RMSD, energy, eval_results.json, trajectory CSV) unchanged

Run
---
  python eval.py                    # evaluate on 1L2Y (default)
  python eval.py --protein 1YRF    # evaluate on any registered protein
  python eval.py --episodes 30     # more episodes for better statistics
  python eval.py --no-plot         # skip Ramachandran plot
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from env.fold_env import FoldEnv, HELIX_PHI_CENTER, HELIX_PSI_CENTER, \
    HELIX_PHI_TOL, HELIX_PSI_TOL, SHEET_PHI_CENTER, SHEET_PSI_CENTER, \
    SHEET_PHI_TOL, SHEET_PSI_TOL
from model.gnn_policy import GNNPolicyNetwork
from agent.ppo import PPOTrainer
from config import MAX_ACTION_DIM, CHECKPOINT_PATH

# ── Config ────────────────────────────────────────────────────
CHECKPOINT    = CHECKPOINT_PATH
EVAL_EPISODES = 20

os.makedirs("logs", exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────

def compute_rmsd(coords: np.ndarray, native: np.ndarray) -> float:
    """Root-mean-square deviation of Cα coordinates."""
    diff = coords - native
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def run_episode(env: FoldEnv, policy: GNNPolicyNetwork,
                deterministic: bool = False) -> tuple:
    """
    Run one full episode with the trained policy.

    Returns
    -------
    trajectory : list of per-step dicts
    rmsd       : final RMSD vs native (Å)
    energy     : final energy (kcal/mol)
    """
    obs, info = env.reset()
    trajectory = []
    done = False

    while not done:
        graph = env.get_graph()
        with torch.no_grad():
            action, _, _, _ = policy.get_action(
                graph, deterministic=deterministic
            )
        action = action % env.action_dim

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        trajectory.append({
            "step"     : info["step"],
            "energy"   : info["energy"],
            "has_clash": info["has_clash"],
            "reward"   : reward,
            "coords"   : env.ca_coords.copy(),
            # Snapshot angles at this step
            "phi"      : env.phi_angles.copy(),
            "psi"      : env.psi_angles.copy(),
        })

    rmsd = compute_rmsd(env.ca_coords, env.native_coords)
    return trajectory, rmsd, info["energy"]


def run_random_episode(env: FoldEnv) -> tuple:
    """
    Run one full episode with a random agent.

    Returns
    -------
    rmsd    : float
    energy  : float
    phi_all : np.ndarray [N] final phi angles
    psi_all : np.ndarray [N] final psi angles
    """
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    rmsd = compute_rmsd(env.ca_coords, env.native_coords)
    return rmsd, info["energy"], env.phi_angles.copy(), env.psi_angles.copy()


# ── Ramachandran ──────────────────────────────────────────────

def collect_all_angles(trajectory: list) -> tuple:
    """
    Extract all phi/psi angles across every step and residue
    from a trajectory produced by run_episode().

    Returns
    -------
    phis : np.ndarray [steps × N]  in degrees
    psis : np.ndarray [steps × N]  in degrees
    """
    phis = np.degrees(np.vstack([t["phi"] for t in trajectory]))
    psis = np.degrees(np.vstack([t["psi"] for t in trajectory]))
    return phis.flatten(), psis.flatten()


def save_best_angles(trajectory: list, env: FoldEnv, path: str):
    """
    Save per-step, per-residue phi/psi angles from the best
    episode to a CSV file.

    Columns: step, residue, phi_deg, psi_deg, ss_type
    ss_type is classified using the same Ramachandran regions
    as fold_env.py.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "residue", "phi_deg", "psi_deg", "ss_type"])
        for t in trajectory:
            for res in range(env.N):
                phi = t["phi"][res]
                psi = t["psi"][res]
                ss  = env._detect_ss(phi, psi)
                writer.writerow([
                    t["step"],
                    res,
                    round(np.degrees(phi), 2),
                    round(np.degrees(psi), 2),
                    ss,
                ])


def save_ramachandran(trained_phis: np.ndarray,
                      trained_psis: np.ndarray,
                      random_phis:  np.ndarray,
                      random_psis:  np.ndarray,
                      path: str,
                      pdb_id: str = ""):
    """
    Generate and save a Ramachandran diagram comparing the trained
    agent's phi/psi distribution against the random baseline.

    Plot elements:
      - Grey scatter : random agent angles (background)
      - Blue scatter : trained agent angles (foreground)
      - Red box      : alpha-helix Ramachandran region
      - Green box    : beta-sheet Ramachandran region
      - Dashed lines : phi=0 and psi=0 axes

    Parameters
    ----------
    trained_phis : phi angles from trained agent (degrees)
    trained_psis : psi angles from trained agent (degrees)
    random_phis  : phi angles from random agent (degrees)
    random_psis  : psi angles from random agent (degrees)
    path         : output file path (PNG)
    pdb_id       : protein name for plot title
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping Ramachandran plot.")
        print("         Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#f8f8f8")

    # ── Random agent (grey, background) ──────────────────────
    ax.scatter(
        random_phis, random_psis,
        s=4, alpha=0.25, color="#aaaaaa", label="Random agent",
        zorder=1,
    )

    # ── Trained agent (blue, foreground) ─────────────────────
    ax.scatter(
        trained_phis, trained_psis,
        s=6, alpha=0.55, color="#1a6fc4", label="Trained agent",
        zorder=2,
    )

    # ── Helix region box (red) ────────────────────────────────
    h_phi_c = np.degrees(HELIX_PHI_CENTER)
    h_psi_c = np.degrees(HELIX_PSI_CENTER)
    h_phi_t = np.degrees(HELIX_PHI_TOL)
    h_psi_t = np.degrees(HELIX_PSI_TOL)

    helix_rect = mpatches.FancyBboxPatch(
        (h_phi_c - h_phi_t, h_psi_c - h_psi_t),
        2 * h_phi_t, 2 * h_psi_t,
        boxstyle="square,pad=0",
        linewidth=1.8, edgecolor="#cc2222",
        facecolor="#cc222222", zorder=3,
        label=f"α-helix region\n(φ={h_phi_c:.0f}°±{h_phi_t:.0f}°, "
              f"ψ={h_psi_c:.0f}°±{h_psi_t:.0f}°)",
    )
    ax.add_patch(helix_rect)

    # ── Sheet region box (green) ──────────────────────────────
    s_phi_c = np.degrees(SHEET_PHI_CENTER)
    s_psi_c = np.degrees(SHEET_PSI_CENTER)
    s_phi_t = np.degrees(SHEET_PHI_TOL)
    s_psi_t = np.degrees(SHEET_PSI_TOL)

    sheet_rect = mpatches.FancyBboxPatch(
        (s_phi_c - s_phi_t, s_psi_c - s_psi_t),
        2 * s_phi_t, 2 * s_psi_t,
        boxstyle="square,pad=0",
        linewidth=1.8, edgecolor="#229922",
        facecolor="#22992222", zorder=3,
        label=f"β-sheet region\n(φ={s_phi_c:.0f}°±{s_phi_t:.0f}°, "
              f"ψ={s_psi_c:.0f}°±{s_psi_t:.0f}°)",
    )
    ax.add_patch(sheet_rect)

    # ── Reference axes ────────────────────────────────────────
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)

    # ── Count how many trained angles fall inside regions ─────
    def count_in_region(phis, psis, phi_c, psi_c, phi_t, psi_t):
        return int(np.sum(
            (np.abs(phis - phi_c) <= phi_t) &
            (np.abs(psis - psi_c) <= psi_t)
        ))

    n_trained = len(trained_phis)
    n_random  = len(random_phis)

    h_trained = count_in_region(trained_phis, trained_psis,
                                 h_phi_c, h_psi_c, h_phi_t, h_psi_t)
    h_random  = count_in_region(random_phis, random_psis,
                                 h_phi_c, h_psi_c, h_phi_t, h_psi_t)
    s_trained = count_in_region(trained_phis, trained_psis,
                                 s_phi_c, s_psi_c, s_phi_t, s_psi_t)
    s_random  = count_in_region(random_phis, random_psis,
                                 s_phi_c, s_psi_c, s_phi_t, s_psi_t)

    pct = lambda n, total: 100 * n / (total + 1e-8)

    stats_text = (
        f"Trained: {pct(h_trained, n_trained):.1f}% helix, "
        f"{pct(s_trained, n_trained):.1f}% sheet\n"
        f"Random:  {pct(h_random,  n_random ):.1f}% helix, "
        f"{pct(s_random,  n_random ):.1f}% sheet"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=9, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    # ── Labels and formatting ─────────────────────────────────
    title = "Ramachandran Diagram — ProteinFold-RL"
    if pdb_id:
        title += f" ({pdb_id})"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("φ (phi) angle  [degrees]", fontsize=11)
    ax.set_ylabel("ψ (psi) angle  [degrees]", fontsize=11)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Ramachandran → {path}")
    print(f"    Trained: {pct(h_trained, n_trained):.1f}% helix | "
          f"{pct(s_trained, n_trained):.1f}% sheet")
    print(f"    Random : {pct(h_random,  n_random ):.1f}% helix | "
          f"{pct(s_random,  n_random ):.1f}% sheet")


# ── Main evaluation ───────────────────────────────────────────

def evaluate(pdb_id: str = "1L2Y",
             n_episodes: int = EVAL_EPISODES,
             make_plot: bool = True):
    """
    Full evaluation: trained agent vs random baseline.

    Saves:
      logs/best_trajectory.csv
      logs/best_angles.csv        (NEW — phi/psi per step per residue)
      logs/ramachandran.png       (NEW — Ramachandran diagram)
      logs/native_coords.npy
      logs/best_coords.npy
      logs/initial_coords.npy
      logs/eval_results.json
    """
    print("=" * 62)
    print("ProteinFold-RL — Evaluation v3")
    print(f"  Protein   : {pdb_id}")
    print(f"  Episodes  : {n_episodes} each (trained + random)")
    print(f"  Checkpoint: {CHECKPOINT}")
    print("=" * 62)

    # ── Setup ─────────────────────────────────────────────────
    if not os.path.exists(CHECKPOINT):
        print(f"\n[ERROR] No checkpoint found at {CHECKPOINT}")
        print("  Run train.py first.")
        sys.exit(1)

    env    = FoldEnv(pdb_id=pdb_id)
    policy = GNNPolicyNetwork(action_dim=MAX_ACTION_DIM)
    trainer = PPOTrainer(policy=policy, action_dim=MAX_ACTION_DIM)
    trainer.load(CHECKPOINT)
    policy.eval()

    # ── Random baseline ───────────────────────────────────────
    print(f"\n[BASELINE] Random agent ({n_episodes} episodes)...")
    random_rmsds, random_energies = [], []
    rand_phis_all, rand_psis_all  = [], []

    for _ in range(n_episodes):
        rmsd, energy, r_phi, r_psi = run_random_episode(env)
        random_rmsds.append(rmsd)
        random_energies.append(energy)
        rand_phis_all.append(np.degrees(r_phi))
        rand_psis_all.append(np.degrees(r_psi))

    r_rmsd_mean   = float(np.mean(random_rmsds))
    r_energy_mean = float(np.mean(random_energies))
    print(f"  Avg RMSD   : {r_rmsd_mean:.3f} Å")
    print(f"  Avg Energy : {r_energy_mean:.3f} kcal/mol")

    # Stack random angles: [n_episodes × N] → flat
    rand_phis_flat = np.concatenate(rand_phis_all)
    rand_psis_flat = np.concatenate(rand_psis_all)

    # ── Trained agent ─────────────────────────────────────────
    print(f"\n[TRAINED] Policy agent ({n_episodes} episodes)...")
    policy_rmsds, policy_energies = [], []
    best_rmsd = float("inf")
    best_traj = None
    trained_phis_all, trained_psis_all = [], []

    for ep in range(n_episodes):
        traj, rmsd, energy = run_episode(env, policy, deterministic=False)
        policy_rmsds.append(rmsd)
        policy_energies.append(energy)

        # Collect angles from this episode
        ep_phis, ep_psis = collect_all_angles(traj)
        trained_phis_all.append(ep_phis)
        trained_psis_all.append(ep_psis)

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_traj = traj

    p_rmsd_mean   = float(np.mean(policy_rmsds))
    p_energy_mean = float(np.mean(policy_energies))
    print(f"  Avg RMSD   : {p_rmsd_mean:.3f} Å")
    print(f"  Avg Energy : {p_energy_mean:.3f} kcal/mol")
    print(f"  Best RMSD  : {best_rmsd:.3f} Å")

    # Stack trained angles: [n_episodes × steps × N] → flat
    trained_phis_flat = np.concatenate(trained_phis_all)
    trained_psis_flat = np.concatenate(trained_psis_all)

    # ── Comparison ────────────────────────────────────────────
    print("\n[COMPARISON]")
    rmsd_imp   = r_rmsd_mean   - p_rmsd_mean
    energy_imp = r_energy_mean - p_energy_mean
    rmsd_pct   = 100 * rmsd_imp   / (r_rmsd_mean   + 1e-8)
    energy_pct = 100 * energy_imp / (r_energy_mean + 1e-8)

    print(f"  RMSD improvement   : {rmsd_imp:+.3f} Å  ({rmsd_pct:+.1f}%)")
    print(f"  Energy improvement : {energy_imp:+.3f} kcal/mol  ({energy_pct:+.1f}%)")

    passed = p_energy_mean < r_energy_mean
    if passed:
        print("  [PASS] Trained agent outperforms random baseline ✅")
    else:
        print("  [WARN] Trained agent did not outperform random.")
        print("         Try training for more episodes.")

    # ── Save best trajectory CSV ──────────────────────────────
    traj_path = "logs/best_trajectory.csv"
    with open(traj_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "energy", "reward", "has_clash"])
        for t in best_traj:
            writer.writerow([
                t["step"],
                round(t["energy"], 4),
                round(t["reward"], 4),
                int(t["has_clash"]),
            ])
    print(f"\n  Best trajectory    → {traj_path}")

    # ── Save best angles CSV (NEW) ────────────────────────────
    angles_path = "logs/best_angles.csv"
    save_best_angles(best_traj, env, angles_path)
    print(f"  Best angles        → {angles_path}")

    # ── Save coords ───────────────────────────────────────────
    np.save("logs/native_coords.npy",  env.native_coords)
    np.save("logs/best_coords.npy",    best_traj[-1]["coords"])
    np.save("logs/initial_coords.npy", best_traj[0]["coords"])
    print(f"  Coords             → logs/")

    # ── Ramachandran diagram (NEW) ────────────────────────────
    if make_plot:
        print(f"\n[RAMACHANDRAN] Generating diagram...")
        save_ramachandran(
            trained_phis=trained_phis_flat,
            trained_psis=trained_psis_flat,
            random_phis=rand_phis_flat,
            random_psis=rand_psis_flat,
            path="logs/ramachandran.png",
            pdb_id=pdb_id,
        )

    # ── Save eval_results.json ────────────────────────────────
    results = {
        "protein"          : pdb_id,
        "n_episodes"       : n_episodes,
        "random": {
            "avg_rmsd"     : round(r_rmsd_mean,   3),
            "avg_energy"   : round(r_energy_mean, 3),
        },
        "trained": {
            "avg_rmsd"     : round(p_rmsd_mean,   3),
            "avg_energy"   : round(p_energy_mean, 3),
            "best_rmsd"    : round(best_rmsd,      3),
        },
        "improvement": {
            "rmsd_abs"     : round(rmsd_imp,   3),
            "rmsd_pct"     : round(rmsd_pct,   1),
            "energy_abs"   : round(energy_imp, 3),
            "energy_pct"   : round(energy_pct, 1),
        },
        "passed"           : passed,
    }

    results_path = "logs/eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Eval results       → {results_path}")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "=" * 62)
    print("Evaluation Complete")
    print(f"  Random  → RMSD {r_rmsd_mean:.3f} Å | Energy {r_energy_mean:.3f}")
    print(f"  Trained → RMSD {p_rmsd_mean:.3f} Å | Energy {p_energy_mean:.3f}")
    print(f"  Improvement: {rmsd_imp:+.3f} Å  |  {energy_imp:+.3f} kcal/mol")
    print("=" * 62)

    return results


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protein", type=str, default="1L2Y",
        help="PDB ID to evaluate on (default: 1L2Y)"
    )
    parser.add_argument(
        "--episodes", type=int, default=EVAL_EPISODES,
        help=f"Episodes per agent (default: {EVAL_EPISODES})"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip Ramachandran diagram generation"
    )
    args = parser.parse_args()
    evaluate(
        pdb_id=args.protein,
        n_episodes=args.episodes,
        make_plot=not args.no_plot,
    )