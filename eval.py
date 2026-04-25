"""
eval.py
ProteinFold-RL — Evaluation (v2)

What changed vs v1
------------------
- Policy loaded at MAX_ACTION_DIM (matches new train.py)
- Action clamped to protein's valid range (same as train.py)
- Saves a full eval_results.json for the dashboard to read
- Cleaner pass/fail output with improvement percentages
- Works with any protein in the registry, not just 1L2Y

Run
---
  python eval.py                    # evaluate on 1L2Y (default)
  python eval.py --protein 1YRF    # evaluate on any registered protein
  python eval.py --episodes 30     # more episodes for better statistics
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from env.fold_env import FoldEnv
from model.gnn_policy import GNNPolicyNetwork
from agent.ppo import PPOTrainer
from config import MAX_ACTION_DIM, CHECKPOINT_PATH

# ── Config ────────────────────────────────────────────────────
CHECKPOINT = CHECKPOINT_PATH
EVAL_EPISODES  = 20


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
    prev_energy = info["energy"]

    while not done:
        graph = env.get_graph()
        with torch.no_grad():
            action, _, _, _ = policy.get_action(
                graph, deterministic=deterministic
            )
        # Clamp to this protein's valid action range
        action = action % env.action_dim

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_rmsd     = compute_rmsd(env.ca_coords, env.native_coords)
        energy_delta  = round(info["energy"] - prev_energy, 4)
        prev_energy   = info["energy"]

        trajectory.append({
            "step"        : info["step"],
            "energy"      : info["energy"],
            "energy_delta": energy_delta,
            "rmsd"        : round(step_rmsd, 4),
            "has_clash"   : info["has_clash"],
            "reward"      : reward,
            "coords"      : env.ca_coords.copy(),
        })

    rmsd = compute_rmsd(env.ca_coords, env.native_coords)
    return trajectory, rmsd, info["energy"]


def run_random_episode(env: FoldEnv) -> tuple:
    """Run one full episode with a random agent."""
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    rmsd = compute_rmsd(env.ca_coords, env.native_coords)
    return rmsd, info["energy"]


# ── Main evaluation ───────────────────────────────────────────

def evaluate(pdb_id: str = "1L2Y", n_episodes: int = EVAL_EPISODES):
    print("=" * 62)
    print("ProteinFold-RL — Evaluation v2")
    print(f"  Protein  : {pdb_id}")
    print(f"  Episodes : {n_episodes} each (trained + random)")
    print(f"  Checkpoint: {CHECKPOINT}")
    print("=" * 62)

    # ── Setup ─────────────────────────────────────────────────
    if not os.path.exists(CHECKPOINT):
        print(f"\n[ERROR] No checkpoint found at {CHECKPOINT}")
        print("  Run train.py first.")
        sys.exit(1)

    env    = FoldEnv(pdb_id=pdb_id)
    policy = GNNPolicyNetwork(action_dim=MAX_ACTION_DIM)
    trainer= PPOTrainer(policy=policy, action_dim=MAX_ACTION_DIM)
    trainer.load(CHECKPOINT)
    policy.eval()

    # ── Random baseline ───────────────────────────────────────
    print(f"\n[BASELINE] Random agent ({n_episodes} episodes)...")
    random_rmsds, random_energies = [], []

    for _ in range(n_episodes):
        rmsd, energy = run_random_episode(env)
        random_rmsds.append(rmsd)
        random_energies.append(energy)

    r_rmsd_mean   = float(np.mean(random_rmsds))
    r_energy_mean = float(np.mean(random_energies))
    print(f"  Avg RMSD   : {r_rmsd_mean:.3f} Å")
    print(f"  Avg Energy : {r_energy_mean:.3f} kcal/mol")

    # ── Trained agent ─────────────────────────────────────────
    print(f"\n[TRAINED] Policy agent ({n_episodes} episodes)...")
    policy_rmsds, policy_energies = [], []
    best_rmsd = float("inf")
    best_traj = None

    for ep in range(n_episodes):
        traj, rmsd, energy = run_episode(env, policy, deterministic=False)
        policy_rmsds.append(rmsd)
        policy_energies.append(energy)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_traj = traj

    p_rmsd_mean   = float(np.mean(policy_rmsds))
    p_energy_mean = float(np.mean(policy_energies))
    print(f"  Avg RMSD   : {p_rmsd_mean:.3f} Å")
    print(f"  Avg Energy : {p_energy_mean:.3f} kcal/mol")
    print(f"  Best RMSD  : {best_rmsd:.3f} Å")

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

    # ── Save best trajectory ──────────────────────────────────
    traj_path = "logs/best_trajectory.csv"
    with open(traj_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "energy", "energy_delta", "rmsd", "reward", "has_clash"])
        for t in best_traj:
            writer.writerow([
                t["step"],
                round(t["energy"],       4),
                round(t["energy_delta"], 4),
                round(t["rmsd"],         4),
                round(t["reward"],       4),
                int(t["has_clash"]),
            ])
    print(f"\n  Best trajectory → {traj_path}")

    # ── Save coords ───────────────────────────────────────────
    np.save("logs/native_coords.npy",  env.native_coords)
    np.save("logs/best_coords.npy",    best_traj[-1]["coords"])
    np.save("logs/initial_coords.npy", best_traj[0]["coords"])
    print(f"  Coords → logs/")

    # ── Save eval_results.json (dashboard reads this) ─────────
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
    print(f"  Eval results → {results_path}")

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
    args = parser.parse_args()
    evaluate(pdb_id=args.protein, n_episodes=args.episodes)