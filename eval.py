import torch
import numpy as np
import os
import sys
import csv
sys.path.insert(0, os.path.dirname(__file__))

from env.fold_env import FoldEnv
from model.gnn_policy import GNNPolicyNetwork
from agent.ppo import PPOTrainer

CHECKPOINT = "checkpoints/policy_final.pt"
EVAL_EPISODES = 20


def compute_rmsd(coords, native):
    diff = coords - native
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def run_episode(env, policy, deterministic=False):
    """Run one full episode, return trajectory data."""
    obs, info = env.reset()
    trajectory = []
    done = False

    while not done:
        graph = env.get_graph()
        with torch.no_grad():
            action, log_prob, value, entropy = policy.get_action(
                graph, deterministic=deterministic
            )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        trajectory.append({
            "step"       : info["step"],
            "energy"     : info["energy"],
            "has_clash"  : info["has_clash"],
            "reward"     : reward,
            "coords"     : env.ca_coords.copy(),
        })

    rmsd = compute_rmsd(env.ca_coords, env.native_coords)
    return trajectory, rmsd, info["energy"]


def evaluate():
    print("=" * 60)
    print("ProteinFold-RL — Evaluation")
    print("=" * 60)

    env    = FoldEnv(pdb_id="1L2Y")
    policy = GNNPolicyNetwork(action_dim=env.action_dim)
    trainer= PPOTrainer(policy=policy, action_dim=env.action_dim)
    trainer.load(CHECKPOINT)
    policy.eval()

    # ── Random baseline ──────────────────────────────────────
    print("\n[BASELINE] Random agent (20 episodes)...")
    random_rmsds, random_energies = [], []
    for _ in range(EVAL_EPISODES):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        random_rmsds.append(compute_rmsd(env.ca_coords, env.native_coords))
        random_energies.append(info["energy"])

    print(f"  Avg RMSD   : {np.mean(random_rmsds):.3f} Å")
    print(f"  Avg Energy : {np.mean(random_energies):.3f} kcal/mol")

    # ── Trained agent ────────────────────────────────────────
    print("\n[TRAINED] Policy agent (20 episodes)...")
    policy_rmsds, policy_energies, best_traj = [], [], None
    best_rmsd = float("inf")

    for ep in range(EVAL_EPISODES):
        traj, rmsd, energy = run_episode(env, policy, deterministic=False)
        policy_rmsds.append(rmsd)
        policy_energies.append(energy)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_traj = traj

    print(f"  Avg RMSD   : {np.mean(policy_rmsds):.3f} Å")
    print(f"  Avg Energy : {np.mean(policy_energies):.3f} kcal/mol")
    print(f"  Best RMSD  : {best_rmsd:.3f} Å")

    # ── Comparison ───────────────────────────────────────────
    print("\n[COMPARISON]")
    rmsd_improvement = np.mean(random_rmsds) - np.mean(policy_rmsds)
    energy_improvement = np.mean(random_energies) - np.mean(policy_energies)
    print(f"  RMSD improvement   : {rmsd_improvement:+.3f} Å")
    print(f"  Energy improvement : {energy_improvement:+.3f} kcal/mol")
    assert np.mean(policy_energies) < np.mean(random_energies), \
        "Trained agent should outperform random!"
    print(f"  [PASS] Trained agent outperforms random baseline ✓")

    # ── Save best trajectory ─────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    traj_path = "logs/best_trajectory.csv"
    with open(traj_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "energy", "reward", "has_clash"])
        for t in best_traj:
            writer.writerow([
                t["step"], round(t["energy"], 4),
                round(t["reward"], 4), int(t["has_clash"])
            ])
    print(f"\n  Best trajectory saved to {traj_path}")

    # ── Save before/after coords ─────────────────────────────
    np.save("logs/native_coords.npy",  env.native_coords)
    np.save("logs/best_coords.npy",    best_traj[-1]["coords"])
    np.save("logs/initial_coords.npy", best_traj[0]["coords"])
    print(f"  Coords saved to logs/")

    print("\n" + "=" * 60)
    print("CHECKPOINT-06 — Proof of learning confirmed.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()