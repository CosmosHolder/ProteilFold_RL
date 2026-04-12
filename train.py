import torch
import numpy as np
import os
import csv
import time
from env.fold_env import FoldEnv
from model.gnn_policy import GNNPolicyNetwork
from agent.ppo import PPOTrainer, HORIZON

# ── Training config ──────────────────────────────────────────
N_EPISODES      = 500
SAVE_EVERY      = 50
LOG_EVERY       = 10
CHECKPOINT_DIR  = "checkpoints"
LOG_FILE        = "logs/training_log.csv"
PDB_ID          = "1L2Y"   # Start with Trp-cage

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


def compute_rmsd(coords: np.ndarray, native: np.ndarray) -> float:
    diff = coords - native
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def train():
    print("=" * 60)
    print("ProteinFold-RL — Training Started")
    print(f"Protein : {PDB_ID} | Episodes: {N_EPISODES}")
    print("=" * 60)

    # ── Setup ────────────────────────────────────────────────
    env     = FoldEnv(pdb_id=PDB_ID)
    policy  = GNNPolicyNetwork(action_dim=env.action_dim)
    trainer = PPOTrainer(policy=policy, action_dim=env.action_dim)

    native_coords = env.native_coords.copy()

    # ── CSV logger ───────────────────────────────────────────
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "total_reward", "final_energy",
            "rmsd", "steps", "clashes",
            "policy_loss", "value_loss", "entropy"
        ])

    best_rmsd   = float("inf")
    best_energy = float("inf")
    step_count  = 0

    for episode in range(1, N_EPISODES + 1):
        obs, info   = env.reset()
        ep_reward   = 0.0
        ep_steps    = 0
        ep_clashes  = 0
        done        = False
        stats       = {}

        while not done:
            graph = env.get_graph()
            action, log_prob, value, entropy = policy.get_action(graph)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            trainer.store(graph, action, reward, log_prob, value, done)
            ep_reward  += reward
            ep_steps   += 1
            ep_clashes += int(info["has_clash"])
            step_count += 1

            # PPO update every HORIZON steps
            if step_count % HORIZON == 0:
                last_val = 0.0
                if not done:
                    with torch.no_grad():
                        _, last_val_t = policy(env.get_graph())
                        last_val = last_val_t.item()
                stats = trainer.update(last_val)

        # ── Episode metrics ──────────────────────────────────
        final_energy = info["energy"]
        rmsd         = compute_rmsd(env.ca_coords, native_coords)

        if rmsd < best_rmsd:
            best_rmsd = rmsd
        if final_energy < best_energy:
            best_energy = final_energy

        # ── Log to CSV ───────────────────────────────────────
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                round(ep_reward, 4),
                round(final_energy, 4),
                round(rmsd, 4),
                ep_steps,
                ep_clashes,
                round(stats.get("policy_loss", 0), 4),
                round(stats.get("value_loss",  0), 4),
                round(stats.get("entropy",     0), 4),
            ])

        # ── Console log ──────────────────────────────────────
        if episode % LOG_EVERY == 0:
            print(
                f"Ep {episode:4d}/{N_EPISODES} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Energy: {final_energy:7.3f} | "
                f"RMSD: {rmsd:.3f}Å | "
                f"Steps: {ep_steps:3d} | "
                f"Clashes: {ep_clashes}"
            )

        # ── Save checkpoint ──────────────────────────────────
        if episode % SAVE_EVERY == 0:
            path = os.path.join(
                CHECKPOINT_DIR, f"policy_ep{episode}.pt"
            )
            trainer.save(path)

    # ── Final summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Best RMSD   : {best_rmsd:.3f} Å")
    print(f"  Best Energy : {best_energy:.3f} kcal/mol")
    print(f"  Log saved   : {LOG_FILE}")
    print("=" * 60)

    # Save final model
    trainer.save(os.path.join(CHECKPOINT_DIR, "policy_final.pt"))
    return trainer


if __name__ == "__main__":
    train()