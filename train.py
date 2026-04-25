"""
train.py
ProteinFold-RL — Curriculum Training Loop (v3)

What changed vs v2
------------------
- PPOTrainer now receives total_steps so cosine scheduler is
  configured correctly for the full run duration
- CSV log gains two new columns: ss_reward, learning_rate
- append_log() records ss_reward per episode + current LR
- Console print shows LR every LOG_EVERY episodes
- init_log() updated header to match new columns
- All curriculum + checkpoint logic unchanged

Run
---
  python train.py              # full curriculum, 500 episodes
  python train.py --episodes 2000   # longer run
  python train.py --protein 1L2Y   # single-protein mode
"""

import argparse
import os
import csv
import time

import numpy as np
import torch

from env.fold_env import FoldEnv, MAX_STEPS
from model.gnn_policy import GNNPolicyNetwork
from agent.ppo import PPOTrainer, HORIZON
from data.curriculum import ProteinCurriculum
from config import MAX_ACTION_DIM, CHECKPOINT_PATH

# ── Config ────────────────────────────────────────────────────
N_EPISODES  = 500
LOG_EVERY   = 5
SAVE_EVERY  = 100

LOG_FILE    = "logs/training_log.csv"
CURR_STATE  = "checkpoints/curriculum_state.json"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────

def compute_rmsd(coords: np.ndarray, native: np.ndarray) -> float:
    """Root-mean-square deviation between two sets of Cα coordinates."""
    diff = coords - native
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def init_log():
    """Write CSV header — called once at the start of a fresh run."""
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "protein", "stage",
            "total_reward", "final_energy", "rmsd",
            "steps", "clashes",
            "policy_loss", "value_loss", "entropy",
            "ss_reward",   # NEW: cumulative secondary-structure reward
            "learning_rate",  # NEW: LR at end of episode
        ])


def append_log(episode, protein, stage, ep_reward,
               final_energy, rmsd, ep_steps, ep_clashes,
               stats, ep_ss_reward, current_lr):
    """Append one episode row to the CSV log."""
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            episode, protein, stage,
            round(ep_reward,     4),
            round(final_energy,  4),
            round(rmsd,          4),
            ep_steps,
            ep_clashes,
            round(stats.get("policy_loss", 0), 4),
            round(stats.get("value_loss",  0), 4),
            round(stats.get("entropy",     0), 4),
            round(ep_ss_reward,  4),
            f"{current_lr:.2e}",
        ])


# ── Main ──────────────────────────────────────────────────────

def train(n_episodes: int = N_EPISODES,
          single_protein: str = None):
    """
    Main training loop.

    Parameters
    ----------
    n_episodes     : total episodes to run
    single_protein : if set, skip curriculum and train on this
                     one protein only (v1 compatibility mode)
    """
    print("=" * 62)
    print("ProteinFold-RL — Curriculum Training v3")
    if single_protein:
        print(f"  Mode     : single-protein ({single_protein})")
    else:
        print(f"  Mode     : 8-protein curriculum")
    print(f"  Episodes : {n_episodes}")
    print("=" * 62)

    # ── Curriculum ────────────────────────────────────────────
    if single_protein:
        curriculum  = None
        current_pdb = single_protein
    else:
        curriculum  = ProteinCurriculum()
        current_pdb = curriculum.current_protein().pdb_id

    # ── Policy — built once at MAX action dim ─────────────────
    policy = GNNPolicyNetwork(action_dim=MAX_ACTION_DIM)

    # Total expected steps — used to set cosine scheduler T_max.
    # Conservative estimate: n_episodes × MAX_STEPS.
    total_steps = n_episodes * MAX_STEPS
    trainer = PPOTrainer(
        policy=policy,
        action_dim=MAX_ACTION_DIM,
        total_steps=total_steps,
    )

    # ── Environment — rebuilt when protein changes ────────────
    env           = FoldEnv(pdb_id=current_pdb)
    native_coords = env.native_coords.copy()

    # ── Logs ──────────────────────────────────────────────────
    init_log()

    # ── State ─────────────────────────────────────────────────
    best_rmsd   = float("inf")
    best_energy = float("inf")
    step_count  = 0
    start_time  = time.time()

    print(f"\n[START] First protein: {current_pdb}")
    print(f"  Initial LR : {trainer.get_lr():.2e}")
    print(f"  Total steps (scheduler T_max): {total_steps}\n")

    # ── Episode loop ──────────────────────────────────────────
    for episode in range(1, n_episodes + 1):

        # Curriculum selects protein for this episode
        if curriculum is not None:
            entry       = curriculum.sample_protein()
            episode_pdb = entry.pdb_id

            # Rebuild env only when protein changes
            if episode_pdb != current_pdb:
                env           = FoldEnv(pdb_id=episode_pdb)
                native_coords = env.native_coords.copy()
                current_pdb   = episode_pdb
        else:
            episode_pdb = single_protein

        # ── Run one episode ───────────────────────────────────
        obs, info  = env.reset()
        ep_reward  = 0.0
        ep_steps   = 0
        ep_clashes = 0
        ep_ss_reward = 0.0   # NEW: accumulated SS reward this episode
        done       = False
        stats      = {}

        while not done:
            graph = env.get_graph()

            action, log_prob, value, entropy = policy.get_action(graph)

            # Clamp action to this protein's valid range.
            action = action % env.action_dim

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            trainer.store(graph, action, reward, log_prob, value, done)
            ep_reward    += reward
            ep_steps     += 1
            ep_clashes   += int(info["has_clash"])
            ep_ss_reward += info.get("ss_reward", 0.0)  # NEW
            step_count   += 1

            # PPO update every HORIZON steps
            if step_count % HORIZON == 0:
                last_val = 0.0
                if not done:
                    with torch.no_grad():
                        _, last_val_t = policy(env.get_graph())
                        last_val = last_val_t.item()
                stats = trainer.update(last_val)

        # ── Episode metrics ───────────────────────────────────
        final_energy = info["energy"]
        rmsd         = compute_rmsd(env.ca_coords, native_coords)
        current_lr   = trainer.get_lr()

        if rmsd         < best_rmsd:   best_rmsd   = rmsd
        if final_energy < best_energy: best_energy = final_energy

        # ── Record in curriculum + check advancement ──────────
        stage = 1
        if curriculum is not None:
            curriculum.record(episode_pdb, rmsd)
            curriculum.maybe_advance()
            stage = curriculum.current_stage

        # ── Log ───────────────────────────────────────────────
        append_log(
            episode, episode_pdb, stage,
            ep_reward, final_energy, rmsd,
            ep_steps, ep_clashes, stats,
            ep_ss_reward, current_lr,      # NEW columns
        )

        # ── Console ───────────────────────────────────────────
        if episode % LOG_EVERY == 0:
            elapsed     = (time.time() - start_time) / 60
            curr_status = curriculum.status() if curriculum else episode_pdb
            print(
                f"Ep {episode:5d}/{n_episodes} | "
                f"Reward {ep_reward:7.2f} | "
                f"Energy {final_energy:7.2f} | "
                f"RMSD {rmsd:.3f}Å | "
                f"SS {ep_ss_reward:.1f} | "
                f"LR {current_lr:.1e} | "
                f"{curr_status} | "
                f"{elapsed:.1f}m"
            )

        # ── Checkpoint ────────────────────────────────────────
        if episode % SAVE_EVERY == 0:
            ckpt = os.path.join("checkpoints", f"policy_ep{episode}.pt")
            trainer.save(ckpt)
            if curriculum is not None:
                curriculum.save(CURR_STATE)
            print(f"  [CKPT] Episode {episode} saved. LR={current_lr:.2e}")

    # ── Final ─────────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    print("\n" + "=" * 62)
    print("Training Complete!")
    print(f"  Episodes    : {n_episodes}")
    print(f"  Best RMSD   : {best_rmsd:.3f} Å")
    print(f"  Best Energy : {best_energy:.3f} kcal/mol")
    print(f"  Final LR    : {trainer.get_lr():.2e}")
    print(f"  Time        : {elapsed:.1f} minutes")
    print(f"  Log         : {LOG_FILE}")
    print("=" * 62)

    if curriculum is not None:
        print(curriculum.summary())

    trainer.save(CHECKPOINT_PATH)
    if curriculum is not None:
        curriculum.save(CURR_STATE)

    return trainer


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Number of training episodes (default {N_EPISODES})"
    )
    parser.add_argument(
        "--protein", type=str, default=None,
        help="Single-protein mode, e.g. --protein 1L2Y"
    )
    args = parser.parse_args()
    train(n_episodes=args.episodes, single_protein=args.protein)