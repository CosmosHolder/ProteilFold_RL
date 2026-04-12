"""
train.py
ProteinFold-RL — Curriculum Training Loop (v2)

Upgraded from single-protein (v1) to full 8-protein hybrid curriculum.

Key upgrades vs v1
------------------
- ProteinCurriculum engine drives protein selection
- Per-protein + global CSV logs
- Performance-gated advancement with forced fallback
- Replay sampling prevents catastrophic forgetting
- Curriculum state saved alongside model checkpoints
- Dynamic FoldEnv creation per protein (handles variable action_dim)
- Action dim mismatch guard — policy always created for max action dim
- All v1 features retained (RMSD, energy, clashes, PPO metrics)

Usage
-----
  python train.py                     # full curriculum run
  python train.py --episodes 2000     # custom episode count
  python train.py --resume            # resume from latest checkpoint
  python train.py --protein 1L2Y     # single-protein mode (v1 compat)

Author : ProteinFold-RL team
"""

import argparse
import os
import csv
import json
import time
import numpy as np
import torch

from env.fold_env import FoldEnv
from model.gnn_policy import GNNPolicyNetwork
from agent.ppo import PPOTrainer, HORIZON
from data.protein_registry import REGISTRY, download_all
from data.curriculum import ProteinCurriculum

# ── Training config ──────────────────────────────────────────
N_EPISODES        = 2000    # total episodes across all proteins
SAVE_EVERY        = 100     # save checkpoint every N episodes
LOG_EVERY         = 10      # console log every N episodes
CURRICULUM_EVERY  = 1       # check curriculum gate every N episodes
CHECKPOINT_DIR    = "checkpoints"
LOG_DIR           = "logs"
GLOBAL_LOG_FILE   = os.path.join(LOG_DIR, "training_log.csv")
CURRICULUM_LOG    = os.path.join(LOG_DIR, "curriculum_log.csv")
CURRICULUM_STATE  = os.path.join(CHECKPOINT_DIR, "curriculum_state.json")

# ── Largest protein in curriculum (hemoglobin α, 141 residues) ──
# Policy is always created for max possible action dim so weights
# are reused across proteins without architecture mismatch.
MAX_RESIDUES      = 141
MAX_ACTION_DIM    = MAX_RESIDUES * 2 * 12   # = 3384

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ── Utility ───────────────────────────────────────────────────

def compute_rmsd(coords: np.ndarray, native: np.ndarray) -> float:
    """Root-mean-square deviation of Cα coordinates."""
    diff = coords - native
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def make_env(pdb_id: str) -> FoldEnv:
    """Create a fresh FoldEnv for the given protein."""
    return FoldEnv(pdb_id=pdb_id)


def init_logs():
    """Initialize CSV log files with headers."""
    with open(GLOBAL_LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "protein", "stage", "total_reward",
            "final_energy", "rmsd", "steps", "clashes",
            "policy_loss", "value_loss", "entropy",
            "gate_rolling_rmsd", "advancement_reason",
        ])

    with open(CURRICULUM_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "from_protein", "to_protein",
            "stage", "reason", "rolling_rmsd",
        ])


def log_episode(episode: int, protein: str, stage: int,
                ep_reward: float, final_energy: float,
                rmsd: float, ep_steps: int, ep_clashes: int,
                stats: dict, gate_rmsd: float,
                advancement_reason: str = ""):
    """Append one row to the global training log."""
    with open(GLOBAL_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            episode, protein, stage,
            round(ep_reward, 4),
            round(final_energy, 4),
            round(rmsd, 4),
            ep_steps, ep_clashes,
            round(stats.get("policy_loss", 0), 4),
            round(stats.get("value_loss",  0), 4),
            round(stats.get("entropy",     0), 4),
            round(gate_rmsd, 4),
            advancement_reason,
        ])


def log_advancement(episode: int, from_pdb: str, to_pdb: str,
                    stage: int, reason: str, rolling_rmsd: float):
    """Append one row to the curriculum advancement log."""
    with open(CURRICULUM_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            episode, from_pdb, to_pdb, stage, reason,
            round(rolling_rmsd, 4),
        ])


# ── Main training loop ────────────────────────────────────────

def train(n_episodes: int = N_EPISODES,
          resume: bool = False,
          single_protein: str = None):
    """
    Main curriculum training loop.

    Parameters
    ----------
    n_episodes     : total episodes to run
    resume         : load latest checkpoint and curriculum state
    single_protein : if set, run single-protein mode (v1 compat)
    """
    print("=" * 65)
    print("ProteinFold-RL — Curriculum Training v2")
    print(f"  Episodes   : {n_episodes}")
    print(f"  Proteins   : {len(REGISTRY)}")
    print(f"  Max action : {MAX_ACTION_DIM}")
    print(f"  Resume     : {resume}")
    if single_protein:
        print(f"  Mode       : single-protein ({single_protein})")
    print("=" * 65)

    # ── Ensure proteins are downloaded ───────────────────────
    print("\n[SETUP] Checking protein downloads...")
    download_all()

    # ── Initialize curriculum ─────────────────────────────────
    if single_protein:
        curriculum = None
        current_pdb = single_protein
    else:
        if resume and os.path.exists(CURRICULUM_STATE):
            curriculum = ProteinCurriculum.load(CURRICULUM_STATE)
        else:
            curriculum = ProteinCurriculum()
        current_pdb = curriculum.current_protein().pdb_id

    # ── Initialize policy (max action dim) ───────────────────
    policy  = GNNPolicyNetwork(action_dim=MAX_ACTION_DIM)
    trainer = PPOTrainer(policy=policy, action_dim=MAX_ACTION_DIM)

    if resume:
        ckpt = os.path.join(CHECKPOINT_DIR, "policy_final.pt")
        if os.path.exists(ckpt):
            trainer.load(ckpt)
            print(f"[RESUME] Loaded checkpoint from {ckpt}")

    # ── Initialize environment for first protein ──────────────
    env          = make_env(current_pdb)
    native_coords = env.native_coords.copy()

    # ── Init logs ─────────────────────────────────────────────
    if not resume:
        init_logs()

    # ── Training state ────────────────────────────────────────
    best_rmsd_global   = float("inf")
    best_energy_global = float("inf")
    step_count         = 0
    start_time         = time.time()

    print(f"\n[TRAIN] Starting on: {current_pdb}\n")

    # ── Episode loop ──────────────────────────────────────────
    for episode in range(1, n_episodes + 1):

        # ── Curriculum: sample protein for this episode ───────
        if curriculum is not None:
            entry       = curriculum.sample_protein()
            episode_pdb = entry.pdb_id

            # Rebuild env only if protein changed
            if episode_pdb != current_pdb:
                env           = make_env(episode_pdb)
                native_coords = env.native_coords.copy()
                current_pdb   = episode_pdb
        else:
            episode_pdb = single_protein

        # ── Run episode ───────────────────────────────────────
        obs, info  = env.reset()
        ep_reward  = 0.0
        ep_steps   = 0
        ep_clashes = 0
        done       = False
        stats      = {}

        while not done:
            graph = env.get_graph()

            # Policy outputs logits for MAX_ACTION_DIM;
            # clamp action to valid range for this protein
            action, log_prob, value, entropy = policy.get_action(graph)
            valid_dim = env.action_dim
            action    = action % valid_dim   # safe clamping

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            trainer.store(graph, action, reward, log_prob, value, done)
            ep_reward  += reward
            ep_steps   += 1
            ep_clashes += int(info["has_clash"])
            step_count += 1

            # PPO update every HORIZON environment steps
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

        if rmsd    < best_rmsd_global:    best_rmsd_global    = rmsd
        if final_energy < best_energy_global: best_energy_global = final_energy

        # ── Record in curriculum ──────────────────────────────
        advancement_reason = ""
        gate_rmsd          = float("inf")

        if curriculum is not None:
            curriculum.record_episode(
                pdb_id=episode_pdb,
                rmsd=rmsd,
                energy=final_energy,
                reward=ep_reward,
            )

            # Gate status for current primary protein
            primary_pdb    = curriculum.current_protein().pdb_id
            gate_st        = curriculum.gate_status(primary_pdb)
            gate_rmsd      = gate_st["rolling_rmsd"]

            # Check + apply advancement
            if episode % CURRICULUM_EVERY == 0:
                should, reason = curriculum.should_advance()
                if should:
                    old_pdb = curriculum.current_protein().pdb_id
                    old_gate = curriculum.gate_status(old_pdb)["rolling_rmsd"]
                    new_entry = curriculum.advance(reason)
                    advancement_reason = reason

                    if new_entry is not None:
                        log_advancement(
                            episode, old_pdb, new_entry.pdb_id,
                            curriculum.current_stage, reason,
                            old_gate,
                        )

        # ── Log to CSV ────────────────────────────────────────
        stage = curriculum.current_stage if curriculum else 1
        log_episode(
            episode, episode_pdb, stage,
            ep_reward, final_energy, rmsd,
            ep_steps, ep_clashes, stats,
            gate_rmsd, advancement_reason,
        )

        # ── Console log ───────────────────────────────────────
        if episode % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            curr_prot = curriculum.current_protein().pdb_id \
                if curriculum else single_protein
            gate_str = f"{gate_rmsd:.2f}Å" if gate_rmsd < 999 else "  N/A "
            print(
                f"Ep {episode:5d}/{n_episodes} | "
                f"{curr_prot:6} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Energy: {final_energy:7.2f} | "
                f"RMSD: {rmsd:.3f}Å | "
                f"Gate: {gate_str} | "
                f"Stage: {stage} | "
                f"T: {elapsed/60:.1f}m"
            )

        # ── Checkpoint ────────────────────────────────────────
        if episode % SAVE_EVERY == 0:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR, f"policy_ep{episode}.pt"
            )
            trainer.save(ckpt_path)

            # Save curriculum state alongside checkpoint
            if curriculum is not None:
                curriculum.save(CURRICULUM_STATE)

            if episode % (SAVE_EVERY * 5) == 0 and curriculum:
                print(curriculum.get_summary())

    # ── Final summary ─────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("Curriculum Training Complete!")
    print(f"  Total episodes   : {n_episodes}")
    print(f"  Best RMSD        : {best_rmsd_global:.3f} Å")
    print(f"  Best Energy      : {best_energy_global:.3f} kcal/mol")
    print(f"  Training time    : {elapsed/60:.1f} minutes")
    print(f"  Global log       : {GLOBAL_LOG_FILE}")
    print("=" * 65)

    if curriculum:
        print(curriculum.get_summary())

    # Save final model + curriculum
    trainer.save(os.path.join(CHECKPOINT_DIR, "policy_final.pt"))
    if curriculum is not None:
        curriculum.save(CURRICULUM_STATE)

    return trainer


# ── CLI ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ProteinFold-RL Training")
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Total training episodes (default: {N_EPISODES})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint + curriculum state"
    )
    parser.add_argument(
        "--protein", type=str, default=None,
        help="Single-protein mode (e.g. 1L2Y). Disables curriculum."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        n_episodes=args.episodes,
        resume=args.resume,
        single_protein=args.protein,
    )