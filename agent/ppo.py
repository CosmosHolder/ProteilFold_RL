"""
agent/ppo.py
ProteinFold-RL — PPO Trainer (v2)

What changed vs v1
------------------
- CosineAnnealingLR scheduler added
    · T_max = total expected training steps (N_EPISODES × MAX_STEPS)
    · eta_min = LR / 10 (decays to 10% of initial LR by end)
    · scheduler.step() called after every optimizer.step()
- scheduler state saved/loaded alongside policy + optimizer
- __init__ accepts optional total_steps to configure T_max
  (defaults to 2000 episodes × 50 steps = 100_000)
- get_lr() helper for logging current LR during training
- All existing PPO logic (GAE, clip, entropy, grad clip) untouched
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.gnn_policy import GNNPolicyNetwork

# ── PPO Hyperparameters ──────────────────────────────────────
GAMMA        = 0.99    # discount factor
LAM          = 0.95    # GAE lambda
CLIP_EPS     = 0.2     # PPO clip epsilon
ENTROPY_COEF = 0.01    # entropy bonus coefficient
VALUE_COEF   = 0.5     # value loss coefficient
MAX_GRAD     = 0.5     # gradient clipping
LR           = 3e-4    # initial learning rate
LR_MIN       = 3e-5    # cosine scheduler minimum LR (10% of LR)
BETAS        = (0.9, 0.999)
HORIZON      = 256     # steps per PPO update
PPO_EPOCHS   = 4       # update epochs per batch

# Default total steps for scheduler: 2000 episodes × 50 steps each
DEFAULT_TOTAL_STEPS = 100_000


class PPOTrainer:
    """
    PPO-clip trainer for ProteinFold-RL.

    Pure PyTorch — no wrappers.
    Includes CosineAnnealingLR scheduler for stable long-run training.
    """

    def __init__(self, policy: GNNPolicyNetwork,
                 action_dim: int,
                 total_steps: int = DEFAULT_TOTAL_STEPS):
        """
        Parameters
        ----------
        policy      : GNNPolicyNetwork instance
        action_dim  : size of action space (MAX_ACTION_DIM)
        total_steps : total environment steps expected over full run,
                      used to set T_max for the cosine scheduler.
                      Default = 2000 episodes × 50 steps = 100_000.
        """
        self.policy      = policy
        self.action_dim  = action_dim
        self.total_steps = total_steps

        self.optimizer = torch.optim.AdamW(
            policy.parameters(), lr=LR, betas=BETAS
        )

        # ── Cosine LR scheduler ──────────────────────────────
        # Decays LR smoothly from LR → LR_MIN over total_steps
        # T_max is in units of scheduler.step() calls = optimizer steps.
        # Each PPO update calls optimizer.step() PPO_EPOCHS times,
        # and updates happen every HORIZON env steps.
        # Total optimizer calls ≈ (total_steps / HORIZON) × PPO_EPOCHS
        t_max = max(1, (total_steps // HORIZON) * PPO_EPOCHS)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=LR_MIN,
        )

        # Trajectory buffer
        self.reset_buffer()

        # Logging
        self.train_stats = {
            "policy_loss" : [],
            "value_loss"  : [],
            "entropy"     : [],
            "total_loss"  : [],
        }

    # ────────────────────────────────────────────────────────
    def reset_buffer(self):
        """Clear trajectory buffer between PPO updates."""
        self.buf_graphs    = []
        self.buf_actions   = []
        self.buf_rewards   = []
        self.buf_log_probs = []
        self.buf_values    = []
        self.buf_dones     = []

    # ────────────────────────────────────────────────────────
    def store(self, graph, action: int, reward: float,
              log_prob: torch.Tensor, value: torch.Tensor,
              done: bool):
        """Store one transition in the trajectory buffer."""
        self.buf_graphs.append(graph)
        self.buf_actions.append(action)
        self.buf_rewards.append(reward)
        self.buf_log_probs.append(log_prob.detach())
        self.buf_values.append(value.detach())
        self.buf_dones.append(done)

    # ────────────────────────────────────────────────────────
    def compute_gae(self, last_value: float = 0.0):
        """
        Generalized Advantage Estimation (GAE-λ).

        Returns normalized advantages and discounted returns.

        Parameters
        ----------
        last_value : bootstrap value for the last state (0 if terminal)

        Returns
        -------
        advantages : torch.Tensor [T]
        returns    : torch.Tensor [T]
        """
        T       = len(self.buf_rewards)
        rewards = np.array(self.buf_rewards, dtype=np.float32)
        values  = torch.stack(self.buf_values).numpy().flatten()
        dones   = np.array(self.buf_dones,   dtype=np.float32)

        # ── Reward normalization — critical for GNN+RL stability ──
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        advantages = np.zeros(T, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(T)):
            next_val  = last_value if t == T - 1 else values[t + 1]
            next_done = dones[t]
            delta     = (rewards[t]
                         + GAMMA * next_val * (1 - next_done)
                         - values[t])
            last_gae  = delta + GAMMA * LAM * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        # Normalize advantages
        advantages = ((advantages - advantages.mean()) /
                      (advantages.std() + 1e-8))

        return (
            torch.tensor(advantages, dtype=torch.float),
            torch.tensor(returns,    dtype=torch.float)
        )

    # ────────────────────────────────────────────────────────
    def update(self, last_value: float = 0.0) -> dict:
        """
        Run PPO-clip update on collected trajectory.

        Steps:
          1. Compute GAE advantages + returns
          2. For PPO_EPOCHS epochs:
               a. Re-evaluate policy on stored (graph, action) pairs
               b. Compute clipped surrogate policy loss
               c. Compute MSE value loss
               d. Compute entropy bonus
               e. Backward + grad clip + optimizer step
               f. Scheduler step (cosine LR decay)
          3. Clear buffer

        Returns
        -------
        dict : average loss stats over PPO_EPOCHS
        """
        if len(self.buf_rewards) == 0:
            return {}

        advantages, returns = self.compute_gae(last_value)
        actions       = torch.tensor(self.buf_actions, dtype=torch.long)
        old_log_probs = torch.stack(self.buf_log_probs)

        stats = {
            "policy_loss": 0.0,
            "value_loss" : 0.0,
            "entropy"    : 0.0,
            "total_loss" : 0.0,
        }

        for _ in range(PPO_EPOCHS):
            # Re-evaluate current policy on stored graphs
            new_log_probs, values, entropy = \
                self.policy.evaluate_actions(self.buf_graphs, actions)

            # ── PPO clip loss ────────────────────────────────
            ratio       = torch.exp(new_log_probs - old_log_probs)
            surr1       = ratio * advantages
            surr2       = torch.clamp(
                ratio, 1 - CLIP_EPS, 1 + CLIP_EPS
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # ── Value loss ───────────────────────────────────
            value_loss  = F.mse_loss(values, returns)

            # ── Entropy bonus ────────────────────────────────
            entropy_loss = -entropy

            # ── Total loss ───────────────────────────────────
            total_loss = (policy_loss
                          + VALUE_COEF   * value_loss
                          + ENTROPY_COEF * entropy_loss)

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD)
            self.optimizer.step()

            # ── Cosine LR step ───────────────────────────────
            self.scheduler.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"]  += value_loss.item()
            stats["entropy"]     += entropy.item()
            stats["total_loss"]  += total_loss.item()

        # Average stats over epochs
        for k in stats:
            stats[k] /= PPO_EPOCHS

        # Append to history
        for k, v in stats.items():
            self.train_stats[k].append(v)

        self.reset_buffer()
        return stats

    # ────────────────────────────────────────────────────────
    def get_lr(self) -> float:
        """Return current learning rate (for logging)."""
        return self.scheduler.get_last_lr()[0]

    # ────────────────────────────────────────────────────────
    def save(self, path: str):
        """
        Save policy weights, optimizer state, and scheduler state.

        All three must be saved together so training can be resumed
        with the correct LR position in the cosine schedule.
        """
        torch.save({
            "policy_state"   : self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }, path)
        print(f"[SAVE] Checkpoint saved to {path}")

    def load(self, path: str):
        """
        Load policy weights, optimizer state, and scheduler state.

        Gracefully handles old checkpoints that pre-date the scheduler
        (scheduler_state key absent) — falls back to fresh scheduler.
        """
        ckpt = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])

        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            print(f"[LOAD] Checkpoint loaded from {path} (with scheduler)")
        else:
            print(f"[LOAD] Checkpoint loaded from {path} "
                  f"(no scheduler state — using fresh scheduler)")


# ── Unit tests ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.fold_env import FoldEnv

    print("=" * 60)
    print("ProteinFold-RL — PPO Trainer Test (v2 with LR scheduler)")
    print("=" * 60)

    env    = FoldEnv(pdb_id="1L2Y")
    obs, _ = env.reset()
    policy = GNNPolicyNetwork(action_dim=env.action_dim)
    trainer = PPOTrainer(policy=policy, action_dim=env.action_dim,
                         total_steps=50 * 50)   # tiny for test

    print(f"\n[TEST] Initial LR : {trainer.get_lr():.2e}")
    print(f"  Scheduler T_max : {trainer.scheduler.T_max}")
    print(f"  LR_min target   : {LR_MIN:.2e}")

    # Collect HORIZON steps
    print(f"\n[TEST] Collecting {HORIZON} steps...")
    obs, _ = env.reset()
    step   = 0

    while step < HORIZON:
        graph = env.get_graph()
        action, log_prob, value, entropy = policy.get_action(graph)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        trainer.store(graph, action, reward, log_prob, value, done)
        step += 1
        if done:
            obs, _ = env.reset()

    print(f"  Buffer size : {len(trainer.buf_rewards)}")

    # PPO update
    print(f"  Running PPO update...")
    stats = trainer.update()
    print(f"  Policy loss    : {stats['policy_loss']:.4f}")
    print(f"  Value loss     : {stats['value_loss']:.4f}")
    print(f"  Entropy        : {stats['entropy']:.4f}")
    print(f"  Total loss     : {stats['total_loss']:.4f}")
    print(f"  LR after update: {trainer.get_lr():.2e}")
    assert stats["total_loss"] != 0, "Loss should not be zero"
    print(f"  [PASS] PPO update completed ✓")

    # Save / load round-trip
    print(f"\n[TEST] Save / load round-trip...")
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    trainer.save(tmp_path)

    # Fresh trainer, load checkpoint
    policy2  = GNNPolicyNetwork(action_dim=env.action_dim)
    trainer2 = PPOTrainer(policy=policy2, action_dim=env.action_dim)
    trainer2.load(tmp_path)

    lr_orig   = trainer.get_lr()
    lr_loaded = trainer2.get_lr()
    assert abs(lr_orig - lr_loaded) < 1e-10, "LR mismatch after load"
    print(f"  LR preserved after reload: {lr_loaded:.2e} ✓")
    _os.unlink(tmp_path)

    print(f"\n{'=' * 60}")
    print("CHECKPOINT — PPO v2 with cosine LR scheduler ready.")
    print("=" * 60)