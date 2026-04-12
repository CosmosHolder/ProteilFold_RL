import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.gnn_policy import GNNPolicyNetwork

# ── PPO Hyperparameters ──────────────────────────────────────
GAMMA       = 0.99   # discount factor
LAM         = 0.95   # GAE lambda
CLIP_EPS    = 0.2    # PPO clip epsilon
ENTROPY_COEF= 0.01   # entropy bonus coefficient
VALUE_COEF  = 0.5    # value loss coefficient
MAX_GRAD    = 0.5    # gradient clipping
LR          = 3e-4   # learning rate
BETAS       = (0.9, 0.999)
HORIZON     = 256    # steps per PPO update
PPO_EPOCHS  = 4      # update epochs per batch


class PPOTrainer:
    """
    PPO-clip trainer for ProteinFold-RL.
    Pure PyTorch — no wrappers.
    """

    def __init__(self, policy: GNNPolicyNetwork, action_dim: int):
        self.policy     = policy
        self.action_dim = action_dim
        self.optimizer  = torch.optim.AdamW(
            policy.parameters(), lr=LR, betas=BETAS
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

    def reset_buffer(self):
        self.buf_graphs    = []
        self.buf_actions   = []
        self.buf_rewards   = []
        self.buf_log_probs = []
        self.buf_values    = []
        self.buf_dones     = []

    def store(self, graph, action: int, reward: float,
              log_prob: torch.Tensor, value: torch.Tensor,
              done: bool):
        """Store one transition in buffer."""
        self.buf_graphs.append(graph)
        self.buf_actions.append(action)
        self.buf_rewards.append(reward)
        self.buf_log_probs.append(log_prob.detach())
        self.buf_values.append(value.detach())
        self.buf_dones.append(done)

    def compute_gae(self, last_value: float = 0.0):
        """
        Generalized Advantage Estimation.
        Returns normalized advantages + discounted returns.
        """
        T       = len(self.buf_rewards)
        rewards = np.array(self.buf_rewards, dtype=np.float32)
        values  = torch.stack(self.buf_values).numpy().flatten()
        dones   = np.array(self.buf_dones,   dtype=np.float32)

        # ── Reward normalization (critical for GNN+RL stability) ─
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        advantages = np.zeros(T, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(T)):
            next_val   = last_value if t == T - 1 else values[t + 1]
            next_done  = dones[t]
            delta      = (rewards[t] + GAMMA * next_val *
                         (1 - next_done) - values[t])
            last_gae   = delta + GAMMA * LAM * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        # Normalize advantages
        advantages = ((advantages - advantages.mean()) /
                      (advantages.std() + 1e-8))

        return (
            torch.tensor(advantages, dtype=torch.float),
            torch.tensor(returns,    dtype=torch.float)
        )

    def update(self, last_value: float = 0.0):
        """
        Run PPO-clip update on collected trajectory.
        Returns dict of loss stats.
        """
        if len(self.buf_rewards) == 0:
            return {}

        advantages, returns = self.compute_gae(last_value)
        actions     = torch.tensor(self.buf_actions, dtype=torch.long)
        old_log_probs = torch.stack(self.buf_log_probs)

        stats = {"policy_loss": 0, "value_loss": 0,
                 "entropy": 0, "total_loss": 0}

        for epoch in range(PPO_EPOCHS):
            # Evaluate current policy on stored graphs
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
                         + VALUE_COEF  * value_loss
                         + ENTROPY_COEF * entropy_loss)

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), MAX_GRAD
            )
            self.optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"]  += value_loss.item()
            stats["entropy"]     += entropy.item()
            stats["total_loss"]  += total_loss.item()

        # Average over epochs
        for k in stats:
            stats[k] /= PPO_EPOCHS

        # Log
        for k, v in stats.items():
            self.train_stats[k].append(v)

        self.reset_buffer()
        return stats

    def save(self, path: str):
        torch.save({
            "policy_state" : self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)
        print(f"[SAVE] Checkpoint saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[LOAD] Checkpoint loaded from {path}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.fold_env import FoldEnv

    print("=" * 50)
    print("ProteinFold-RL — PPO Trainer Test")
    print("=" * 50)

    env    = FoldEnv(pdb_id="1L2Y")
    obs, _ = env.reset()
    policy = GNNPolicyNetwork(action_dim=env.action_dim)
    trainer= PPOTrainer(policy=policy, action_dim=env.action_dim)

    print(f"\n[TEST] Collecting {HORIZON} steps...")
    obs, _ = env.reset()
    step   = 0

    while step < HORIZON:
        graph               = env.get_graph()
        action, log_prob, value, entropy = policy.get_action(graph)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        trainer.store(graph, action, reward, log_prob, value, done)
        step += 1
        if done:
            obs, _ = env.reset()

    print(f"  Buffer size    : {len(trainer.buf_rewards)}")
    print(f"  Running PPO update...")

    stats = trainer.update()
    print(f"  Policy loss    : {stats['policy_loss']:.4f}")
    print(f"  Value loss     : {stats['value_loss']:.4f}")
    print(f"  Entropy        : {stats['entropy']:.4f}")
    print(f"  Total loss     : {stats['total_loss']:.4f}")
    assert stats["total_loss"] != 0, "Loss should not be zero"
    print(f"  [PASS] PPO update completed ✓")

    print(f"\n{'=' * 50}")
    print("CHECKPOINT-04 — PPO trainer ready.")
    print("=" * 50)