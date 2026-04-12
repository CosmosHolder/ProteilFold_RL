import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.features import NodeEncoder, EdgeEncoder
from model.mpnn import MPNNStack


class GNNPolicyNetwork(nn.Module):
    """
    Full GNN Policy + Value Network for ProteinFold-RL.

    Architecture:
        NodeEncoder  → 128-dim node embeddings
        EdgeEncoder  → 64-dim edge embeddings
        MPNNStack    → 4 message passing layers → 256-dim global embedding
        Policy head  → Linear(256, action_dim) → action logits
        Value head   → MLP(256, 128, 1)        → scalar state value

    Input  : PyG Data object (current protein graph)
    Output : (action_logits [action_dim], state_value [1])
    """

    def __init__(self, action_dim: int,
                 node_input_dim: int = 23,
                 edge_input_dim: int = 4,
                 hidden_dim: int = 128,
                 edge_dim: int = 64,
                 n_layers: int = 4):
        super().__init__()

        self.action_dim = action_dim

        # Encoders
        self.node_encoder = NodeEncoder(
            input_dim=node_input_dim, hidden_dim=hidden_dim
        )
        self.edge_encoder = EdgeEncoder(
            input_dim=edge_input_dim, edge_dim=edge_dim
        )

        # Message passing
        self.mpnn = MPNNStack(
            node_dim=hidden_dim, edge_dim=edge_dim, n_layers=n_layers
        )

        # Policy head
        self.policy_head = nn.Linear(256, action_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, graph: Data):
        """
        Args:
            graph : PyG Data with x [N,23], edge_index [2,E],
                    edge_attr [E,4], optional batch [N]
        Returns:
            logits : [1, action_dim]  action logits
            value  : [1, 1]           state value estimate
        """
        x         = self.node_encoder(graph.x)
        edge_attr = self.edge_encoder(graph.edge_attr)
        batch     = graph.batch if hasattr(graph, 'batch') else None

        global_emb = self.mpnn(x, graph.edge_index, edge_attr, batch)

        logits = self.policy_head(global_emb)   # [1, action_dim]
        value  = self.value_head(global_emb)    # [1, 1]

        return logits, value

    def get_action(self, graph: Data, deterministic: bool = False):
        """
        Sample action from policy.

        Returns:
            action   : int
            log_prob : torch.Tensor scalar
            value    : torch.Tensor scalar
            entropy  : torch.Tensor scalar
        """
        logits, value = self.forward(graph)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))

        if deterministic:
            action = logits.argmax(dim=-1).item()
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action))
        entropy  = dist.entropy()

        return action, log_prob, value.squeeze(), entropy

    def evaluate_actions(self, graphs: list, actions: torch.Tensor):
        """
        Evaluate a batch of (graph, action) pairs for PPO update.

        Args:
            graphs  : list of PyG Data objects
            actions : [T] tensor of action indices

        Returns:
            log_probs : [T]
            values    : [T]
            entropy   : scalar
        """
        log_probs, values, entropies = [], [], []

        for i, graph in enumerate(graphs):
            logits, value = self.forward(graph)
            dist = torch.distributions.Categorical(
                logits=logits.squeeze(0)
            )
            log_probs.append(dist.log_prob(actions[i]))
            values.append(value.squeeze())
            entropies.append(dist.entropy())

        return (
            torch.stack(log_probs),
            torch.stack(values),
            torch.stack(entropies).mean()
        )


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.protein_graph import pdb_to_graph
    from env.fold_env import FoldEnv

    print("=" * 50)
    print("ProteinFold-RL — GNN Policy Network Test")
    print("=" * 50)

    # Build env to get action_dim
    env    = FoldEnv(pdb_id="1L2Y")
    obs, _ = env.reset()

    policy = GNNPolicyNetwork(action_dim=env.action_dim)
    graph  = env.get_graph()

    # Forward pass
    logits, value = policy(graph)
    print(f"\n  Action dim     : {env.action_dim}")
    print(f"  Logits shape   : {logits.shape}")
    print(f"  Value shape    : {value.shape}")
    assert logits.shape == (1, env.action_dim), "Logits shape wrong"
    assert value.shape  == (1, 1),              "Value shape wrong"
    print(f"  [PASS] Forward pass ✓")

    # Action sampling
    action, log_prob, val, entropy = policy.get_action(graph)
    print(f"\n  Sampled action : {action}")
    print(f"  Log prob       : {log_prob.item():.4f}")
    print(f"  Value estimate : {val.item():.4f}")
    print(f"  Entropy        : {entropy.item():.4f}")
    assert 0 <= action < env.action_dim, "Action out of range"
    print(f"  [PASS] Action sampling ✓")

    print(f"\n{'=' * 50}")
    print("CHECKPOINT-03 — GNN policy ready.")
    print("=" * 50)