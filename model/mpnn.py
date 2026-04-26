"""
model/mpnn.py
ProteinFold-RL — Message Passing Neural Network (v2)

Upgraded from v1 (plain mean aggregation) to:
  - Edge-gated attention  : each node learns which neighbours matter
  - Layer normalisation   : stable training across proteins of all sizes
  - Residual connections  : gradients flow cleanly through 4 layers

Interface is identical to v1 — gnn_policy.py needs zero changes.

  MPNNStack(node_dim=128, edge_dim=64, n_layers=4)
  → forward(x, edge_index, edge_attr, batch=None)
  → global embedding [1, 256]

Run : python model/mpnn.py   (self-test)
this is the running mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class EdgeGatedMPNNLayer(nn.Module):
    """
    One message passing layer with edge-gated attention.

    For each node i:
      1. Compute a message from each neighbour j using node + edge features.
      2. Compute an attention gate (0-1) for each neighbour j.
      3. Aggregate: weighted sum of messages using gates as weights.
      4. Update node embedding with a GRU-style residual.
      5. Apply layer norm.

    This teaches the agent WHICH contacts matter (gate) and HOW they
    influence the node (message) — much richer than plain mean pooling.
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.node_dim = node_dim

        # Message network: neighbour node + edge → message vector
        self.message_net = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )

        # Gate network: neighbour node + edge → scalar attention weight
        self.gate_net = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )

        # Update: combine aggregated message with current node embedding
        self.update_net = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )

        # Layer norm applied after update (keeps activations stable
        # across proteins with very different numbers of residues)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [N, node_dim]  node embeddings
        edge_index : [2, E]         source / destination indices
        edge_attr  : [E, edge_dim]  edge features

        Returns
        -------
        x_new : [N, node_dim]  updated node embeddings
        """
        src, dst = edge_index   # src→dst direction

        # ── Messages ─────────────────────────────────────────
        # Each edge carries: source node embedding + edge features
        src_feats = x[src]                          # [E, node_dim]
        edge_input = torch.cat([src_feats,
                                edge_attr], dim=-1) # [E, node_dim+edge_dim]

        messages = self.message_net(edge_input)     # [E, node_dim]
        gates    = self.gate_net(edge_input)        # [E, 1]

        # Gated messages
        gated_messages = messages * gates           # [E, node_dim]

        # ── Aggregation (sum over incoming edges per node) ────
        aggregated = torch.zeros_like(x)            # [N, node_dim]
        aggregated.scatter_add_(
            0,
            dst.unsqueeze(-1).expand_as(gated_messages),
            gated_messages,
        )

        # ── Update with residual ──────────────────────────────
        combined = torch.cat([x, aggregated], dim=-1)   # [N, node_dim*2]
        update   = self.update_net(combined)            # [N, node_dim]

        # Residual connection + layer norm
        x_new = self.norm(x + update)

        return x_new


class MPNNStack(nn.Module):
    """
    Stack of EdgeGatedMPNNLayers followed by global pooling.

    Produces a single fixed-size graph embedding regardless of
    how many residues the protein has — essential for the policy
    head which must output a fixed action_dim.

    Output projection: node_dim (128) → 256 for policy + value heads.
    This matches the v1 interface exactly.
    """

    def __init__(self, node_dim: int = 128,
                 edge_dim: int = 64,
                 n_layers: int = 4):
        super().__init__()
        self.node_dim = node_dim

        # Stack of message passing layers
        self.layers = nn.ModuleList([
            EdgeGatedMPNNLayer(node_dim=node_dim, edge_dim=edge_dim)
            for _ in range(n_layers)
        ])

        # Project pooled embedding to 256-dim
        # (matches v1 output — gnn_policy.py unchanged)
        self.output_proj = nn.Sequential(
            nn.Linear(node_dim, 256),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [N, node_dim]   node embeddings from NodeEncoder
        edge_index : [2, E]          graph connectivity
        edge_attr  : [E, edge_dim]   edge embeddings from EdgeEncoder
        batch      : [N]             batch assignment (None = single graph)

        Returns
        -------
        global_emb : [1, 256]   graph-level embedding
        """
        # Pass through all message passing layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Global mean pooling → one vector per graph
        if batch is None:
            # Single graph: mean over all nodes
            pooled = x.mean(dim=0, keepdim=True)   # [1, node_dim]
        else:
            pooled = global_mean_pool(x, batch)    # [B, node_dim]

        # Project to 256
        global_emb = self.output_proj(pooled)      # [1, 256]

        return global_emb


# ── Self-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.fold_env import FoldEnv

    print("=" * 52)
    print("ProteinFold-RL — MPNN v2 Self-Test")
    print("=" * 52)

    NODE_DIM = 128
    EDGE_DIM = 64
    N_LAYERS = 4

    env    = FoldEnv(pdb_id="1L2Y")
    obs, _ = env.reset()
    graph  = env.get_graph()

    N = graph.x.shape[0]
    E = graph.edge_index.shape[1]
    print(f"\n  Graph: {N} nodes, {E} edges")

    # ── Layer test ────────────────────────────────────────────
    print("\n[TEST 1] EdgeGatedMPNNLayer forward pass...")
    layer = EdgeGatedMPNNLayer(node_dim=NODE_DIM, edge_dim=EDGE_DIM)

    x_fake    = torch.randn(N, NODE_DIM)
    ea_fake   = torch.randn(E, EDGE_DIM)
    x_out     = layer(x_fake, graph.edge_index, ea_fake)

    assert x_out.shape == (N, NODE_DIM), \
        f"Expected ({N}, {NODE_DIM}), got {x_out.shape}"
    assert not torch.isnan(x_out).any(), "NaN in layer output"
    print(f"  Output shape : {x_out.shape}  ✅")

    # ── Stack test ────────────────────────────────────────────
    print("\n[TEST 2] MPNNStack forward pass...")
    stack  = MPNNStack(node_dim=NODE_DIM, edge_dim=EDGE_DIM, n_layers=N_LAYERS)
    x_fake = torch.randn(N, NODE_DIM)
    ea_fake= torch.randn(E, EDGE_DIM)

    emb = stack(x_fake, graph.edge_index, ea_fake, batch=None)
    assert emb.shape == (1, 256), \
        f"Expected (1, 256), got {emb.shape}"
    assert not torch.isnan(emb).any(), "NaN in stack output"
    print(f"  Output shape : {emb.shape}  ✅")

    # ── Residual check ────────────────────────────────────────
    print("\n[TEST 3] Residual connection (output != input)...")
    assert not torch.allclose(x_fake, x_out), \
        "Layer output should differ from input"
    print("  Residual active  ✅")

    # ── Gradient flow ─────────────────────────────────────────
    print("\n[TEST 4] Gradient flow through all 4 layers...")
    x_grad  = torch.randn(N, NODE_DIM, requires_grad=True)
    ea_grad = torch.randn(E, EDGE_DIM)
    emb_g   = stack(x_grad, graph.edge_index, ea_grad)
    loss    = emb_g.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradient reached input"
    assert not torch.isnan(x_grad.grad).any(), "NaN in gradients"
    print("  Gradients flow cleanly  ✅")

    # ── Parameter count ───────────────────────────────────────
    n_params = sum(p.numel() for p in stack.parameters())
    print(f"\n  Parameters : {n_params:,}")

    print("\n" + "=" * 52)
    print("MPNN v2 — all tests passed ✅")
    print("Next: python model/gnn_policy.py  (verify full network)")
    print("=" * 52)
