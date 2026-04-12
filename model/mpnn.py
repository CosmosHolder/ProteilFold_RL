import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class MPNNLayer(MessagePassing):
    """
    Single Message Passing layer.
    Aggregates neighbor messages into node updates.

    Message : MLP(src_feat + dst_feat + edge_feat) → 64-dim
    Update  : MLP(node_feat + aggregated_msg)      → 128-dim
    """

    def __init__(self, node_dim: int = 128, edge_dim: int = 64):
        super().__init__(aggr="add")  # sum aggregation

        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim + edge_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
            nn.GELU(),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : [N, node_dim]
            edge_index : [2, E]
            edge_attr  : [E, edge_dim]
        Returns:
            x_out      : [N, node_dim]
        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + out)  # residual connection

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute message from node j to node i."""
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        """Update node features with aggregated messages."""
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class MPNNStack(nn.Module):
    """
    Stack of 4 MPNN layers with global pooling.
    Output: 256-dim graph-level embedding.
    """

    def __init__(self, node_dim: int = 128, edge_dim: int = 64,
                 n_layers: int = 4):
        super().__init__()

        self.layers = nn.ModuleList([
            MPNNLayer(node_dim=node_dim, edge_dim=edge_dim)
            for _ in range(n_layers)
        ])

        # Project to 256-dim global embedding
        self.global_proj = nn.Sequential(
            nn.Linear(node_dim, 256),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x          : [N, node_dim]
            edge_index : [2, E]
            edge_attr  : [E, edge_dim]
            batch      : [N] batch assignment (None = single graph)
        Returns:
            global_emb : [1, 256] or [B, 256] if batched
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Mean pooling over all nodes
        if batch is None:
            global_emb = x.mean(dim=0, keepdim=True)  # [1, node_dim]
        else:
            from torch_geometric.nn import global_mean_pool
            global_emb = global_mean_pool(x, batch)   # [B, node_dim]

        return self.global_proj(global_emb)            # [1/B, 256]


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.protein_graph import pdb_to_graph
    from model.features import NodeEncoder, EdgeEncoder

    print("=" * 50)
    print("ProteinFold-RL — MPNN Stack Test")
    print("=" * 50)

    path  = "data/structures/1L2Y.pdb"
    graph = pdb_to_graph(path)

    node_enc = NodeEncoder(input_dim=23,  hidden_dim=128)
    edge_enc = EdgeEncoder(input_dim=4,   edge_dim=64)
    mpnn     = MPNNStack  (node_dim=128,  edge_dim=64, n_layers=4)

    x         = node_enc(graph.x)
    edge_attr = edge_enc(graph.edge_attr)
    global_emb= mpnn(x, graph.edge_index, edge_attr)

    print(f"\n  Node features  : {graph.x.shape}")
    print(f"  After encoding : {x.shape}")
    print(f"  Edge features  : {graph.edge_attr.shape}")
    print(f"  After encoding : {edge_attr.shape}")
    print(f"  Global embed   : {global_emb.shape}")

    assert x.shape         == (graph.num_nodes, 128), "Node dim wrong"
    assert edge_attr.shape == (graph.edge_index.shape[1], 64), "Edge dim wrong"
    assert global_emb.shape== (1, 256), "Global embedding dim wrong"
    print(f"\n  [PASS] All shape assertions passed ✓")

    print("\n" + "=" * 50)
    print("MPNN stack ready.")
    print("=" * 50)