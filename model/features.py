import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    """
    Encodes raw node features (23-dim) into hidden representation.
    Input : [N, 23]  — 20-dim AA one-hot + 3-dim Cα coords
    Output: [N, hidden_dim]
    """
    def __init__(self, input_dim: int = 23, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EdgeEncoder(nn.Module):
    """
    Encodes raw edge features (4-dim) into hidden representation.
    Input : [E, 4]  — distance + dx + dy + peptide_flag
    Output: [E, edge_dim]
    """
    def __init__(self, input_dim: int = 4, edge_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
            nn.GELU(),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.encoder(edge_attr)


if __name__ == "__main__":
    print("=" * 50)
    print("ProteinFold-RL — Feature Encoders Test")
    print("=" * 50)

    N, E = 20, 140
    x         = torch.randn(N, 23)
    edge_attr = torch.randn(E, 4)

    node_enc = NodeEncoder(input_dim=23, hidden_dim=128)
    edge_enc = EdgeEncoder(input_dim=4,  edge_dim=64)

    node_out = node_enc(x)
    edge_out = edge_enc(edge_attr)

    print(f"\n  Node input  : {x.shape}")
    print(f"  Node output : {node_out.shape}")
    assert node_out.shape == (N, 128), "Node encoder output shape wrong"
    print(f"  [PASS] Node encoder ✓")

    print(f"\n  Edge input  : {edge_attr.shape}")
    print(f"  Edge output : {edge_out.shape}")
    assert edge_out.shape == (E, 64), "Edge encoder output shape wrong"
    print(f"  [PASS] Edge encoder ✓")

    print("\n" + "=" * 50)
    print("Feature encoders ready.")
    print("=" * 50)