"""
model/features.py
ProteinFold-RL — Feature Encoders (v2)

What changed vs v1
------------------
NodeEncoder:
  - Amino acid type (20-dim one-hot) and Cα coordinates (3-dim) are
    encoded separately then merged. They carry completely different
    information and should not be mixed in the first linear layer.
  - Layer norm on the final output — stabilises training across
    proteins with different numbers of residues.

EdgeEncoder:
  - Layer norm on the final output — edge distances vary across
    proteins of different sizes (small vs large contact maps).

Interface is identical to v1 — gnn_policy.py needs zero changes.

  NodeEncoder(input_dim=23, hidden_dim=128)
  EdgeEncoder(input_dim=4,  edge_dim=64)

Run : python model/features.py
"""

import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    """
    Encodes raw node features into 128-dim node embeddings.

    Input  : [N, 23]  — 20-dim AA one-hot + 3-dim Cα coords
    Output : [N, hidden_dim]

    AA type and coordinates are encoded in separate branches
    then concatenated and projected to hidden_dim.
    This avoids the first linear layer trying to mix two
    fundamentally different feature types.
    """

    AA_DIM    = 20   # one-hot amino acid type
    COORD_DIM = 3    # Cα x, y, z

    def __init__(self, input_dim: int = 23, hidden_dim: int = 128):
        super().__init__()

        assert input_dim == self.AA_DIM + self.COORD_DIM, \
            f"input_dim must be {self.AA_DIM + self.COORD_DIM}, got {input_dim}"

        # Branch 1 — amino acid type
        aa_hidden = hidden_dim // 2   # 64
        self.aa_branch = nn.Sequential(
            nn.Linear(self.AA_DIM, aa_hidden),
            nn.GELU(),
            nn.Linear(aa_hidden, aa_hidden),
            nn.GELU(),
        )

        # Branch 2 — Cα coordinates
        coord_hidden = hidden_dim // 2   # 64
        self.coord_branch = nn.Sequential(
            nn.Linear(self.COORD_DIM, coord_hidden),
            nn.GELU(),
            nn.Linear(coord_hidden, coord_hidden),
            nn.GELU(),
        )

        # Merge: concat both branches → project to hidden_dim
        self.merge = nn.Sequential(
            nn.Linear(aa_hidden + coord_hidden, hidden_dim),
            nn.GELU(),
        )

        # Layer norm — stabilises across proteins of different sizes
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [N, 23]  raw node features

        Returns
        -------
        [N, hidden_dim]  encoded node embeddings
        """
        aa_feat    = x[:, :self.AA_DIM]      # [N, 20]
        coord_feat = x[:, self.AA_DIM:]      # [N, 3]

        aa_emb    = self.aa_branch(aa_feat)       # [N, 64]
        coord_emb = self.coord_branch(coord_feat) # [N, 64]

        merged = torch.cat([aa_emb, coord_emb], dim=-1)  # [N, 128]
        out    = self.merge(merged)                       # [N, hidden_dim]
        return self.norm(out)


class EdgeEncoder(nn.Module):
    """
    Encodes raw edge features into 64-dim edge embeddings.

    Input  : [E, 4]  — distance + dx + dy + peptide_flag
    Output : [E, edge_dim]

    Layer norm added to stabilise distances across proteins
    of very different sizes (small vs large contact maps).
    """

    def __init__(self, input_dim: int = 4, edge_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
            nn.GELU(),
        )

        # Layer norm — edge distances vary across protein sizes
        self.norm = nn.LayerNorm(edge_dim)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        edge_attr : [E, 4]  raw edge features

        Returns
        -------
        [E, edge_dim]  encoded edge embeddings
        """
        return self.norm(self.encoder(edge_attr))


# ── Self-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("ProteinFold-RL — Feature Encoders v2 Test")
    print("=" * 52)

    N, E = 20, 146   # 1L2Y: 20 nodes, ~146 edges

    x         = torch.randn(N, 23)
    edge_attr = torch.randn(E, 4)

    node_enc = NodeEncoder(input_dim=23, hidden_dim=128)
    edge_enc = EdgeEncoder(input_dim=4,  edge_dim=64)

    # ── Shape tests ───────────────────────────────────────────
    print("\n[TEST 1] Output shapes...")
    node_out = node_enc(x)
    edge_out = edge_enc(edge_attr)

    assert node_out.shape == (N, 128), \
        f"Node encoder: expected ({N}, 128), got {node_out.shape}"
    assert edge_out.shape == (E, 64), \
        f"Edge encoder: expected ({E}, 64), got {edge_out.shape}"
    print(f"  Node: {x.shape} → {node_out.shape}  ✅")
    print(f"  Edge: {edge_attr.shape} → {edge_out.shape}  ✅")

    # ── NaN check ─────────────────────────────────────────────
    print("\n[TEST 2] No NaN in outputs...")
    assert not torch.isnan(node_out).any(), "NaN in node encoder output"
    assert not torch.isnan(edge_out).any(), "NaN in edge encoder output"
    print("  No NaN  ✅")

    # ── Branch separation check ───────────────────────────────
    print("\n[TEST 3] AA and coord branches are separate...")
    x2 = x.clone()
    x2[:, :20] = torch.zeros(N, 20)   # zero out AA features
    out_no_aa = node_enc(x2)
    assert not torch.allclose(node_out, out_no_aa), \
        "AA branch has no effect — check split"
    print("  AA branch active  ✅")

    x3 = x.clone()
    x3[:, 20:] = torch.zeros(N, 3)    # zero out coord features
    out_no_coord = node_enc(x3)
    assert not torch.allclose(node_out, out_no_coord), \
        "Coord branch has no effect — check split"
    print("  Coord branch active  ✅")

    # ── Gradient flow ─────────────────────────────────────────
    print("\n[TEST 4] Gradient flow...")
    x_g = torch.randn(N, 23, requires_grad=True)
    out = node_enc(x_g)
    out.sum().backward()
    assert x_g.grad is not None, "No gradient reached node input"
    assert not torch.isnan(x_g.grad).any(), "NaN in gradients"
    print("  Gradients flow cleanly  ✅")

    # ── Parameter count ───────────────────────────────────────
    n_node = sum(p.numel() for p in node_enc.parameters())
    n_edge = sum(p.numel() for p in edge_enc.parameters())
    print(f"\n  NodeEncoder params : {n_node:,}")
    print(f"  EdgeEncoder params : {n_edge:,}")

    print("\n" + "=" * 52)
    print("Feature encoders v2 — all tests passed ✅")
    print("Next: python model/gnn_policy.py")
    print("=" * 52)