import numpy as np
import torch
from torch_geometric.data import Data
from Bio import PDB
from Bio.PDB import PPBuilder
import warnings
warnings.filterwarnings("ignore")

# 20 standard amino acids
AA_LIST = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
CONTACT_THRESHOLD = 8.0  # Angstroms


def one_hot_aa(resname: str) -> list:
    vec = [0.0] * 20
    idx = AA_TO_IDX.get(resname.upper(), -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def get_backbone_angles(residue) -> list:
    """Extract phi/psi angles — return [0,0] if not computable."""
    try:
        phi = residue.internal_coord.get_angle("phi") or 0.0
        psi = residue.internal_coord.get_angle("psi") or 0.0
        return [
            np.sin(np.radians(phi)),
            np.cos(np.radians(phi)),
            np.sin(np.radians(psi)),
            np.cos(np.radians(psi))
        ]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def pdb_to_graph(pdb_path: str, model_idx: int = 0) -> Data:
    """
    Convert a PDB file to a PyTorch Geometric Data object.

    Node features (23-dim):
        - 20-dim one-hot amino acid type
        - 3-dim Cα coordinates (x, y, z)

    Edge features (4-dim):
        - distance between Cα atoms
        - sin/cos of relative orientation (dx, dy normalized)
        - 1-dim peptide bond flag (1 = sequential neighbor)

    Returns:
        torch_geometric.data.Data with:
            x          : [N, 23]  node features
            edge_index : [2, E]   edge connectivity
            edge_attr  : [E, 4]   edge features
            pos        : [N, 3]   Cα coordinates
            seq        : list of residue names
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = list(structure.get_models())[model_idx]

    residues = []
    ca_coords = []

    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != " ":
                continue  # skip HETATMs
            if "CA" not in residue:
                continue  # skip residues without Cα
            residues.append(residue)
            ca_coords.append(residue["CA"].get_vector().get_array())

    N = len(residues)
    assert N > 0, f"No valid residues found in {pdb_path}"

    ca_coords = np.array(ca_coords, dtype=np.float32)  # [N, 3]

    # ── Node features ──────────────────────────────────────────
    node_feats = []
    for i, res in enumerate(residues):
        aa_oh = one_hot_aa(res.get_resname())       # 20-dim
        coord = ca_coords[i].tolist()               # 3-dim
        node_feats.append(aa_oh + coord)            # 23-dim

    x = torch.tensor(node_feats, dtype=torch.float)  # [N, 23]

    # ── Edges — spatial contacts within 8Å ────────────────────
    edge_src, edge_dst, edge_attrs = [], [], []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = ca_coords[j] - ca_coords[i]
            dist = float(np.linalg.norm(diff))

            if dist <= CONTACT_THRESHOLD:
                is_peptide = 1.0 if abs(i - j) == 1 else 0.0
                diff_norm = diff / (dist + 1e-8)

                edge_src.append(i)
                edge_dst.append(j)
                edge_attrs.append([
                    dist,
                    float(diff_norm[0]),
                    float(diff_norm[1]),
                    is_peptide
                ])

    edge_index = torch.tensor(
        [edge_src, edge_dst], dtype=torch.long
    )  # [2, E]

    edge_attr = torch.tensor(
        edge_attrs, dtype=torch.float
    )  # [E, 4]

    pos = torch.tensor(ca_coords, dtype=torch.float)  # [N, 3]
    seq = [r.get_resname() for r in residues]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        seq=seq,
        num_nodes=N
    )

    return data


if __name__ == "__main__":
    import os

    for pdb_id, name in [("1L2Y", "Trp-cage"), ("1YRF", "Villin")]:
        path = os.path.join(
            os.path.dirname(__file__),
            f"../data/structures/{pdb_id}.pdb"
        )
        print(f"\n[TEST] {name} ({pdb_id})")
        graph = pdb_to_graph(path)
        print(f"  Residues (nodes) : {graph.num_nodes}")
        print(f"  Edges            : {graph.edge_index.shape[1]}")
        print(f"  Node feat shape  : {graph.x.shape}")
        print(f"  Edge feat shape  : {graph.edge_attr.shape}")
        print(f"  Pos shape        : {graph.pos.shape}")
        assert graph.x.shape[1] == 23,      "Node features must be 23-dim"
        assert graph.edge_attr.shape[1] == 4, "Edge features must be 4-dim"
        assert graph.edge_index.shape[0] == 2, "Edge index must be [2, E]"
        print(f"  [PASS] All shape assertions passed ✓")

    print("\nCHECKPOINT-01 — Protein graphs ready.")