import numpy as np


# ── Lennard-Jones parameters ────────────────────────────────
LJ_EPSILON = 0.238  # kcal/mol (typical carbon-carbon)
LJ_SIGMA   = 3.8    # Angstroms (Cα-Cα effective radius)

# ── Torsion energy parameters ───────────────────────────────
TORSION_WEIGHT = 0.5

# ── Preferred backbone angles (Ramachandran favored) ────────
# Alpha helix: phi=-57, psi=-47
# Beta sheet:  phi=-119, psi=113
HELIX_PHI  = np.radians(-57.0)
HELIX_PSI  = np.radians(-47.0)
SHEET_PHI  = np.radians(-119.0)
SHEET_PSI  = np.radians(113.0)


def lj_potential(dist: float) -> float:
    """
    Simplified Lennard-Jones potential between two Cα atoms.
    V(r) = 4ε [(σ/r)^12 - (σ/r)^6]
    Clamped to avoid explosion at very short distances.
    """
    if dist < 1e-3:
        return 100.0  # hard wall
    ratio = LJ_SIGMA / dist
    ratio6  = ratio ** 6
    ratio12 = ratio6 ** 2
    v = 4.0 * LJ_EPSILON * (ratio12 - ratio6)
    return float(np.clip(v, -10.0, 100.0))


def torsion_energy(phi: float, psi: float) -> float:
    """
    Simplified torsion energy — penalizes deviation from
    favored Ramachandran regions (helix or sheet).
    Returns the minimum energy across both favored regions.
    """
    def angle_diff(a, b):
        d = abs(a - b) % (2 * np.pi)
        return min(d, 2 * np.pi - d)

    helix_cost = (
        angle_diff(phi, HELIX_PHI) ** 2 +
        angle_diff(psi, HELIX_PSI) ** 2
    )
    sheet_cost = (
        angle_diff(phi, SHEET_PHI) ** 2 +
        angle_diff(psi, SHEET_PSI) ** 2
    )
    return TORSION_WEIGHT * float(min(helix_cost, sheet_cost))


def compute_energy(ca_coords: np.ndarray, phi_angles: np.ndarray,
                   psi_angles: np.ndarray) -> float:
    """
    Total energy of a protein conformation.

    Args:
        ca_coords  : [N, 3] Cα coordinates in Angstroms
        phi_angles : [N]    phi backbone dihedral angles in radians
        psi_angles : [N]    psi backbone dihedral angles in radians

    Returns:
        total_energy : float (lower = more stable)
    """
    N = len(ca_coords)
    lj_energy = 0.0
    tor_energy = 0.0

    # ── Pairwise LJ (non-bonded, skip i±1 neighbors) ────────
    for i in range(N):
        for j in range(i + 2, N):  # skip peptide bond neighbors
            dist = float(np.linalg.norm(ca_coords[i] - ca_coords[j]))
            lj_energy += lj_potential(dist)

    # ── Torsion energy per residue ───────────────────────────
    for i in range(N):
        tor_energy += torsion_energy(phi_angles[i], psi_angles[i])

    return lj_energy + tor_energy


def compute_energy_delta(energy_old: float, energy_new: float) -> float:
    """Returns energy change. Negative = improvement."""
    return energy_new - energy_old


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.protein_graph import pdb_to_graph

    print("=" * 50)
    print("ProteinFold-RL — Energy Function Test")
    print("=" * 50)

    for pdb_id, name in [("1L2Y", "Trp-cage"), ("1YRF", "Villin")]:
        path = os.path.join(
            os.path.dirname(__file__),
            f"../data/structures/{pdb_id}.pdb"
        )
        graph = pdb_to_graph(path)
        N = graph.num_nodes
        ca_coords = graph.pos.numpy()

        # Random angles as placeholder (env will track real angles)
        phi = np.random.uniform(-np.pi, np.pi, N)
        psi = np.random.uniform(-np.pi, np.pi, N)

        energy = compute_energy(ca_coords, phi, psi)
        print(f"\n[{name}] ({pdb_id})")
        print(f"  Residues : {N}")
        print(f"  Energy   : {energy:.4f} kcal/mol")
        assert isinstance(energy, float), "Energy must be a float"
        assert not np.isnan(energy),      "Energy must not be NaN"
        assert not np.isinf(energy),      "Energy must not be Inf"
        print(f"  [PASS] Energy function valid ✓")

    print("\n" + "=" * 50)
    print("Energy function ready.")
    print("=" * 50)