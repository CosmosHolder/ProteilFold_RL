import numpy as np

# Minimum allowed distance between non-bonded Cα atoms
CLASH_THRESHOLD = 1.5   # Angstroms — hard clash
SOFT_THRESHOLD  = 3.0   # Angstroms — soft warning (not penalized, just tracked)


def detect_clashes(ca_coords: np.ndarray) -> dict:
    """
    Detect steric clashes between non-bonded Cα atoms.

    Args:
        ca_coords : [N, 3] Cα coordinates in Angstroms

    Returns:
        dict with keys:
            has_clash     : bool   — True if any hard clash exists
            clash_count   : int    — number of clashing pairs
            soft_count    : int    — number of soft (warning) pairs
            min_dist      : float  — minimum pairwise distance found
            clash_pairs   : list   — list of (i, j) clashing pairs
    """
    N = len(ca_coords)
    clash_pairs = []
    soft_pairs  = []
    min_dist    = float('inf')

    for i in range(N):
        for j in range(i + 2, N):  # skip peptide bond neighbors (i±1)
            diff = ca_coords[i] - ca_coords[j]
            dist = float(np.linalg.norm(diff))

            if dist < min_dist:
                min_dist = dist

            if dist < CLASH_THRESHOLD:
                clash_pairs.append((i, j))
            elif dist < SOFT_THRESHOLD:
                soft_pairs.append((i, j))

    return {
        "has_clash"   : len(clash_pairs) > 0,
        "clash_count" : len(clash_pairs),
        "soft_count"  : len(soft_pairs),
        "min_dist"    : min_dist if min_dist < float('inf') else 0.0,
        "clash_pairs" : clash_pairs
    }


def is_valid_conformation(ca_coords: np.ndarray) -> bool:
    """Quick boolean check — True if no hard clashes."""
    result = detect_clashes(ca_coords)
    return not result["has_clash"]


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.protein_graph import pdb_to_graph

    print("=" * 50)
    print("ProteinFold-RL — Clash Detection Test")
    print("=" * 50)

    # ── Test 1: Native structure (should have no clashes) ───
    for pdb_id, name in [("1L2Y", "Trp-cage"), ("1YRF", "Villin")]:
        path = os.path.join(
            os.path.dirname(__file__),
            f"../data/structures/{pdb_id}.pdb"
        )
        graph = pdb_to_graph(path)
        ca_coords = graph.pos.numpy()
        result = detect_clashes(ca_coords)

        print(f"\n[{name}] Native structure:")
        print(f"  Hard clashes : {result['clash_count']}")
        print(f"  Soft clashes : {result['soft_count']}")
        print(f"  Min dist     : {result['min_dist']:.3f} Å")
        print(f"  Valid        : {not result['has_clash']}")
        assert not result["has_clash"], f"Native structure should not clash!"
        print(f"  [PASS] No clashes in native structure ✓")

    # ── Test 2: Artificial clash (should be detected) ───────
    print(f"\n[TEST] Artificial clash detection:")
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],   # peptide neighbor — skip
        [1.0, 0.0, 0.0],   # non-bonded, same position = clash
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ], dtype=np.float32)
    result = detect_clashes(coords)
    assert result["has_clash"], "Should detect artificial clash!"
    print(f"  Clash detected: {result['clash_count']} pair(s) ✓")
    print(f"  [PASS] Clash detection working ✓")

    print("\n" + "=" * 50)
    print("Clash detection ready.")
    print("=" * 50)