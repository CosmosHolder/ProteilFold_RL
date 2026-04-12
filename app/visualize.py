import numpy as np
import os


def coords_to_pdb_string(ca_coords: np.ndarray, seq: list) -> str:
    """
    Convert Cα coordinates + sequence to minimal PDB format string
    for py3Dmol rendering.

    Args:
        ca_coords : [N, 3] numpy array
        seq       : list of residue names (e.g. ['ALA', 'GLY', ...])

    Returns:
        pdb_string : str in PDB ATOM format
    """
    lines = []
    for i, (coord, resname) in enumerate(zip(ca_coords, seq)):
        x, y, z = coord
        lines.append(
            f"ATOM  {i+1:5d}  CA  {resname:3s} A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines)


def load_trajectory(traj_path: str) -> list:
    """
    Load best trajectory from CSV.
    Returns list of dicts with step, energy, reward.
    """
    import csv
    trajectory = []
    with open(traj_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory.append({
                "step"      : int(row["step"]),
                "energy"    : float(row["energy"]),
                "reward"    : float(row["reward"]),
                "has_clash" : int(row["has_clash"]),
            })
    return trajectory


def load_training_log(log_path: str) -> dict:
    """
    Load training log CSV.
    Returns dict of lists for each metric.
    """
    import csv
    log = {
        "episode"      : [],
        "total_reward" : [],
        "final_energy" : [],
        "rmsd"         : [],
        "steps"        : [],
    }
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            log["episode"].append(int(row["episode"]))
            log["total_reward"].append(float(row["total_reward"]))
            log["final_energy"].append(float(row["final_energy"]))
            log["rmsd"].append(float(row["rmsd"]))
            log["steps"].append(int(row["steps"]))
    return log


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.protein_graph import pdb_to_graph

    print("=" * 50)
    print("ProteinFold-RL — Visualize Utility Test")
    print("=" * 50)

    graph = pdb_to_graph("data/structures/1L2Y.pdb")
    coords = graph.pos.numpy()
    seq    = graph.seq

    pdb_str = coords_to_pdb_string(coords, seq)
    print(f"\n  PDB string lines : {len(pdb_str.splitlines())}")
    assert "ATOM" in pdb_str, "PDB string missing ATOM records"
    assert "END"  in pdb_str, "PDB string missing END record"
    print(f"  [PASS] PDB string generation ✓")

    log = load_training_log("logs/training_log.csv")
    print(f"\n  Training episodes loaded : {len(log['episode'])}")
    assert len(log["episode"]) == 500, "Expected 500 episodes"
    print(f"  [PASS] Training log loaded ✓")

    traj = load_trajectory("logs/best_trajectory.csv")
    print(f"\n  Trajectory steps loaded  : {len(traj)}")
    print(f"  [PASS] Trajectory loaded ✓")

    print("\n" + "=" * 50)
    print("Visualize utilities ready.")
    print("=" * 50)