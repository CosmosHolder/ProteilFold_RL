"""
api/fold_runner.py
ProteinFold-RL — Inference engine.

Wraps FoldEnv + GNNPolicyNetwork into a clean run_fold() function
that returns structured data.  No FastAPI imports here — this layer
is independently testable.

Author : ProteinFold-RL team
"""

from __future__ import annotations

import types
import uuid
import logging
from typing import List, Tuple

import numpy as np
import torch

from env.fold_env import FoldEnv
from model.gnn_policy import GNNPolicyNetwork
from api.schemas import FoldRequest, FoldResponse, StepSnapshot

logger = logging.getLogger("proteinfold.fold_runner")

# ── Constants ──────────────────────────────────────────────────
MAX_ACTION_DIM = 141 * 2 * 12   # must match model_manager


# ── PDB string builder ─────────────────────────────────────────

# Standard three-letter to one-letter AA mapping (not used for output
# but kept here for reference; we use one-letter codes in the PDB ATOM
# records for simplicity because the frontend only reads coordinates).
_ONE_TO_THREE: dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
_FALLBACK_THREE = "GLY"

# Standard 20 AA one-letter codes (index matches one-hot encoding in FoldEnv)
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def coords_to_pdb_string(
    coords: np.ndarray,
    seq: str,
) -> str:
    """
    Convert Cα coordinates + sequence to a minimal PDB ATOM string.

    Parameters
    ----------
    coords : np.ndarray  shape [N, 3]
    seq    : str         one-letter AA sequence of length N

    Returns
    -------
    str  — multi-line PDB ATOM record string.
    """
    lines: List[str] = []
    for i, (xyz, aa) in enumerate(zip(coords, seq), start=1):
        three = _ONE_TO_THREE.get(aa.upper(), _FALLBACK_THREE)
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        lines.append(
            f"ATOM  {i:5d}  CA  {three} A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines)


def _decode_sequence_from_graph(env: FoldEnv) -> str:
    """
    Recover one-letter sequence from the one-hot node features.
    Node features are [N, 23]: first 20 dims = one-hot AA type.
    """
    aa_onehot = env.native_graph.x[:, :20].numpy()   # [N, 20]
    indices   = aa_onehot.argmax(axis=1)              # [N]
    return "".join(_AA_ALPHABET[i] for i in indices)


def _compute_rmsd(coords: np.ndarray, native: np.ndarray) -> float:
    """Root-mean-square deviation of Cα coordinates."""
    diff = coords - native
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ── Core inference function ────────────────────────────────────
def build_custom_env(sequence: str) -> FoldEnv:
    """
    Build a minimal FoldEnv-like object for a custom sequence.
    Uses extended chain initialization (Option A):
    all Cα atoms placed in a straight line, 3.8Å apart.
    No native structure available — RMSD will return 0.0.
    """
    import torch
    from torch_geometric.data import Data
    from env.energy import compute_energy
    from env.fold_env import (
        N_ANGLES, N_INCREMENTS, MAX_STEPS,
        MAX_CLASHES, ENERGY_CONVERGE,
    )

    N = len(sequence)
    BOND_LENGTH = 3.8

    # Extended chain coordinates — straight line along x-axis
    ca_coords = np.array(
        [[i * BOND_LENGTH, 0.0, 0.0] for i in range(N)],
        dtype=np.float32,
    )

    # Random backbone angles (unfolded-like)
    phi_angles = np.random.uniform(-np.pi, np.pi, N)
    psi_angles = np.random.uniform(-np.pi, np.pi, N)

    # One-hot encode the sequence
    aa_onehot = np.zeros((N, 20), dtype=np.float32)
    for i, aa in enumerate(sequence):
        idx = _AA_ALPHABET.find(aa)
        if idx >= 0:
            aa_onehot[i, idx] = 1.0

    # Build PyG graph
    coords_t = torch.tensor(ca_coords, dtype=torch.float)
    aa_t     = torch.tensor(aa_onehot, dtype=torch.float)
    x        = torch.cat([aa_t, coords_t], dim=1)  # [N, 23]

    edge_src, edge_dst, edge_attrs = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = ca_coords[j] - ca_coords[i]
            dist = float(np.linalg.norm(diff))
            if dist <= 8.0:
                is_peptide = 1.0 if abs(i - j) == 1 else 0.0
                dn = diff / (dist + 1e-8)
                edge_src.append(i)
                edge_dst.append(j)
                edge_attrs.append([dist, dn[0], dn[1], is_peptide])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)
    graph      = Data(
        x=x, edge_index=edge_index,
        edge_attr=edge_attr, pos=coords_t, num_nodes=N,
    )

    # Build a lightweight namespace that mimics FoldEnv's interface
    env             = types.SimpleNamespace()
    env.N           = N
    env.action_dim  = N * N_ANGLES * N_INCREMENTS
    env.ca_coords   = ca_coords
    env.phi_angles  = phi_angles
    env.psi_angles  = psi_angles
    env.native_coords = ca_coords.copy()  # no native — use initial as ref
    env.current_energy = float(compute_energy(
        ca_coords, phi_angles, psi_angles
    ))
    env.energy_history = np.full(5, env.current_energy, dtype=np.float32)
    env.step_count  = 0
    env.clash_count = 0
    env.graph       = graph
    env.sequence    = sequence

    # Attach get_graph method
    env.get_graph = lambda: env.graph

    # Attach step() method — mirrors FoldEnv.step() logic
    def _step(action: int):
        from env.fold_env import (
            N_INCREMENTS, N_ANGLES, ANGLE_STEP,
            R_STEP_PENALTY, R_NO_CLASH, R_ENERGY_BIG,
            R_ENERGY_SMALL, R_ENERGY_UP, ENERGY_DROP_BIG,
            MAX_STEPS, MAX_CLASHES,
        )
        from env.clash_detect import detect_clashes
        from env.energy import compute_energy, compute_energy_delta

        increment = action % N_INCREMENTS
        remainder = action // N_INCREMENTS
        angle_type = remainder % N_ANGLES
        residue_idx = min(remainder // N_ANGLES, env.N - 1)

        old_phi = env.phi_angles.copy()
        old_psi = env.psi_angles.copy()

        delta = ANGLE_STEP * (increment - N_INCREMENTS // 2)
        if angle_type == 0:
            env.phi_angles[residue_idx] += delta
            env.phi_angles[residue_idx] = float(
                (env.phi_angles[residue_idx] + np.pi) % (2 * np.pi) - np.pi
            )
        else:
            env.psi_angles[residue_idx] += delta
            env.psi_angles[residue_idx] = float(
                (env.psi_angles[residue_idx] + np.pi) % (2 * np.pi) - np.pi
            )

        clash_result = detect_clashes(env.ca_coords)
        has_clash = clash_result["has_clash"]

        if has_clash:
            env.clash_count += 1
            env.phi_angles = old_phi
            env.psi_angles = old_psi

        new_energy = compute_energy(
            env.ca_coords, env.phi_angles, env.psi_angles
        )
        energy_delta = compute_energy_delta(env.current_energy, new_energy)

        reward = R_STEP_PENALTY
        if has_clash:
            reward += -2.0
        else:
            reward += R_NO_CLASH
            if energy_delta < -ENERGY_DROP_BIG:
                reward += R_ENERGY_BIG
            elif energy_delta < 0:
                reward += R_ENERGY_SMALL
            else:
                reward += R_ENERGY_UP

        if not has_clash:
            env.current_energy = new_energy

        env.energy_history = np.roll(env.energy_history, -1)
        env.energy_history[-1] = env.current_energy
        env.step_count += 1

        converged = (
                len(set(np.round(env.energy_history, 3))) == 1
                and env.step_count > 5
        )
        terminated = (
                env.step_count >= MAX_STEPS or
                env.clash_count >= MAX_CLASHES or
                converged
        )

        info = {
            "energy": env.current_energy,
            "energy_delta": energy_delta,
            "step": env.step_count,
            "clash_count": env.clash_count,
            "has_clash": has_clash,
        }
        return {}, reward, terminated, False, info

    env.step = _step

    return env

def run_fold(
    request: FoldRequest,
    policy: GNNPolicyNetwork,
    env: FoldEnv,
) -> FoldResponse:
    """
    Run the trained agent on a protein for `request.n_steps` steps.

    Parameters
    ----------
    request : FoldRequest   — validated Pydantic request object.
    policy  : GNNPolicyNetwork — loaded, eval-mode policy network.
    env     : FoldEnv          — fresh env for the requested protein.

    Returns
    -------
    FoldResponse — fully populated result object.
    """
    job_id = str(uuid.uuid4())
    logger.info(
        "[FoldRunner] job=%s protein=%s steps=%d deterministic=%s",
        job_id, request.pdb_id or "custom",
        request.n_steps, request.deterministic,
    )

    # ── Reset environment ──────────────────────────────────────
    # Custom envs (SimpleNamespace) don't have .reset() — skip it
    if hasattr(env, 'reset'):
        obs, info = env.reset()

    # ── Capture initial state ──────────────────────────────────
    # Custom sequence: use sequence directly from request
    # Known protein: decode from graph node features
    if request.sequence:
        seq = request.sequence
    else:
        seq = _decode_sequence_from_graph(env)

    initial_coords = env.ca_coords.copy()
    initial_energy = float(env.current_energy)
    initial_rmsd = _compute_rmsd(initial_coords, env.native_coords)

    initial_pdb = coords_to_pdb_string(initial_coords, seq)
    # No native structure for custom sequences
    native_pdb = coords_to_pdb_string(env.native_coords, seq) \
        if not request.sequence else ""

    # ── Trajectory containers ──────────────────────────────────
    trajectory: List[StepSnapshot] = []
    trajectory.append(StepSnapshot(
        step=0,
        energy=initial_energy,
        rmsd=initial_rmsd,
        has_clash=False,
        reward=0.0,
        coords=[[round(float(x), 3) for x in row]
                for row in initial_coords.tolist()],
    ))

    best_rmsd   = initial_rmsd
    converged   = False
    steps_run   = 0

    # ── Agent loop ─────────────────────────────────────────────
    done = False
    while not done and steps_run < request.n_steps:
        graph = env.get_graph()

        with torch.no_grad():
            action, _log_prob, _value, _entropy = policy.get_action(
                graph, deterministic=request.deterministic
            )

        # Clamp action to valid range for this protein
        action = action % env.action_dim

        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        steps_run += 1

        current_rmsd = _compute_rmsd(env.ca_coords, env.native_coords)
        if current_rmsd < best_rmsd:
            best_rmsd = current_rmsd

        trajectory.append(StepSnapshot(
            step=steps_run,
            energy=float(step_info["energy"]),
            rmsd=round(current_rmsd, 4),
            has_clash=bool(step_info["has_clash"]),
            reward=round(float(reward), 4),
            coords=[[round(float(x), 3) for x in row]
                    for row in env.ca_coords.tolist()],
        ))

        # Check convergence flag
        if terminated and step_info["step"] < env.action_dim:
            # terminated before MAX_STEPS → convergence or clash limit
            converged = not (step_info["clash_count"] >= 5)

    # ── Final metrics ──────────────────────────────────────────
    final_coords  = env.ca_coords.copy()
    final_energy  = float(env.current_energy)
    final_rmsd    = _compute_rmsd(final_coords, env.native_coords)
    final_pdb     = coords_to_pdb_string(final_coords, seq)
    energy_drop   = round(initial_energy - final_energy, 4)

    logger.info(
        "[FoldRunner] job=%s done. steps=%d energy %.3f→%.3f "
        "rmsd=%.3fÅ best_rmsd=%.3fÅ converged=%s",
        job_id, steps_run,
        initial_energy, final_energy,
        final_rmsd, best_rmsd, converged,
    )

    return FoldResponse(
        job_id=job_id,
        protein=request.pdb_id.value if request.pdb_id else "custom",
        n_residues=env.N,
        steps_run=steps_run,
        initial_energy=round(initial_energy, 4),
        final_energy=round(final_energy, 4),
        energy_drop=energy_drop,
        final_rmsd=round(final_rmsd, 4),
        best_rmsd=round(best_rmsd, 4),
        trajectory=trajectory,
        energy_curve=[[t.step, t.energy] for t in trajectory],
        initial_pdb=initial_pdb,
        final_pdb=final_pdb,
        native_pdb=native_pdb,
        native_coords=[[round(float(x), 3) for x in row]
                       for row in env.native_coords.tolist()],
        converged=converged,
    )


# ── Comparison runner ──────────────────────────────────────────

def run_comparison(
    policy: GNNPolicyNetwork,
    pdb_id: str = "1L2Y",
    n_episodes: int = 20,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Run N episodes each of trained agent and random baseline.

    Returns
    -------
    trained_rmsds, trained_energies, random_rmsds, random_energies
    """
    from env.fold_env import FoldEnv as _FoldEnv  # local to avoid circular

    trained_rmsds, trained_energies = [], []
    random_rmsds,  random_energies  = [], []

    for _ in range(n_episodes):
        # ── Trained agent ────────────────────────────────────
        env_t = _FoldEnv(pdb_id=pdb_id)
        obs, info = env_t.reset()
        done = False
        while not done:
            graph = env_t.get_graph()
            with torch.no_grad():
                action, _, _, _ = policy.get_action(graph, deterministic=False)
            action = action % env_t.action_dim
            obs, reward, terminated, truncated, info = env_t.step(action)
            done = terminated or truncated
        trained_rmsds.append(_compute_rmsd(env_t.ca_coords, env_t.native_coords))
        trained_energies.append(float(info["energy"]))

        # ── Random baseline ───────────────────────────────────
        env_r = _FoldEnv(pdb_id=pdb_id)
        obs, info = env_r.reset()
        done = False
        while not done:
            action = env_r.action_space.sample()
            obs, reward, terminated, truncated, info = env_r.step(action)
            done = terminated or truncated
        random_rmsds.append(_compute_rmsd(env_r.ca_coords, env_r.native_coords))
        random_energies.append(float(info["energy"]))

    return trained_rmsds, trained_energies, random_rmsds, random_energies