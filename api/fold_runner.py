"""
api/fold_runner.py
ProteinFold-RL — Inference engine.

Wraps FoldEnv + GNNPolicyNetwork into a clean run_fold() function
that returns structured data.  No FastAPI imports here — this layer
is independently testable.

Author : ProteinFold-RL team
"""

from __future__ import annotations

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
    obs, info = env.reset()

    # ── Capture initial state ──────────────────────────────────
    seq             = _decode_sequence_from_graph(env)
    initial_coords  = env.ca_coords.copy()
    initial_energy  = float(env.current_energy)
    initial_rmsd    = _compute_rmsd(initial_coords, env.native_coords)

    initial_pdb = coords_to_pdb_string(initial_coords, seq)
    native_pdb  = coords_to_pdb_string(env.native_coords, seq)

    # ── Trajectory containers ──────────────────────────────────
    trajectory: List[StepSnapshot] = []
    trajectory.append(StepSnapshot(
        step=0,
        energy=initial_energy,
        rmsd=initial_rmsd,
        has_clash=False,
        reward=0.0,
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
        initial_pdb=initial_pdb,
        final_pdb=final_pdb,
        native_pdb=native_pdb,
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