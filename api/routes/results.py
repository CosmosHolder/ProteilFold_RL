"""
api/routes/results.py
Training results endpoints:
  GET /results       — full training log (all episodes)
  GET /best-episode  — best recorded trajectory
  GET /compare       — trained agent vs random baseline stats

All data is read from the CSV files written by train.py and eval.py.
No database required — files are the source of truth.

Author : ProteinFold-RL team
"""

from __future__ import annotations

import csv
import logging
import os
from typing import List
import torch

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status

from api.model_manager import get_model_manager
from api.fold_runner import run_comparison
from api.schemas import (
    AgentComparisonResponse,
    BestEpisodeResponse,
    EpisodeSummary,
    TrainingResultsResponse,
    TrajectoryStep,
    ErrorResponse,
)

logger = logging.getLogger("proteinfold.routes.results")
router = APIRouter(tags=["Results"])

# ── File paths ─────────────────────────────────────────────────
_PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAINING_LOG     = os.path.join(_PROJECT_ROOT, "logs", "training_log.csv")
BEST_TRAJ_LOG    = os.path.join(_PROJECT_ROOT, "logs", "best_trajectory.csv")


# ── Helpers ───────────────────────────────────────────────────

def _safe_float(val: str, default: float = 0.0) -> float:
    """Parse a CSV string to float, returning `default` on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: str, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _load_training_log() -> List[EpisodeSummary]:
    """
    Read logs/training_log.csv and return a list of EpisodeSummary.
    Raises FileNotFoundError if the log doesn't exist.
    """
    if not os.path.exists(TRAINING_LOG):
        raise FileNotFoundError(
            f"Training log not found at {TRAINING_LOG}. "
            "Run train.py first."
        )

    episodes: List[EpisodeSummary] = []
    with open(TRAINING_LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(EpisodeSummary(
                episode      = _safe_int(row.get("episode", "0")),
                protein      = row.get("protein", "unknown"),
                total_reward = _safe_float(row.get("total_reward", "0")),
                final_energy = _safe_float(row.get("final_energy", "0")),
                rmsd         = _safe_float(row.get("rmsd", "0")),
                steps        = _safe_int(row.get("steps", "0")),
                policy_loss  = _safe_float(row.get("policy_loss", "0")),
                value_loss   = _safe_float(row.get("value_loss", "0")),
                entropy      = _safe_float(row.get("entropy", "0")),
            ))

    return episodes


def _load_best_trajectory() -> List[TrajectoryStep]:
    """
    Read logs/best_trajectory.csv.
    Raises FileNotFoundError if the file doesn't exist.
    """
    if not os.path.exists(BEST_TRAJ_LOG):
        raise FileNotFoundError(
            f"Best trajectory log not found at {BEST_TRAJ_LOG}. "
            "Run eval.py first."
        )

    steps: List[TrajectoryStep] = []
    with open(BEST_TRAJ_LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(TrajectoryStep(
                step      = _safe_int(row.get("step", "0")),
                energy    = _safe_float(row.get("energy", "0")),
                reward    = _safe_float(row.get("reward", "0")),
                has_clash = bool(_safe_int(row.get("has_clash", "0"))),
            ))

    return steps


# ── GET /results ───────────────────────────────────────────────

@router.get(
    "/results",
    response_model=TrainingResultsResponse,
    summary="Full training log",
    description=(
        "Returns all episode summaries from `logs/training_log.csv`. "
        "Use `limit` to cap the number of episodes returned "
        "(default 500, max 5000)."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Training log not found."},
    },
)
def get_results(
    limit: int = Query(
        default=500,
        ge=1,
        le=5000,
        description="Maximum number of episodes to return.",
    ),
) -> TrainingResultsResponse:
    """Return training metrics for all recorded episodes."""
    try:
        episodes = _load_training_log()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"detail": str(exc), "code": "LOG_NOT_FOUND"},
        ) from exc

    # Apply limit (most recent episodes first for the dashboard)
    episodes_limited = episodes[-limit:]

    all_rmsds    = [e.rmsd         for e in episodes]
    all_energies = [e.final_energy for e in episodes]
    last50_rmsds    = [e.rmsd         for e in episodes[-50:]] or [0.0]
    last50_energies = [e.final_energy for e in episodes[-50:]] or [0.0]

    return TrainingResultsResponse(
        total_episodes    = len(episodes),
        best_rmsd         = round(float(np.min(all_rmsds)),     4),
        best_energy       = round(float(np.min(all_energies)),  4),
        avg_rmsd_last50   = round(float(np.mean(last50_rmsds)),    4),
        avg_energy_last50 = round(float(np.mean(last50_energies)), 4),
        episodes          = episodes_limited,
    )


# ── GET /best-episode ─────────────────────────────────────────

@router.get(
    "/best-episode",
    response_model=BestEpisodeResponse,
    summary="Best recorded trajectory",
    description=(
        "Returns the step-by-step energy/reward trace for the best "
        "episode found during evaluation (lowest RMSD)."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Trajectory log not found."},
    },
)
def get_best_episode() -> BestEpisodeResponse:
    """Return energy trajectory of the best recorded episode."""
    try:
        traj = _load_best_trajectory()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"detail": str(exc), "code": "TRAJECTORY_NOT_FOUND"},
        ) from exc

    if not traj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "detail": "Trajectory file is empty.",
                "code": "TRAJECTORY_EMPTY",
            },
        )

    energies = [t.energy for t in traj]
    best_energy = float(np.min(energies))

    # RMSD isn't in best_trajectory.csv (produced by eval.py)
    # We load it from the training log as a proxy.
    best_rmsd = 0.0
    try:
        episodes = _load_training_log()
        if episodes:
            best_rmsd = float(np.min([e.rmsd for e in episodes]))
    except FileNotFoundError:
        pass  # Return 0.0 if training log is also missing

    return BestEpisodeResponse(
        best_rmsd=round(best_rmsd, 4),
        best_energy=round(best_energy, 4),
        trajectory=traj,
    )


# ── GET /compare ──────────────────────────────────────────────

@router.get(
    "/compare",
    response_model=AgentComparisonResponse,
    summary="Trained agent vs random baseline",
    description=(
        "Runs `n_episodes` episodes of the trained agent AND a random "
        "baseline, then returns comparative metrics. "
        "**Warning:** this is computationally expensive "
        "(default 10 episodes each). Keep `n_episodes` ≤ 20 on HF Spaces."
    ),
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded."},
        500: {"model": ErrorResponse, "description": "Comparison run failed."},
    },
)
def compare_agents(
    pdb_id: str = Query(
        default="1L2Y",
        description="Which protein to compare on (1L2Y or 1YRF).",
    ),
    n_episodes: int = Query(
        default=10,
        ge=1,
        le=20,
        description="Episodes per agent (1–20).",
    ),
) -> AgentComparisonResponse:
    """Live comparison: trained agent vs random baseline."""
    mm = get_model_manager()

    if not mm.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": "Model not loaded yet.", "code": "MODEL_NOT_LOADED"},
        )

    if pdb_id not in ("1L2Y", "1YRF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": f"Unknown protein '{pdb_id}'. Use 1L2Y or 1YRF.",
                "code": "UNKNOWN_PROTEIN",
            },
        )

    try:
        tr_rmsds, tr_energies, rnd_rmsds, rnd_energies = run_comparison(
            policy=mm.policy,
            pdb_id=pdb_id,
            n_episodes=n_episodes,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[/compare] Error during comparison: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": str(exc), "code": "COMPARISON_FAILED"},
        ) from exc

    return AgentComparisonResponse(
        random_avg_rmsd    = round(float(np.mean(rnd_rmsds)),    3),
        random_avg_energy  = round(float(np.mean(rnd_energies)), 3),
        trained_avg_rmsd   = round(float(np.mean(tr_rmsds)),     3),
        trained_avg_energy = round(float(np.mean(tr_energies)),  3),
        trained_best_rmsd  = round(float(np.min(tr_rmsds)),      3),
        rmsd_improvement   = round(
            float(np.mean(rnd_rmsds)) - float(np.mean(tr_rmsds)), 3
        ),
        energy_improvement = round(
            float(np.mean(rnd_energies)) - float(np.mean(tr_energies)), 3
        ),
    )
# ── GET /training-log-json ─────────────────────────────────────

@router.get(
    "/training-log-json",
    summary="Training log as JSON array for dashboard charts",
    description="Returns training_log.csv as a JSON array. "
                "The frontend charts.js fetches this directly.",
)
def get_training_log_json(
    limit: int = Query(
        default=1500,
        ge=1,
        le=5000,
        description="Maximum episodes to return.",
    ),
):
    """Return training log as JSON array for the frontend dashboard."""
    try:
        episodes = _load_training_log()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"detail": str(exc), "code": "LOG_NOT_FOUND"},
        ) from exc

    episodes_limited = episodes[-limit:]

    return [
        {
            "episode"     : e.episode,
            "protein"     : e.protein,
            "stage"       : 1,
            "total_reward": e.total_reward,
            "final_energy": e.final_energy,
            "rmsd"        : e.rmsd,
            "steps"       : e.steps,
            "clashes"     : 0,
            "policy_loss" : e.policy_loss,
            "value_loss"  : e.value_loss,
            "entropy"     : e.entropy,
        }
        for e in episodes_limited
    ]
# ── GET /ramachandran ──────────────────────────────────────────

@router.get(
    "/ramachandran",
    summary="Phi/psi angles for Ramachandran plot",
    description=(
        "Runs 20 episodes of trained agent + random baseline on 1L2Y. "
        "Collects all phi/psi angles per step. Also returns native angles. "
        "Feeds the Ramachandran plot on the frontend."
    ),
)
def get_ramachandran():
    """Return phi/psi angles for native, trained, and random agents."""
    mm = get_model_manager()

    if not mm.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": "Model not loaded.", "code": "MODEL_NOT_LOADED"},
        )

    from env.fold_env import FoldEnv as _FoldEnv

    N_EPISODES = 20
    pdb_id     = "1L2Y"

    native_angles  = []
    trained_angles = []
    random_angles  = []

    # ── Native angles from the native graph ───────────────────
    env_native = _FoldEnv(pdb_id=pdb_id)
    env_native.reset()
    for phi, psi in zip(env_native.phi_angles, env_native.psi_angles):
        native_angles.append({
            "phi": round(float(phi), 4),
            "psi": round(float(psi), 4),
        })

    # ── Trained agent ─────────────────────────────────────────
    for _ in range(N_EPISODES):
        env_t = _FoldEnv(pdb_id=pdb_id)
        env_t.reset()
        done = False
        while not done:
            graph = env_t.get_graph()
            with torch.no_grad():
                action, _, _, _ = mm.policy.get_action(
                    graph, deterministic=False
                )
            action = action % env_t.action_dim
            _, _, terminated, truncated, _ = env_t.step(action)
            done = terminated or truncated
            for phi, psi in zip(env_t.phi_angles, env_t.psi_angles):
                trained_angles.append({
                    "phi": round(float(phi), 4),
                    "psi": round(float(psi), 4),
                })

    # ── Random baseline ───────────────────────────────────────
    for _ in range(N_EPISODES):
        env_r = _FoldEnv(pdb_id=pdb_id)
        env_r.reset()
        done = False
        while not done:
            action = env_r.action_space.sample()
            _, _, terminated, truncated, _ = env_r.step(action)
            done = terminated or truncated
            for phi, psi in zip(env_r.phi_angles, env_r.psi_angles):
                random_angles.append({
                    "phi": round(float(phi), 4),
                    "psi": round(float(psi), 4),
                })

    return {
        "native" : native_angles,
        "trained": trained_angles,
        "random" : random_angles,
    }


# ── GET /coords ────────────────────────────────────────────────

@router.get(
    "/coords",
    summary="3D Cα coordinates for molecular viewer",
    description=(
        "Returns best and native Cα coordinates from "
        "logs/best_coords.npy and logs/native_coords.npy. "
        "Feeds the 3D viewer on the frontend."
    ),
)
def get_coords():
    """Return best and native 3D Cα coordinates as JSON."""
    best_path   = os.path.join(_PROJECT_ROOT, "logs", "best_coords.npy")
    native_path = os.path.join(_PROJECT_ROOT, "logs", "native_coords.npy")

    if not os.path.exists(best_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "detail": "best_coords.npy not found. Run eval.py first.",
                "code": "COORDS_NOT_FOUND",
            },
        )

    if not os.path.exists(native_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "detail": "native_coords.npy not found. Run eval.py first.",
                "code": "COORDS_NOT_FOUND",
            },
        )

    best_coords   = np.load(best_path).tolist()
    native_coords = np.load(native_path).tolist()

    return {
        "best"  : [[round(x, 4) for x in row] for row in best_coords],
        "native": [[round(x, 4) for x in row] for row in native_coords],
    }