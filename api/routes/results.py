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