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
# ── Ramachandran cache ─────────────────────────────────────────
_ramachandran_cache = None

@router.get(
    "/ramachandran",
    summary="Phi/psi angles for Ramachandran plot",
)
def get_ramachandran():
    """Return phi/psi angles — computed once, cached forever."""
    global _ramachandran_cache

    # Return cached result immediately if available
    if _ramachandran_cache is not None:
        return _ramachandran_cache

    mm = get_model_manager()
    if not mm.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": "Model not loaded.", "code": "MODEL_NOT_LOADED"},
        )

    from env.fold_env import FoldEnv as _FoldEnv

    N_EPISODES = 5  # reduced from 20 — still meaningful, much faster
    pdb_id     = "1L2Y"

    native_angles  = []
    trained_angles = []
    random_angles  = []

    # Native angles
    env_native = _FoldEnv(pdb_id=pdb_id)
    env_native.reset()
    for phi, psi in zip(env_native.phi_angles, env_native.psi_angles):
        native_angles.append({
            "phi": round(float(np.degrees(phi)), 2),
            "psi": round(float(np.degrees(psi)), 2),
        })

    # Trained agent — 5 episodes
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
                    "phi": round(float(np.degrees(phi)), 2),
                    "psi": round(float(np.degrees(psi)), 2),
                })

    # Random baseline — 5 episodes
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
                    "phi": round(float(np.degrees(phi)), 2),
                    "psi": round(float(np.degrees(psi)), 2),
                })

    _ramachandran_cache = {
        "native" : native_angles,
        "trained": trained_angles,
        "random" : random_angles,
    }

    return _ramachandran_cache


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
# ── GET /critical-points ──────────────────────────────────────

@router.get(
    "/critical-points",
    summary="Critical folding events for the best episode",
)
def get_critical_points():
    mm = get_model_manager()
    if not mm.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": "Model not loaded.", "code": "MODEL_NOT_LOADED"},
        )

    from env.fold_env import FoldEnv as _FoldEnv

    pdb_id = "1L2Y"
    env    = _FoldEnv(pdb_id=pdb_id)
    obs, info = env.reset()

    steps_data = []
    done       = False

    while not done:
        graph = env.get_graph()
        with torch.no_grad():
            action, _, _, _ = mm.policy.get_action(graph, deterministic=True)
        action = action % env.action_dim
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        diff = env.ca_coords - env.native_coords
        rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

        steps_data.append({
            "step"        : int(step_info["step"]),
            "energy"      : float(step_info["energy"]),
            "energy_delta": float(step_info.get("energy_delta", 0.0)),
            "has_clash"   : bool(step_info["has_clash"]),
            "reward"      : float(reward),
            "rmsd"        : round(rmsd, 4),
        })

    if not steps_data:
        raise HTTPException(
            status_code=500,
            detail={"detail": "Episode produced no steps.", "code": "EMPTY_EPISODE"},
        )

    start_energy     = steps_data[0]["energy"]
    end_energy       = steps_data[-1]["energy"]
    start_rmsd       = steps_data[0]["rmsd"]
    rmsd_threshold   = start_rmsd * 0.80
    critical_points  = []
    best_rmsd_so_far = start_rmsd
    energies         = [s["energy"] for s in steps_data]

    for i, s in enumerate(steps_data):

        if s["energy_delta"] < -1.0:
            critical_points.append({
                "step"        : s["step"],
                "type"        : "big",
                "badge"       : "Big Energy Drop",
                "title"       : f"Energy fell {abs(s['energy_delta']):.1f} kcal/mol",
                "description" : (
                    f"A single dihedral angle adjustment dropped energy by "
                    f"{abs(s['energy_delta']):.2f} kcal/mol — the agent found a "
                    f"significantly more stable backbone conformation."
                ),
                "energy"      : s["energy"],
                "energy_delta": s["energy_delta"],
                "reward"      : s["reward"],
                "rmsd"        : s["rmsd"],
            })

        elif s["rmsd"] < best_rmsd_so_far * 0.90 and s["rmsd"] > 0:
            label = "RMSD <2Å" if s["rmsd"] < 2.0 else "RMSD Improved"
            critical_points.append({
                "step"        : s["step"],
                "type"        : "rmsd",
                "badge"       : label,
                "title"       : f"RMSD improved to {s['rmsd']:.3f} Å",
                "description" : (
                    f"Cα RMSD vs native structure dropped to {s['rmsd']:.3f} Å — "
                    f"a {((best_rmsd_so_far - s['rmsd']) / best_rmsd_so_far * 100):.0f}% "
                    f"improvement from the previous best ({best_rmsd_so_far:.2f} Å)."
                    + (" Near-native conformation — +15.0 RMSD bonus triggered!"
                       if s["rmsd"] < 2.0 else
                       f" Best RMSD so far: {s['rmsd']:.3f} Å.")
                ),
                "energy"      : s["energy"],
                "energy_delta": s["energy_delta"],
                "reward"      : s["reward"],
                "rmsd"        : s["rmsd"],
            })
            best_rmsd_so_far = s["rmsd"]

        elif s["has_clash"]:
            critical_points.append({
                "step"        : s["step"],
                "type"        : "clash",
                "badge"       : "Steric Clash",
                "title"       : "Atoms came within 1.5Å — move reverted",
                "description" : (
                    "Two Cα atoms collided below the 1.5Å clash threshold. "
                    "The dihedral change was reverted. The agent received a "
                    "−2.0 penalty and learned to avoid this region."
                ),
                "energy"      : s["energy"],
                "energy_delta": s["energy_delta"],
                "reward"      : s["reward"],
                "rmsd"        : s["rmsd"],
            })

        if i >= 5:
            window   = energies[i - 5 : i]
            max_diff = max(window) - min(window)
            if max_diff < 0.5:
                already_conv = any(cp["type"] == "conv" for cp in critical_points)
                if not already_conv:
                    critical_points.append({
                        "step"        : s["step"],
                        "type"        : "conv",
                        "badge"       : "Convergence",
                        "title"       : f"Energy stabilised at {s['energy']:.1f} kcal/mol",
                        "description" : (
                            f"Energy varied by only {max_diff:.3f} kcal/mol over 5 "
                            f"consecutive steps — local energy minimum found. "
                            f"Episode terminated at step {s['step']}."
                        ),
                        "energy"      : s["energy"],
                        "energy_delta": s["energy_delta"],
                        "reward"      : s["reward"],
                        "rmsd"        : s["rmsd"],
                    })

    critical_points.sort(key=lambda x: x["step"])

    big_drops    = sum(1 for cp in critical_points if cp["type"] == "big")
    rmsd_crosses = sum(1 for cp in critical_points if cp["type"] == "rmsd")
    clashes      = sum(1 for cp in critical_points if cp["type"] == "clash")
    convergences = sum(1 for cp in critical_points if cp["type"] == "conv")

    summary = {
        "protein"            : pdb_id,
        "total_steps"        : len(steps_data),
        "total_events"       : len(critical_points),
        "big_drops"          : big_drops,
        "rmsd_crossings"     : rmsd_crosses,
        "clashes"            : clashes,
        "convergences"       : convergences,
        "start_energy"       : round(start_energy, 3),
        "end_energy"         : round(end_energy, 3),
        "total_drop"         : round(start_energy - end_energy, 3),
        "best_rmsd"          : round(min(s["rmsd"] for s in steps_data), 3),
        "start_rmsd"         : round(start_rmsd, 3),
        "rmsd_threshold_used": round(rmsd_threshold, 3),
    }

    return {
        "summary"        : summary,
        "trajectory"     : steps_data,
        "critical_points": critical_points,
    }


# ── GET /critical-points ───────────────────────────────────────

# Detection thresholds — must match fold_env.py constants
_BIG_ENERGY_DROP   = 1.0    # kcal/mol — same as ENERGY_DROP_BIG in fold_env
_RMSD_THRESHOLD    = 2.0    # Å — same as RMSD_THRESHOLD in fold_env
_CONVERGE_WINDOW   = 5      # steps — same as energy_history length
_CONVERGE_EPSILON  = 0.01   # kcal/mol — same as ENERGY_CONVERGE in fold_env

# Type labels sent to the frontend
_TYPE_BIG   = "big"
_TYPE_RMSD  = "rmsd"
_TYPE_CLASH = "clash"
_TYPE_CONV  = "conv"


def _load_best_trajectory_rich() -> list:
    """
    Load best_trajectory.csv into a list of dicts.
    Requires the new columns written by the updated eval.py:
      step, energy, energy_delta, rmsd, reward, has_clash
    Falls back gracefully if old columns are present.
    """
    if not os.path.exists(BEST_TRAJ_LOG):
        raise FileNotFoundError(
            f"Best trajectory log not found at {BEST_TRAJ_LOG}. "
            "Run eval.py first."
        )

    rows = []
    with open(BEST_TRAJ_LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step"        : _safe_int(row.get("step", "0")),
                "energy"      : _safe_float(row.get("energy", "0")),
                "energy_delta": _safe_float(row.get("energy_delta", "0")),
                "rmsd"        : _safe_float(row.get("rmsd", "0")),
                "reward"      : _safe_float(row.get("reward", "0")),
                "has_clash"   : bool(_safe_int(row.get("has_clash", "0"))),
            })
    return rows


def _detect_critical_points(rows: list) -> list:
    """
    Scan trajectory rows and detect critical events.

    Detection logic (in priority order per step — one event per step):
      1. Steric clash          → has_clash == True
      2. RMSD threshold cross  → rmsd drops below _RMSD_THRESHOLD
                                  for the first time
      3. Big energy drop       → energy_delta < -_BIG_ENERGY_DROP
      4. Convergence           → last _CONVERGE_WINDOW steps all within
                                  _CONVERGE_EPSILON of each other
    """
    events = []
    rmsd_crossed = False   # track first crossing only
    n = len(rows)

    for i, row in enumerate(rows):
        step         = row["step"]
        energy       = row["energy"]
        energy_delta = row["energy_delta"]
        rmsd         = row["rmsd"]
        reward       = row["reward"]
        has_clash    = row["has_clash"]

        # ── 1. Clash ──────────────────────────────────────────
        if has_clash:
            events.append({
                "step"        : step,
                "type"        : _TYPE_CLASH,
                "badge"       : "Steric Clash",
                "energy"      : energy,
                "energy_delta": energy_delta,
                "rmsd"        : rmsd,
                "reward"      : reward,
            })
            continue  # one event per step

        # ── 2. RMSD threshold crossing (first time only) ──────
        if not rmsd_crossed and rmsd < _RMSD_THRESHOLD:
            rmsd_crossed = True
            events.append({
                "step"        : step,
                "type"        : _TYPE_RMSD,
                "badge"       : "RMSD Threshold",
                "energy"      : energy,
                "energy_delta": energy_delta,
                "rmsd"        : rmsd,
                "reward"      : reward,
            })
            continue

        # ── 3. Big energy drop ────────────────────────────────
        if energy_delta < -_BIG_ENERGY_DROP:
            events.append({
                "step"        : step,
                "type"        : _TYPE_BIG,
                "badge"       : "Big Energy Drop",
                "energy"      : energy,
                "energy_delta": energy_delta,
                "rmsd"        : rmsd,
                "reward"      : reward,
            })
            continue

        # ── 4. Convergence (check last CONVERGE_WINDOW steps) ─
        if i >= _CONVERGE_WINDOW:
            window = [rows[j]["energy"] for j in
                      range(i - _CONVERGE_WINDOW + 1, i + 1)]
            span   = max(window) - min(window)
            if span < _CONVERGE_EPSILON:
                events.append({
                    "step"        : step,
                    "type"        : _TYPE_CONV,
                    "badge"       : "Convergence",
                    "energy"      : energy,
                    "energy_delta": energy_delta,
                    "rmsd"        : rmsd,
                    "reward"      : reward,
                })
                # Only record first convergence detection
                break

    return events


def _build_summary(rows: list, events: list) -> dict:
    """Compute summary statistics for the strip at the top of the page."""
    if not rows:
        return {}

    start_energy = rows[0]["energy"]
    end_energy   = rows[-1]["energy"]
    total_drop   = round(start_energy - end_energy, 3)

    big_drops    = sum(1 for e in events if e["type"] == _TYPE_BIG)
    rmsd_crosses = sum(1 for e in events if e["type"] == _TYPE_RMSD)
    clashes      = sum(1 for e in events if e["type"] == _TYPE_CLASH)
    convs        = sum(1 for e in events if e["type"] == _TYPE_CONV)

    best_rmsd    = min(r["rmsd"] for r in rows) if rows else 0.0

    return {
        "total_steps"    : len(rows),
        "total_events"   : len(events),
        "big_drops"      : big_drops,
        "rmsd_crossings" : rmsd_crosses,
        "clashes"        : clashes,
        "convergences"   : convs,
        "start_energy"   : round(start_energy, 3),
        "end_energy"     : round(end_energy,   3),
        "total_drop"     : total_drop,
        "best_rmsd"      : round(best_rmsd,    4),
    }


@router.get(
    "/critical-points",
    summary="Critical folding events from the best episode",
    description=(
        "Reads `logs/best_trajectory.csv` (written by eval.py) and returns "
        "detected critical events: big energy drops, RMSD threshold crossings, "
        "steric clashes, and the convergence point. "
        "Also returns the full per-step trajectory for the main chart. "
        "Run eval.py first to generate the file."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Trajectory log not found."},
    },
)
def get_critical_points():
    """
    Detect and return critical folding events from the best episode.

    Response shape
    --------------
    {
      "summary": {
        "total_steps": int,
        "total_events": int,
        "big_drops": int,
        "rmsd_crossings": int,
        "clashes": int,
        "convergences": int,
        "start_energy": float,
        "end_energy": float,
        "total_drop": float,
        "best_rmsd": float
      },
      "trajectory": [
        {"step": int, "energy": float, "energy_delta": float,
         "rmsd": float, "reward": float, "has_clash": bool}
      ],
      "critical_points": [
        {"step": int, "type": str, "badge": str,
         "energy": float, "energy_delta": float,
         "rmsd": float, "reward": float}
      ]
    }
    """
    try:
        rows = _load_best_trajectory_rich()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"detail": str(exc), "code": "TRAJECTORY_NOT_FOUND"},
        ) from exc

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"detail": "Trajectory file is empty.", "code": "TRAJECTORY_EMPTY"},
        )

    events  = _detect_critical_points(rows)
    summary = _build_summary(rows, events)

    return {
        "summary"        : summary,
        "trajectory"     : rows,
        "critical_points": events,
    }
