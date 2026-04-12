"""
api/routes/fold.py
POST /fold — run the trained agent on a protein.

This is the most important endpoint.  It:
  1. Validates the request (Pydantic does this automatically).
  2. Checks the model is loaded.
  3. Gets a fresh FoldEnv for the requested protein.
  4. Calls fold_runner.run_fold() and returns the result.

All heavy logic lives in fold_runner.py — this file stays thin.

Author : ProteinFold-RL team
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from api.fold_runner import run_fold
from api.model_manager import get_model_manager
from api.schemas import FoldRequest, FoldResponse, ErrorResponse

logger = logging.getLogger("proteinfold.routes.fold")
router = APIRouter(tags=["Folding"])


@router.post(
    "/fold",
    response_model=FoldResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the folding agent on a protein",
    description=(
        "Send a protein (by PDB ID or custom sequence) and the number of "
        "agent steps to run. Returns per-step energy/RMSD trajectory, "
        "before/after/native PDB strings, and summary metrics.\n\n"
        "**Supported PDB IDs:** `1L2Y` (Trp-cage, 20 res), "
        "`1YRF` (Villin, 35 res).\n\n"
        "**Custom sequences:** 5–50 residues, standard AA single-letter codes."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Bad request (invalid input)."},
        503: {"model": ErrorResponse, "description": "Model not loaded yet."},
        500: {"model": ErrorResponse, "description": "Internal error during folding."},
    },
)
def fold_protein(request: FoldRequest) -> FoldResponse:
    """
    Run the trained PPO agent on the requested protein.

    - **pdb_id**: `1L2Y` or `1YRF`
    - **sequence**: custom AA string (5–50 residues)
    - **n_steps**: 1–200
    - **deterministic**: greedy action selection if `true`
    """
    mm = get_model_manager()

    # ── Guard: model must be loaded ────────────────────────────
    if not mm.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "detail": "Model is still loading. Retry in a few seconds.",
                "code": "MODEL_NOT_LOADED",
            },
        )

    # ── Get protein environment ────────────────────────────────
    pdb_id = request.pdb_id.value if request.pdb_id else None

    if pdb_id is None:
        # Custom sequence — not yet supported in this checkpoint
        # (requires dynamic graph construction from sequence alone,
        # which needs a secondary structure predictor for coordinates).
        # We return a clear error rather than silently using wrong coords.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": (
                    "Custom sequence folding is not yet available. "
                    "Use pdb_id='1L2Y' or pdb_id='1YRF'."
                ),
                "code": "CUSTOM_SEQUENCE_UNSUPPORTED",
            },
        )

    try:
        env = mm.get_env(pdb_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": str(exc), "code": "UNKNOWN_PROTEIN"},
        ) from exc

    # ── Run agent ──────────────────────────────────────────────
    try:
        result = run_fold(
            request=request,
            policy=mm.policy,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[/fold] Unhandled error during fold: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": f"Folding failed: {exc}",
                "code": "FOLD_FAILED",
            },
        ) from exc

    return result