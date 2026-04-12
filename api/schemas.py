"""
api/schemas.py
ProteinFold-RL — Pydantic request/response schemas.

Every endpoint's input and output is defined here.
The frontend should treat this file as the ground truth contract.

Author : ProteinFold-RL team
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
import re

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────

class ProteinID(str, Enum):
    """Supported proteins in the current model checkpoint."""
    TRP_CAGE = "1L2Y"
    VILLIN   = "1YRF"


# ── /fold ─────────────────────────────────────────────────────

class FoldRequest(BaseModel):
    """
    Request body for POST /fold.

    Either `pdb_id` (known protein) or `sequence` (custom) must be
    provided — not both, and not neither.
    """
    pdb_id    : Optional[ProteinID] = Field(
        default=None,
        description="PDB ID of a known protein (1L2Y or 1YRF).",
        examples=["1L2Y"],
    )
    sequence  : Optional[str] = Field(
        default=None,
        description=(
            "Custom amino acid sequence in single-letter code "
            "(A-Z, uppercase, 5–50 residues). "
            "Leave null to use pdb_id."
        ),
        examples=["NLYIQWLKDGGPSSGRPPPS"],
    )
    n_steps   : int = Field(
        default=50,
        ge=1,
        le=200,
        description="Number of agent steps to run (1–200).",
    )
    deterministic: bool = Field(
        default=False,
        description=(
            "If True the agent picks the highest-probability action "
            "at every step (greedy). False = stochastic sampling."
        ),
    )

    @field_validator("sequence", mode="before")
    @classmethod
    def validate_sequence(cls, v: Optional[str]) -> Optional[str]:
        """Uppercase, strip whitespace, and validate amino acid alphabet."""
        if v is None:
            return v
        v = v.strip().upper()
        if not (5 <= len(v) <= 50):
            raise ValueError(
                f"Sequence length must be 5–50 residues, got {len(v)}."
            )
        invalid = set(re.sub(r"[ACDEFGHIKLMNPQRSTVWY]", "", v))
        if invalid:
            raise ValueError(
                f"Invalid amino acid characters: {sorted(invalid)}. "
                "Use standard single-letter codes."
            )
        return v

    def model_post_init(self, __context) -> None:  # noqa: N802
        """Enforce exactly one of pdb_id / sequence."""
        if self.pdb_id is None and self.sequence is None:
            raise ValueError(
                "Provide either 'pdb_id' (1L2Y or 1YRF) "
                "or 'sequence' (custom amino acid string)."
            )
        if self.pdb_id is not None and self.sequence is not None:
            raise ValueError(
                "Provide 'pdb_id' OR 'sequence', not both."
            )


class StepSnapshot(BaseModel):
    """Energy/RMSD snapshot at a single agent step."""
    step      : int   = Field(description="Step index (0 = initial).")
    energy    : float = Field(description="Energy in kcal/mol.")
    rmsd      : float = Field(description="RMSD vs native structure in Å.")
    has_clash : bool  = Field(description="Whether a steric clash was detected.")
    reward    : float = Field(description="Reward received at this step.")


class FoldResponse(BaseModel):
    """
    Full response from POST /fold.

    Contains per-step trajectory, before/after PDB strings,
    and a summary of key metrics.
    """
    job_id          : str   = Field(description="Unique identifier for this fold run.")
    protein          : str   = Field(description="PDB ID or 'custom'.")
    n_residues       : int   = Field(description="Number of residues in the chain.")
    steps_run        : int   = Field(description="Actual steps executed.")
    initial_energy   : float = Field(description="Energy at step 0 (kcal/mol).")
    final_energy     : float = Field(description="Energy at last step (kcal/mol).")
    energy_drop      : float = Field(description="initial_energy − final_energy.")
    final_rmsd       : float = Field(description="RMSD vs native at last step (Å).")
    best_rmsd        : float = Field(description="Lowest RMSD achieved during run (Å).")
    trajectory       : List[StepSnapshot] = Field(
        description="Per-step energy/RMSD/clash/reward snapshots."
    )
    initial_pdb      : str   = Field(description="PDB string of starting conformation.")
    final_pdb        : str   = Field(description="PDB string of final conformation.")
    native_pdb       : str   = Field(
        description=(
            "PDB string of native structure. "
            "Empty string for custom sequences (no reference)."
        )
    )
    converged        : bool  = Field(
        description="True if agent terminated early due to energy convergence."
    )


# ── /results ──────────────────────────────────────────────────

class EpisodeSummary(BaseModel):
    """One row of the training log."""
    episode      : int
    protein      : str
    total_reward : float
    final_energy : float
    rmsd         : float
    steps        : int
    policy_loss  : float
    value_loss   : float
    entropy      : float


class TrainingResultsResponse(BaseModel):
    """Response from GET /results."""
    total_episodes  : int
    best_rmsd       : float
    best_energy     : float
    avg_rmsd_last50 : float
    avg_energy_last50: float
    episodes        : List[EpisodeSummary]


# ── /best-episode ─────────────────────────────────────────────

class TrajectoryStep(BaseModel):
    """One step of the best recorded trajectory."""
    step      : int
    energy    : float
    reward    : float
    has_clash : bool


class BestEpisodeResponse(BaseModel):
    """Response from GET /best-episode."""
    best_rmsd   : float
    best_energy : float
    trajectory  : List[TrajectoryStep]


# ── /health ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response from GET /health."""
    status          : str   = Field(examples=["ok"])
    model_loaded    : bool
    checkpoint_path : str
    supported_proteins: List[str]
    version         : str   = Field(examples=["2.0.0"])


# ── /compare ─────────────────────────────────────────────────

class AgentComparisonResponse(BaseModel):
    """Response from GET /compare — trained vs random baseline stats."""
    random_avg_rmsd    : float
    random_avg_energy  : float
    trained_avg_rmsd   : float
    trained_avg_energy : float
    trained_best_rmsd  : float
    rmsd_improvement   : float
    energy_improvement : float


# ── Error ─────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standardised error envelope returned on 4xx / 5xx."""
    detail  : str
    code    : str = Field(
        description="Machine-readable error code.",
        examples=["INVALID_SEQUENCE", "MODEL_NOT_LOADED", "FOLD_FAILED"],
    )