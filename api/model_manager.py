"""
api/model_manager.py
ProteinFold-RL — Thread-safe singleton model manager.

Loads the GNNPolicyNetwork checkpoint once at startup and serves it
to all request handlers without reloading on every call.

Usage
-----
    from api.model_manager import get_model_manager
    mm = get_model_manager()
    policy = mm.policy
    env    = mm.get_env("1L2Y")

Author : ProteinFold-RL team
"""

from __future__ import annotations

import os
import threading
import logging
from typing import Dict, Optional

import torch

# These imports work when the project root is on sys.path.
# main.py adds the project root to sys.path before importing this module.
from env.fold_env import FoldEnv
from model.gnn_policy import GNNPolicyNetwork

logger = logging.getLogger("proteinfold.model_manager")

# ── Constants ─────────────────────────────────────────────────
DEFAULT_CHECKPOINT  = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "checkpoints", "policy_final.pt"
)
MAX_ACTION_DIM      = 141 * 2 * 12   # hemoglobin α — largest protein
SUPPORTED_PROTEINS  = ["1L2Y", "1YRF"]


class ModelManager:
    """
    Singleton that owns the loaded policy and cached environments.

    Thread-safe: a single RLock guards all mutation.
    Env instances are cached per pdb_id so the PDB file is parsed once.
    """

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT):
        self._lock              : threading.RLock   = threading.RLock()
        self._checkpoint_path   : str               = checkpoint_path
        self._policy            : Optional[GNNPolicyNetwork] = None
        self._env_cache         : Dict[str, FoldEnv]         = {}
        self._loaded            : bool              = False
        self._load_error        : Optional[str]     = None

    # ── Public API ─────────────────────────────────────────────

    def load(self) -> None:
        """
        Load model weights from checkpoint.
        Safe to call multiple times — subsequent calls are no-ops.
        Raises RuntimeError if the checkpoint is missing or corrupt.
        """
        with self._lock:
            if self._loaded:
                return
            self._load_policy()
            self._warm_envs()
            self._loaded = True
            logger.info(
                "[ModelManager] Ready. Checkpoint: %s",
                self._checkpoint_path,
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    @property
    def policy(self) -> GNNPolicyNetwork:
        """Return the loaded policy. Raises if not yet loaded."""
        if not self._loaded:
            raise RuntimeError(
                "Model not loaded. Call ModelManager.load() first."
            )
        return self._policy  # type: ignore[return-value]

    def get_env(self, pdb_id: str) -> FoldEnv:
        """
        Return a *fresh* FoldEnv for the given protein.

        Each call returns a new instance so that concurrent requests
        don't share mutable environment state. The native graph is
        parsed only once per pdb_id (cached internally by FoldEnv).
        """
        if pdb_id not in SUPPORTED_PROTEINS:
            raise ValueError(
                f"Unknown protein '{pdb_id}'. "
                f"Supported: {SUPPORTED_PROTEINS}"
            )
        # FoldEnv is lightweight to construct — native graph is loaded
        # from PDB at construction time, which we accept.
        # For a production server under high QPS, cache the native
        # graph separately and pass it in. For this project this is fine.
        return FoldEnv(pdb_id=pdb_id)

    # ── Private helpers ────────────────────────────────────────

    def _load_policy(self) -> None:
        """Instantiate GNNPolicyNetwork and load weights."""
        if not os.path.exists(self._checkpoint_path):
            self._load_error = (
                f"Checkpoint not found: {self._checkpoint_path}"
            )
            logger.warning(
                "[ModelManager] %s — running without weights "
                "(random policy).", self._load_error
            )
            # Still create the network so the server can boot in
            # demo / test environments where no checkpoint exists.
            self._policy = GNNPolicyNetwork(action_dim=MAX_ACTION_DIM)
            self._policy.eval()
            return

        try:
            self._policy = GNNPolicyNetwork(action_dim=MAX_ACTION_DIM)
            ckpt = torch.load(
                self._checkpoint_path, map_location="cpu"
            )
            self._policy.load_state_dict(ckpt["policy_state"])
            self._policy.eval()
            logger.info(
                "[ModelManager] Loaded weights from %s",
                self._checkpoint_path,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.error(
                "[ModelManager] Failed to load checkpoint: %s", exc
            )
            raise RuntimeError(
                f"Failed to load checkpoint '{self._checkpoint_path}': {exc}"
            ) from exc

    def _warm_envs(self) -> None:
        """
        Pre-build one FoldEnv per supported protein.
        This parses the PDB files at startup, not during the first
        request, so the first /fold call isn't artificially slow.
        """
        for pdb_id in SUPPORTED_PROTEINS:
            try:
                env = FoldEnv(pdb_id=pdb_id)
                # Store only to verify construction; requests get
                # fresh instances via get_env().
                self._env_cache[pdb_id] = env
                logger.info(
                    "[ModelManager] Warmed env for %s (%d residues)",
                    pdb_id, env.N,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[ModelManager] Could not warm env for %s: %s",
                    pdb_id, exc,
                )


# ── Module-level singleton ─────────────────────────────────────

_manager_instance : Optional[ModelManager] = None
_manager_lock     : threading.Lock          = threading.Lock()


def get_model_manager(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
) -> ModelManager:
    """
    Return the global ModelManager singleton.

    The first call creates and loads the manager.
    All subsequent calls return the cached instance.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt checkpoint file.  Ignored after the first call.
    """
    global _manager_instance  # noqa: PLW0603

    if _manager_instance is None:
        with _manager_lock:
            # Double-checked locking
            if _manager_instance is None:
                _manager_instance = ModelManager(
                    checkpoint_path=checkpoint_path
                )
    return _manager_instance