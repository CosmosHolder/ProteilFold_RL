"""
config.py
ProteinFold-RL — Shared Configuration

Single source of truth for constants used across multiple files.
Import from here instead of hardcoding values in each file.

Usage
-----
  from config import MAX_ACTION_DIM, CHECKPOINT_PATH
"""

import os

# ── Action space ──────────────────────────────────────────────
# Largest protein in curriculum is 2HHB: 141 residues
# Action dim = n_residues × 2 angles × 12 increments
# All files (train.py, eval.py, gradio_app.py) must use this value
# so checkpoints load without size mismatch errors.
MAX_ACTION_DIM = 141 * 2 * 12   # 3384

# ── Paths ─────────────────────────────────────────────────────
CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "policy_final.pt")
LOG_DIR         = "logs"