"""
data/curriculum.py
ProteinFold-RL — Curriculum Learning Engine

Implements performance-gated curriculum advancement with fixed-schedule fallback.

Strategy
--------
1.  The agent trains on proteins in Stage order (1 → 2 → 3 → 4).
2.  Within a stage, proteins are sampled in round-robin order.
3.  Advancement gate: agent must achieve rolling mean RMSD < protein.rmsd_gate
    over the last GATE_WINDOW episodes on that protein.
4.  Fallback: if GATE_PATIENCE episodes pass without gating, advance anyway.
    This prevents the agent from getting permanently stuck.
5.  Stage sampling: once stage is unlocked, proteins from ALL unlocked stages
    are sampled, with exponentially decaying weight on older stages.
    This prevents catastrophic forgetting.

State is serializable so training can be resumed from checkpoint.

Author : ProteinFold-RL team
"""

import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

from data.protein_registry import (
    ProteinEntry, REGISTRY, STAGE_MAP,
    GATE_WINDOW, GATE_PATIENCE, MIN_EPISODES,
    get_stage,
)

# ── Sampling weights for rehearsal of older stages ───────────
# Stage k gets weight REPLAY_DECAY^(current_stage - k)
REPLAY_DECAY = 0.4   # older stages get much less time — focus on new


class ProteinCurriculum:
    """
    Curriculum manager for ProteinFold-RL.

    Usage
    -----
    curriculum = ProteinCurriculum()
    protein = curriculum.current_protein()   # ProteinEntry to train on
    curriculum.record_episode(rmsd=2.3)      # log episode result
    if curriculum.should_advance():
        curriculum.advance()
    protein = curriculum.sample_protein()    # may replay old stages

    Serialization
    -------------
    curriculum.save(path)
    curriculum = ProteinCurriculum.load(path)
    """

    def __init__(self):
        # ── Stage tracking ────────────────────────────────────
        self.current_stage          : int  = 1
        self.current_stage_idx      : int  = 0   # index within stage proteins
        self.global_episode         : int  = 0

        # ── Per-protein performance history ───────────────────
        # pdb_id → deque of recent RMSD values (max GATE_WINDOW)
        self.rmsd_history           : Dict[str, deque] = {
            p.pdb_id: deque(maxlen=GATE_WINDOW) for p in REGISTRY
        }

        # ── Per-protein episode counts ────────────────────────
        self.episode_counts         : Dict[str, int] = {
            p.pdb_id: 0 for p in REGISTRY
        }

        # ── Patience counter (resets on advancement) ──────────
        self.patience_counter       : int  = 0

        # ── Stage unlock log ──────────────────────────────────
        self.stage_unlock_episodes  : Dict[int, int] = {1: 0}

        # ── Advancement log ───────────────────────────────────
        self.advancement_log        : List[Dict] = []

        # ── Cached protein list for current stage ─────────────
        self._stage_proteins        : List[ProteinEntry] = get_stage(1)

    # ── Core API ─────────────────────────────────────────────

    def current_protein(self) -> ProteinEntry:
        """Return the protein the agent should train on next."""
        proteins = self._stage_proteins
        idx = self.current_stage_idx % len(proteins)
        return proteins[idx]

    def sample_protein(self) -> ProteinEntry:
        """
        Sample a protein for the next episode using a weighted
        mixture of all unlocked stages (rehearsal of older stages
        prevents catastrophic forgetting).

        Returns
        -------
        ProteinEntry — the protein to train on
        """
        import random

        # Build weighted list across all unlocked stages
        candidates = []
        weights    = []

        for stage in range(1, self.current_stage + 1):
            stage_weight = REPLAY_DECAY ** (self.current_stage - stage)
            stage_proteins = get_stage(stage)
            for p in stage_proteins:
                candidates.append(p)
                weights.append(stage_weight)

        # Normalize
        total = sum(weights)
        norm_weights = [w / total for w in weights]

        chosen = random.choices(candidates, weights=norm_weights, k=1)[0]
        return chosen

    def record_episode(self, pdb_id: str, rmsd: float,
                       energy: float, reward: float) -> None:
        """
        Record episode result for a given protein.

        Parameters
        ----------
        pdb_id  : protein trained this episode
        rmsd    : final RMSD vs native (Å)
        energy  : final energy (kcal/mol)
        reward  : total episode reward
        """
        self.rmsd_history[pdb_id].append(rmsd)
        self.episode_counts[pdb_id] += 1
        self.global_episode += 1
        self.patience_counter += 1

    def gate_status(self, pdb_id: Optional[str] = None) -> Dict:
        """
        Return gating status for a protein (default: current protein).

        Returns dict with:
          rolling_rmsd   : mean RMSD over last GATE_WINDOW episodes
          gate           : target RMSD threshold
          n_episodes     : episodes on this protein so far
          gate_met       : bool — rolling mean < gate
          patience_used  : episodes since last advancement
          patience_left  : episodes before forced advancement
          force_advance  : bool — patience exhausted
        """
        if pdb_id is None:
            pdb_id = self.current_protein().pdb_id

        from data.protein_registry import get_protein
        entry = get_protein(pdb_id)
        history = self.rmsd_history[pdb_id]
        n_ep    = self.episode_counts[pdb_id]

        rolling_rmsd = (
            sum(history) / len(history) if history else float("inf")
        )

        gate_met     = (
            n_ep >= MIN_EPISODES
            and len(history) >= GATE_WINDOW
            and rolling_rmsd < entry.rmsd_gate
        )
        force_advance = self.patience_counter >= GATE_PATIENCE

        return {
            "pdb_id"        : pdb_id,
            "rolling_rmsd"  : rolling_rmsd,
            "gate"          : entry.rmsd_gate,
            "n_episodes"    : n_ep,
            "gate_met"      : gate_met,
            "patience_used" : self.patience_counter,
            "patience_left" : max(0, GATE_PATIENCE - self.patience_counter),
            "force_advance" : force_advance,
        }

    def should_advance(self) -> Tuple[bool, str]:
        """
        Check whether the curriculum should advance to the next protein.

        Returns
        -------
        (bool, reason_string)
          True + "gate"    — performance gate met
          True + "patience" — patience exhausted, forced advance
          False + ""       — keep training on current protein
        """
        # Already at final stage + final protein
        if self._is_curriculum_complete():
            return False, "complete"

        status = self.gate_status()

        if status["gate_met"]:
            return True, "gate"
        if status["force_advance"]:
            return True, "patience"
        return False, ""

    def advance(self, reason: str = "gate") -> Optional[ProteinEntry]:
        """
        Advance curriculum to the next protein / stage.

        Returns the new current ProteinEntry, or None if complete.
        """
        if self._is_curriculum_complete():
            return None

        old_protein = self.current_protein()
        self.patience_counter = 0   # reset patience

        # Try advancing within current stage
        stage_proteins = self._stage_proteins
        if self.current_stage_idx + 1 < len(stage_proteins):
            self.current_stage_idx += 1
        else:
            # Move to next stage
            next_stage = self.current_stage + 1
            if next_stage in STAGE_MAP:
                self.current_stage         = next_stage
                self.current_stage_idx     = 0
                self._stage_proteins       = get_stage(next_stage)
                self.stage_unlock_episodes[next_stage] = self.global_episode

        new_protein = self.current_protein()

        log_entry = {
            "global_episode"  : self.global_episode,
            "from_protein"    : old_protein.pdb_id,
            "to_protein"      : new_protein.pdb_id,
            "reason"          : reason,
            "stage"           : self.current_stage,
        }
        self.advancement_log.append(log_entry)

        print(
            f"\n{'─' * 50}\n"
            f"  [CURRICULUM] Advancing!\n"
            f"  From : {old_protein.pdb_id} ({old_protein.name})\n"
            f"  To   : {new_protein.pdb_id} ({new_protein.name})\n"
            f"  Stage: {self.current_stage}  |  Reason: {reason}\n"
            f"  Global episode: {self.global_episode}\n"
            f"{'─' * 50}\n"
        )
        return new_protein

    def get_summary(self) -> str:
        """Return a formatted summary of curriculum progress."""
        lines = [
            "═" * 55,
            "  ProteinFold-RL — Curriculum Progress",
            "═" * 55,
            f"  Global episodes : {self.global_episode}",
            f"  Current stage   : {self.current_stage} / 4",
            f"  Current protein : {self.current_protein().pdb_id}"
            f" ({self.current_protein().name})",
            "",
            f"  {'PDB':6} {'Episodes':9} {'Best RMSD':10} {'Gate':7} {'Status':8}",
            "  " + "-" * 45,
        ]
        for p in REGISTRY:
            hist = self.rmsd_history[p.pdb_id]
            best = min(hist) if hist else float("inf")
            roll = sum(hist) / len(hist) if hist else float("inf")
            gated = "✅" if (roll < p.rmsd_gate and
                             self.episode_counts[p.pdb_id] >= MIN_EPISODES) \
                    else "🔄" if self.episode_counts[p.pdb_id] > 0 \
                    else "⬜"
            lines.append(
                f"  {p.pdb_id:6} {self.episode_counts[p.pdb_id]:9d} "
                f"{best:10.3f} {p.rmsd_gate:7.1f} {gated}"
            )
        lines.append("═" * 55)

        if self.advancement_log:
            lines.append(f"\n  Advancements: {len(self.advancement_log)}")
            for adv in self.advancement_log[-3:]:   # last 3
                lines.append(
                    f"    Ep {adv['global_episode']:5d}: "
                    f"{adv['from_protein']} → {adv['to_protein']} "
                    f"({adv['reason']})"
                )
        return "\n".join(lines)

    # ── Serialization ─────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return a JSON-serializable state dict for checkpointing."""
        return {
            "current_stage"         : self.current_stage,
            "current_stage_idx"     : self.current_stage_idx,
            "global_episode"        : self.global_episode,
            "patience_counter"      : self.patience_counter,
            "episode_counts"        : dict(self.episode_counts),
            "rmsd_history"          : {
                k: list(v) for k, v in self.rmsd_history.items()
            },
            "stage_unlock_episodes" : self.stage_unlock_episodes,
            "advancement_log"       : self.advancement_log,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore curriculum state from a dict."""
        self.current_stage          = state["current_stage"]
        self.current_stage_idx      = state["current_stage_idx"]
        self.global_episode         = state["global_episode"]
        self.patience_counter       = state["patience_counter"]
        self.episode_counts         = state["episode_counts"]
        self.rmsd_history           = {
            k: deque(v, maxlen=GATE_WINDOW)
            for k, v in state["rmsd_history"].items()
        }
        self.stage_unlock_episodes  = {
            int(k): v for k, v in state["stage_unlock_episodes"].items()
        }
        self.advancement_log        = state["advancement_log"]
        self._stage_proteins        = get_stage(self.current_stage)

    def save(self, path: str) -> None:
        """Save curriculum state to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)
        print(f"[CURRICULUM] State saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ProteinCurriculum":
        """Load curriculum state from JSON file."""
        curriculum = cls()
        with open(path, "r") as f:
            state = json.load(f)
        curriculum.load_state_dict(state)
        print(f"[CURRICULUM] State loaded from {path}")
        return curriculum

    # ── Internal helpers ──────────────────────────────────────

    def _is_curriculum_complete(self) -> bool:
        max_stage = max(STAGE_MAP.keys())
        if self.current_stage < max_stage:
            return False
        return self.current_stage_idx >= len(self._stage_proteins) - 1


# ── Self-test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    print("=" * 55)
    print("ProteinFold-RL — Curriculum Engine Test")
    print("=" * 55)

    c = ProteinCurriculum()

    print(f"\n[TEST 1] Initial state...")
    p = c.current_protein()
    print(f"  Current protein : {p.pdb_id} ({p.name})")
    print(f"  Stage           : {c.current_stage}")
    assert p.pdb_id == "1L2Y", f"Expected 1L2Y, got {p.pdb_id}"
    print("  [PASS] Starts on 1L2Y ✓")

    print(f"\n[TEST 2] Record {MIN_EPISODES} poor episodes (no gate)...")
    for i in range(MIN_EPISODES):
        c.record_episode("1L2Y", rmsd=8.0, energy=300.0, reward=-5.0)
    advance, reason = c.should_advance()
    assert not advance, "Should not advance on bad RMSD"
    print("  [PASS] Does not advance with poor RMSD ✓")

    print(f"\n[TEST 3] Record {GATE_WINDOW} good episodes (gate met)...")
    for i in range(GATE_WINDOW):
        c.record_episode("1L2Y", rmsd=2.0, energy=50.0, reward=10.0)
    advance, reason = c.should_advance()
    assert advance and reason == "gate", f"Expected gate, got ({advance}, {reason})"
    new_p = c.advance(reason)
    assert new_p.pdb_id == "1YRF", f"Expected 1YRF, got {new_p.pdb_id}"
    print(f"  [PASS] Gated correctly → advanced to {new_p.pdb_id} ✓")

    print(f"\n[TEST 4] Patience / forced advancement...")
    c2 = ProteinCurriculum()
    for i in range(GATE_PATIENCE):
        c2.record_episode("1L2Y", rmsd=9.0, energy=400.0, reward=-10.0)
    advance, reason = c2.should_advance()
    assert advance and reason == "patience", \
        f"Expected patience advancement, got ({advance}, {reason})"
    print("  [PASS] Patience fallback triggers correctly ✓")

    print(f"\n[TEST 5] Serialization round-trip...")
    import tempfile, json
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_path = tf.name
    c.save(tmp_path)
    c_loaded = ProteinCurriculum.load(tmp_path)
    assert c_loaded.current_stage == c.current_stage
    assert c_loaded.global_episode == c.global_episode
    os.unlink(tmp_path)
    print("  [PASS] Serialization round-trip correct ✓")

    print(f"\n[TEST 6] Sample protein (rehearsal)...")
    samples = [c.sample_protein().pdb_id for _ in range(50)]
    # Should see stage 1 proteins (1L2Y, 1YRF) and stage 2 (1VII, 2GB1)
    assert "1YRF" in samples or "1VII" in samples, \
        "Sampling should include unlocked proteins"
    print(f"  Sample distribution: {set(samples)}")
    print("  [PASS] Sampling includes multiple proteins ✓")

    print()
    print(c.get_summary())

    print("\n" + "=" * 55)
    print("CHECKPOINT — Curriculum engine verified.")
    print("=" * 55)