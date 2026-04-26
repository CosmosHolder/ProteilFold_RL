"""
data/curriculum.py
ProteinFold-RL — Curriculum Engine

One job: decide which protein the agent trains on, and when to advance.

Advancement logic
-----------------
- Agent trains on proteins in stage order (Stage 1 → 2 → 3 → 4).
- Within a stage, proteins are visited in order.
- GATE: agent must achieve rolling mean RMSD < protein.rmsd_gate
  over the last GATE_WINDOW episodes to advance.
- FALLBACK: if GATE_PATIENCE episodes pass without hitting the gate,
  advance anyway. Agent never gets permanently stuck.
- MINIMUM: at least MIN_EPISODES on a protein before gate is checked.

Replay (anti-forgetting)
------------------------
- Once a stage is unlocked, proteins from ALL unlocked stages are
  sampled each episode, with older stages getting less weight.
- This prevents the agent from forgetting what it already learned.

Serialization
-------------
- Full state saved to JSON alongside model checkpoints.
- Training can be resumed exactly from where it stopped.

Run : python data/curriculum.py   (self-test)
this is the running mechanism
"""

import json
import os
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.protein_registry import (
    REGISTRY, BY_STAGE, ProteinEntry, get_stage, get_protein
)

# ── Advancement constants ─────────────────────────────────────
GATE_WINDOW   = 10    # rolling window size (episodes)
MIN_EPISODES  = 30    # minimum episodes before gate is checked
GATE_PATIENCE = 100   # max episodes before forced advancement

# ── Replay weight for older stages ───────────────────────────
# Stage k gets weight REPLAY_DECAY^(current_stage - k)
# e.g. current=3, stage=1 → 0.3^2 = 0.09  (mostly current stage)
REPLAY_DECAY  = 0.3


class ProteinCurriculum:
    """
    Curriculum manager for ProteinFold-RL.

    Typical usage in training loop
    -------------------------------
    curriculum = ProteinCurriculum()

    for episode in range(N_EPISODES):
        protein = curriculum.sample_protein()       # which protein to use
        # ... run episode ...
        curriculum.record(protein.pdb_id, rmsd)    # log the result
        curriculum.maybe_advance()                 # advance if gate met
    """

    def __init__(self):
        self.current_stage     : int = 1
        self.current_idx       : int = 0   # index within current stage
        self.global_episode    : int = 0
        self.patience_counter  : int = 0

        # Per-protein RMSD history (rolling window)
        self.rmsd_history: Dict[str, deque] = {
            p.pdb_id: deque(maxlen=GATE_WINDOW) for p in REGISTRY
        }

        # Per-protein episode counts
        self.episode_counts: Dict[str, int] = {
            p.pdb_id: 0 for p in REGISTRY
        }

        # Log of every advancement that happened
        self.advancement_log: List[dict] = []

    # ── Primary protein ───────────────────────────────────────

    def current_protein(self) -> ProteinEntry:
        """The protein the agent is currently focused on."""
        proteins = get_stage(self.current_stage)
        idx = min(self.current_idx, len(proteins) - 1)
        return proteins[idx]

    # ── Sampling (with replay) ────────────────────────────────

    def sample_protein(self) -> ProteinEntry:
        """
        Sample a protein for one episode.

        Current stage proteins are sampled most often.
        Older stages are sampled with decaying weight
        to prevent catastrophic forgetting.
        """
        import random

        candidates : List[ProteinEntry] = []
        weights    : List[float]        = []

        for stage in range(1, self.current_stage + 1):
            w = REPLAY_DECAY ** (self.current_stage - stage)
            for p in get_stage(stage):
                candidates.append(p)
                weights.append(w)

        return random.choices(candidates, weights=weights, k=1)[0]

    # ── Recording ─────────────────────────────────────────────

    def record(self, pdb_id: str, rmsd: float) -> None:
        """
        Record the RMSD result of one episode.

        Parameters
        ----------
        pdb_id : protein that was trained this episode
        rmsd   : final RMSD vs native structure (Å)
        """
        self.rmsd_history[pdb_id].append(rmsd)
        self.episode_counts[pdb_id] += 1
        self.global_episode        += 1
        self.patience_counter      += 1

    # ── Advancement ───────────────────────────────────────────

    def gate_met(self) -> bool:
        """
        True if the rolling mean RMSD for the current protein
        is below its gate threshold, and minimum episodes are done.
        """
        p       = self.current_protein()
        history = self.rmsd_history[p.pdb_id]
        n       = self.episode_counts[p.pdb_id]

        if n < MIN_EPISODES or len(history) < GATE_WINDOW:
            return False

        return (sum(history) / len(history)) < p.rmsd_gate

    def patience_exhausted(self) -> bool:
        """True if agent has been stuck too long — force advance."""
        return self.patience_counter >= GATE_PATIENCE

    def maybe_advance(self) -> Tuple[bool, str]:
        """
        Check gate and patience. Advance if either is triggered.

        Returns
        -------
        (advanced: bool, reason: str)
          reason is "gate", "patience", or "" (no advancement)
        """
        if self._is_complete():
            return False, ""

        reason = ""
        if self.gate_met():
            reason = "gate"
        elif self.patience_exhausted():
            reason = "patience"

        if reason:
            self._advance(reason)
            return True, reason

        return False, ""

    def _advance(self, reason: str) -> None:
        """Move to the next protein or stage."""
        old = self.current_protein()
        self.patience_counter = 0   # reset patience on every advance

        stage_proteins = get_stage(self.current_stage)

        if self.current_idx + 1 < len(stage_proteins):
            # Next protein within same stage
            self.current_idx += 1
        else:
            # Move to next stage if it exists
            next_stage = self.current_stage + 1
            if next_stage in BY_STAGE:
                self.current_stage = next_stage
                self.current_idx   = 0

        new = self.current_protein()

        self.advancement_log.append({
            "global_episode" : self.global_episode,
            "from"           : old.pdb_id,
            "to"             : new.pdb_id,
            "stage"          : self.current_stage,
            "reason"         : reason,
        })

        print(
            f"\n{'─' * 52}\n"
            f"  [CURRICULUM] Advancing!\n"
            f"  From  : {old.pdb_id} ({old.name})\n"
            f"  To    : {new.pdb_id} ({new.name})\n"
            f"  Stage : {self.current_stage}   Reason: {reason}\n"
            f"  Episode: {self.global_episode}\n"
            f"{'─' * 52}\n"
        )

    def _is_complete(self) -> bool:
        """True if agent has finished the final protein in stage 4."""
        max_stage = max(BY_STAGE.keys())
        if self.current_stage < max_stage:
            return False
        return self.current_idx >= len(get_stage(max_stage)) - 1

    # ── Status ────────────────────────────────────────────────

    def status(self) -> str:
        """One-line status string for console logging."""
        p       = self.current_protein()
        history = self.rmsd_history[p.pdb_id]
        rolling = (sum(history) / len(history)) if history else float("inf")
        return (
            f"Stage {self.current_stage} | "
            f"{p.pdb_id} | "
            f"Rolling RMSD: {rolling:.3f}Å / gate {p.rmsd_gate}Å | "
            f"Patience: {self.patience_counter}/{GATE_PATIENCE}"
        )

    def summary(self) -> str:
        """Full formatted progress table."""
        lines = [
            "═" * 58,
            "  ProteinFold-RL — Curriculum Progress",
            "═" * 58,
            f"  Global episodes : {self.global_episode}",
            f"  Current stage   : {self.current_stage} / 4",
            f"  Current protein : {self.current_protein().pdb_id}"
            f" ({self.current_protein().name})",
            "",
            f"  {'PDB':6} {'N ep':6} {'Roll RMSD':10} {'Gate':6} {'Status'}",
            "  " + "─" * 44,
        ]
        for p in REGISTRY:
            hist = self.rmsd_history[p.pdb_id]
            n    = self.episode_counts[p.pdb_id]
            roll = (sum(hist) / len(hist)) if hist else float("inf")
            if n == 0:
                icon = "⬜"
            elif roll < p.rmsd_gate and n >= MIN_EPISODES:
                icon = "✅"
            else:
                icon = "🔄"
            roll_str = f"{roll:.3f}" if roll < 999 else "  —  "
            lines.append(
                f"  {p.pdb_id:6} {n:6d} {roll_str:10} "
                f"{p.rmsd_gate:6.1f} {icon}"
            )
        lines.append("═" * 58)
        if self.advancement_log:
            lines.append(f"\n  Advancements ({len(self.advancement_log)}):")
            for a in self.advancement_log[-5:]:
                lines.append(
                    f"    Ep {a['global_episode']:5d}: "
                    f"{a['from']} → {a['to']}  ({a['reason']})"
                )
        return "\n".join(lines)

    # ── Serialization ─────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return JSON-serializable state for checkpointing."""
        return {
            "current_stage"   : self.current_stage,
            "current_idx"     : self.current_idx,
            "global_episode"  : self.global_episode,
            "patience_counter": self.patience_counter,
            "episode_counts"  : dict(self.episode_counts),
            "rmsd_history"    : {k: list(v)
                                 for k, v in self.rmsd_history.items()},
            "advancement_log" : self.advancement_log,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore curriculum from a saved state dict."""
        self.current_stage    = state["current_stage"]
        self.current_idx      = state["current_idx"]
        self.global_episode   = state["global_episode"]
        self.patience_counter = state["patience_counter"]
        self.episode_counts   = state["episode_counts"]
        self.rmsd_history     = {
            k: deque(v, maxlen=GATE_WINDOW)
            for k, v in state["rmsd_history"].items()
        }
        self.advancement_log  = state["advancement_log"]

    def save(self, path: str) -> None:
        """Save curriculum state to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)
        print(f"[CURRICULUM] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ProteinCurriculum":
        """Load curriculum state from a JSON file."""
        c = cls()
        with open(path) as f:
            c.load_state_dict(json.load(f))
        print(f"[CURRICULUM] Loaded ← {path}")
        return c


# ── Self-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=" * 52)
    print("ProteinFold-RL — Curriculum Engine Test")
    print("=" * 52)

    # Test 1: starts on 1L2Y
    c = ProteinCurriculum()
    assert c.current_protein().pdb_id == "1L2Y", "Should start on 1L2Y"
    print("\n[PASS] Starts on 1L2Y ✅")

    # Test 2: gate not met with bad RMSD
    for _ in range(MIN_EPISODES + GATE_WINDOW):
        c.record("1L2Y", rmsd=9.0)
    advanced, reason = c.maybe_advance()
    assert not advanced, "Should not advance with bad RMSD"
    print("[PASS] Does not advance on bad RMSD ✅")

    # Test 3: gate met with good RMSD
    c2 = ProteinCurriculum()
    for _ in range(MIN_EPISODES):
        c2.record("1L2Y", rmsd=9.0)
    for _ in range(GATE_WINDOW):
        c2.record("1L2Y", rmsd=2.0)   # well below gate of 3.5
    advanced, reason = c2.maybe_advance()
    assert advanced and reason == "gate", f"Expected gate, got ({advanced}, {reason})"
    assert c2.current_protein().pdb_id == "1YRF", "Should advance to 1YRF"
    print("[PASS] Gate advancement works → 1YRF ✅")

    # Test 4: patience fallback
    c3 = ProteinCurriculum()
    for _ in range(GATE_PATIENCE):
        c3.record("1L2Y", rmsd=9.0)
    advanced, reason = c3.maybe_advance()
    assert advanced and reason == "patience", \
        f"Expected patience, got ({advanced}, {reason})"
    print("[PASS] Patience fallback works ✅")

    # Test 5: serialization round-trip
    c4 = ProteinCurriculum()
    for _ in range(15):
        c4.record("1L2Y", rmsd=5.0)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    c4.save(tmp)
    c5 = ProteinCurriculum.load(tmp)
    assert c5.global_episode   == c4.global_episode
    assert c5.current_stage    == c4.current_stage
    assert c5.patience_counter == c4.patience_counter
    os.unlink(tmp)
    print("[PASS] Serialization round-trip ✅")

    # Test 6: sample_protein returns valid entries
    import collections
    c6 = ProteinCurriculum()
    c6.current_stage = 2          # simulate stage 2 unlocked
    samples = [c6.sample_protein().pdb_id for _ in range(100)]
    counts  = collections.Counter(samples)
    # Stage 2 proteins should appear more than stage 1
    stage2  = {"1VII", "2GB1"}
    stage1  = {"1L2Y", "1YRF"}
    stage2_count = sum(counts[p] for p in stage2)
    stage1_count = sum(counts[p] for p in stage1)
    assert stage2_count > stage1_count, "Current stage should be sampled more"
    print("[PASS] Replay sampling favours current stage ✅")

    print()
    print(c2.summary())

    print("\n" + "=" * 52)
    print("All curriculum tests passed.")
    print("Next step: upgrade train.py")
    print("=" * 52)
