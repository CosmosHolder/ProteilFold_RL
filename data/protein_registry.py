"""
data/protein_registry.py
ProteinFold-RL — Protein Registry

Single source of truth for every protein in the curriculum.
Defines what proteins exist, their metadata, and where they live on disk.
Does NOT download. Does NOT train. Does NOT advance curriculum.
Those are separate files with separate jobs.

8 proteins across 4 stages — hybrid curriculum:
  Stage 1 — tiny, helix-only          (warmup, agent builds intuition)
  Stage 2 — small, introduces sheets  (generalization begins)
  Stage 3 — medium, mixed topology    (real challenge)
  Stage 4 — larger, mastery test      (proves the agent learned physics)

File location : data/protein_registry.py
Run directly  : python data/protein_registry.py  (self-test + print table)
this is the running mechanism
"""

import os
from dataclasses import dataclass
from typing import List, Dict

# ── Path to PDB files — must match fold_env.PDB_PATHS pattern ─
STRUCTURES_DIR = os.path.join(os.path.dirname(__file__), "structures")

# ── RCSB download template ────────────────────────────────────
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


@dataclass(frozen=True)
class ProteinEntry:
    """
    Metadata record for one protein.

    Attributes
    ----------
    pdb_id          : 4-char RCSB identifier  e.g. "1L2Y"
    name            : human-readable name
    n_residues      : number of Cα residues
    ss_type         : dominant secondary structure
                      "helix" | "sheet" | "mixed"
    curriculum_stage: 1-4 (1 = easiest)
    difficulty      : 1-4 (finer-grained than stage)
    rmsd_gate       : rolling-mean RMSD (Å) agent must reach to advance
    description     : one-line biological note
    """
    pdb_id           : str
    name             : str
    n_residues       : int
    ss_type          : str
    curriculum_stage : int
    difficulty       : int
    rmsd_gate        : float
    description      : str

    @property
    def url(self) -> str:
        """Direct RCSB download URL."""
        return RCSB_URL.format(pdb_id=self.pdb_id)

    @property
    def local_path(self) -> str:
        """Absolute path where the PDB file lives on disk."""
        return os.path.join(STRUCTURES_DIR, f"{self.pdb_id}.pdb")

    @property
    def is_downloaded(self) -> bool:
        """True if the PDB file exists on disk."""
        return os.path.isfile(self.local_path)


# ─────────────────────────────────────────────────────────────
# THE 8-PROTEIN CURRICULUM
# ─────────────────────────────────────────────────────────────
#
# Protein selection rationale
# ───────────────────────────
# Stage 1 — Micro proteins, pure helix
#   1L2Y  Trp-cage (20 res)    — already trained, warmup anchor
#   1YRF  Villin HP35 (35 res) — 3-helix bundle, fast folder, ideal RL target
#
# Stage 2 — Small, introduces beta-sheet
#   1VII  Villin HP36 (36 res) — same family as 1YRF, different loop geometry
#   2GB1  Protein G (56 res)   — helix + sheet, classic benchmark
#
# Stage 3 — Medium, fully mixed topology
#   1ENH  Engrailed HD (54 res)  — homeodomain, 3-helix, DNA-binding
#   1UBQ  Ubiquitin (76 res)     — alpha+beta, universal eukaryotic protein
#
# Stage 4 — Larger, mastery challenge
#   1BDD  Protein A (58 res)     — tight 3-helix packing, high difficulty
#   2HHB  Hemoglobin α (141 res) — largest in set, disease-relevant
#
REGISTRY: List[ProteinEntry] = [

    # ── Stage 1 ──────────────────────────────────────────────
    ProteinEntry(
        pdb_id           = "1L2Y",
        name             = "Trp-cage miniprotein",
        n_residues       = 20,
        ss_type          = "helix",
        curriculum_stage = 1,
        difficulty       = 1,
        rmsd_gate        = 3.5,
        description      = "Smallest known autonomously folding protein. "
                           "Gold standard for folding benchmarks.",
    ),
    ProteinEntry(
        pdb_id           = "1YRF",
        name             = "Villin headpiece HP35",
        n_residues       = 35,
        ss_type          = "helix",
        curriculum_stage = 1,
        difficulty       = 2,
        rmsd_gate        = 4.5,
        description      = "Three-helix bundle, microsecond folder. "
                           "Ideal RL target — fast dynamics, well-studied.",
    ),

    # ── Stage 2 ──────────────────────────────────────────────
    ProteinEntry(
        pdb_id           = "1VII",
        name             = "Villin headpiece HP36",
        n_residues       = 36,
        ss_type          = "helix",
        curriculum_stage = 2,
        difficulty       = 2,
        rmsd_gate        = 4.5,
        description      = "HP36 variant of Villin. Tests generalization "
                           "within the same protein family.",
    ),
    ProteinEntry(
        pdb_id           = "2GB1",
        name             = "Protein G B1 domain",
        n_residues       = 56,
        ss_type          = "mixed",
        curriculum_stage = 2,
        difficulty       = 3,
        rmsd_gate        = 5.5,
        description      = "Mixed helix + sheet. Forces agent to learn "
                           "both secondary structure types.",
    ),

    # ── Stage 3 ──────────────────────────────────────────────
    ProteinEntry(
        pdb_id           = "1ENH",
        name             = "Engrailed homeodomain",
        n_residues       = 54,
        ss_type          = "helix",
        curriculum_stage = 3,
        difficulty       = 3,
        rmsd_gate        = 5.5,
        description      = "Drosophila homeodomain, DNA-binding protein. "
                           "Well-characterized folding pathway.",
    ),
    ProteinEntry(
        pdb_id           = "1UBQ",
        name             = "Ubiquitin",
        n_residues       = 76,
        ss_type          = "mixed",
        curriculum_stage = 3,
        difficulty       = 4,
        rmsd_gate        = 6.5,
        description      = "76-residue alpha+beta protein, universal in "
                           "eukaryotes. Central to protein degradation.",
    ),

    # ── Stage 4 ──────────────────────────────────────────────
    ProteinEntry(
        pdb_id           = "1BDD",
        name             = "Protein A B-domain",
        n_residues       = 58,
        ss_type          = "helix",
        curriculum_stage = 4,
        difficulty       = 4,
        rmsd_gate        = 6.0,
        description      = "Staphylococcal protein A, tight 3-helix packing. "
                           "High difficulty from constrained geometry.",
    ),
    ProteinEntry(
        pdb_id           = "2HHB",
        name             = "Hemoglobin alpha chain",
        n_residues       = 141,
        ss_type          = "mixed",
        curriculum_stage = 4,
        difficulty       = 4,
        rmsd_gate        = 8.0,
        description      = "Human hemoglobin alpha chain, 141 residues. "
                           "Sickle cell disease target. Mastery challenge.",
    ),
]

# ─────────────────────────────────────────────────────────────
# LOOKUP HELPERS
# ─────────────────────────────────────────────────────────────

# Fast lookup by PDB ID
BY_ID: Dict[str, ProteinEntry] = {p.pdb_id: p for p in REGISTRY}

# Proteins grouped by curriculum stage
BY_STAGE: Dict[int, List[ProteinEntry]] = {}
for _p in REGISTRY:
    BY_STAGE.setdefault(_p.curriculum_stage, []).append(_p)


def get_protein(pdb_id: str) -> ProteinEntry:
    """Return ProteinEntry by PDB ID. Raises KeyError if unknown."""
    if pdb_id not in BY_ID:
        raise KeyError(
            f"Unknown PDB ID '{pdb_id}'. "
            f"Available: {list(BY_ID.keys())}"
        )
    return BY_ID[pdb_id]


def get_stage(stage: int) -> List[ProteinEntry]:
    """Return all proteins in a given curriculum stage (1-4)."""
    if stage not in BY_STAGE:
        raise KeyError(
            f"Stage {stage} not found. "
            f"Valid stages: {sorted(BY_STAGE)}"
        )
    return BY_STAGE[stage]


def all_pdb_ids() -> List[str]:
    """Return all PDB IDs in curriculum order."""
    return [p.pdb_id for p in REGISTRY]


# ─────────────────────────────────────────────────────────────
# PRETTY PRINT
# ─────────────────────────────────────────────────────────────

def print_registry() -> None:
    """Print the full registry as a formatted table."""
    div = "─" * 72
    print("═" * 72)
    print("  ProteinFold-RL — Protein Registry (8 proteins, 4 stages)")
    print("═" * 72)
    print(f"  {'PDB':6} {'Name':28} {'Res':4} {'SS':7} "
          f"{'Stg':4} {'Dif':4} {'Gate':6}  On disk")
    print(f"  {div}")
    for stage in sorted(BY_STAGE):
        for p in BY_STAGE[stage]:
            status = "✅" if p.is_downloaded else "⬜  need download"
            print(
                f"  {p.pdb_id:6} {p.name:28} {p.n_residues:4d} "
                f"{p.ss_type:7} {p.curriculum_stage:4d} {p.difficulty:4d} "
                f"{p.rmsd_gate:6.1f}  {status}"
            )
        print(f"  {div}")
    downloaded = sum(1 for p in REGISTRY if p.is_downloaded)
    print(f"\n  {downloaded}/{len(REGISTRY)} proteins on disk")
    print("═" * 72)


# ─────────────────────────────────────────────────────────────
# SELF-TEST  —  python data/protein_registry.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_registry()

    print("\n[TEST] Registry integrity checks...")
    assert len(REGISTRY) == 8,                        "Must have 8 proteins"
    assert len(BY_STAGE)  == 4,                        "Must have 4 stages"
    assert all(p.difficulty in range(1, 5)
               for p in REGISTRY),                    "Difficulty must be 1-4"
    assert all(p.curriculum_stage in range(1, 5)
               for p in REGISTRY),                    "Stage must be 1-4"
    assert all(p.rmsd_gate > 0
               for p in REGISTRY),                    "Gate must be positive"
    assert all(p.pdb_id in p.url
               for p in REGISTRY),                    "URL must contain PDB ID"
    assert all(p.pdb_id in p.local_path
               for p in REGISTRY),                    "Path must contain PDB ID"
    assert get_protein("1L2Y").curriculum_stage == 1,  "1L2Y must be stage 1"
    assert get_protein("2HHB").difficulty       == 4,  "2HHB must be difficulty 4"
    assert get_stage(1)[0].pdb_id == "1L2Y",           "Stage 1 starts with 1L2Y"
    assert len(get_stage(4)) == 2,                     "Stage 4 has 2 proteins"

    print("  All checks passed ✅")
    print("\nNext step: python data/download_proteins.py")
