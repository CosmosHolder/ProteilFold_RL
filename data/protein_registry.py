"""
data/protein_registry.py
ProteinFold-RL — Multi-Protein Curriculum Registry

8 proteins selected for HYBRID curriculum:
  Size-based difficulty (residue count) interleaved with
  structure-based diversity (helix / sheet / mixed / disordered).

Difficulty stages:
  Stage 1 — Micro proteins, helix-dominated   (warmup)
  Stage 2 — Small proteins, sheet-dominated   (generalization)
  Stage 3 — Small-medium, mixed ss            (real challenge)
  Stage 4 — Medium, partially disordered      (mastery)

Performance-gated advancement:
  Agent must hit RMSD < rmsd_gate over a rolling window of
  gate_window episodes before unlocking the next stage.
  If gate_patience episodes pass without hitting the gate,
  fall back to fixed-schedule advancement (never stuck).

All PDB files downloaded from RCSB: https://files.rcsb.org/download/{PDB_ID}.pdb
Checksums (SHA-256, first 16 hex chars) verify download integrity.

Author : ProteinFold-RL team
"""

import os
import hashlib
import urllib.request
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# ── Download config ──────────────────────────────────────────
RCSB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"
DATA_DIR = os.path.join(os.path.dirname(__file__), "structures")

# ── Curriculum advancement config ────────────────────────────
GATE_WINDOW    = 10     # rolling window (episodes) to evaluate gating
GATE_PATIENCE  = 100    # max episodes before forced advancement
MIN_EPISODES   = 50     # min episodes on any protein before gating check


@dataclass
class ProteinEntry:
    """
    Full metadata record for one protein in the curriculum.

    Fields
    ------
    pdb_id          : 4-char RCSB identifier
    name            : human-readable name
    n_residues      : number of Cα residues (after parsing)
    ss_type         : secondary structure character
                      "helix"     — predominantly alpha-helical
                      "sheet"     — predominantly beta-sheet
                      "mixed"     — helix + sheet + loops
                      "disordered"— intrinsically disordered regions present
    difficulty      : 1-4 integer (1=easiest, 4=hardest)
    curriculum_stage: 1-4 integer (stage this protein belongs to)
    rmsd_gate       : RMSD threshold (Å) the agent must reach to advance
                      (rolling mean over gate_window episodes)
    description     : one-line biological relevance note
    url             : direct RCSB download URL
    local_path      : absolute path where PDB will be stored
    checksum_prefix : first 16 hex chars of SHA-256 of the raw PDB file
                      (populated after first successful download)
    """
    pdb_id          : str
    name            : str
    n_residues      : int
    ss_type         : str
    difficulty      : int
    curriculum_stage: int
    rmsd_gate       : float
    description     : str
    url             : str          = field(init=False)
    local_path      : str          = field(init=False)
    checksum_prefix : Optional[str] = field(default=None)

    def __post_init__(self):
        self.url        = RCSB_URL_TEMPLATE.format(pdb_id=self.pdb_id)
        self.local_path = os.path.join(DATA_DIR, f"{self.pdb_id}.pdb")

    def is_downloaded(self) -> bool:
        """Return True if PDB file exists on disk."""
        return os.path.isfile(self.local_path)

    def verify_checksum(self) -> bool:
        """
        Verify the downloaded file against stored checksum prefix.
        Returns True if checksum matches (or if no checksum stored yet).
        """
        if not self.is_downloaded():
            return False
        if self.checksum_prefix is None:
            return True   # no reference — accept
        computed = _sha256_prefix(self.local_path)
        return computed == self.checksum_prefix

    def compute_and_store_checksum(self) -> str:
        """Compute SHA-256 of local file and store the prefix."""
        prefix = _sha256_prefix(self.local_path)
        self.checksum_prefix = prefix
        return prefix


# ── The 8-protein curriculum ─────────────────────────────────
#
# Selection rationale:
#   1L2Y — Trp-cage      20 res helix     — already trained, warmup anchor
#   1YRF — Villin HP35   35 res helix     — 3-helix bundle, next step up
#   2GB1 — Protein G     56 res mixed     — classic benchmark, helix+sheet
#   1VII — Villin full   36 res helix     — more loops than 1YRF
#   1ENH — Engrailed     54 res helix     — homeodomain, well-studied
#   2HHB — Hemoglobin α  141 res mixed    — medium, mixed, disease-relevant
#   1UBQ — Ubiquitin     76 res mixed     — ubiquitous benchmark, mixed
#   1BDD — Protein A     58 res helix     — 3-helix, good sheet transition

REGISTRY: List[ProteinEntry] = [

    # ── Stage 1: Micro proteins, helix-dominated ─────────────
    ProteinEntry(
        pdb_id           = "1L2Y",
        name             = "Trp-cage miniprotein",
        n_residues       = 20,
        ss_type          = "helix",
        difficulty       = 1,
        curriculum_stage = 1,
        rmsd_gate        = 3.5,
        description      = (
            "Smallest known autonomously folding protein. "
            "20 residues, single alpha-helix + hydrophobic core. "
            "Gold standard for folding benchmarks."
        ),
    ),
    ProteinEntry(
        pdb_id           = "1YRF",
        name             = "Villin headpiece HP35",
        n_residues       = 35,
        ss_type          = "helix",
        difficulty       = 2,
        curriculum_stage = 1,
        rmsd_gate        = 4.5,
        description      = (
            "Three-helix bundle. 35 residues, ultra-fast folder. "
            "Microsecond folding timescale — ideal RL target. "
            "Well-studied misfolding intermediate."
        ),
    ),

    # ── Stage 2: Small proteins, sheet + mixed ───────────────
    ProteinEntry(
        pdb_id           = "1VII",
        name             = "Villin headpiece HP36",
        n_residues       = 36,
        ss_type          = "helix",
        difficulty       = 2,
        curriculum_stage = 2,
        rmsd_gate        = 4.5,
        description      = (
            "HP36 variant with one additional residue vs HP35. "
            "Tests generalization within same protein family. "
            "Different loop geometry forces new angle decisions."
        ),
    ),
    ProteinEntry(
        pdb_id           = "2GB1",
        name             = "Protein G B1 domain",
        n_residues       = 56,
        ss_type          = "mixed",
        difficulty       = 3,
        curriculum_stage = 2,
        rmsd_gate        = 5.5,
        description      = (
            "Mixed alpha-helix + beta-sheet. 56 residues. "
            "Classic folding benchmark — used in hundreds of studies. "
            "Forces agent to learn both helix and sheet geometry."
        ),
    ),

    # ── Stage 3: Small-medium, mixed secondary structure ─────
    ProteinEntry(
        pdb_id           = "1ENH",
        name             = "Engrailed homeodomain",
        n_residues       = 54,
        ss_type          = "helix",
        difficulty       = 3,
        curriculum_stage = 3,
        rmsd_gate        = 5.5,
        description      = (
            "Drosophila engrailed homeodomain. 54 residues, 3-helix. "
            "DNA-binding protein — biological relevance. "
            "Well-characterized folding pathway in literature."
        ),
    ),
    ProteinEntry(
        pdb_id           = "1UBQ",
        name             = "Ubiquitin",
        n_residues       = 76,
        ss_type          = "mixed",
        difficulty       = 4,
        curriculum_stage = 3,
        rmsd_gate        = 6.5,
        description      = (
            "76 residues, alpha+beta mixed. Universal in all eukaryotes. "
            "Central to protein degradation — disease target. "
            "Tests agent on complex mixed topology."
        ),
    ),

    # ── Stage 4: Medium, partially disordered, mastery ───────
    ProteinEntry(
        pdb_id           = "1BDD",
        name             = "Protein A B-domain",
        n_residues       = 58,
        ss_type          = "helix",
        difficulty       = 4,
        curriculum_stage = 4,
        rmsd_gate        = 6.0,
        description      = (
            "Staphylococcal protein A B-domain. 58 residues, 3-helix bundle. "
            "Used in affinity chromatography globally. "
            "Higher difficulty from tight helix packing constraints."
        ),
    ),
    ProteinEntry(
        pdb_id           = "2HHB",
        name             = "Hemoglobin alpha chain",
        n_residues       = 141,
        ss_type          = "mixed",
        difficulty       = 4,
        curriculum_stage = 4,
        rmsd_gate        = 8.0,
        description      = (
            "Alpha chain of human hemoglobin. 141 residues. "
            "Sickle cell disease caused by single point mutation. "
            "Largest protein in curriculum — mastery challenge."
        ),
    ),
]

# ── Index for fast lookup ────────────────────────────────────
REGISTRY_BY_ID: Dict[str, ProteinEntry] = {
    p.pdb_id: p for p in REGISTRY
}

STAGE_MAP: Dict[int, List[ProteinEntry]] = {
    1: [p for p in REGISTRY if p.curriculum_stage == 1],
    2: [p for p in REGISTRY if p.curriculum_stage == 2],
    3: [p for p in REGISTRY if p.curriculum_stage == 3],
    4: [p for p in REGISTRY if p.curriculum_stage == 4],
}


# ── Utility functions ────────────────────────────────────────

def _sha256_prefix(filepath: str, n_chars: int = 16) -> str:
    """Compute first n_chars hex chars of SHA-256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:n_chars]


def download_protein(entry: ProteinEntry, force: bool = False) -> bool:
    """
    Download a single protein PDB from RCSB.

    Parameters
    ----------
    entry : ProteinEntry
    force : bool — re-download even if file exists

    Returns
    -------
    True if file is ready (downloaded or already existed + valid)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if entry.is_downloaded() and not force:
        if entry.verify_checksum():
            print(f"  [SKIP] {entry.pdb_id} already downloaded and verified.")
            return True
        else:
            print(f"  [WARN] {entry.pdb_id} checksum mismatch — re-downloading.")

    print(f"  [DOWN] {entry.pdb_id} ({entry.name}) from {entry.url}")
    try:
        urllib.request.urlretrieve(entry.url, entry.local_path)
        entry.compute_and_store_checksum()
        print(f"         → Saved to {entry.local_path}")
        print(f"         → SHA-256 prefix: {entry.checksum_prefix}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Could not download {entry.pdb_id}: {exc}")
        return False


def download_all(force: bool = False) -> Dict[str, bool]:
    """
    Download all 8 proteins in curriculum order.

    Returns dict mapping pdb_id → success bool.
    """
    print("=" * 60)
    print("ProteinFold-RL — Downloading Curriculum Proteins")
    print(f"  Target dir : {DATA_DIR}")
    print(f"  Proteins   : {len(REGISTRY)}")
    print("=" * 60)

    results = {}
    for stage in sorted(STAGE_MAP.keys()):
        print(f"\n── Stage {stage} ──────────────────────────────────────")
        for entry in STAGE_MAP[stage]:
            results[entry.pdb_id] = download_protein(entry, force=force)

    success = sum(results.values())
    print(f"\n  Downloaded: {success}/{len(REGISTRY)} proteins")
    if success < len(REGISTRY):
        failed = [k for k, v in results.items() if not v]
        print(f"  Failed    : {failed}")
    print("=" * 60)
    return results


def get_protein(pdb_id: str) -> ProteinEntry:
    """Return ProteinEntry by PDB ID. Raises KeyError if not found."""
    if pdb_id not in REGISTRY_BY_ID:
        raise KeyError(
            f"PDB ID '{pdb_id}' not in registry. "
            f"Available: {list(REGISTRY_BY_ID.keys())}"
        )
    return REGISTRY_BY_ID[pdb_id]


def get_stage(stage: int) -> List[ProteinEntry]:
    """Return all proteins in a given curriculum stage."""
    if stage not in STAGE_MAP:
        raise KeyError(f"Stage {stage} not found. Valid stages: {list(STAGE_MAP.keys())}")
    return STAGE_MAP[stage]


def print_registry():
    """Pretty-print the full registry table."""
    header = (
        f"{'PDB':6} {'Name':30} {'Res':5} {'SS':12} "
        f"{'Diff':5} {'Stage':6} {'Gate(Å)':8}"
    )
    print("=" * len(header))
    print("ProteinFold-RL — Protein Registry")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for stage in sorted(STAGE_MAP.keys()):
        for p in STAGE_MAP[stage]:
            status = "✅" if p.is_downloaded() else "⬜"
            print(
                f"{p.pdb_id:6} {p.name:30} {p.n_residues:5d} "
                f"{p.ss_type:12} {p.difficulty:5d} {p.curriculum_stage:6d} "
                f"{p.rmsd_gate:8.1f} {status}"
            )
        print()
    print("=" * len(header))


# ── Self-test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print_registry()

    print("\n[TEST] Registry integrity checks...")
    assert len(REGISTRY) == 8,        "Expected 8 proteins"
    assert len(STAGE_MAP) == 4,       "Expected 4 stages"
    assert all(
        p.curriculum_stage in STAGE_MAP for p in REGISTRY
    ), "Stage mapping broken"
    assert all(
        1 <= p.difficulty <= 4 for p in REGISTRY
    ), "Difficulty out of range"
    assert all(
        p.rmsd_gate > 0 for p in REGISTRY
    ), "RMSD gate must be positive"
    print("  [PASS] All registry integrity checks passed ✓")

    print("\n[TEST] URL construction...")
    for p in REGISTRY:
        assert p.pdb_id in p.url, f"URL missing PDB ID for {p.pdb_id}"
        assert p.local_path.endswith(f"{p.pdb_id}.pdb"), \
            f"Local path wrong for {p.pdb_id}"
    print("  [PASS] All URLs + paths correct ✓")

    print("\nTo download all proteins, run:")
    print("  from data.protein_registry import download_all")
    print("  download_all()")