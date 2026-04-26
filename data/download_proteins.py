"""
data/download_proteins.py
ProteinFold-RL — PDB Downloader

One job: download any missing proteins from RCSB, verify each file
is a valid PDB, and report what happened.

Does NOT define proteins (that is protein_registry.py).
Does NOT train anything.

Run : python data/download_proteins.py
      python data/download_proteins.py --force   (re-download everything)
      this is the running mechanism
"""

import argparse
import os
import sys
import urllib.request
import urllib.error

# ── Make sure project root is on path ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.protein_registry import REGISTRY, STRUCTURES_DIR, print_registry


# ── Validation ────────────────────────────────────────────────

def is_valid_pdb(filepath: str) -> bool:
    """
    Minimal PDB validity check.
    A valid PDB file must contain at least one ATOM record.
    """
    try:
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("ATOM"):
                    return True
        return False
    except OSError:
        return False


# ── Download one protein ──────────────────────────────────────

def download_one(pdb_id: str, url: str, local_path: str,
                 force: bool = False) -> bool:
    """
    Download a single PDB file.

    Parameters
    ----------
    pdb_id     : e.g. "1L2Y"  (used only for display)
    url        : direct RCSB URL
    local_path : destination on disk
    force      : if True, re-download even if file exists

    Returns
    -------
    True if file is ready and valid after this call.
    """
    # Already on disk and not forcing
    if os.path.isfile(local_path) and not force:
        if is_valid_pdb(local_path):
            print(f"  [SKIP] {pdb_id} — already on disk ✅")
            return True
        else:
            print(f"  [WARN] {pdb_id} — file exists but is invalid, re-downloading...")

    print(f"  [DOWN] {pdb_id} — {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
    except urllib.error.URLError as e:
        print(f"  [FAIL] {pdb_id} — network error: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] {pdb_id} — unexpected error: {e}")
        return False

    # Verify immediately after download
    if is_valid_pdb(local_path):
        size_kb = os.path.getsize(local_path) // 1024
        print(f"         ✅  saved  ({size_kb} KB)  →  {local_path}")
        return True
    else:
        print(f"  [FAIL] {pdb_id} — downloaded but no ATOM records found.")
        os.remove(local_path)   # remove corrupt file
        return False


# ── Download all ──────────────────────────────────────────────

def download_all(force: bool = False) -> dict:
    """
    Download every protein in the registry that is missing (or all if force).

    Returns
    -------
    dict mapping pdb_id → True/False (success)
    """
    os.makedirs(STRUCTURES_DIR, exist_ok=True)

    print("=" * 60)
    print("ProteinFold-RL — Downloading Proteins")
    print(f"  Structures dir : {STRUCTURES_DIR}")
    print(f"  Force          : {force}")
    print("=" * 60)

    results = {}

    for entry in REGISTRY:
        results[entry.pdb_id] = download_one(
            pdb_id     = entry.pdb_id,
            url        = entry.url,
            local_path = entry.local_path,
            force      = force,
        )

    # ── Summary ───────────────────────────────────────────────
    n_ok   = sum(results.values())
    n_fail = len(results) - n_ok

    print("\n" + "=" * 60)
    print(f"  Done: {n_ok}/{len(REGISTRY)} proteins ready")
    if n_fail:
        failed = [k for k, v in results.items() if not v]
        print(f"  Failed: {failed}")
        print("  Check your internet connection and try again.")
    else:
        print("  All proteins downloaded and verified ✅")
    print("=" * 60)

    return results


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PDB files for ProteinFold-RL curriculum"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download all proteins even if already on disk"
    )
    args = parser.parse_args()

    results = download_all(force=args.force)

    # Print updated registry table so the user can see the new ✅ marks
    print()
    print_registry()

    # Exit with error code if any download failed
    if not all(results.values()):
        sys.exit(1)
