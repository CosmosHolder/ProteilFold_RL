import urllib.request
import os

PDB_IDS = {
    "1L2Y": "trp_cage",    # Trp-cage — 20 residues
    "1YRF": "villin"       # Villin headpiece — 35 residues
}

SAVE_DIR = os.path.join(os.path.dirname(__file__), "structures")


def download_pdb(pdb_id: str, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")

    if os.path.exists(save_path):
        print(f"[SKIP] {pdb_id}.pdb already exists.")
        return save_path

    print(f"[DOWNLOAD] Fetching {pdb_id} from RCSB...")
    urllib.request.urlretrieve(url, save_path)
    print(f"[OK] Saved to {save_path}")
    return save_path


def verify_pdb(path: str) -> bool:
    with open(path, "r") as f:
        lines = f.readlines()
    atom_lines = [l for l in lines if l.startswith("ATOM")]
    print(f"[VERIFY] {os.path.basename(path)} — {len(atom_lines)} ATOM records found.")
    return len(atom_lines) > 0


if __name__ == "__main__":
    print("=" * 50)
    print("ProteinFold-RL — PDB Data Preparation")
    print("=" * 50)

    for pdb_id, name in PDB_IDS.items():
        path = download_pdb(pdb_id, SAVE_DIR)
        ok = verify_pdb(path)
        if ok:
            print(f"[READY] {name} ({pdb_id}) ✓")
        else:
            print(f"[ERROR] {name} ({pdb_id}) — no ATOM records!")

    print("=" * 50)
    print("All PDB files ready. Proceed to CHECKPOINT-01.")
    print("=" * 50)

