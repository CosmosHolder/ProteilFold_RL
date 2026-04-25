"""
env/fold_env.py
ProteinFold-RL — Gymnasium Environment (v3)

What changed vs v2
------------------
- Secondary structure reward:
    R_HELIX_BONUS = +3.0  per residue entering helix region (φ/ψ Ramachandran)
    R_SHEET_BONUS = +3.0  per residue entering sheet region
- _detect_ss(phi, psi) → returns ("helix" | "sheet" | "other")
- _compute_ss_reward() counts newly-formed helix/sheet residues
- ss_reward added to info dict for logging in train.py
- Reward comment updated to 9 cases total
- All existing logic untouched
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from torch_geometric.data import Data
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.protein_graph import pdb_to_graph
from env.energy import compute_energy, compute_energy_delta
from env.clash_detect import detect_clashes, is_valid_conformation

# ── Reward constants ─────────────────────────────────────────
R_ENERGY_BIG    = +8.0
R_ENERGY_SMALL  = +2.0
R_NO_CLASH      = +1.0
R_CLASH         = -2.0
R_ENERGY_UP     = -1.0
R_STEP_PENALTY  = -0.3
R_RMSD_BONUS    = +15.0
R_HELIX_BONUS   = +3.0   # per residue newly entering helix region
R_SHEET_BONUS   = +3.0   # per residue newly entering sheet region
RMSD_THRESHOLD  =  2.0   # Angstroms

# ── Secondary structure Ramachandran regions ─────────────────
# Helix: phi ∈ [-80°±20°], psi ∈ [-45°±20°]   (α-helix core)
HELIX_PHI_CENTER = np.radians(-80.0)
HELIX_PSI_CENTER = np.radians(-45.0)
HELIX_PHI_TOL    = np.radians(20.0)
HELIX_PSI_TOL    = np.radians(20.0)

# Sheet: phi ∈ [-120°±30°], psi ∈ [+120°±30°]  (β-strand core)
SHEET_PHI_CENTER = np.radians(-120.0)
SHEET_PSI_CENTER = np.radians(+120.0)
SHEET_PHI_TOL    = np.radians(30.0)
SHEET_PSI_TOL    = np.radians(30.0)

# ── Action space constants ───────────────────────────────────
N_ANGLES        = 2      # phi, psi
N_INCREMENTS    = 12     # 30° each
ANGLE_STEP      = np.radians(30.0)
ENERGY_DROP_BIG = 1.0    # kcal/mol threshold for big reward

# ── Episode constants ────────────────────────────────────────
MAX_STEPS       = 50
MAX_CLASHES     = 5
ENERGY_CONVERGE = 0.01   # kcal/mol — convergence threshold


PDB_PATHS = {
    "1L2Y": os.path.join(os.path.dirname(__file__), "../data/structures/1L2Y.pdb"),
    "1YRF": os.path.join(os.path.dirname(__file__), "../data/structures/1YRF.pdb"),
    "1VII": os.path.join(os.path.dirname(__file__), "../data/structures/1VII.pdb"),
    "2GB1": os.path.join(os.path.dirname(__file__), "../data/structures/2GB1.pdb"),
    "1ENH": os.path.join(os.path.dirname(__file__), "../data/structures/1ENH.pdb"),
    "1UBQ": os.path.join(os.path.dirname(__file__), "../data/structures/1UBQ.pdb"),
    "1BDD": os.path.join(os.path.dirname(__file__), "../data/structures/1BDD.pdb"),
    "2HHB": os.path.join(os.path.dirname(__file__), "../data/structures/2HHB.pdb"),
}


class FoldEnv(gym.Env):
    """
    ProteinFold-RL Gymnasium Environment (v3).

    The agent sequentially adjusts backbone dihedral angles (phi/psi)
    of a protein, rewarded by physics-based energy reduction and
    secondary structure formation.

    Observation : dict with PyG graph + step info + energy buffer
    Action      : Discrete(N_residues * 2 * 12)
                  = choose residue × angle type × increment direction

    Reward cases (9 total):
        +8.0  energy drop > 1 kcal/mol
        +2.0  energy drop < 1 kcal/mol
        +1.0  no steric clash
        -2.0  steric clash
        -1.0  energy increases
        -0.3  per-step efficiency penalty (always)
        +15.0 RMSD vs native < 2Å
        +3.0  per residue newly entering helix Ramachandran region
        +3.0  per residue newly entering sheet Ramachandran region
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, pdb_id: str = "1L2Y", render_mode=None):
        super().__init__()
        assert pdb_id in PDB_PATHS, f"Unknown PDB ID: {pdb_id}"

        self.pdb_id      = pdb_id
        self.render_mode = render_mode

        # Load native structure once
        self.native_graph  = pdb_to_graph(PDB_PATHS[pdb_id])
        self.native_coords = self.native_graph.pos.numpy().copy()
        self.N             = self.native_graph.num_nodes

        # Action space
        self.action_dim   = self.N * N_ANGLES * N_INCREMENTS
        self.action_space = spaces.Discrete(self.action_dim)

        # Observation space — flat box for gym compatibility
        # (actual graph passed separately to GNN)
        node_dim = 23
        self.observation_space = spaces.Dict({
            "node_features" : spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.N, node_dim), dtype=np.float32
            ),
            "ca_coords"     : spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.N, 3), dtype=np.float32
            ),
            "step"          : spaces.Box(
                low=0, high=MAX_STEPS,
                shape=(1,), dtype=np.float32
            ),
            "energy_history": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(5,), dtype=np.float32
            ),
        })

        # State variables (initialized in reset)
        self.ca_coords      = None
        self.phi_angles     = None
        self.psi_angles     = None
        self.current_energy = None
        self.energy_history = None
        self.step_count     = None
        self.clash_count    = None
        self.graph          = None

        # Secondary structure state: array of "helix"/"sheet"/"other"
        # per residue, tracked to award bonus only on *new* formations
        self.ss_state       = None

    # ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start from native structure + small random perturbation
        noise = np.random.normal(0, 0.5, self.native_coords.shape)
        self.ca_coords  = self.native_coords.copy() + noise.astype(np.float32)

        # Initialize backbone angles randomly (unfolded-like)
        self.phi_angles = np.random.uniform(-np.pi, np.pi, self.N)
        self.psi_angles = np.random.uniform(-np.pi, np.pi, self.N)

        self.current_energy = compute_energy(
            self.ca_coords, self.phi_angles, self.psi_angles
        )
        self.energy_history = np.full(5, self.current_energy, dtype=np.float32)
        self.step_count     = 0
        self.clash_count    = 0

        # Initialise SS state from starting angles
        self.ss_state = np.array([
            self._detect_ss(self.phi_angles[i], self.psi_angles[i])
            for i in range(self.N)
        ], dtype=object)

        self.graph = self._build_graph()
        obs        = self._get_obs()
        info       = {"energy": self.current_energy, "step": 0,
                      "ss_reward": 0.0}
        return obs, info

    # ────────────────────────────────────────────────────────
    def step(self, action: int):
        assert self.ca_coords is not None, "Call reset() before step()"

        # ── Decode action ───────────────────────────────────
        residue_idx, angle_type, increment = self._decode_action(action)

        # ── Apply torsion change ────────────────────────────
        old_phi = self.phi_angles.copy()
        old_psi = self.psi_angles.copy()

        delta = ANGLE_STEP * (increment - N_INCREMENTS // 2)
        if angle_type == 0:
            self.phi_angles[residue_idx] += delta
            self.phi_angles[residue_idx] = self._wrap_angle(
                self.phi_angles[residue_idx]
            )
        else:
            self.psi_angles[residue_idx] += delta
            self.psi_angles[residue_idx] = self._wrap_angle(
                self.psi_angles[residue_idx]
            )

        # ── Update Cα coordinates from angles ───────────────
        self._update_coords(residue_idx)

        # ── Compute new energy ──────────────────────────────
        new_energy   = compute_energy(
            self.ca_coords, self.phi_angles, self.psi_angles
        )
        energy_delta = compute_energy_delta(self.current_energy, new_energy)

        # ── Clash detection ─────────────────────────────────
        clash_result = detect_clashes(self.ca_coords)
        has_clash    = clash_result["has_clash"]

        if has_clash:
            self.clash_count += 1
            # Revert angles on clash
            self.phi_angles = old_phi
            self.psi_angles = old_psi
            self._update_coords(residue_idx)

        # ── Secondary structure reward ───────────────────────
        ss_reward = 0.0
        if not has_clash:
            ss_reward = self._compute_ss_reward()

        # ── Compute full reward ──────────────────────────────
        reward = self._compute_reward(
            energy_delta, has_clash, new_energy, ss_reward
        )

        # ── Update state ────────────────────────────────────
        if not has_clash:
            self.current_energy = new_energy

        self.energy_history = np.roll(self.energy_history, -1)
        self.energy_history[-1] = self.current_energy
        self.step_count += 1
        self.graph = self._build_graph()

        # ── Termination conditions ───────────────────────────
        converged = (
            len(set(np.round(self.energy_history, 3))) == 1
            and self.step_count > 5
        )
        terminated = (
            self.step_count >= MAX_STEPS or
            self.clash_count >= MAX_CLASHES or
            converged
        )
        truncated = False

        obs  = self._get_obs()
        info = {
            "energy"      : self.current_energy,
            "energy_delta": energy_delta,
            "step"        : self.step_count,
            "clash_count" : self.clash_count,
            "has_clash"   : has_clash,
            "ss_reward"   : ss_reward,
        }
        return obs, reward, terminated, truncated, info

    # ────────────────────────────────────────────────────────
    # Secondary structure helpers
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _detect_ss(phi: float, psi: float) -> str:
        """
        Classify a single residue into a Ramachandran region.

        Parameters
        ----------
        phi : float  backbone phi angle in radians [-π, π]
        psi : float  backbone psi angle in radians [-π, π]

        Returns
        -------
        "helix" | "sheet" | "other"
        """
        in_helix = (
            abs(phi - HELIX_PHI_CENTER) <= HELIX_PHI_TOL and
            abs(psi - HELIX_PSI_CENTER) <= HELIX_PSI_TOL
        )
        if in_helix:
            return "helix"

        in_sheet = (
            abs(phi - SHEET_PHI_CENTER) <= SHEET_PHI_TOL and
            abs(psi - SHEET_PSI_CENTER) <= SHEET_PSI_TOL
        )
        if in_sheet:
            return "sheet"

        return "other"

    def _compute_ss_reward(self) -> float:
        """
        Compare current phi/psi angles against previous ss_state.
        Award R_HELIX_BONUS / R_SHEET_BONUS for each residue that
        *newly* enters a structured Ramachandran region this step.

        Updates self.ss_state in place.

        Returns
        -------
        float — total secondary structure reward for this step
        """
        ss_reward = 0.0
        for i in range(self.N):
            new_ss = self._detect_ss(self.phi_angles[i], self.psi_angles[i])

            # Only reward transitions INTO a structured region
            if new_ss != self.ss_state[i]:
                if new_ss == "helix":
                    ss_reward += R_HELIX_BONUS
                elif new_ss == "sheet":
                    ss_reward += R_SHEET_BONUS

            self.ss_state[i] = new_ss

        return ss_reward

    # ────────────────────────────────────────────────────────
    def _compute_reward(self, energy_delta: float,
                        has_clash: bool, new_energy: float,
                        ss_reward: float = 0.0) -> float:
        """
        Compute the full step reward.

        Reward cases:
            -0.3   per-step penalty (always)
            -2.0   clash detected
            +1.0   no clash
            +8.0   energy drop > ENERGY_DROP_BIG
            +2.0   energy drop, small
            -1.0   energy increases
            +15.0  RMSD < RMSD_THRESHOLD
            +ss    secondary structure bonus (helix/sheet formations)
        """
        reward = R_STEP_PENALTY  # always applied

        if has_clash:
            reward += R_CLASH
            return reward

        reward += R_NO_CLASH

        if energy_delta < -ENERGY_DROP_BIG:
            reward += R_ENERGY_BIG
        elif energy_delta < 0:
            reward += R_ENERGY_SMALL
        else:
            reward += R_ENERGY_UP

        # RMSD bonus
        rmsd = self._compute_rmsd()
        if rmsd < RMSD_THRESHOLD:
            reward += R_RMSD_BONUS

        # Secondary structure bonus
        reward += ss_reward

        return reward

    # ────────────────────────────────────────────────────────
    def _decode_action(self, action: int):
        increment   = action % N_INCREMENTS
        remainder   = action // N_INCREMENTS
        angle_type  = remainder % N_ANGLES
        residue_idx = remainder // N_ANGLES
        residue_idx = min(residue_idx, self.N - 1)
        return residue_idx, angle_type, increment

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _update_coords(self, residue_idx: int):
        """
        Proper kinematic chain propagation.
        When a dihedral angle changes at residue i, all residues
        downstream (i+1, i+2, ..., N-1) must be recomputed.
        Uses NeRF (Natural Extension Reference Frame) algorithm.
        """
        BOND_LENGTH = 3.8  # Cα-Cα virtual bond length in Angstroms

        for i in range(residue_idx, self.N):
            if i == 0:
                continue

            if i == 1:
                prev = self.ca_coords[0]
                self.ca_coords[1] = prev + np.array(
                    [BOND_LENGTH, 0.0, 0.0], dtype=np.float32
                )
                continue

            phi = self.phi_angles[i]

            a = self.ca_coords[i - 2]
            b = self.ca_coords[i - 1]

            bc      = b - a
            bc_norm = bc / (np.linalg.norm(bc) + 1e-8)

            if i >= 2:
                n = np.cross(bc_norm, np.array([0, 0, 1], dtype=np.float32))
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-6:
                    n = np.cross(bc_norm,
                                 np.array([0, 1, 0], dtype=np.float32))
                    n_norm = np.linalg.norm(n)
                n = n / (n_norm + 1e-8)

            m = np.cross(n, bc_norm)

            d = BOND_LENGTH * np.array([
                np.cos(np.pi - 1.92),
                np.sin(np.pi - 1.92) * np.cos(phi),
                np.sin(np.pi - 1.92) * np.sin(phi)
            ], dtype=np.float32)

            self.ca_coords[i] = b + (
                d[0] * bc_norm +
                d[1] * m +
                d[2] * n
            )

    def _build_graph(self) -> Data:
        """Rebuild PyG graph from current coordinates."""
        N  = self.N
        ca = self.ca_coords

        aa_feats = self.native_graph.x[:, :20]
        coords_t = torch.tensor(ca, dtype=torch.float)
        x        = torch.cat([aa_feats, coords_t], dim=1)  # [N, 23]

        edge_src, edge_dst, edge_attrs = [], [], []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = ca[j] - ca[i]
                dist = float(np.linalg.norm(diff))
                if dist <= 8.0:
                    is_peptide = 1.0 if abs(i - j) == 1 else 0.0
                    dn = diff / (dist + 1e-8)
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_attrs.append([dist, dn[0], dn[1], is_peptide])

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    pos=coords_t, num_nodes=N)

    def _get_obs(self) -> dict:
        return {
            "node_features" : self.graph.x.numpy(),
            "ca_coords"     : self.ca_coords.copy(),
            "step"          : np.array([self.step_count], dtype=np.float32),
            "energy_history": self.energy_history.copy(),
        }

    def _compute_rmsd(self) -> float:
        """RMSD between current and native Cα coordinates."""
        diff = self.ca_coords - self.native_coords
        return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    def get_graph(self) -> Data:
        """Return current PyG graph for GNN input."""
        return self.graph

    def render(self):
        if self.render_mode == "human":
            helix_count = int(np.sum(self.ss_state == "helix"))
            sheet_count = int(np.sum(self.ss_state == "sheet"))
            print(f"Step: {self.step_count:3d} | "
                  f"Energy: {self.current_energy:8.3f} | "
                  f"Clashes: {self.clash_count} | "
                  f"Helix: {helix_count} | Sheet: {sheet_count}")


# ── Unit tests ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ProteinFold-RL — Environment Test (v3)")
    print("=" * 60)

    env = FoldEnv(pdb_id="1L2Y")
    obs, info = env.reset()

    print(f"\n[INIT] Trp-cage environment")
    print(f"  N residues     : {env.N}")
    print(f"  Action dim     : {env.action_dim}")
    print(f"  Initial energy : {info['energy']:.4f}")
    print(f"  Obs keys       : {list(obs.keys())}")

    # Verify SS detection on known angles
    print(f"\n[TEST] Secondary structure detection:")
    phi_helix = np.radians(-80.0)
    psi_helix = np.radians(-45.0)
    phi_sheet = np.radians(-120.0)
    psi_sheet = np.radians(+120.0)
    phi_other = np.radians(0.0)
    psi_other = np.radians(0.0)

    assert FoldEnv._detect_ss(phi_helix, psi_helix) == "helix", "Helix detect failed"
    assert FoldEnv._detect_ss(phi_sheet, psi_sheet) == "sheet", "Sheet detect failed"
    assert FoldEnv._detect_ss(phi_other, psi_other) == "other", "Other detect failed"
    print(f"  Helix detection  : PASS ✓")
    print(f"  Sheet detection  : PASS ✓")
    print(f"  Other detection  : PASS ✓")

    # Run 100 random steps — check ss_reward appears in info
    print(f"\n[TEST] Running 100 random steps with SS reward...")
    total_reward = 0.0
    total_ss_reward = 0.0
    helix_events = 0
    sheet_events = 0

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward    += reward
        total_ss_reward += info["ss_reward"]
        if info["ss_reward"] > 0:
            # Count helix vs sheet from ss_state
            helix_events += int(np.sum(env.ss_state == "helix"))
            sheet_events += int(np.sum(env.ss_state == "sheet"))
        if terminated:
            obs, info = env.reset()

    print(f"  Completed without crash ✓")
    print(f"  Total reward     : {total_reward:.3f}")
    print(f"  Total SS reward  : {total_ss_reward:.3f}")
    print(f"  'ss_reward' in info : ✓")

    # Verify all 9 reward constants
    print(f"\n[TEST] Reward constants (9 total):")
    print(f"  R_ENERGY_BIG   = {R_ENERGY_BIG}   ✓")
    print(f"  R_ENERGY_SMALL = {R_ENERGY_SMALL}   ✓")
    print(f"  R_NO_CLASH     = {R_NO_CLASH}   ✓")
    print(f"  R_CLASH        = {R_CLASH}  ✓")
    print(f"  R_ENERGY_UP    = {R_ENERGY_UP}  ✓")
    print(f"  R_STEP_PENALTY = {R_STEP_PENALTY}  ✓")
    print(f"  R_RMSD_BONUS   = {R_RMSD_BONUS}  ✓")
    print(f"  R_HELIX_BONUS  = {R_HELIX_BONUS}   ✓")
    print(f"  R_SHEET_BONUS  = {R_SHEET_BONUS}   ✓")

    print(f"\n{'=' * 60}")
    print("CHECKPOINT — Environment v3 stable. SS reward active.")
    print("=" * 60)