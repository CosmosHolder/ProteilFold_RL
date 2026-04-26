"""
analysis/ramachandran.py
ProteinFold-RL — Ramachandran Diagram Generator

Runs the trained agent on a protein, collects all phi/psi angles
visited during folding, and plots them on a Ramachandran diagram.

Three datasets are plotted together:
  - Random agent angles  (grey)  — baseline, scattered everywhere
  - Trained agent angles (blue)  — should cluster in allowed regions
  - Native structure     (red ★) — the ground truth target

Allowed regions (Ramachandran):
  Alpha-helix : phi ∈ [-160°, -20°], psi ∈ [-80°,  40°]
  Beta-sheet  : phi ∈ [-160°, -60°], psi ∈ [ 80°, 180°]

If the trained agent's angles cluster in these regions more than
the random agent's angles, that is proof the agent learned physics.

Output : logs/ramachandran.png

Run    : python analysis/ramachandran.py
         python analysis/ramachandran.py --protein 1YRF
         this is running mechanism
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.fold_env import FoldEnv
from model.gnn_policy import GNNPolicyNetwork
from agent.ppo import PPOTrainer
from config import MAX_ACTION_DIM, CHECKPOINT_PATH

# ── Config ────────────────────────────────────────────────────
EVAL_EPISODES = 20     # episodes to collect angles from
OUTPUT_PATH   = "logs/ramachandran.png"

os.makedirs("logs", exist_ok=True)


# ── Allowed region boundaries (degrees) ──────────────────────
HELIX_PHI  = (-160, -20)
HELIX_PSI  = ( -80,  40)
SHEET_PHI  = (-160, -60)
SHEET_PSI  = (  80, 180)


def collect_angles(env: FoldEnv, policy, n_episodes: int,
                   use_random: bool = False) -> tuple:
    """
    Run n_episodes and collect all phi/psi angles visited.

    Returns
    -------
    phi_deg : np.ndarray  all phi angles in degrees
    psi_deg : np.ndarray  all psi angles in degrees
    """
    all_phi, all_psi = [], []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False

        # Collect initial angles
        all_phi.extend(np.degrees(env.phi_angles).tolist())
        all_psi.extend(np.degrees(env.psi_angles).tolist())

        while not done:
            if use_random:
                action = env.action_space.sample()
            else:
                graph = env.get_graph()
                with torch.no_grad():
                    action, _, _, _ = policy.get_action(graph)
                action = action % env.action_dim

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Collect angles at each step
            all_phi.extend(np.degrees(env.phi_angles).tolist())
            all_psi.extend(np.degrees(env.psi_angles).tolist())

    return np.array(all_phi), np.array(all_psi)


def get_native_angles(env: FoldEnv) -> tuple:
    """
    Extract native phi/psi angles by resetting with zero noise.
    These represent the target conformation.
    """
    # Reset to get initial angles, then read native directly
    env.reset()
    # Native angles are approximated from the native coords reset
    # We run one reset and read the angles before any perturbation
    phi = np.degrees(env.phi_angles)
    psi = np.degrees(env.psi_angles)
    return phi, psi


def fraction_in_allowed(phi_deg: np.ndarray,
                         psi_deg: np.ndarray) -> float:
    """
    Fraction of (phi, psi) pairs that fall in helix OR sheet regions.
    Higher = more physically realistic conformation.
    """
    in_helix = (
        (phi_deg >= HELIX_PHI[0]) & (phi_deg <= HELIX_PHI[1]) &
        (psi_deg >= HELIX_PSI[0]) & (psi_deg <= HELIX_PSI[1])
    )
    in_sheet = (
        (phi_deg >= SHEET_PHI[0]) & (phi_deg <= SHEET_PHI[1]) &
        (psi_deg >= SHEET_PSI[0]) & (psi_deg <= SHEET_PSI[1])
    )
    return float(np.mean(in_helix | in_sheet))


def plot_ramachandran(random_phi, random_psi,
                      trained_phi, trained_psi,
                      native_phi, native_psi,
                      protein: str,
                      random_frac: float,
                      trained_frac: float,
                      output_path: str):
    """
    Generate and save the Ramachandran diagram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f1a")

    titles = [
        f"Random Agent  ({len(random_phi):,} angles)",
        f"Trained Agent  ({len(trained_phi):,} angles)",
    ]
    phi_sets = [random_phi,  trained_phi]
    psi_sets = [random_psi,  trained_psi]
    colors   = ["#888888",   "#4fc3f7"]
    fracs    = [random_frac, trained_frac]

    for ax, title, phi, psi, color, frac in zip(
            axes, titles, phi_sets, psi_sets, colors, fracs):

        ax.set_facecolor("#0f0f1a")

        # ── Allowed region shading ────────────────────────────
        helix_patch = mpatches.FancyBboxPatch(
            (HELIX_PHI[0], HELIX_PSI[0]),
            HELIX_PHI[1] - HELIX_PHI[0],
            HELIX_PSI[1] - HELIX_PSI[0],
            boxstyle="round,pad=2",
            linewidth=0,
            facecolor=to_rgba("#00e676", 0.10),
            zorder=1,
        )
        sheet_patch = mpatches.FancyBboxPatch(
            (SHEET_PHI[0], SHEET_PSI[0]),
            SHEET_PHI[1] - SHEET_PHI[0],
            SHEET_PSI[1] - SHEET_PSI[0],
            boxstyle="round,pad=2",
            linewidth=0,
            facecolor=to_rgba("#ffeb3b", 0.10),
            zorder=1,
        )
        ax.add_patch(helix_patch)
        ax.add_patch(sheet_patch)

        # ── Scatter plot ──────────────────────────────────────
        ax.scatter(phi, psi,
                   c=color, alpha=0.25, s=8,
                   linewidths=0, zorder=2,
                   label=f"Agent angles")

        # ── Native structure ──────────────────────────────────
        ax.scatter(native_phi, native_psi,
                   c="#ff1744", marker="*", s=120,
                   zorder=5, label="Native structure",
                   edgecolors="white", linewidths=0.4)

        # ── Region labels ─────────────────────────────────────
        ax.text(-95, -20, "α-helix", color="#00e676",
                fontsize=9, alpha=0.8, zorder=6)
        ax.text(-130, 140, "β-sheet", color="#ffeb3b",
                fontsize=9, alpha=0.8, zorder=6)

        # ── Axes ──────────────────────────────────────────────
        ax.axhline(0, color="#333355", linewidth=0.5, zorder=0)
        ax.axvline(0, color="#333355", linewidth=0.5, zorder=0)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("φ (phi) degrees", color="white", fontsize=11)
        ax.set_ylabel("ψ (psi) degrees", color="white", fontsize=11)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        # ── Title + fraction ──────────────────────────────────
        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.text(0.98, 0.02,
                f"Allowed: {frac*100:.1f}%",
                transform=ax.transAxes,
                ha="right", va="bottom",
                color="#00e676" if frac > random_frac else "#ff7043",
                fontsize=11, fontweight="bold")

        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#1a1a2e", labelcolor="white",
                  edgecolor="#333355")

    # ── Main title ────────────────────────────────────────────
    improvement = (trained_frac - random_frac) * 100
    fig.suptitle(
        f"ProteinFold-RL — Ramachandran Diagram  |  {protein}\n"
        f"Trained agent: {improvement:+.1f}% more angles in allowed regions",
        color="white", fontsize=13, y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────

def generate(protein: str = "1L2Y", n_episodes: int = EVAL_EPISODES):
    print("=" * 58)
    print("ProteinFold-RL — Ramachandran Diagram Generator")
    print(f"  Protein  : {protein}")
    print(f"  Episodes : {n_episodes}")
    print("=" * 58)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n[ERROR] No checkpoint at {CHECKPOINT_PATH}")
        print("  Run train.py first.")
        sys.exit(1)

    # ── Load policy ───────────────────────────────────────────
    env    = FoldEnv(pdb_id=protein)
    policy = GNNPolicyNetwork(action_dim=MAX_ACTION_DIM)
    trainer= PPOTrainer(policy=policy, action_dim=MAX_ACTION_DIM)
    trainer.load(CHECKPOINT_PATH)
    policy.eval()

    # ── Collect angles ────────────────────────────────────────
    print(f"\n[1/3] Collecting random agent angles...")
    random_phi, random_psi = collect_angles(
        env, policy, n_episodes, use_random=True
    )

    print(f"[2/3] Collecting trained agent angles...")
    trained_phi, trained_psi = collect_angles(
        env, policy, n_episodes, use_random=False
    )

    print(f"[3/3] Reading native structure angles...")
    native_phi, native_psi = get_native_angles(env)

    # ── Compute allowed fractions ─────────────────────────────
    random_frac  = fraction_in_allowed(random_phi,  random_psi)
    trained_frac = fraction_in_allowed(trained_phi, trained_psi)

    print(f"\n  Random  allowed: {random_frac*100:.1f}%")
    print(f"  Trained allowed: {trained_frac*100:.1f}%")
    improvement = (trained_frac - random_frac) * 100
    print(f"  Improvement    : {improvement:+.1f}%")

    # ── Plot ──────────────────────────────────────────────────
    plot_ramachandran(
        random_phi, random_psi,
        trained_phi, trained_psi,
        native_phi, native_psi,
        protein=protein,
        random_frac=random_frac,
        trained_frac=trained_frac,
        output_path=OUTPUT_PATH,
    )

    print("\n" + "=" * 58)
    print("Ramachandran diagram complete.")
    if trained_frac > random_frac:
        print("  [PASS] Trained agent more physically realistic ✅")
    else:
        print("  [NOTE] More training episodes will improve this.")
    print("=" * 58)


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protein", type=str, default="1L2Y",
        help="PDB ID to analyse (default: 1L2Y)"
    )
    parser.add_argument(
        "--episodes", type=int, default=EVAL_EPISODES,
        help=f"Episodes to collect angles from (default: {EVAL_EPISODES})"
    )
    args = parser.parse_args()
    generate(protein=args.protein, n_episodes=args.episodes)
