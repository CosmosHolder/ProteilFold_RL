import gradio as gr
import numpy as np
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.fold_env import FoldEnv
from model.gnn_policy import GNNPolicyNetwork
from agent.ppo import PPOTrainer
from app.visualize import (
    coords_to_pdb_string,
    load_training_log,
    load_trajectory
)

# ── Load trained model once ──────────────────────────────────
env    = FoldEnv(pdb_id="1L2Y")
policy = GNNPolicyNetwork(action_dim=env.action_dim)
trainer= PPOTrainer(policy=policy, action_dim=env.action_dim)
trainer.load("checkpoints/policy_final.pt")
policy.eval()

NATIVE_COORDS = env.native_coords.copy()
NATIVE_SEQ    = env.native_graph.seq


def get_energy_chart():
    """Load training log and return energy + reward curves."""
    log = load_training_log("logs/training_log.csv")

    episodes = log["episode"]
    energies = log["final_energy"]
    rewards  = log["total_reward"]
    rmsds    = log["rmsd"]

    # Smooth with rolling average (window=10)
    def smooth(arr, w=10):
        return [
            float(np.mean(arr[max(0, i-w):i+1]))
            for i in range(len(arr))
        ]

    energy_data = list(zip(episodes, smooth(energies)))
    reward_data = list(zip(episodes, smooth(rewards)))
    rmsd_data   = list(zip(episodes, smooth(rmsds)))

    return energy_data, reward_data, rmsd_data


def run_folding_demo(pdb_choice: str, n_steps: int):
    """
    Run trained agent on selected protein.
    Returns energy curve + before/after PDB strings.
    """
    env_demo = FoldEnv(pdb_id=pdb_choice)
    obs, _   = env_demo.reset()

    initial_coords = env_demo.ca_coords.copy()
    initial_pdb    = coords_to_pdb_string(
        initial_coords, env_demo.native_graph.seq
    )

    energies = [env_demo.current_energy]
    steps    = [0]

    done = False
    step = 0
    while not done and step < n_steps:
        graph = env_demo.get_graph()
        with torch.no_grad():
            action, _, _, _ = policy.get_action(graph, deterministic=False)
        obs, reward, terminated, truncated, info = env_demo.step(action)
        done = terminated or truncated
        energies.append(info["energy"])
        steps.append(step + 1)
        step += 1

    final_coords = env_demo.ca_coords.copy()
    final_pdb    = coords_to_pdb_string(
        final_coords, env_demo.native_graph.seq
    )
    native_pdb   = coords_to_pdb_string(
        env_demo.native_coords, env_demo.native_graph.seq
    )

    # RMSD
    diff         = final_coords - env_demo.native_coords
    final_rmsd   = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    energy_curve = list(zip(steps, energies))

    summary = (
        f"**Protein:** {pdb_choice}\n\n"
        f"**Steps run:** {step}\n\n"
        f"**Initial energy:** {energies[0]:.3f} kcal/mol\n\n"
        f"**Final energy:** {energies[-1]:.3f} kcal/mol\n\n"
        f"**Energy drop:** {energies[0] - energies[-1]:.3f} kcal/mol\n\n"
        f"**Final RMSD vs native:** {final_rmsd:.3f} Å"
    )

    return (
        energy_curve,
        initial_pdb,
        final_pdb,
        native_pdb,
        summary
    )


def get_trajectory_replay():
    """Load best trajectory energy curve."""
    traj = load_trajectory("logs/best_trajectory.csv")
    return [(t["step"], t["energy"]) for t in traj]


# ── Gradio UI ────────────────────────────────────────────────
with gr.Blocks(
    title="ProteinFold-RL",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🧬 ProteinFold-RL
    ### *AlphaFold shows the destination. We discover the journey.*

    An RL agent that learns **how** proteins fold — not just where they end up.
    Rewarded by the laws of chemistry. No human labels. Physics is the teacher.
    """)

    # ── Tab 1: Training Results ──────────────────────────────
    with gr.Tab("📈 Training Results"):
        gr.Markdown("### Proof of Learning — 500 Episodes on Trp-cage")

        with gr.Row():
            energy_plot = gr.LinePlot(
                label="Energy vs Episodes (lower = better folding)",
                x="Episode", y="Energy"
            )
            rmsd_plot = gr.LinePlot(
                label="RMSD vs Episodes (lower = closer to native)",
                x="Episode", y="RMSD"
            )

        load_btn = gr.Button("Load Training Results", variant="primary")


        def load_results():
            import pandas as pd
            log = load_training_log("logs/training_log.csv")
            e_data = pd.DataFrame({
                "Episode": log["episode"],
                "Energy": log["final_energy"]
            })
            r_data = pd.DataFrame({
                "Episode": log["episode"],
                "RMSD": log["rmsd"]
            })
            return e_data, r_data

        load_btn.click(load_results, outputs=[energy_plot, rmsd_plot])

    # ── Tab 2: Live Demo ─────────────────────────────────────
    with gr.Tab("🔬 Live Folding Demo"):
        gr.Markdown("### Watch the agent fold a protein in real time")

        with gr.Row():
            pdb_dropdown = gr.Dropdown(
                choices=["1L2Y", "1YRF"],
                value="1L2Y",
                label="Select Protein (1L2Y=Trp-cage, 1YRF=Villin)"
            )
            steps_slider = gr.Slider(
                minimum=10, maximum=50, value=50, step=5,
                label="Number of folding steps"
            )

        run_btn = gr.Button("▶ Run Folding Agent", variant="primary")

        with gr.Row():
            demo_energy_plot = gr.LinePlot(
                label="Energy during folding",
                x="Step", y="Energy"
            )
            demo_summary = gr.Markdown("Run the agent to see results...")

        with gr.Row():
            before_pdb = gr.Textbox(
                label="Initial conformation (PDB)",
                lines=5, max_lines=10
            )
            after_pdb = gr.Textbox(
                label="Final conformation (PDB)",
                lines=5, max_lines=10
            )
            native_pdb_box = gr.Textbox(
                label="Native structure (PDB)",
                lines=5, max_lines=10
            )


        def run_demo(pdb_choice, n_steps):
            import pandas as pd
            curve, init_pdb, fin_pdb, nat_pdb, summary = \
                run_folding_demo(pdb_choice, int(n_steps))
            plot_data = pd.DataFrame({
                "Step": [s for s, e in curve],
                "Energy": [e for s, e in curve]
            })
            return plot_data, init_pdb, fin_pdb, nat_pdb, summary

        run_btn.click(
            run_demo,
            inputs=[pdb_dropdown, steps_slider],
            outputs=[
                demo_energy_plot, before_pdb,
                after_pdb, native_pdb_box, demo_summary
            ]
        )

    # ── Tab 3: Comparison ────────────────────────────────────
    with gr.Tab("🏆 Agent vs Random"):
        gr.Markdown("""
        ### Trained Agent vs Random Baseline

        | Metric | Random Agent | Trained Agent | Improvement |
        |--------|-------------|---------------|-------------|
        | Avg RMSD | 7.637 Å | 3.145 Å | **+4.49 Å** |
        | Avg Energy | 267 kcal/mol | 148 kcal/mol | **+118 kcal/mol** |
        | Best RMSD | — | 1.313 Å | **< 2Å threshold** ✅ |
        """)

    # ── Tab 4: About ─────────────────────────────────────────
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ### Why ProteinFold-RL?

        **AlphaFold2** (Nobel Prize 2024) predicts where proteins end up.
        It is completely silent about **how they get there**.

        The folding **pathway** is where disease lives:
        - 🧠 Alzheimer's disease
        - 🧠 Parkinson's disease
        - 💉 Type 2 Diabetes

        ### How it works
        - **State** → protein as a graph (nodes = residues, edges = contacts)
        - **Action** → adjust backbone dihedral angles (φ/ψ)
        - **Reward** → energy drop (physics, not labels)
        - **Algorithm** → PPO + Graph Neural Network

        ### Tech Stack
        `PyTorch` · `PyTorch Geometric` · `Gymnasium` · `BioPython` · `Gradio`
        """)


if __name__ == "__main__":
    print("=" * 60)
    print("ProteinFold-RL — Launching Dashboard")
    print("=" * 60)
    demo.launch(share=False, show_error=True)