"""
gradio_app.py
ProteinFold-RL — Dashboard (v3)

What changed vs v2
------------------
- All 8 proteins available in Live Demo dropdown
- Policy loaded at MAX_ACTION_DIM (shared across all proteins)
- Protein info card shows residues + type on selection
- Comparison tab dynamically loads eval_results.json per protein
- Tab 3 has a per-protein breakdown table + run eval button
"""

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
    load_trajectory,
)
from config import MAX_ACTION_DIM, CHECKPOINT_PATH

# ── Protein registry — all 8 proteins ───────────────────────
PROTEIN_INFO = {
    "1L2Y": {"name": "Trp-cage",          "residues": 20,  "type": "Helix + turn",       "difficulty": "⭐ Easy"},
    "1YRF": {"name": "Villin headpiece",   "residues": 35,  "type": "3-helix bundle",     "difficulty": "⭐⭐ Medium"},
    "1VII": {"name": "Villin HP36",        "residues": 36,  "type": "3-helix bundle",     "difficulty": "⭐⭐ Medium"},
    "2GB1": {"name": "GB1 hairpin",        "residues": 56,  "type": "β-hairpin + helix",  "difficulty": "⭐⭐ Medium"},
    "1ENH": {"name": "Engrailed homeodomain","residues": 54,"type": "3-helix bundle",     "difficulty": "⭐⭐⭐ Hard"},
    "1UBQ": {"name": "Ubiquitin",          "residues": 76,  "type": "Mixed α/β",          "difficulty": "⭐⭐⭐ Hard"},
    "1BDD": {"name": "BBL domain",         "residues": 47,  "type": "3-helix bundle",     "difficulty": "⭐⭐ Medium"},
    "2HHB": {"name": "Hemoglobin (α chain)","residues": 141,"type": "All-α helical",      "difficulty": "⭐⭐⭐⭐ Expert"},
}

DROPDOWN_CHOICES = [
    f"{pdb} — {info['name']} ({info['residues']} res)"
    for pdb, info in PROTEIN_INFO.items()
]
DROPDOWN_MAP = {
    f"{pdb} — {info['name']} ({info['residues']} res)": pdb
    for pdb, info in PROTEIN_INFO.items()
}

# ── Load trained policy once at MAX_ACTION_DIM ───────────────
print("[INIT] Loading policy...")
_dummy_env = FoldEnv(pdb_id="1L2Y")
policy     = GNNPolicyNetwork(action_dim=MAX_ACTION_DIM)
trainer    = PPOTrainer(policy=policy, action_dim=MAX_ACTION_DIM)
if os.path.exists(CHECKPOINT_PATH):
    trainer.load(CHECKPOINT_PATH)
    print(f"[INIT] Checkpoint loaded: {CHECKPOINT_PATH}")
else:
    print(f"[WARN] No checkpoint found at {CHECKPOINT_PATH} — using untrained policy")
policy.eval()


# ── Helper: protein info card markdown ───────────────────────

def protein_info_card(label: str) -> str:
    pdb  = DROPDOWN_MAP.get(label, "1L2Y")
    info = PROTEIN_INFO[pdb]
    return (
        f"**{info['name']}** (`{pdb}`)\n\n"
        f"- Residues: **{info['residues']}**\n"
        f"- Type: {info['type']}\n"
        f"- Difficulty: {info['difficulty']}\n"
        f"- Action space: {info['residues'] * 2 * 12} discrete actions"
    )


# ── Core folding function ─────────────────────────────────────

def run_folding_demo(label: str, n_steps: int):
    """
    Run trained agent on selected protein.
    Returns: plot_df, init_pdb, final_pdb, native_pdb, summary_md
    """
    import pandas as pd

    pdb      = DROPDOWN_MAP.get(label, "1L2Y")
    env_demo = FoldEnv(pdb_id=pdb)
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
        # Clamp to this protein's valid action range
        action = action % env_demo.action_dim
        obs, reward, terminated, truncated, info = env_demo.step(action)
        done = terminated or truncated
        energies.append(info["energy"])
        steps.append(step + 1)
        step += 1

    final_coords = env_demo.ca_coords.copy()
    final_pdb    = coords_to_pdb_string(
        final_coords, env_demo.native_graph.seq
    )
    native_pdb = coords_to_pdb_string(
        env_demo.native_coords, env_demo.native_graph.seq
    )

    diff       = final_coords - env_demo.native_coords
    final_rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    energy_drop = energies[0] - energies[-1]

    plot_df = pd.DataFrame({
        "Step":   steps,
        "Energy": energies,
    })

    info_obj = PROTEIN_INFO[pdb]
    summary = (
        f"### Results — {info_obj['name']} (`{pdb}`)\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Protein | {info_obj['name']} ({pdb}) |\n"
        f"| Residues | {info_obj['residues']} |\n"
        f"| Steps run | {step} |\n"
        f"| Initial energy | {energies[0]:.3f} kcal/mol |\n"
        f"| Final energy | {energies[-1]:.3f} kcal/mol |\n"
        f"| Energy drop | {energy_drop:.3f} kcal/mol |\n"
        f"| Final RMSD vs native | {final_rmsd:.3f} Å |\n"
        f"| RMSD < 2Å | {'✅ Yes' if final_rmsd < 2.0 else '❌ Not yet'} |"
    )

    return plot_df, initial_pdb, final_pdb, native_pdb, summary


# ── Training log loader ───────────────────────────────────────

def load_results():
    import pandas as pd
    log   = load_training_log("logs/training_log.csv")
    e_df  = pd.DataFrame({"Episode": log["episode"], "Energy": log["final_energy"]})
    r_df  = pd.DataFrame({"Episode": log["episode"], "RMSD":   log["rmsd"]})
    return e_df, r_df


# ── Per-protein eval results loader ──────────────────────────

def load_per_protein_results():
    """
    Load eval_results.json (written by eval.py).
    Returns a markdown table.
    """
    import json
    path = "logs/eval_results.json"
    if not os.path.exists(path):
        return (
            "**No eval results found.**\n\n"
            "Run `python eval.py` to generate results,\n"
            "or run eval for each protein:\n"
            "`python eval.py --protein 1L2Y`"
        )

    with open(path) as f:
        r = json.load(f)

    pdb  = r.get("protein", "?")
    info = PROTEIN_INFO.get(pdb, {})

    lines = [
        f"### Evaluation Results — {info.get('name', pdb)} (`{pdb}`)\n",
        f"Episodes per agent: **{r['n_episodes']}**\n",
        "| Metric | Random | Trained | Improvement |",
        "|--------|--------|---------|-------------|",
        f"| Avg RMSD (Å) | {r['random']['avg_rmsd']} | {r['trained']['avg_rmsd']} | {r['improvement']['rmsd_abs']:+.3f} Å ({r['improvement']['rmsd_pct']:+.1f}%) |",
        f"| Avg Energy | {r['random']['avg_energy']} | {r['trained']['avg_energy']} | {r['improvement']['energy_abs']:+.3f} ({r['improvement']['energy_pct']:+.1f}%) |",
        f"| Best RMSD | — | {r['trained']['best_rmsd']} Å | {'✅ < 2Å' if r['trained']['best_rmsd'] < 2.0 else '—'} |",
        f"\n**Verdict:** {'✅ PASS — Trained agent outperforms random baseline' if r['passed'] else '⚠️ WARN — More training needed'}",
    ]
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="ProteinFold-RL", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧬 ProteinFold-RL
    ### *AlphaFold shows the destination. We discover the journey.*

    An RL agent that learns **how** proteins fold — not just where they end up.
    Rewarded by the laws of chemistry. No human labels. Physics is the teacher.
    """)

    # ── Tab 1: Training Results ──────────────────────────────
    with gr.Tab("📈 Training Results"):
        gr.Markdown("### Proof of Learning — Curriculum Training across 8 Proteins")

        with gr.Row():
            energy_plot = gr.LinePlot(
                label="Energy vs Episodes (lower = better folding)",
                x="Episode", y="Energy"
            )
            rmsd_plot = gr.LinePlot(
                label="RMSD vs Episodes (lower = closer to native)",
                x="Episode", y="RMSD"
            )

        load_btn = gr.Button("📂 Load Training Results", variant="primary")
        load_btn.click(load_results, outputs=[energy_plot, rmsd_plot])

    # ── Tab 2: Live Demo — ALL 8 proteins ────────────────────
    with gr.Tab("🔬 Live Folding Demo"):
        gr.Markdown("### Watch the agent fold any of the 8 training proteins")

        with gr.Row():
            with gr.Column(scale=2):
                pdb_dropdown = gr.Dropdown(
                    choices=DROPDOWN_CHOICES,
                    value=DROPDOWN_CHOICES[0],
                    label="Select Protein",
                )
                steps_slider = gr.Slider(
                    minimum=10, maximum=50, value=50, step=5,
                    label="Number of folding steps"
                )
                run_btn = gr.Button("▶ Run Folding Agent", variant="primary")

            with gr.Column(scale=1):
                protein_card = gr.Markdown(
                    value=protein_info_card(DROPDOWN_CHOICES[0]),
                    label="Protein Info"
                )

        # Update protein card on dropdown change
        pdb_dropdown.change(
            fn=protein_info_card,
            inputs=pdb_dropdown,
            outputs=protein_card,
        )

        with gr.Row():
            demo_energy_plot = gr.LinePlot(
                label="Energy during folding",
                x="Step", y="Energy"
            )
            demo_summary = gr.Markdown("Select a protein and click **Run Folding Agent**.")

        with gr.Row():
            before_pdb = gr.Textbox(
                label="Initial conformation (PDB)", lines=5, max_lines=10
            )
            after_pdb = gr.Textbox(
                label="Final conformation (PDB)", lines=5, max_lines=10
            )
            native_pdb_box = gr.Textbox(
                label="Native structure (PDB)", lines=5, max_lines=10
            )

        run_btn.click(
            fn=run_folding_demo,
            inputs=[pdb_dropdown, steps_slider],
            outputs=[demo_energy_plot, before_pdb, after_pdb, native_pdb_box, demo_summary],
        )

    # ── Tab 3: Comparison — dynamic from eval_results.json ───
    with gr.Tab("🏆 Agent vs Random"):
        gr.Markdown("### Trained Agent vs Random Baseline")

        gr.Markdown("""
        > Results below are loaded from the last `eval.py` run.
        > To evaluate a specific protein: `python eval.py --protein 2GB1`
        """)

        eval_md = gr.Markdown(value=load_per_protein_results())
        refresh_btn = gr.Button("🔄 Refresh Results", variant="secondary")
        refresh_btn.click(fn=load_per_protein_results, outputs=eval_md)

        gr.Markdown("---")
        gr.Markdown("""
        ### All 8 Proteins — Curriculum Overview

        | PDB  | Protein              | Residues | Type              | Difficulty     |
        |------|----------------------|----------|-------------------|----------------|
        | 1L2Y | Trp-cage             | 20       | Helix + turn      | ⭐ Easy        |
        | 1YRF | Villin headpiece     | 35       | 3-helix bundle    | ⭐⭐ Medium    |
        | 1VII | Villin HP36          | 36       | 3-helix bundle    | ⭐⭐ Medium    |
        | 2GB1 | GB1 hairpin          | 56       | β-hairpin + helix | ⭐⭐ Medium    |
        | 1BDD | BBL domain           | 47       | 3-helix bundle    | ⭐⭐ Medium    |
        | 1ENH | Engrailed homeodomain| 54       | 3-helix bundle    | ⭐⭐⭐ Hard    |
        | 1UBQ | Ubiquitin            | 76       | Mixed α/β         | ⭐⭐⭐ Hard    |
        | 2HHB | Hemoglobin α chain   | 141      | All-α helical     | ⭐⭐⭐⭐ Expert|

        The curriculum starts with 1L2Y (easiest) and advances as the agent masters each protein.
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
        - **Curriculum** → 8 proteins, easy → hard

        ### Architecture
        ```
        PDB file → BioPython → PyG graph [N×23 nodes, E×4 edges]
                → NodeEncoder(23→128) + EdgeEncoder(4→64)
                → 4-layer MPNN → 256-dim global embedding
                → Policy head (→ action logits)
                → Value head  (→ scalar V(s))
                → PPO-clip update every 256 steps
        ```

        ### Tech Stack
        `PyTorch` · `PyTorch Geometric` · `Gymnasium` · `BioPython` · `Gradio`
        """)


if __name__ == "__main__":
    print("=" * 60)
    print("ProteinFold-RL — Launching Dashboard v3")
    print("  All 8 proteins available in Live Demo")
    print("=" * 60)
    demo.launch(share=False, show_error=True)