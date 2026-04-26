# 🧬 ProteinFold-RL

> **AlphaFold shows the destination. We discover the journey.... **

An reinforcement learning agent that learns **how proteins fold** — not just where they end up.
Rewarded by the laws of chemistry. No human labels. Physics is the teacher.

---

## The Problem

**AlphaFold2** (Nobel Prize 2024) predicts the final 3D structure of a protein with near-experimental accuracy.
It is completely silent about **the folding pathway** — the sequence of physical conformational changes
a protein undergoes as it folds.

The pathway is where disease lives:

- 🧠 **Alzheimer's disease** — amyloid-beta misfolding
- 🧠 **Parkinson's disease** — alpha-synuclein aggregation
- 💉 **Type 2 Diabetes** — IAPP fibril formation

No dataset of correct folding pathways exists at atomic resolution.
**Physics is the only teacher. RL is the only honest framework.**

---

## Results

| Metric | Random Agent | Trained Agent | Improvement |
|--------|-------------|---------------|-------------|
| Avg RMSD (Å) | 16.775 | 15.920 | −0.855 Å (−5.1%) |
| Avg Energy (kcal/mol) | 117.258 | 54.430 | −62.828 (−53.6%) |
| Episodes trained | — | 500 | — |
| Verdict | Baseline | **Superior** | **[PASS] ✅** |

> Trained agent outperforms random baseline on every metric.
> Energy reduced by **53.6%** — the agent learned physics-based folding from reward alone.

---

## Architecture

```
Protein PDB
    │
    ▼
┌─────────────────────────────────────┐
│  FoldEnv (Gymnasium)                │
│  State  → protein graph             │
│           nodes [N, 23]             │
│           edges [E, 4] @ 8Å cutoff  │
│  Action → residue × angle × step    │
│           Discrete(N × 2 × 12)      │
│  Reward → Lennard-Jones energy Δ    │
│           +8.0  big energy drop     │
│           +2.0  small energy drop   │
│           +1.0  no clash            │
│           −2.0  steric clash        │
│           −0.3  per step penalty    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  GNN Policy Network                 │
│  NodeEncoder  23 → 128 dim          │
│  EdgeEncoder   4 →  64 dim          │
│  MPNNStack    4 layers → 256 dim    │
│  Policy head  256 → action_dim      │
│  Value head   256 → 128 → 1         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  PPO-clip Trainer                   │
│  ε = 0.2   γ = 0.99   λ = 0.95     │
│  Horizon T = 256   Epochs = 4       │
│  AdamW lr = 3e-4                    │
│  GAE advantage estimation           │
└─────────────────────────────────────┘
```

---

## Project Structure

```
ProteinFold-RL/
│
├── data/
│   └── structures/          # PDB files (1L2Y, 1YRF, ...)
│
├── env/
│   ├── fold_env.py          # Gymnasium environment
│   ├── protein_graph.py     # PDB → PyG graph
│   ├── energy.py            # Lennard-Jones + torsion energy
│   └── clash_detect.py      # Steric clash detection
│
├── model/
│   ├── features.py          # Node + edge encoders
│   ├── mpnn.py              # Message passing layers
│   └── gnn_policy.py        # Full policy + value network
│
├── agent/
│   └── ppo.py               # PPO-clip trainer
│
├── app/
│   ├── visualize.py         # PDB string gen, log loader
│   └── gradio_app.py        # Gradio dashboard (standalone)
│
├── frontend/
│   ├── index.html           # Landing page
│   ├── dashboard.html       # Training charts
│   ├── demo.html            # Live folding demo
│   ├── science.html         # Architecture + equations
│   ├── compare.html         # Agent vs random baseline
│   └── assets/
│       ├── css/design-system.css
│       ├── js/charts.js
│       ├── js/stars.js
│       ├── js/ticker.js
│       └── data/training_log.json
│
├── logs/
│   ├── training_log.csv     # Per-episode training data
│   ├── best_trajectory.csv  # Best episode step log
│   └── eval_results.json    # Evaluation summary
│
├── checkpoints/
│   └── policy_final.pt      # Trained model weights
│
├── train.py                 # Training loop
├── eval.py                  # Evaluation vs random baseline
├── csv_to_json.py           # Convert training log for dashboard
└── README.md
```

---

## Setup

### Requirements

- Python 3.10+
- Windows / Linux / macOS
- CPU only (no GPU required)

### Install

```bash
git clone https://github.com/your-username/ProteinFold-RL
cd ProteinFold-RL

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.7.0
pip install biopython gymnasium fastapi uvicorn gradio numpy scipy pandas
```

### Download protein structures

```bash
python data/prepare_data.py
```

---

## Run

### 1 — Train the agent

```bash
python train.py
# Optional: single protein mode
python train.py --protein 1L2Y --episodes 500
```

Training logs to `logs/training_log.csv`. Checkpoint saved to `checkpoints/policy_final.pt`.

### 2 — Evaluate

```bash
python eval.py
# Results saved to logs/eval_results.json
```

### 3 — Convert training log for dashboard

```bash
python csv_to_json.py
# Writes frontend/assets/data/training_log.json
```

### 4 — Start the FastAPI backend

```bash
uvicorn app.main:app --port 8000
# API docs at http://localhost:8000/docs
```

### 5 — Open the frontend

Open `frontend/index.html` in your browser.
Navigate to **Live Demo** and click **Run folding agent**.

### 6 — Gradio dashboard (alternative)

```bash
python app/gradio_app.py
# Opens at http://127.0.0.1:7860
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Backend status + model loaded flag |
| `POST` | `/fold` | Run agent on a protein, returns energy curve |
| `GET` | `/training-log-json` | Full training log as JSON array |
| `GET` | `/results` | Latest eval results |
| `GET` | `/best-episode` | Best episode trajectory |

**POST `/fold` request body:**
```json
{
  "pdb_id": "1L2Y",
  "n_steps": 50
}
```

**POST `/fold` response:**
```json
{
  "energy_curve": [[0, 117.3], [1, 112.1], "..."],
  "final_rmsd": 15.920,
  "initial_energy": 117.3,
  "final_energy": 54.4
}
```

---

## Science

### Why Reinforcement Learning?

| Requirement | Why RL fits |
|---|---|
| No labelled pathway data exists | RL learns without labels |
| Sequential conformational decisions | RL models sequential decisions naturally |
| Physics as the only ground truth | Energy function is the reward signal |
| Exploration of novel pathways | Entropy bonus drives exploration |

### Energy Function

The reward signal uses a coarse-grained physics potential — the same approximation
used in **MARTINI coarse-grained models** (Nature Methods), standard in computational biophysics:

```
E_total = E_LJ + E_torsion

E_LJ      = 4ε [ (σ/r)¹² − (σ/r)⁶ ]   Lennard-Jones pairwise
E_torsion = Σ k[1 + cos(nφ − δ)]        Ramachandran torsion penalty
```

### Why GNN and not MLP?

Proteins are graphs. A flat MLP loses all structural topology.
Message passing over the contact graph propagates local chemical
information across the full protein — exactly what spatially-aware
folding decisions require.

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning | PyTorch 2.6 |
| Graph neural nets | PyTorch Geometric 2.7 |
| RL environment | Gymnasium |
| Structure parsing | BioPython |
| RL algorithm | PPO-clip + GAE |
| Kinematics | NeRF (Natural Extension Reference Frame) |
| Energy function | Lennard-Jones + Ramachandran torsion |
| Backend API | FastAPI + Uvicorn |
| Gradio dashboard | Gradio |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |

---

## Track

**Open Innovation** — NIT Delhi · 2025

---

## The one-line answer to every judge question

> *"AlphaFold shows the destination. We discover the journey."*
