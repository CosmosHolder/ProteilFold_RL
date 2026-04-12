# ProteilFold_RL
nit delhi project
# ProteinFold-RL — Backend API v2.0

> **AlphaFold shows the destination. We discover the journey.**

FastAPI backend for the ProteinFold-RL project.  
Runs on **port 8000** locally. Deployed to **HuggingFace Spaces**.

---

## Project structure

```
ProteilFold_RL/
├── api/
│   ├── __init__.py
│   ├── main.py              ← FastAPI app, CORS, lifespan
│   ├── model_manager.py     ← Singleton model loader
│   ├── fold_runner.py       ← Inference engine (no FastAPI deps)
│   ├── schemas.py           ← All Pydantic request/response models
│   └── routes/
│       ├── __init__.py
│       ├── health.py        ← GET /health
│       ├── fold.py          ← POST /fold
│       └── results.py       ← GET /results, /best-episode, /compare
└── api/tests/
    └── test_api.py          ← Full test suite (no weights needed)
```

---

## Quick start

```powershell
# From project root
.venv\Scripts\activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs  
ReDoc:       http://localhost:8000/redoc

---

## Endpoints

### `GET /health`
Liveness + readiness probe.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true,
  "checkpoint_path": "checkpoints/policy_final.pt",
  "supported_proteins": ["1L2Y", "1YRF"],
  "version": "2.0.0"
}
```

---

### `POST /fold`
Run the trained agent on a protein.

**Request body**
```json
{
  "pdb_id": "1L2Y",
  "n_steps": 50,
  "deterministic": false
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `pdb_id` | `"1L2Y"` \| `"1YRF"` | one of pdb_id/sequence | Known protein |
| `sequence` | string | one of pdb_id/sequence | 5–50 AA, uppercase |
| `n_steps` | int 1–200 | No (default 50) | Agent steps to run |
| `deterministic` | bool | No (default false) | Greedy if true |

**Response**
```json
{
  "job_id": "a3f1c2d4-...",
  "protein": "1L2Y",
  "n_residues": 20,
  "steps_run": 50,
  "initial_energy": 284.312,
  "final_energy": 148.737,
  "energy_drop": 135.575,
  "final_rmsd": 3.145,
  "best_rmsd": 1.313,
  "converged": false,
  "trajectory": [
    { "step": 0, "energy": 284.312, "rmsd": 6.2, "has_clash": false, "reward": 0.0 },
    { "step": 1, "energy": 278.100, "rmsd": 5.8, "has_clash": false, "reward": 8.7 }
  ],
  "initial_pdb": "ATOM      1  CA  ASN A   1 ...\nEND",
  "final_pdb":   "ATOM      1  CA  ASN A   1 ...\nEND",
  "native_pdb":  "ATOM      1  CA  ASN A   1 ...\nEND"
}
```

**Error codes**

| HTTP | `code` | Meaning |
|------|--------|---------|
| 400 | `CUSTOM_SEQUENCE_UNSUPPORTED` | Custom seq not yet available |
| 400 | `UNKNOWN_PROTEIN` | pdb_id not in supported list |
| 422 | — | Validation error (Pydantic) |
| 503 | `MODEL_NOT_LOADED` | Checkpoint still loading |
| 500 | `FOLD_FAILED` | Internal error |

---

### `GET /results?limit=500`
Full training log from `logs/training_log.csv`.

**Response**
```json
{
  "total_episodes": 500,
  "best_rmsd": 1.313,
  "best_energy": 22.228,
  "avg_rmsd_last50": 3.145,
  "avg_energy_last50": 148.737,
  "episodes": [
    {
      "episode": 1, "protein": "1L2Y", "total_reward": 12.3,
      "final_energy": 280.1, "rmsd": 5.8, "steps": 50,
      "policy_loss": 0.12, "value_loss": 0.34, "entropy": 0.05
    }
  ]
}
```

---

### `GET /best-episode`
Step-by-step trace of the best recorded episode.

**Response**
```json
{
  "best_rmsd": 1.313,
  "best_energy": 22.228,
  "trajectory": [
    { "step": 0, "energy": 284.3, "reward": 0.0, "has_clash": false },
    { "step": 1, "energy": 276.1, "reward": 8.7, "has_clash": false }
  ]
}
```

---

### `GET /compare?pdb_id=1L2Y&n_episodes=10`
Live trained agent vs random baseline comparison.

> ⚠️ Computationally expensive. Keep `n_episodes` ≤ 10 on HF Spaces.

**Response**
```json
{
  "random_avg_rmsd": 7.637,
  "random_avg_energy": 267.092,
  "trained_avg_rmsd": 3.145,
  "trained_avg_energy": 148.737,
  "trained_best_rmsd": 1.313,
  "rmsd_improvement": 4.492,
  "energy_improvement": 118.355
}
```

---

## Frontend integration (fetch examples)

```javascript
// Check model is ready before showing UI
const health = await fetch("http://localhost:8000/health").then(r => r.json());
if (!health.model_loaded) { /* show loading spinner */ }

// Run fold
const result = await fetch("http://localhost:8000/fold", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ pdb_id: "1L2Y", n_steps: 50 })
}).then(r => r.json());

// Plot energy curve
const energyCurve = result.trajectory.map(t => ({ x: t.step, y: t.energy }));

// Render 3D structure (py3Dmol / NGL)
viewer.addModel(result.final_pdb, "pdb");

// Load training results
const results = await fetch("http://localhost:8000/results?limit=500").then(r => r.json());
```

---

## Running tests

```powershell
cd C:\Users\Kavinder\Desktop\ProteilFold_RL
.venv\Scripts\activate
pytest api/tests/test_api.py -v
```

Tests mock the model — no checkpoint required to run the test suite.

---

## HuggingFace Spaces deployment

Set the environment variable `HF_SPACE_URL` to your Space URL so CORS works:

```
HF_SPACE_URL=https://kavinder-proteinfold-rl.hf.space
```

The `Dockerfile` / `app.py` should launch:
```
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

HF Spaces only exposes port 7860 — use that instead of 8000.