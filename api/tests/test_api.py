"""
api/tests/test_api.py
ProteinFold-RL — API unit tests.

Uses FastAPI's TestClient (built on httpx/requests — no running server
needed). Tests are intentionally isolated: no real model weights
required — the model_manager is monkey-patched with a lightweight mock.

Run
---
    cd C:\\Users\\Kavinder\\Desktop\\ProteilFold_RL
    .venv\\Scripts\\activate
    pytest api/tests/test_api.py -v

Author : ProteinFold-RL team
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ── Ensure project root is on path ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi.testclient import TestClient


# ── Fixtures & mocks ──────────────────────────────────────────

N_RESIDUES   = 20   # Trp-cage
ACTION_DIM   = N_RESIDUES * 2 * 12

# Fake policy: always returns action=0, log_prob=0, value=0, entropy=0
def _fake_get_action(graph, deterministic=False):
    return (
        0,                              # action
        torch.tensor(0.0),             # log_prob
        torch.tensor(0.0),             # value
        torch.tensor(1.0),             # entropy
    )


def _make_mock_manager(tmp_log_dir: str):
    """
    Build a ModelManager mock that:
      - reports is_loaded = True
      - has a .policy with a fake get_action method
      - has a .get_env() that returns a real FoldEnv for 1L2Y
    """
    from env.fold_env import FoldEnv

    mock_policy = MagicMock()
    mock_policy.get_action = _fake_get_action

    mock_mm = MagicMock()
    mock_mm.is_loaded = True
    mock_mm.checkpoint_path = "checkpoints/policy_final.pt"
    mock_mm.policy = mock_policy
    mock_mm.get_env.side_effect = lambda pdb_id: FoldEnv(pdb_id=pdb_id)
    return mock_mm


@pytest.fixture()
def tmp_logs(tmp_path):
    """
    Create temporary training_log.csv and best_trajectory.csv
    so results endpoints don't need real logs.
    """
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # training_log.csv — 5 fake episodes
    tlog = logs_dir / "training_log.csv"
    with open(tlog, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "protein", "stage", "total_reward",
            "final_energy", "rmsd", "steps", "clashes",
            "policy_loss", "value_loss", "entropy",
            "gate_rolling_rmsd", "advancement_reason",
        ])
        for i in range(1, 6):
            w.writerow([
                i, "1L2Y", 1, round(10.0 - i, 2),
                round(300.0 - i * 10, 2), round(5.0 - i * 0.5, 3),
                50, 0, 0.1, 0.2, 0.05, 99.0, "",
            ])

    # best_trajectory.csv — 5 steps
    btraj = logs_dir / "best_trajectory.csv"
    with open(btraj, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "energy", "reward", "has_clash"])
        for i in range(5):
            w.writerow([i, round(300.0 - i * 20, 2), round(1.0 + i, 2), 0])

    return str(tmp_path)


@pytest.fixture()
def client(tmp_logs):
    """
    TestClient with model_manager mocked and log paths patched
    to the temporary directory.
    """
    mock_mm = _make_mock_manager(tmp_logs)

    with patch("api.model_manager.get_model_manager", return_value=mock_mm), \
         patch("api.routes.fold.get_model_manager",    return_value=mock_mm), \
         patch("api.routes.results.get_model_manager", return_value=mock_mm), \
         patch("api.routes.health.get_model_manager",  return_value=mock_mm), \
         patch(
             "api.routes.results.TRAINING_LOG",
             os.path.join(tmp_logs, "logs", "training_log.csv"),
         ), \
         patch(
             "api.routes.results.BEST_TRAJ_LOG",
             os.path.join(tmp_logs, "logs", "best_trajectory.csv"),
         ):
        from api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── GET /health ────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "1L2Y" in body["supported_proteins"]
        assert "1YRF" in body["supported_proteins"]

    def test_health_has_version(self, client):
        r = client.get("/health")
        assert r.json()["version"] == "2.0.0"


# ── POST /fold ─────────────────────────────────────────────────

class TestFold:
    def test_fold_1l2y_default(self, client):
        r = client.post("/fold", json={"pdb_id": "1L2Y", "n_steps": 5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["protein"] == "1L2Y"
        assert body["n_residues"] == N_RESIDUES
        assert body["steps_run"] <= 5
        assert len(body["trajectory"]) >= 1
        assert "initial_pdb" in body
        assert "final_pdb" in body
        assert "native_pdb" in body
        assert isinstance(body["job_id"], str)

    def test_fold_1yrf(self, client):
        r = client.post("/fold", json={"pdb_id": "1YRF", "n_steps": 3})
        assert r.status_code == 200, r.text
        assert r.json()["protein"] == "1YRF"

    def test_fold_deterministic(self, client):
        r = client.post(
            "/fold",
            json={"pdb_id": "1L2Y", "n_steps": 3, "deterministic": True},
        )
        assert r.status_code == 200

    def test_fold_energy_drop_field(self, client):
        r = client.post("/fold", json={"pdb_id": "1L2Y", "n_steps": 5})
        body = r.json()
        expected_drop = round(body["initial_energy"] - body["final_energy"], 4)
        assert abs(body["energy_drop"] - expected_drop) < 0.01

    def test_fold_trajectory_steps_in_order(self, client):
        r = client.post("/fold", json={"pdb_id": "1L2Y", "n_steps": 10})
        traj = r.json()["trajectory"]
        steps = [t["step"] for t in traj]
        assert steps == list(range(len(steps))), "Steps must be sequential"

    def test_fold_pdb_string_format(self, client):
        r = client.post("/fold", json={"pdb_id": "1L2Y", "n_steps": 2})
        pdb = r.json()["initial_pdb"]
        assert "ATOM" in pdb
        assert "END" in pdb

    # ── Validation errors ──────────────────────────────────────

    def test_fold_missing_both_fields_is_422(self, client):
        r = client.post("/fold", json={"n_steps": 5})
        assert r.status_code == 422

    def test_fold_both_fields_is_422(self, client):
        r = client.post(
            "/fold",
            json={"pdb_id": "1L2Y", "sequence": "ACDEFG", "n_steps": 5},
        )
        assert r.status_code == 422

    def test_fold_invalid_pdb_id_is_422(self, client):
        r = client.post("/fold", json={"pdb_id": "XXXX"})
        assert r.status_code == 422

    def test_fold_n_steps_too_large_is_422(self, client):
        r = client.post("/fold", json={"pdb_id": "1L2Y", "n_steps": 9999})
        assert r.status_code == 422

    def test_fold_n_steps_zero_is_422(self, client):
        r = client.post("/fold", json={"pdb_id": "1L2Y", "n_steps": 0})
        assert r.status_code == 422

    def test_fold_invalid_sequence_chars(self, client):
        r = client.post("/fold", json={"sequence": "XXXXXBBBBB"})
        assert r.status_code == 422

    def test_fold_sequence_too_short(self, client):
        r = client.post("/fold", json={"sequence": "ACG"})
        assert r.status_code == 422

    def test_fold_custom_sequence_returns_200(self, client):
        # Custom sequences are now supported via extended chain
        r = client.post("/fold", json={"sequence": "ACDEFGHIKLMNPQRSTVWY", "n_steps": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["protein"] == "custom"
        assert body["n_residues"] == 20
        assert body["native_pdb"] == ""


# ── GET /results ───────────────────────────────────────────────

class TestResults:
    def test_results_ok(self, client):
        r = client.get("/results")
        assert r.status_code == 200
        body = r.json()
        assert body["total_episodes"] == 5
        assert len(body["episodes"]) == 5
        assert body["best_rmsd"] > 0
        assert body["best_energy"] > 0

    def test_results_limit(self, client):
        r = client.get("/results?limit=3")
        assert r.status_code == 200
        assert len(r.json()["episodes"]) == 3

    def test_results_limit_too_large_is_422(self, client):
        r = client.get("/results?limit=99999")
        assert r.status_code == 422

    def test_results_episode_fields(self, client):
        ep = client.get("/results").json()["episodes"][0]
        for field in ["episode", "protein", "total_reward",
                      "final_energy", "rmsd", "steps",
                      "policy_loss", "value_loss", "entropy"]:
            assert field in ep, f"Missing field: {field}"

    def test_results_404_when_no_log(self, tmp_path):
        """Without the log file, /results must return 404."""
        mock_mm = _make_mock_manager(str(tmp_path))
        with patch("api.routes.fold.get_model_manager",    return_value=mock_mm), \
             patch("api.routes.results.get_model_manager", return_value=mock_mm), \
             patch("api.routes.health.get_model_manager",  return_value=mock_mm), \
             patch(
                 "api.routes.results.TRAINING_LOG",
                 str(tmp_path / "nonexistent.csv"),
             ), \
             patch(
                 "api.routes.results.BEST_TRAJ_LOG",
                 str(tmp_path / "nonexistent2.csv"),
             ):
            from api.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.get("/results")
                assert r.status_code == 404


# ── GET /best-episode ──────────────────────────────────────────

class TestBestEpisode:
    def test_best_episode_ok(self, client):
        r = client.get("/best-episode")
        assert r.status_code == 200
        body = r.json()
        assert len(body["trajectory"]) == 5
        assert body["best_energy"] > 0

    def test_best_episode_trajectory_fields(self, client):
        traj_step = client.get("/best-episode").json()["trajectory"][0]
        for field in ["step", "energy", "reward", "has_clash"]:
            assert field in traj_step


# ── GET / ──────────────────────────────────────────────────────

class TestRoot:
    def test_root_ok(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["project"] == "ProteinFold-RL"
        assert "/docs" in body["docs"]


# ── 404 handler ────────────────────────────────────────────────

class TestErrors:
    def test_unknown_route_404(self, client):
        r = client.get("/nonexistent-route-xyz")
        assert r.status_code == 404