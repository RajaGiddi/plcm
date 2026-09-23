"""
Offline tests for E12's era-checkpoint storage plan (docs/E12_prereg.md sec 3).

These run the real ContinualTrainer path on a tiny synthetic benchmark, no
download and no GPU, so the mechanism is verified before an L4 is booked. What
they cannot cover — the ViT readout, P3a/P3b, and the size of the fp16 rounding
on 86M pretrained parameters — is what scripts/e12_ckpt_smoke.py measures.

The load-bearing one is test_fp32_shadow_reloads_exactly: it is the control that
splits an fp16 reload delta into rounding versus anything state_dict() drops,
and only the second kind is a defect (catch 29).
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM
from src.training.trainer import ContinualTrainer


class _TinyBenchmark:
    """Two tasks of random [N, 8, 6] sequences over 3 classes."""

    num_tasks = 2

    def __init__(self, n: int = 96, batch_size: int = 32, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self._tr, self._te = [], []
        for t in range(self.num_tasks):
            x = torch.randn(n, 8, 6, generator=gen)
            y = torch.randint(0, 3, (n,), generator=gen)
            # Signal across timesteps, not only the last row — the cell state is
            # an accumulator, so a last-row-only task is unlearnable from it.
            x[:, :, 0] += y.float().unsqueeze(1)
            self._tr.append(DataLoader(TensorDataset(x, y), batch_size=batch_size))
            self._te.append(DataLoader(TensorDataset(x, y), batch_size=batch_size))

    def get_task_loaders(self, task_id, num_workers=0):
        return self._tr[task_id], self._te[task_id]

    def get_all_test_loaders(self, num_workers=0):
        return list(self._te)


def _config(tmp_path: Path, **logging) -> dict:
    """Full config sections: the checkpoint carries this and load_era rebuilds
    the model from it, so a partial one would fail for a reason the sweep never
    meets."""
    return {
        "seed": 42,
        "model": {"input_size": 6, "hidden_size": 16, "num_classes": 3,
                  "num_layers": 1, "dropout": 0.0, "use_task_heads": True},
        "memory": {"enabled": False, "capacity": 8, "key_dim": 8, "top_k": 2,
                   "num_heads": 1, "write_threshold": 0.5,
                   "consolidation_interval": 50, "consolidation_clusters": 4},
        "composition": {"mode": "ggc"},
        "plcm_features": {"coord_align": False},
        "training": {"num_tasks": 2, "epochs_per_task": 1, "learning_rate": 1e-3,
                     "grad_clip": 1.0, "batch_size": 32},
        "logging": {"log_dir": str(tmp_path / "runs"), "verbose": False,
                    "checkpoint_dir": str(tmp_path / "ckpt"),
                    "era_checkpoints": True, "checkpoint_fp16": True, **logging},
    }


def _run(tmp_path: Path, **logging) -> dict:
    cfg = _config(tmp_path, **logging)
    trainer = ContinualTrainer(PLCM.from_config(cfg), "plcm",
                               torch.device("cpu"), cfg)
    return trainer.run(_TinyBenchmark())


@pytest.fixture(scope="module")
def shadow_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("shadow"), checkpoint_fp32_shadow=True)


def test_one_checkpoint_per_task_not_per_epoch(shadow_run):
    eras = [t["era_checkpoint"] for t in shadow_run["task_history"]]
    assert len(eras) == 2
    assert [e["task"] for e in eras] == [0, 1]


def test_stored_floats_are_fp16(shadow_run):
    """'fp16' has to be a fact about the file, or the 35GB estimate is a label."""
    for era in (t["era_checkpoint"] for t in shadow_run["task_history"]):
        state = torch.load(era["path"], weights_only=True,
                           map_location="cpu")["model_state"]
        floats = [v.dtype for v in state.values()
                  if torch.is_tensor(v) and v.is_floating_point()]
        assert floats and set(floats) == {torch.float16}


def test_audit_records_load_and_both_deltas(shadow_run):
    """Boundary values are authoritative; the reload disagreement is measured."""
    for era in (t["era_checkpoint"] for t in shadow_run["task_history"]):
        assert era["loads"] is True, era.get("error")
        assert era["acc_refit_ceiling_boundary"] is not None
        assert era["delta"] == pytest.approx(era["acc_reload"] - era["acc_boundary"])
        assert era["delta_refit"] == pytest.approx(
            era["acc_refit_ceiling_reload"] - era["acc_refit_ceiling_boundary"])


def test_fp32_shadow_reloads_exactly(shadow_run):
    """The gate that separates rounding from a serialization defect."""
    for era in (t["era_checkpoint"] for t in shadow_run["task_history"]):
        sh = era["fp32_shadow"]
        assert "error" not in sh, sh.get("error")
        assert sh["delta"] == 0.0
        assert sh["delta_refit"] == 0.0


def test_fp16_file_is_actually_smaller(shadow_run):
    """The storage plan's whole justification, measured on the file."""
    for era in (t["era_checkpoint"] for t in shadow_run["task_history"]):
        assert era["bytes"] < era["fp32_shadow"]["bytes"]


def test_audit_fires_on_a_truncated_checkpoint(shadow_run, tmp_path):
    """Positive control: 'audited by loading' has to be able to fail.

    Same helper the smoke uses, so the control and the thing it certifies
    cannot drift apart.
    """
    from scripts.e12_ckpt_smoke import truncation_control

    era = shadow_run["task_history"][0]["era_checkpoint"]
    assert truncation_control(era["path"], tmp_path)["fired"] is True


def test_p3_gate_skip_is_recorded_not_silent(shadow_run):
    for era in (t["era_checkpoint"] for t in shadow_run["task_history"]):
        assert "skipped" in era["p3_live"] and "skipped" in era["p3_reload"]


def test_arm_provenance_reads_the_model_not_the_config(shadow_run):
    arm = shadow_run["arm"]
    assert arm["use_task_heads"] is True and arm["n_task_heads"] == 2
    assert arm["backbone"] == "lstm" and arm["use_memory"] is False


def test_era_and_per_epoch_checkpoints_cannot_collide(tmp_path):
    cfg = _config(tmp_path, save_checkpoints=True)
    with pytest.raises(AssertionError, match="collide"):
        ContinualTrainer(PLCM.from_config(cfg), "plcm", torch.device("cpu"), cfg)
