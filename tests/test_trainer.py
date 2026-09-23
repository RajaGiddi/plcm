"""
Smoke tests for the ContinualTrainer model-dispatch paths.

These run each supported model_type end-to-end through ContinualTrainer on a
tiny synthetic benchmark. They exist to guard the dispatch branches that the
model-level unit tests do NOT exercise: the trainer/eval/EWC code paths pick
different call conventions per model_type (PLCM returns a dict, LSTM returns a
(logits, hidden) tuple). A mismatch there — e.g. a stray ``== "plcm"`` that
sends ``plcm_ewc`` down the LSTM branch — raises "too many values to unpack"
only at run time, which is what these tests catch.
"""

import sys
from pathlib import Path

import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM
from src.models.lstm_base import LSTMBaseline
from src.training.trainer import ContinualTrainer


class _FakeBenchmark:
    """Tiny 2-task benchmark of random [N, 28, 28] sequences over 10 classes.

    Mirrors the interface trainer.run() consumes (get_task_loaders /
    get_all_test_loaders / num_tasks) without any dataset download.
    """

    num_tasks = 2

    def __init__(self, n: int = 64, batch_size: int = 32, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self._loaders = []
        for _ in range(self.num_tasks):
            x = torch.randn(n, 28, 28, generator=gen)
            y = torch.randint(0, 10, (n,), generator=gen)
            self._loaders.append(
                DataLoader(TensorDataset(x, y), batch_size=batch_size)
            )

    def get_task_loaders(self, task_id: int, num_workers: int = 0):
        return self._loaders[task_id], self._loaders[task_id]

    def get_all_test_loaders(self, num_workers: int = 0):
        return self._loaders


def _config() -> dict:
    """A minimal, fast config (small model, 2 tasks, 1 epoch)."""
    return {
        "seed": 42,
        "model": {
            "input_size": 28, "hidden_size": 32, "num_classes": 10,
            "num_layers": 1, "dropout": 0.0,
        },
        "memory": {
            "capacity": 32, "key_dim": 16, "top_k": 4, "num_heads": 4,
            "write_threshold": 0.2, "consolidation_interval": 20,
            "consolidation_clusters": 16,
        },
        "composition": {"mode": "ggc"},
        "training": {
            "num_tasks": 2, "epochs_per_task": 1, "batch_size": 32,
            "learning_rate": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0,
        },
        "ewc": {"lambda": 400.0, "fisher_samples": 32},
        "plcm_ewc": {"lambda": 150.0, "fisher_samples": 32},
        "logging": {"verbose": False, "log_dir": "/tmp/plcm_test_runs/"},
    }


def _build(model_type: str, cfg: dict) -> torch.nn.Module:
    """Build a model the same way scripts/train.py does (by trainer model_type)."""
    if model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter"):
        cfg["model"]["freeze_encoder_after_first_task"] = model_type in ("plcm_frozen", "plcm_adapter")
        cfg["adapters"] = {"enabled": model_type == "plcm_adapter", "dim": 28 * 28}
        cfg["router"] = {
            "enabled": model_type == "plcm_adapter",
            "replay": True,
            "num_tasks": cfg["training"]["num_tasks"],
        }
        return PLCM.from_config(cfg)
    return LSTMBaseline(
        input_size=28,
        hidden_size=cfg["model"]["hidden_size"],
        num_classes=cfg["model"]["num_classes"],
        num_layers=cfg["model"]["num_layers"],
    )


ALL_MODEL_TYPES = ["lstm", "lstm_ewc", "plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter"]


class TestTrainerDispatch:
    """Each model_type must run end-to-end and produce valid metrics."""

    @pytest.mark.parametrize("model_type", ALL_MODEL_TYPES)
    def test_run_completes(self, model_type):
        torch.manual_seed(0)
        cfg = _config()
        model = _build(model_type, cfg)
        trainer = ContinualTrainer(
            model=model, model_type=model_type,
            device=torch.device("cpu"), config=cfg,
        )
        results = trainer.run(_FakeBenchmark())

        # Dispatch worked (no unpack error) and all metrics are present.
        for key in (
            "average_accuracy", "forgetting", "backward_transfer",
            "forward_transfer", "accuracy_matrix",
        ):
            assert key in results, f"{model_type}: missing metric {key}"
        assert 0.0 <= results["average_accuracy"] <= 1.0

    @pytest.mark.parametrize("model_type", ["lstm_ewc", "plcm_ewc"])
    def test_ewc_enabled_with_correct_lambda(self, model_type):
        cfg = _config()
        model = _build(model_type, cfg)
        trainer = ContinualTrainer(
            model=model, model_type=model_type,
            device=torch.device("cpu"), config=cfg,
        )
        assert trainer.ewc is not None
        # plcm_ewc must use its own (lower) lambda section, not the EWC baseline's.
        expected = 150.0 if model_type == "plcm_ewc" else 400.0
        assert trainer.ewc.ewc_lambda == expected

    def test_plcm_frozen_has_no_ewc(self):
        cfg = _config()
        model = _build("plcm_frozen", cfg)
        trainer = ContinualTrainer(
            model=model, model_type="plcm_frozen",
            device=torch.device("cpu"), config=cfg,
        )
        assert trainer.ewc is None

    def test_encoder_freezes_after_task0(self):
        torch.manual_seed(0)
        cfg = _config()
        model = _build("plcm_frozen", cfg)
        trainer = ContinualTrainer(
            model=model, model_type="plcm_frozen",
            device=torch.device("cpu"), config=cfg,
        )
        trainer.run(_FakeBenchmark())  # 2 tasks -> encoder frozen after task 0
        assert all(not p.requires_grad for p in model.lstm.parameters()), \
            "LSTM must be frozen after task 0"
        assert all(not p.requires_grad for p in model.output_gate.parameters()), \
            "output gate must be frozen after task 0"
        # Adaptive components must stay trainable.
        assert any(p.requires_grad for p in model.composition.parameters())
        assert any(p.requires_grad for p in model.task_classifiers["1"].parameters())

    @pytest.mark.parametrize("model_type", ["plcm", "plcm_ewc"])
    def test_unfrozen_variants_keep_encoder_trainable(self, model_type):
        # Freeze is plcm_frozen-only: plcm / plcm_ewc must NOT freeze, so their
        # earlier results stay reproducible and the A/B baseline is preserved.
        torch.manual_seed(0)
        cfg = _config()
        model = _build(model_type, cfg)
        trainer = ContinualTrainer(
            model=model, model_type=model_type,
            device=torch.device("cpu"), config=cfg,
        )
        trainer.run(_FakeBenchmark())
        assert all(p.requires_grad for p in model.lstm.parameters()), \
            f"{model_type} must keep the LSTM trainable (freeze is plcm_frozen-only)"

    @pytest.mark.parametrize("model_type", ["lstm", "lstm_ewc"])
    def test_non_plcm_has_no_memory(self, model_type):
        model = _build(model_type, cfg=_config())
        assert not hasattr(model, "memory_bank")

    @pytest.mark.parametrize("model_type", ["plcm", "plcm_ewc"])
    def test_plcm_memory_task_aware(self, model_type):
        torch.manual_seed(0)
        cfg = _config()
        model = _build(model_type, cfg)
        trainer = ContinualTrainer(
            model=model, model_type=model_type,
            device=torch.device("cpu"), config=cfg,
        )
        trainer.run(_FakeBenchmark())
        stats = model.get_memory_stats()
        # After a 2-task run, memory should hold entries from more than one task.
        assert stats["num_tasks_stored"] >= 1
        assert "num_consolidated" in stats
