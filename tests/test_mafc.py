"""
Structural tests for MAFC (pre-registration v2).

Scope per prereg §5: shape and gradient-flow only. These tests NEVER predict
outcomes — no mock simulation of learning dynamics (4/4 historical failure
rate). They exist to verify that the two terms have the gradient structure the
contract claims, because that structure is the whole reason the loss is split.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM
from src.training.trainer import ContinualTrainer


def _model(use_task_heads=False):
    torch.manual_seed(0)
    return PLCM(
        input_size=28, hidden_size=32, num_classes=10, memory_capacity=32,
        key_dim=16, top_k=4, num_heads=4, write_threshold=0.2,
        use_input_adapters=True, adapter_dim=784, use_task_heads=use_task_heads,
    )


def _cfg(**mafc):
    m = dict(lambda_=0.0)
    return {
        "seed": 0,
        "model": {"input_size": 28, "hidden_size": 32, "num_classes": 10,
                  "num_layers": 1, "dropout": 0.0, "use_task_heads": False},
        "memory": {"capacity": 32, "key_dim": 16, "top_k": 4, "num_heads": 4,
                   "write_threshold": 0.2, "consolidation_interval": 50,
                   "consolidation_clusters": 16},
        "composition": {"mode": "ggc"},
        "adapters": {"enabled": True, "dim": 784},
        "training": {"num_tasks": 2, "epochs_per_task": 1, "batch_size": 16,
                     "learning_rate": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0},
        "mafc": {"lambda": 1.0, "temperature": 2.0, "encoder_term": True,
                 "readout_term": True, "adapter_probes": True, "bank_batch": 16,
                 **mafc},
        "logging": {"verbose": False},
    }


class TestGradientStructure:
    """The contract's §1.1 claim: the two terms have DIFFERENT reachable params."""

    def test_readout_term_cannot_reach_encoder(self):
        """L_read on stored bank vectors must give theta exactly zero gradient.

        This is the structural fact that forced the two-term split: a stored
        state is a constant w.r.t. theta, so no bank-sourced loss can ever
        constrain encoder drift.
        """
        m = _model()
        c = torch.randn(8, 32)
        gate = torch.rand(8, 32)
        target = torch.randn(8, 10)
        m.zero_grad()
        loss = nn.functional.kl_div(
            nn.functional.log_softmax(m.readout_from_state(c, gate), -1),
            nn.functional.softmax(target, -1), reduction="batchmean",
        )
        loss.backward()
        assert all(p.grad is None or p.grad.abs().sum() == 0
                   for p in m.lstm.parameters()), "L_read must NOT reach theta"
        assert m.classifier.weight.grad.abs().sum() > 0, "L_read must reach phi"

    def test_encoder_term_reaches_encoder_and_readout(self):
        m = _model()
        m.set_task(0)
        m.zero_grad()
        m.functional_forward(torch.randn(8, 28, 28), adapter_key="0").sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.lstm.parameters()), "L_enc must reach theta"
        assert m.classifier.weight.grad.abs().sum() > 0, "L_enc must reach phi"

    def test_functional_forward_bypasses_memory(self):
        """The anchor is on the parametric function only, so a changing bank
        must not move the target."""
        m = _model()
        m.set_task(0)
        m.eval()
        x = torch.randn(4, 28, 28)
        with torch.no_grad():
            before = m.functional_forward(x, adapter_key="0")
            m.memory_bank.write(torch.randn(8, 32), torch.rand(8), task_id=0,
                                gate_vectors=torch.rand(8, 32),
                                logit_vectors=torch.randn(8, 10))
            after = m.functional_forward(x, adapter_key="0")
        assert torch.allclose(before, after), "functional_forward must ignore the bank"

    def test_shared_readout_when_heads_disabled(self):
        m = _model(use_task_heads=False)
        m.set_task(0)
        m.set_task(1)
        assert len(m.task_classifiers) == 0, "MAFC requires a shared readout"
        assert len(m.task_adapters) == 2, "adapters are still per-task"
        assert m.classifier.weight.requires_grad, "shared phi must be trainable"


class TestArmWiring:
    """Each prereg §4 arm must activate exactly the intended terms."""

    @pytest.mark.parametrize("arm,enc,read,adapt", [
        ("full", True, True, True),
        ("enc", True, False, True),
        ("lwf", True, True, False),
    ])
    def test_arm_flags(self, arm, enc, read, adapt):
        cfg = _cfg(encoder_term=enc, readout_term=read, adapter_probes=adapt)
        t = ContinualTrainer(PLCM.from_config(cfg), "mafc",
                             torch.device("cpu"), cfg)
        assert t.mafc_encoder_term is enc
        assert t.mafc_readout_term is read
        assert t.mafc_adapter_probes is adapt

    def test_lambda0_produces_no_mafc_gradient(self):
        cfg = _cfg()
        cfg["mafc"]["lambda"] = 0.0
        t = ContinualTrainer(PLCM.from_config(cfg), "mafc", torch.device("cpu"), cfg)
        assert t.mafc_lambda == 0.0

    def test_snapshot_is_frozen_and_stationary(self):
        cfg = _cfg()
        m = PLCM.from_config(cfg)
        t = ContinualTrainer(m, "mafc", torch.device("cpu"), cfg)
        m.set_task(0)
        t._take_mafc_snapshot(0)
        snap = t.mafc_snapshots[0]
        assert all(not p.requires_grad for p in snap.parameters())
        x = torch.randn(4, 28, 28)
        with torch.no_grad():
            before = snap.functional_forward(x, adapter_key="0")
        # Mutate the live model; the snapshot target must not move.
        with torch.no_grad():
            for p in m.lstm.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        with torch.no_grad():
            after = snap.functional_forward(x, adapter_key="0")
        assert torch.allclose(before, after), "snapshot targets must be stationary"


class TestLowRankAdapters:
    """Low-rank residual adapters A(x) = x + U V x."""

    def test_exact_identity_at_init(self):
        """Must be EXACTLY identity, not approximately.

        The full-rank adapters relied on identity init. If a low-rank variant
        started away from identity, a poor result could be blamed on
        initialization rather than capacity — which would make the
        pre-registered rank branches uninterpretable.
        """
        from src.models.plcm import LowRankAdapter
        a = LowRankAdapter(784, 32)
        x = torch.randn(16, 784)
        assert torch.equal(a(x), x), "low-rank adapter must be exact identity at init"

    def test_storage_is_2dr(self):
        from src.models.plcm import LowRankAdapter
        d, r = 784, 32
        a = LowRankAdapter(d, r)
        assert sum(p.numel() for p in a.parameters()) == 2 * d * r

    @pytest.mark.parametrize("rank,expected_type", [(0, "Linear"), (32, "LowRankAdapter")])
    def test_trainer_logs_dist_from_identity_for_both_types(self, rank, expected_type):
        """Regression: LowRankAdapter has no `.weight`, which crashed the
        per-epoch adapter diagnostic with AttributeError."""
        cfg = _cfg()
        cfg["adapters"] = {"enabled": True, "dim": 784, "rank": rank}
        cfg["mafc"]["lambda"] = 0.0
        model = PLCM.from_config(cfg)
        trainer = ContinualTrainer(model, "mafc", torch.device("cpu"), cfg)

        gen = torch.Generator().manual_seed(0)

        class Bench:
            num_tasks = 2

            def __init__(self):
                self.l = [DataLoader(TensorDataset(
                    torch.randn(32, 28, 28, generator=gen),
                    torch.randint(0, 10, (32,), generator=gen),
                ), batch_size=16) for _ in range(2)]

            def get_task_loaders(self, t, nw=0):
                return self.l[t], self.l[t]

            def get_all_test_loaders(self, nw=0):
                return self.l

        res = trainer.run(Bench())
        assert type(model.task_adapters["0"]).__name__ == expected_type
        vals = [e.get("adapter_dist_from_identity")
                for t in res["task_history"] for e in t["epochs"]]
        assert vals and all(v is not None for v in vals), \
            "adapter_dist_from_identity must be logged for both adapter types"


class TestEndToEnd:
    """The mafc model_type must dispatch and train without error."""

    def test_mafc_runs_through_trainer(self):
        cfg = _cfg()
        gen = torch.Generator().manual_seed(0)

        class Bench:
            num_tasks = 2

            def __init__(self):
                self.l = [
                    DataLoader(TensorDataset(
                        torch.randn(32, 28, 28, generator=gen),
                        torch.randint(0, 10, (32,), generator=gen),
                    ), batch_size=16) for _ in range(2)
                ]

            def get_task_loaders(self, t, nw=0):
                return self.l[t], self.l[t]

            def get_all_test_loaders(self, nw=0):
                return self.l

        m = PLCM.from_config(cfg)
        t = ContinualTrainer(m, "mafc", torch.device("cpu"), cfg)
        res = t.run(Bench())
        assert "average_accuracy" in res
        # A snapshot must exist after task 0 so task 1 has an anchor.
        assert 0 in t.mafc_snapshots
