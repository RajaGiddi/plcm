"""E5d method v2 — adapter-first warmup. Contract: docs/E5D_prereg.md.

The warmup's whole claim is "during warmup the adapter is the ONLY loss-reducing
path." If anything else can still train, the mechanism is partially bypassed and
every downstream number is blurred — and it would blur them *quietly*, because a
trainable readout works well enough to look plausible.

So these tests assert the freeze contract directly, param by param:

  1. v2-ON warmup   -> ONLY task_adapters.<k> has requires_grad
  2. v2-control     -> ONLY the readout has it; the adapter is frozen
  3. both arms freeze the encoder for the SAME epochs (the attribution control
     in prereg §3a is worthless if the arms differ in encoder training)
  4. task 0 NEVER warms up (prereg §7 — an adapter fitted to a random encoder)
  5. grad state is restored exactly afterwards, so warmup cannot leak into the
     rest of training
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM
from src.training.trainer import ContinualTrainer


def _cfg(warmup_epochs=1, warmup_mode="adapter"):
    return {
        "seed": 0,
        "model": {"input_size": 9, "hidden_size": 16, "num_classes": 6,
                  "num_layers": 1, "dropout": 0.0, "use_task_heads": False},
        "memory": {"capacity": 16, "key_dim": 8, "top_k": 2, "num_heads": 2,
                   "write_threshold": 0.2, "consolidation_interval": 50,
                   "consolidation_clusters": 8},
        "composition": {"mode": "ggc"},
        "adapters": {"enabled": True, "dim": 9, "mode": "per_step"},
        "training": {"num_tasks": 5, "epochs_per_task": 3, "batch_size": 8,
                     "learning_rate": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0,
                     "warmup_epochs": warmup_epochs, "warmup_mode": warmup_mode},
        "logging": {"verbose": False},
    }


def _trainer(warmup_mode="adapter", warmup_epochs=1):
    torch.manual_seed(0)
    cfg = _cfg(warmup_epochs, warmup_mode)
    m = PLCM.from_config(cfg)
    return ContinualTrainer(model=m, model_type="mafc",
                            device=torch.device("cpu"), config=cfg), m


def _trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


class TestFreezeContract:
    def test_v2_on_trains_only_the_new_adapter(self):
        t, m = _trainer("adapter")
        for k in (0, 1):
            m.set_task(k)
        t._enter_warmup(1)
        assert _trainable(m) == {"task_adapters.1.weight"}, \
            "v2-ON warmup must leave EXACTLY the new adapter trainable"

    def test_v2_control_trains_only_readout_and_freezes_adapter(self):
        t, m = _trainer("control")
        for k in (0, 1):
            m.set_task(k)
        t._enter_warmup(1)
        tr = _trainable(m)
        assert tr, "control arm must train something"
        assert all(n.startswith(t.READOUT_PREFIXES) for n in tr), \
            f"control warmup must train only the readout, got {sorted(tr)}"
        assert not any(n.startswith("task_adapters.") for n in tr), \
            "control arm must FREEZE the adapter — that is the whole control"

    def test_shared_classifier_is_frozen_in_v2_on(self):
        """Catch 4: a trainable readout is a bypass around the mechanism."""
        t, m = _trainer("adapter")
        for k in (0, 1):
            m.set_task(k)
        t._enter_warmup(1)
        assert not m.classifier.weight.requires_grad
        assert not m.classifier.bias.requires_grad

    def test_encoder_frozen_in_BOTH_arms(self):
        """Prereg §3a: the arms must be matched on encoder training."""
        for mode in ("adapter", "control"):
            t, m = _trainer(mode)
            for k in (0, 1):
                m.set_task(k)
            t._enter_warmup(1)
            enc = [n for n in _trainable(m) if n.startswith("lstm.")]
            assert not enc, f"{mode}: encoder must be frozen during warmup, got {enc}"

    def test_warmup_never_resurrects_an_already_frozen_param(self):
        """Warmup may only RESTRICT the trainable set, never widen it."""
        t, m = _trainer("adapter")
        for k in (0, 1):
            m.set_task(k)
        m.task_adapters["1"].weight.requires_grad_(False)   # arm froze it already
        t._enter_warmup(1)
        assert not m.task_adapters["1"].weight.requires_grad


class TestRestore:
    def test_exit_restores_exact_state(self):
        t, m = _trainer("adapter")
        for k in (0, 1):
            m.set_task(k)
        before = {n: p.requires_grad for n, p in m.named_parameters()}
        saved = t._enter_warmup(1)
        assert _trainable(m) != {n for n, v in before.items() if v}
        t._exit_warmup(saved)
        after = {n: p.requires_grad for n, p in m.named_parameters()}
        assert before == after, "warmup must not leak into post-warmup training"


class TestSchedule:
    @pytest.mark.parametrize("mode", ["adapter", "control"])
    def test_task_zero_never_warms_up(self, mode):
        """Prereg §7: an adapter fitted against a random encoder bakes in noise.

        Drives the real task loop and asserts via the recorded per-epoch flag.
        """
        t, m = _trainer(mode)
        X = torch.randn(24, 8, 9)
        y = torch.randint(0, 6, (24,))
        from torch.utils.data import DataLoader, TensorDataset
        ld = DataLoader(TensorDataset(X, y), batch_size=8)
        res = t.train_task(0, ld, [ld]*5)
        assert all(e["warmup"] is False for e in res["epochs"]), \
            "task 0 must never enter warmup"

    def test_later_task_warms_up_for_exactly_W_epochs(self):
        t, m = _trainer("adapter", warmup_epochs=2)
        X = torch.randn(24, 8, 9)
        y = torch.randint(0, 6, (24,))
        from torch.utils.data import DataLoader, TensorDataset
        ld = DataLoader(TensorDataset(X, y), batch_size=8)
        t.train_task(0, ld, [ld]*5)
        res = t.train_task(1, ld, [ld]*5)
        flags = [e["warmup"] for e in res["epochs"]]
        assert flags == [True, True, False], f"expected 2 warmup epochs, got {flags}"

    def test_zero_warmup_is_v1_behaviour(self):
        t, m = _trainer("adapter", warmup_epochs=0)
        X = torch.randn(24, 8, 9)
        y = torch.randint(0, 6, (24,))
        from torch.utils.data import DataLoader, TensorDataset
        ld = DataLoader(TensorDataset(X, y), batch_size=8)
        t.train_task(0, ld, [ld]*5)
        res = t.train_task(1, ld, [ld]*5)
        assert all(e["warmup"] is False for e in res["epochs"])
        assert m.lstm.weight_ih_l0.requires_grad, "v1 must leave the encoder training"


class TestAdapterActuallyMoves:
    """The point of warmup: the adapter must move, and only in the v2-ON arm.

    These drive the REAL train_task path rather than calling _train_epoch
    directly, because train_task is what registers a lazily-created adapter
    into the optimizer. A test that skips that step reports "adapter frozen"
    for both arms and would have hidden a genuine failure behind a fixture bug.

    epochs_per_task=1 with warmup_epochs=1 means task 1 consists of exactly one
    warmup epoch, so any change is attributable to warmup alone.
    """

    @staticmethod
    def _run(mode):
        t, m = _trainer(mode, warmup_epochs=1)
        t.epochs_per_task = 1
        t.config["training"]["epochs_per_task"] = 1
        X = torch.randn(64, 8, 9)
        y = torch.randint(0, 6, (64,))
        from torch.utils.data import DataLoader, TensorDataset
        ld = DataLoader(TensorDataset(X, y), batch_size=8)
        t.train_task(0, ld, [ld] * 5)
        before = {k: v.weight.detach().clone() for k, v in m.task_adapters.items()}
        enc_before = m.lstm.weight_ih_l0.detach().clone()
        res = t.train_task(1, ld, [ld] * 5)
        assert [e["warmup"] for e in res["epochs"]] == [True]
        return m, before, enc_before

    # Adapters are created lazily inside train_task and are EXACTLY
    # identity-initialized, so "did it move" is "does it differ from I" — a
    # stronger assertion than a captured snapshot, and it needs no snapshot.

    def test_adapter_moves_under_v2_on_warmup(self):
        m, _, _ = self._run("adapter")
        A = m.task_adapters["1"].weight
        assert not torch.equal(A, torch.eye(A.shape[0])), \
            "the new adapter must receive gradient during v2-ON warmup"

    def test_adapter_frozen_under_control_warmup(self):
        m, _, _ = self._run("control")
        A = m.task_adapters["1"].weight
        assert torch.equal(A, torch.eye(A.shape[0])), \
            "control arm's adapter must NOT move during warmup"

    @pytest.mark.parametrize("mode", ["adapter", "control"])
    def test_encoder_does_not_move_in_either_arm(self, mode):
        """The attribution control: both arms must rest the encoder equally."""
        m, _, enc_before = self._run(mode)
        assert torch.equal(enc_before, m.lstm.weight_ih_l0), \
            f"{mode}: encoder must not move during a warmup-only task"

    def test_old_adapter_never_moves(self):
        """Task 0's adapter is frozen after its task; warmup must not revive it."""
        m, before, _ = self._run("adapter")
        assert torch.equal(before["0"], m.task_adapters["0"].weight)
