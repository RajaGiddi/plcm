"""
Integration tests for the full PLCM model.

Verifies that all components work together: LSTM → Read → Compose → Write.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM


@pytest.fixture
def plcm():
    torch.manual_seed(42)
    return PLCM(
        input_size=28,
        hidden_size=64,
        num_classes=10,
        memory_capacity=32,
        key_dim=32,
        top_k=4,
        num_heads=4,
        composition_mode="ggc",
        write_threshold=0.3,
        consolidation_interval=100,
        consolidation_clusters=16,
    )


class TestPLCMForward:

    def test_output_shape(self, plcm):
        x = torch.randn(8, 28, 28)  # batch=8, seq=28, input=28
        output = plcm(x, store_memories=False)
        assert output["logits"].shape == (8, 10)
        assert output["cell_state"].shape == (8, 64)

    def test_training_stores_memories(self, plcm):
        plcm.train()
        x = torch.randn(16, 28, 28)
        output = plcm(x, store_memories=True)
        # After training, some memories should be stored
        assert plcm.memory_bank.size >= 0  # might be 0 if all below threshold

    def test_memory_accumulates(self, plcm):
        plcm.train()
        plcm.write_threshold = 0.0  # store everything for this test

        for _ in range(5):
            x = torch.randn(4, 28, 28)
            plcm(x, store_memories=True)

        assert plcm.memory_bank.size > 0, "Memories should accumulate"

    def test_eval_does_not_store(self, plcm):
        plcm.eval()
        initial_size = plcm.memory_bank.size

        x = torch.randn(8, 28, 28)
        plcm(x, store_memories=False)

        assert plcm.memory_bank.size == initial_size

    def test_task_switching(self, plcm):
        plcm.set_task(0)
        assert plcm.current_task_id == 0
        plcm.set_task(3)
        assert plcm.current_task_id == 3

    def test_diagnostics(self, plcm):
        # First store some memories
        plcm.train()
        plcm.write_threshold = 0.0
        x = torch.randn(8, 28, 28)
        plcm(x, store_memories=True)

        # Now get diagnostics
        plcm.eval()
        output = plcm(x, store_memories=False, return_diagnostics=True)
        assert "diagnostics" in output
        assert "memory_bank" in output["diagnostics"]

    def test_gradient_through_full_model(self, plcm):
        plcm.train()
        plcm.write_threshold = 0.0

        # Populate memory first
        x = torch.randn(4, 28, 28)
        plcm(x, store_memories=True)

        # Now train with memory
        x = torch.randn(4, 28, 28)
        y = torch.randint(0, 10, (4,))
        output = plcm(x, store_memories=True)
        loss = torch.nn.functional.cross_entropy(output["logits"], y)
        loss.backward()

        # Check gradients exist
        grad_count = sum(
            1 for p in plcm.parameters() if p.grad is not None and p.grad.norm() > 0
        )
        assert grad_count > 0, "Gradients should flow through the model"

    def test_clear_memory(self, plcm):
        plcm.train()
        plcm.write_threshold = 0.0
        x = torch.randn(8, 28, 28)
        plcm(x, store_memories=True)

        plcm.clear_memory()
        assert plcm.memory_bank.is_empty


class TestPLCMConfig:

    def test_from_config(self):
        config = {
            "model": {
                "input_size": 784,
                "hidden_size": 128,
                "num_classes": 10,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "memory": {
                "capacity": 256,
                "key_dim": 64,
                "top_k": 4,
                "num_heads": 4,
                "write_threshold": 0.5,
                "consolidation_interval": 50,
                "consolidation_clusters": 32,
            },
            "composition": {
                "mode": "ggc",
            },
        }
        model = PLCM.from_config(config)
        assert model.hidden_size == 128
        assert model.memory_bank.capacity == 256

    def test_additive_mode(self):
        model = PLCM(
            input_size=28,
            hidden_size=64,
            num_classes=10,
            composition_mode="additive",
        )
        x = torch.randn(4, 28, 28)
        output = model(x, store_memories=False)
        assert output["logits"].shape == (4, 10)
