"""
Tests for the Memory Bank.

Verifies storage, retrieval, eviction, and capacity management.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.memory_bank import MemoryBank


@pytest.fixture
def bank():
    torch.manual_seed(42)
    return MemoryBank(capacity=10, key_dim=32, value_dim=64)


class TestMemoryBankBasics:

    def test_empty_bank(self, bank):
        assert bank.is_empty
        assert bank.size == 0

    def test_write_single(self, bank):
        cell = torch.randn(1, 64)
        importance = torch.tensor([0.8])
        bank.write(cell, importance, task_id=0)
        assert bank.size == 1
        assert not bank.is_empty

    def test_write_batch(self, bank):
        cells = torch.randn(5, 64)
        importance = torch.rand(5)
        bank.write(cells, importance, task_id=0)
        assert bank.size == 5

    def test_capacity_limit(self, bank):
        # Write more than capacity
        cells = torch.randn(15, 64)
        importance = torch.arange(15, dtype=torch.float)
        bank.write(cells, importance, task_id=0)
        assert bank.size == 10  # capped at capacity

    def test_eviction_keeps_important(self, bank):
        # Fill bank with low importance
        low = torch.randn(10, 64)
        low_imp = torch.full((10,), 0.1)
        bank.write(low, low_imp, task_id=0)

        # Write high-importance entry (should evict lowest)
        high = torch.randn(1, 64)
        high_imp = torch.tensor([0.9])
        bank.write(high, high_imp, task_id=1)

        # High-importance entry should be in the bank
        assert bank.size == 10
        assert bank.importance.max() >= 0.9


class TestMemoryBankRetrieval:

    def test_read_empty(self, bank):
        query = torch.randn(4, 64)
        values, gates, logits, task_ids, weights, indices = bank.read(query, top_k=3)
        assert values.shape == (4, 3, 64)
        assert weights.shape == (4, 3)
        assert (indices == -1).all()

    def test_read_returns_correct_shape(self, bank):
        cells = torch.randn(8, 64)
        importance = torch.rand(8)
        bank.write(cells, importance, task_id=0)

        query = torch.randn(4, 64)
        values, gates, logits, task_ids, weights, indices = bank.read(query, top_k=3)
        assert values.shape == (4, 3, 64)
        assert weights.shape == (4, 3)

    def test_read_weights_sum_to_one(self, bank):
        cells = torch.randn(8, 64)
        importance = torch.rand(8)
        bank.write(cells, importance, task_id=0)

        query = torch.randn(4, 64)
        _, _, _, _, weights, _ = bank.read(query, top_k=3)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_read_similar_query_retrieves_similar(self, bank):
        # Store a known vector
        target = torch.randn(1, 64)
        noise = [torch.randn(1, 64) for _ in range(5)]
        cells = torch.cat([target] + noise, dim=0)
        importance = torch.ones(6)
        bank.write(cells, importance, task_id=0)

        # Query with something similar to target
        query = target + 0.01 * torch.randn(1, 64)
        values, gates, logits, task_ids, weights, indices = bank.read(query, top_k=1)

        # The retrieved value should be close to target
        retrieved = values[0, 0]
        cosine_sim = torch.nn.functional.cosine_similarity(
            retrieved.unsqueeze(0), target
        )
        assert cosine_sim > 0.8, "Should retrieve similar memory"

    def test_top_k_larger_than_bank(self, bank):
        cells = torch.randn(3, 64)
        importance = torch.rand(3)
        bank.write(cells, importance, task_id=0)

        query = torch.randn(2, 64)
        values, gates, logits, task_ids, weights, indices = bank.read(query, top_k=8)
        # Should pad to top_k
        assert values.shape == (2, 8, 64)


class TestImportanceDecay:
    """Regression tests for the importance explosion bug (v1)."""

    def test_importance_never_exceeds_one(self, bank):
        cells = torch.randn(8, 64)
        importance = torch.ones(8) * 0.9
        bank.write(cells, importance, task_id=0)
        for step in range(500):
            bank.global_step += 1
            query = torch.randn(2, 64)
            bank.read(query, top_k=5)
            bank.decay_importance()
        n = bank.size
        assert (bank.importance[:n] <= 1.0).all()
        assert (bank.importance[:n] >= 0.0).all()
        assert torch.isfinite(bank.importance[:n]).all()

    def test_importance_decays_over_time(self, bank):
        cells = torch.randn(5, 64)
        importance = torch.ones(5) * 0.8
        bank.write(cells, importance, task_id=0)
        initial_imp = bank.importance[:5].clone()
        for _ in range(50):
            bank.global_step += 1
            bank.decay_importance()
        assert (bank.importance[:5] < initial_imp).all()

    def test_used_memories_decay_slower(self, bank):
        cells = torch.randn(2, 64)
        importance = torch.ones(2) * 0.8
        bank.write(cells, importance, task_id=0)
        for _ in range(20):
            bank.global_step += 1
            bank.usage_count[0] += 10
            bank.decay_importance()
        assert bank.importance[0] > bank.importance[1]

    def test_importance_clamped_on_write(self, bank):
        cells = torch.randn(2, 64)
        importance = torch.tensor([1.5, -0.3])
        bank.write(cells, importance, task_id=0)
        assert bank.importance[0] <= 1.0
        assert bank.importance[1] >= 0.0


class TestTaskAwareMemory:
    """Tests for task-aware memory management."""

    def test_rebalance_distributes_evenly(self, bank):
        """After rebalance, each task should have at most slots_per_task entries."""
        cells_t0 = torch.randn(8, 64)
        bank.write(cells_t0, torch.ones(8) * 0.8, task_id=0)

        stats = bank.rebalance(slots_per_task=3)
        assert stats[0]["after"] == 3
        assert bank.consolidated[:bank.size].sum().item() == 3

    def test_consolidated_immune_to_eviction(self, bank):
        """Consolidated entries must not be evicted by new writes."""
        cells = torch.randn(10, 64)
        bank.write(cells, torch.ones(10) * 0.8, task_id=0)
        bank.rebalance(slots_per_task=5)

        new_cells = torch.randn(10, 64)
        bank.write(new_cells, torch.ones(10) * 0.9, task_id=1)

        t0_count = (bank.task_ids[:bank.size] == 0).sum().item()
        assert t0_count >= 5, "Consolidated T0 entries must survive T1 writes"

    def test_consolidated_immune_to_decay(self, bank):
        """Consolidated entries must not lose importance via decay."""
        cells = torch.randn(5, 64)
        bank.write(cells, torch.ones(5) * 0.8, task_id=0)
        bank.rebalance(slots_per_task=5)

        initial_imp = bank.importance[:5].clone()
        for _ in range(100):
            bank.global_step += 1
            bank.decay_importance()

        assert torch.allclose(bank.importance[:5], initial_imp), (
            "Consolidated entries must not decay"
        )

    def test_multi_task_diversity(self, bank):
        """After 3 tasks with rebalancing, all tasks should be represented."""
        bank_big = MemoryBank(capacity=30, key_dim=32, value_dim=64)

        bank_big.write(torch.randn(15, 64), torch.ones(15)*0.7, task_id=0)
        bank_big.rebalance(slots_per_task=10)

        bank_big.write(torch.randn(20, 64), torch.ones(20)*0.7, task_id=1)
        bank_big.rebalance(slots_per_task=10)

        bank_big.write(torch.randn(20, 64), torch.ones(20)*0.7, task_id=2)

        n = bank_big.size
        tasks_present = bank_big.task_ids[:n].unique()
        tasks_present = tasks_present[tasks_present >= 0]
        assert len(tasks_present) >= 3, (
            f"All 3 tasks should be in the bank, found {tasks_present.tolist()}"
        )


class TestMemoryBankStats:

    def test_stats_empty(self, bank):
        stats = bank.get_stats()
        assert stats["size"] == 0

    def test_stats_populated(self, bank):
        cells = torch.randn(5, 64)
        importance = torch.rand(5)
        bank.write(cells, importance, task_id=0)

        stats = bank.get_stats()
        assert stats["size"] == 5
        assert stats["utilization"] == 0.5
        assert stats["num_tasks_stored"] == 1

    def test_clear(self, bank):
        cells = torch.randn(5, 64)
        importance = torch.rand(5)
        bank.write(cells, importance, task_id=0)
        bank.clear()
        assert bank.is_empty
        assert bank.size == 0
