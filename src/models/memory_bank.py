"""
Memory Bank for the Thought Space.

A fixed-capacity key-value store holding cell state vectors from previous
tasks/timesteps. Keys are projected context embeddings; values are the
cell states themselves. Uses importance-weighted eviction when full.

The memory bank is NOT a parameter — it's a persistent buffer that lives
outside the optimization loop. This is what allows it to survive across
tasks without being overwritten by gradient updates.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MemoryBank(nn.Module):
    """
    Fixed-capacity memory bank for cell state vectors.

    Stores (key, value, importance, task_id) tuples. Keys are used for
    retrieval via attention; values are the actual cell states; importance
    scores determine eviction priority; task_ids enable analysis.

    The bank uses a write cursor that wraps around. When capacity is reached,
    new entries evict the lowest-importance existing entry.

    Args:
        capacity: Maximum number of stored vectors
        key_dim: Dimension of key vectors (projected from hidden)
        value_dim: Dimension of value vectors (= hidden_size)
    """

    def __init__(self, capacity: int, key_dim: int, value_dim: int, num_classes: int = 10):
        super().__init__()
        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_classes = num_classes

        # Persistent buffers (not parameters, not updated by optimizer)
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.register_buffer("gate_values", torch.zeros(capacity, value_dim))
        self.register_buffer("logits", torch.zeros(capacity, num_classes))
        self.register_buffer("importance", torch.zeros(capacity))
        self.register_buffer("task_ids", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("timestamps", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("usage_count", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("consolidated", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        # Key projection: hidden_size → key_dim
        self.key_proj = nn.Linear(value_dim, key_dim)

    @property
    def size(self) -> int:
        """Current number of stored entries."""
        return min(self.count.item(), self.capacity)

    @property
    def is_empty(self) -> bool:
        return self.count.item() == 0

    def project_key(self, cell_state: torch.Tensor) -> torch.Tensor:
        """Project a cell state to key space.

        Args:
            cell_state: [batch, value_dim] or [value_dim]

        Returns:
            key: [batch, key_dim] or [key_dim], L2-normalized
        """
        key = self.key_proj(cell_state)
        return torch.nn.functional.normalize(key, dim=-1)

    def _store(
        self,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        importances: torch.Tensor,
        task_id: int,
        gates: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        """Scatter a batch of entries into the given slots (vectorized).

        Args:
            slots: Destination slot indices [m]
            keys: Projected keys [m, key_dim]
            values: Cell states [m, value_dim]
            importances: Importance scores [m]
            task_id: Task identifier for all m entries
            gates: Optional recovered output-gate vectors [m, value_dim]
            logits: Optional stored logit predictions [m, num_classes]
        """
        if slots.numel() == 0:
            return
        # Stored bank contents are constants — never carry an autograd graph.
        self.keys[slots] = keys.detach()
        self.values[slots] = values.detach()
        # Fix 1b: clamp importance to [0, 1] on entry (applied here so both the
        # fill and eviction paths of the vectorized write() are covered).
        self.importance[slots] = importances.detach().clamp(0.0, 1.0)
        self.task_ids[slots] = task_id
        self.timestamps[slots] = self.global_step
        self.usage_count[slots] = 0
        # Newly written entries belong to the current task and stay evictable.
        self.consolidated[slots] = False
        if gates is not None:
            self.gate_values[slots] = gates.detach()
        if logits is not None:
            self.logits[slots] = logits.detach()

    def write(
        self,
        cell_states: torch.Tensor,
        importance_scores: torch.Tensor,
        task_id: int = -1,
        gate_vectors: Optional[torch.Tensor] = None,
        logit_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write cell states to memory bank.

        Fully vectorized (no per-entry Python loop). Two phases:
          1. Fill any empty slots with the leading entries.
          2. If entries remain (bank full), evict in a single batched pass over
             the NON-consolidated slots — pair the highest-importance incoming
             entries with the lowest-importance evictable slots and overwrite
             where the incoming entry wins. Same top-by-importance result as a
             greedy one-at-a-time eviction, but in O(1) Python steps.

        Args:
            cell_states: Cell states to store [num_entries, value_dim]
            importance_scores: Importance of each entry [num_entries]
            task_id: Current task identifier
            gate_vectors: Optional recovered output-gate context to store
                alongside each entry [num_entries, value_dim]

        Returns:
            write_indices: Slot each entry was written to, or -1 if not stored
                [num_entries]
        """
        device = cell_states.device
        num_entries = cell_states.shape[0]
        write_indices = torch.full((num_entries,), -1, dtype=torch.long, device=device)

        if num_entries == 0:
            self.global_step += 1
            return write_indices

        cell_states = cell_states.detach()
        importance_scores = importance_scores.detach()
        keys = self.project_key(cell_states)

        count = int(self.count.item())
        cursor = 0

        # Phase 1: fill empty slots (unconditional, like the original).
        n_empty = self.capacity - count
        if n_empty > 0:
            n_fill = min(n_empty, num_entries)
            slots = torch.arange(count, count + n_fill, device=device)
            sl = slice(0, n_fill)
            self._store(
                slots, keys[sl], cell_states[sl], importance_scores[sl], task_id,
                gates=gate_vectors[sl] if gate_vectors is not None else None,
                logits=logit_vectors[sl] if logit_vectors is not None else None,
            )
            write_indices[sl] = slots
            count += n_fill
            cursor = n_fill
        self.count.fill_(count)

        # Phase 2: batched eviction for any remaining entries (bank now full).
        # Only NON-consolidated slots are eviction candidates — consolidated
        # entries (protected old-task memories set by rebalance()) are immune.
        n_rem = num_entries - cursor
        if n_rem > 0:
            evict_pool = (~self.consolidated).nonzero(as_tuple=True)[0]  # evictable slots
            if evict_pool.numel() > 0:
                # Incoming remaining, highest importance first.
                inc_vals, inc_order = torch.sort(
                    importance_scores[cursor:], descending=True
                )
                inc_indices = inc_order + cursor  # positions in the incoming arrays
                # Lowest-importance evictable slots, lowest first.
                pool_imp = self.importance[evict_pool]
                k = min(n_rem, evict_pool.numel())
                low_vals, low_local = torch.topk(pool_imp, k, largest=False, sorted=True)
                evict_slots = evict_pool[low_local]
                # Pair biggest incoming with smallest evictable; keep where it wins.
                replace = inc_vals[:k] > low_vals
                sel = replace.nonzero(as_tuple=True)[0]
                if sel.numel() > 0:
                    src = inc_indices[sel]
                    dst = evict_slots[sel]
                    self._store(
                        dst, keys[src], cell_states[src], importance_scores[src], task_id,
                        gates=gate_vectors[src] if gate_vectors is not None else None,
                        logits=logit_vectors[src] if logit_vectors is not None else None,
                    )
                    write_indices[src] = dst

        self.global_step += 1
        return write_indices

    def read(
        self,
        query: torch.Tensor,
        top_k: int = 8,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Retrieve most relevant memories via attention over keys.

        Args:
            query: Query vector(s) [batch, value_dim]
            top_k: Number of memories to retrieve

        Returns:
            retrieved_values: Top-k memory values [batch, top_k, value_dim]
            retrieved_gates: Stored output-gate context [batch, top_k, value_dim]
            retrieved_logits: Stored logit predictions [batch, top_k, num_classes]
            retrieved_task_ids: Task id of each retrieved entry [batch, top_k]
            attention_weights: Attention over top-k [batch, top_k]
            indices: Which memory slots were accessed [batch, top_k]
        """
        if self.is_empty:
            batch = query.shape[0]
            device = query.device
            return (
                torch.zeros(batch, top_k, self.value_dim, device=device),
                torch.zeros(batch, top_k, self.value_dim, device=device),
                torch.zeros(batch, top_k, self.num_classes, device=device),
                torch.full((batch, top_k), -1, dtype=torch.long, device=device),
                torch.zeros(batch, top_k, device=device),
                torch.full((batch, top_k), -1, dtype=torch.long, device=device),
            )

        # Project query to key space
        query_key = self.project_key(query)  # [batch, key_dim]

        # Only attend over filled slots
        n = self.size
        active_keys = self.keys[:n]  # [n, key_dim]

        # Compute attention scores (scaled dot product)
        scale = math.sqrt(self.key_dim)
        scores = torch.matmul(query_key, active_keys.T) / scale  # [batch, n]

        # Select top-k
        k = min(top_k, n)
        topk_scores, topk_indices = scores.topk(k, dim=-1)  # [batch, k]

        # Softmax over top-k only (not full memory — sharper retrieval)
        topk_weights = torch.softmax(topk_scores, dim=-1)  # [batch, k]

        # Gather values and their stored gate / logit / task context
        topk_values = self.values[topk_indices]  # [batch, k, value_dim]
        topk_gates = self.gate_values[topk_indices]  # [batch, k, value_dim]
        topk_logits = self.logits[topk_indices]  # [batch, k, num_classes]
        topk_task_ids = self.task_ids[topk_indices]  # [batch, k]

        # Update usage counts
        unique_indices = topk_indices.unique()
        self.usage_count[unique_indices] += 1

        # Pad if fewer memories than top_k
        if k < top_k:
            batch = query.shape[0]
            device = query.device
            pad_vals = torch.zeros(batch, top_k - k, self.value_dim, device=device)
            pad_gates = torch.zeros(batch, top_k - k, self.value_dim, device=device)
            pad_logits = torch.zeros(batch, top_k - k, self.num_classes, device=device)
            pad_task_ids = torch.full(
                (batch, top_k - k), -1, dtype=torch.long, device=device
            )
            pad_weights = torch.zeros(batch, top_k - k, device=device)
            pad_indices = torch.full(
                (batch, top_k - k), -1, dtype=torch.long, device=device
            )
            topk_values = torch.cat([topk_values, pad_vals], dim=1)
            topk_gates = torch.cat([topk_gates, pad_gates], dim=1)
            topk_logits = torch.cat([topk_logits, pad_logits], dim=1)
            topk_task_ids = torch.cat([topk_task_ids, pad_task_ids], dim=1)
            topk_weights = torch.cat([topk_weights, pad_weights], dim=1)
            topk_indices = torch.cat([topk_indices, pad_indices], dim=1)

        return (
            topk_values, topk_gates, topk_logits,
            topk_task_ids, topk_weights, topk_indices,
        )

    def decay_importance(self, base_decay: float = 0.995) -> None:
        """Apply temporal decay to non-consolidated entries only.

        Consolidated (frozen) entries from previous tasks are immune to decay.
        Only current-task entries decay, with usage slowing the rate.
        """
        n = self.size
        if n == 0:
            return

        active = ~self.consolidated[:n]
        if not active.any():
            return

        usage_signal = self.usage_count[:n].float() / (self.usage_count[:n].float() + 10.0)
        adjusted_decay = base_decay + (1.0 - base_decay) * usage_signal

        decayed = (self.importance[:n] * adjusted_decay).clamp_(0.0, 1.0)
        self.importance[:n] = torch.where(active, decayed, self.importance[:n])
        self.usage_count[:n] = torch.where(
            active,
            torch.zeros_like(self.usage_count[:n]),
            self.usage_count[:n],
        )

    @torch.no_grad()
    def rebalance(self, slots_per_task: int) -> dict:
        """Rebalance so each task keeps at most slots_per_task entries.

        Called at task boundaries. For each task with more entries than
        the budget, keeps the highest-importance ones and marks them as
        consolidated (immune to decay and eviction). Excess entries are
        freed for the incoming task.

        Args:
            slots_per_task: Max entries per task after rebalancing

        Returns:
            Per-task before/after counts
        """
        n = self.size
        if n == 0:
            return {}

        stats = {}
        unique_tasks = self.task_ids[:n].unique()
        unique_tasks = unique_tasks[unique_tasks >= 0]

        for task_id in unique_tasks:
            tid = task_id.item()
            task_mask = (self.task_ids[:n] == tid)
            task_indices = task_mask.nonzero(as_tuple=True)[0]
            count_before = len(task_indices)

            if count_before <= slots_per_task:
                self.consolidated[task_indices] = True
                stats[tid] = {"before": count_before, "after": count_before}
            else:
                task_importances = self.importance[task_indices]
                _, top_local = task_importances.topk(slots_per_task)
                keep_indices = task_indices[top_local]

                self.consolidated[keep_indices] = True

                keep_set = set(keep_indices.tolist())
                for idx in task_indices.tolist():
                    if idx not in keep_set:
                        self.consolidated[idx] = False
                        self.importance[idx] = 0.0
                        self.task_ids[idx] = -1
                        self.keys[idx].zero_()
                        self.values[idx].zero_()
                        self.gate_values[idx].zero_()
                        self.logits[idx].zero_()

                stats[tid] = {"before": count_before, "after": slots_per_task}

        return stats

    def get_stats(self) -> dict:
        """Return diagnostic statistics about the memory bank."""
        n = self.size
        if n == 0:
            return {"size": 0, "capacity": self.capacity}

        unique_tasks = self.task_ids[:n].unique()
        unique_tasks = unique_tasks[unique_tasks >= 0]

        return {
            "size": n,
            "capacity": self.capacity,
            "utilization": n / self.capacity,
            "mean_importance": self.importance[:n].mean().item(),
            "min_importance": self.importance[:n].min().item(),
            "max_importance": self.importance[:n].max().item(),
            "num_tasks_stored": len(unique_tasks),
            "num_consolidated": self.consolidated[:n].sum().item(),
            "num_active": (~self.consolidated[:n]).sum().item(),
            "mean_usage": self.usage_count[:n].float().mean().item(),
            "mean_age": (self.global_step - self.timestamps[:n]).float().mean().item(),
        }

    def clear(self) -> None:
        """Reset the memory bank."""
        self.keys.zero_()
        self.values.zero_()
        self.importance.zero_()
        self.task_ids.fill_(-1)
        self.timestamps.zero_()
        self.usage_count.zero_()
        self.consolidated.zero_()
        self.gate_values.zero_()
        self.logits.zero_()
        self.count.zero_()
