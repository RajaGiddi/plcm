"""
Memory Consolidation Module.

Periodically compresses the memory bank to prevent unbounded growth of
similar vectors. Uses k-means clustering to merge nearby cell states,
preserving the highest-importance representative from each cluster.

Inspired by memory consolidation in biological systems, where similar
episodic memories are merged into more abstract semantic memories during
sleep.
"""

import torch
import torch.nn as nn
from typing import Optional

from .memory_bank import MemoryBank


class MemoryConsolidation:
    """
    Periodic memory consolidation via clustering and eviction.

    Two consolidation strategies:
    1. Cluster-merge: Group similar memories, keep highest-importance per cluster
    2. Importance-decay: Reduce importance of old, unused memories over time

    Args:
        num_clusters: Target number of clusters (should be < capacity)
        max_iterations: K-means iterations
        similarity_threshold: Below this cosine similarity, memories are "different enough"
    """

    def __init__(
        self,
        num_clusters: int = 64,
        max_iterations: int = 10,
        similarity_threshold: float = 0.95,
    ):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold

    @torch.no_grad()
    def consolidate(self, memory_bank: MemoryBank) -> dict:
        """Run consolidation on the memory bank.

        Merges highly similar memories and evicts low-importance ones.

        Args:
            memory_bank: Memory bank to consolidate (modified in place)

        Returns:
            stats: Consolidation statistics
        """
        n = memory_bank.size
        if n <= self.num_clusters:
            return {"merged": 0, "evicted": 0, "kept": n}

        values = memory_bank.values[:n]  # [n, value_dim]
        importance = memory_bank.importance[:n]  # [n]
        task_ids = memory_bank.task_ids[:n]  # [n]
        timestamps = memory_bank.timestamps[:n]

        # Step 1: Find clusters via k-means
        cluster_assignments, centroids = self._kmeans(
            values, min(self.num_clusters, n)
        )

        # Step 2: For each cluster, keep the highest-importance member.
        # Vectorized: order entries by descending importance, then mark the
        # first time each cluster appears (that cluster's max-importance member).
        # The one-hot cumsum gives, per entry, how many earlier entries share
        # its cluster — zero means "first occurrence".
        device = values.device
        num_clusters_actual = centroids.shape[0]
        order = torch.argsort(importance, descending=True)  # high -> low importance
        clusters_ordered = cluster_assignments[order]  # [n]
        onehot = torch.zeros(n, num_clusters_actual, device=device)
        onehot[torch.arange(n, device=device), clusters_ordered] = 1.0
        earlier_same = (onehot.cumsum(0) - onehot).gather(
            1, clusters_ordered.unsqueeze(1)
        ).squeeze(1)  # [n]
        is_first = earlier_same == 0  # max-importance member of each cluster
        keep_mask = torch.zeros(n, dtype=torch.bool, device=device)
        keep_mask[order[is_first]] = True

        # Consolidated (protected) entries are ALWAYS kept — the periodic
        # k-means merge must never drop a task-protected memory.
        keep_mask = keep_mask | memory_bank.consolidated[:n]

        # Step 3: Compact the memory bank
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        num_kept = len(keep_indices)

        # Rewrite bank with kept entries. The consolidated flag is carried
        # alongside so protection stays aligned with its entry after compaction.
        memory_bank.keys[:num_kept] = memory_bank.keys[keep_indices]
        memory_bank.values[:num_kept] = values[keep_indices]
        memory_bank.importance[:num_kept] = importance[keep_indices]
        memory_bank.task_ids[:num_kept] = task_ids[keep_indices]
        memory_bank.timestamps[:num_kept] = timestamps[keep_indices]
        memory_bank.consolidated[:num_kept] = memory_bank.consolidated[keep_indices]
        memory_bank.gate_values[:num_kept] = memory_bank.gate_values[keep_indices]
        memory_bank.logits[:num_kept] = memory_bank.logits[keep_indices]

        # Zero out freed slots
        memory_bank.keys[num_kept:n].zero_()
        memory_bank.values[num_kept:n].zero_()
        memory_bank.importance[num_kept:n].zero_()
        memory_bank.task_ids[num_kept:n].fill_(-1)
        memory_bank.timestamps[num_kept:n].zero_()
        memory_bank.usage_count[num_kept:n].zero_()
        memory_bank.consolidated[num_kept:n] = False
        memory_bank.gate_values[num_kept:n].zero_()
        memory_bank.logits[num_kept:n].zero_()

        memory_bank.count.fill_(num_kept)

        # Re-project keys for kept entries
        with torch.no_grad():
            memory_bank.keys[:num_kept] = memory_bank.project_key(
                memory_bank.values[:num_kept]
            )

        return {
            "merged": n - num_kept,
            "evicted": 0,
            "kept": num_kept,
            "original_size": n,
        }

    def _kmeans(
        self,
        data: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple k-means clustering.

        Args:
            data: Vectors to cluster [n, dim]
            k: Number of clusters

        Returns:
            assignments: Cluster assignment per vector [n]
            centroids: Cluster centers [k, dim]
        """
        n, dim = data.shape
        device = data.device

        # Initialize centroids with k-means++. Keep a running minimum distance
        # to the nearest chosen centroid so each step only measures distance to
        # the newest centroid (O(n)) instead of recomputing against all chosen.
        first = torch.randint(n, (1,), device=device).item()
        indices = [first]
        min_dists = torch.cdist(data, data[first:first + 1]).squeeze(1)  # [n]
        for _ in range(k - 1):
            # Probabilistic selection weighted by squared-ish distance
            probs = min_dists / (min_dists.sum() + 1e-8)
            next_idx = torch.multinomial(probs, 1).item()
            indices.append(next_idx)
            new_dists = torch.cdist(data, data[next_idx:next_idx + 1]).squeeze(1)  # [n]
            min_dists = torch.minimum(min_dists, new_dists)

        centroids = data[indices].clone()  # [k, dim]

        ones = torch.ones(n, device=device, dtype=data.dtype)
        for _ in range(self.max_iterations):
            # Assign to nearest centroid
            dists = torch.cdist(data, centroids)  # [n, k]
            assignments = dists.argmin(dim=1)  # [n]

            # Update centroids as the per-cluster mean, vectorized via index_add
            # (replaces the per-cluster Python loop). Empty clusters keep their
            # previous centroid.
            counts = torch.zeros(k, device=device, dtype=data.dtype)
            counts.index_add_(0, assignments, ones)  # [k]
            sums = torch.zeros_like(centroids)  # [k, dim]
            sums.index_add_(0, assignments, data)
            nonempty = counts > 0  # [k]
            new_centroids = centroids.clone()
            new_centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)

            # Check convergence
            shift = (new_centroids - centroids).norm(dim=1).max()
            centroids = new_centroids
            if shift < 1e-6:
                break

        return assignments, centroids

    @torch.no_grad()
    def deduplicate(
        self,
        memory_bank: MemoryBank,
    ) -> int:
        """Remove near-duplicate memories (fast, runs every write batch).

        Uses cosine similarity threshold rather than full clustering.

        Args:
            memory_bank: Memory bank to deduplicate

        Returns:
            num_removed: Number of duplicates removed
        """
        n = memory_bank.size
        if n < 2:
            return 0

        values = memory_bank.values[:n]
        normed = torch.nn.functional.normalize(values, dim=-1)
        similarity = torch.matmul(normed, normed.T)  # [n, n]

        # Mask diagonal
        similarity.fill_diagonal_(0.0)

        # Find pairs above threshold
        duplicates = (similarity > self.similarity_threshold).any(dim=1)
        if not duplicates.any():
            return 0

        # For each group of duplicates, keep highest importance
        removed = 0
        processed = torch.zeros(n, dtype=torch.bool, device=values.device)

        for i in range(n):
            if processed[i] or not duplicates[i]:
                continue

            # Find all entries similar to i
            group = (similarity[i] > self.similarity_threshold).nonzero(
                as_tuple=True
            )[0]
            group = torch.cat([torch.tensor([i], device=values.device), group])
            group = group[~processed[group]]

            if len(group) <= 1:
                continue

            # Keep highest importance, mark rest for removal
            group_importance = memory_bank.importance[group]
            best = group[group_importance.argmax()]

            for idx in group:
                if idx != best:
                    memory_bank.importance[idx] = 0.0
                    removed += 1
                processed[idx] = True

        return removed
