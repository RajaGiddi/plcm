"""
Read and Write Controllers for the Thought Space.

Read Controller: Multi-head attention over the memory bank.
    Given a query (current cell state), retrieves and aggregates the
    most relevant stored memories into a single vector for composition.

Write Controller: Importance scoring for storage decisions.
    After composition, evaluates whether the new cell state should be
    stored in the memory bank and with what importance score.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .memory_bank import MemoryBank


class ReadController(nn.Module):
    """
    Multi-head attention read controller.

    Queries the memory bank with the current cell state, retrieves the
    top-k most relevant memories, and aggregates them into a single
    context vector for the composition operator.

    Uses multi-head attention to capture different "reasons" for retrieval:
    one head might attend to task-similar memories while another attends
    to recently stored ones.

    Args:
        hidden_size: Cell state dimension
        num_heads: Number of attention heads
        top_k: Number of memories to retrieve
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, top_k: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.top_k = top_k

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )
        self.head_dim = hidden_size // num_heads

        # Multi-head query projection
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        # Value re-projection (transform retrieved values per head)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        # Output projection (merge heads)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # Layer norm for stable retrieval
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        cell_state: torch.Tensor,
        memory_bank: MemoryBank,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, Optional[torch.Tensor]]:
        """
        Read from memory bank using multi-head attention.

        Args:
            cell_state: Current cell state [batch, hidden]
            memory_bank: Memory bank to query
            return_attention: If True, return attention weights

        Returns:
            context: Aggregated memory context [batch, hidden]
            gate_context: Aggregated stored output-gate context [batch, hidden]
            retrieval_info: Dict with aggregated logits, bank_weights, task_ids
            attention: Optional attention weights [batch, num_heads, top_k]
        """
        if memory_bank.is_empty:
            # No memories — return zeros (composition gate will handle this)
            empty_info = {"logits": None, "bank_weights": None, "task_ids": None}
            return (
                torch.zeros_like(cell_state),
                torch.zeros_like(cell_state),
                empty_info,
                None,
            )

        batch = cell_state.shape[0]
        device = cell_state.device

        # Retrieve top-k from memory bank (uses bank's own key projection)
        (
            retrieved_values, retrieved_gates, retrieved_logits,
            retrieved_task_ids, bank_weights, indices,
        ) = memory_bank.read(cell_state, top_k=self.top_k)
        # retrieved_values / retrieved_gates: [batch, top_k, hidden]
        # retrieved_logits: [batch, top_k, num_classes]
        # retrieved_task_ids, bank_weights: [batch, top_k]
        #
        # bank_weights derives from a matmul over a view of the key buffer that
        # write() mutates in-place. The memory is a constant here (gate mask /
        # stored predictions / distribution stats), so detach once at the source
        # to keep the loss graph off that buffer (avoids an autograd version bug).
        bank_weights = bank_weights.detach()

        # Multi-head re-attention over retrieved values
        # Project query into multiple heads
        Q = self.query_proj(cell_state)  # [batch, hidden]
        Q = Q.view(batch, self.num_heads, self.head_dim)  # [batch, heads, head_dim]

        # Project retrieved values
        V = self.value_proj(retrieved_values)  # [batch, top_k, hidden]
        V = V.view(batch, self.top_k, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)  # [batch, heads, top_k, head_dim]

        # Use retrieved values as keys too (content-based attention)
        K = retrieved_values.view(batch, self.top_k, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3)  # [batch, heads, top_k, head_dim]

        # Scaled dot-product attention per head
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(
            Q.unsqueeze(2), K.transpose(-2, -1)  # [batch, heads, 1, head_dim] x [batch, heads, head_dim, top_k]
        ).squeeze(2) / scale  # [batch, heads, top_k]

        # Mask out padding (indices == -1)
        if (indices < 0).any():
            mask = (indices < 0).unsqueeze(1).expand(-1, self.num_heads, -1)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch, heads, top_k]

        # Handle all-masked case (prevent NaN from softmax of all -inf)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Aggregate: weighted sum of values per head
        context = torch.matmul(
            attn_weights.unsqueeze(2), V  # [batch, heads, 1, top_k] x [batch, heads, top_k, head_dim]
        ).squeeze(2)  # [batch, heads, head_dim]

        # Merge heads
        context = context.reshape(batch, self.hidden_size)  # [batch, hidden]
        context = self.out_proj(context)  # [batch, hidden]
        context = self.layer_norm(context)

        # Aggregate the stored gate and logit context with the bank's single-head
        # weights (raw values preserved — a memory's gate acts as a per-dimension
        # relevance mask, its stored logits as a memory-based prediction).
        # bank_weights was already detached above (memory treated as constant).
        gate_context = torch.bmm(
            bank_weights.unsqueeze(1),  # [batch, 1, top_k]
            retrieved_gates,            # [batch, top_k, hidden]
        ).squeeze(1)                    # [batch, hidden]
        logit_context = torch.bmm(
            bank_weights.unsqueeze(1),  # [batch, 1, top_k]
            retrieved_logits,           # [batch, top_k, num_classes]
        ).squeeze(1)                    # [batch, num_classes]

        retrieval_info = {
            "logits": logit_context,
            "bank_weights": bank_weights,
            "task_ids": retrieved_task_ids,
        }

        if return_attention:
            return context, gate_context, retrieval_info, attn_weights
        return context, gate_context, retrieval_info, None


class WriteController(nn.Module):
    """
    Importance scoring write controller.

    Evaluates whether a composed cell state should be stored in the
    memory bank. Uses multiple signals:
    - Gate activation patterns (how actively the LSTM gates engaged)
    - Novelty (dissimilarity from existing memories)
    - Prediction confidence change (did this state help?)

    Outputs a scalar importance score per sample in the batch.

    Args:
        hidden_size: Cell state dimension
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Importance scoring network
        # Takes [cell_state; cell_state_before; gate_stats]
        self.score_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        # Novelty detector: how different is this from existing memories
        self.novelty_proj = nn.Linear(hidden_size, hidden_size)

    def compute_novelty(
        self,
        cell_state: torch.Tensor,
        memory_bank: MemoryBank,
    ) -> torch.Tensor:
        """Compute novelty score: how different is this from stored memories.

        High novelty → more worth storing (it's new information).

        Args:
            cell_state: Candidate state [batch, hidden]
            memory_bank: Current memory bank

        Returns:
            novelty: Novelty scores [batch]
        """
        if memory_bank.is_empty:
            return torch.ones(cell_state.shape[0], device=cell_state.device)

        # Project to comparison space
        projected = self.novelty_proj(cell_state)  # [batch, hidden]

        # Compare against stored memories
        n = memory_bank.size
        stored = memory_bank.values[:n]  # [n, hidden]
        stored_proj = self.novelty_proj(stored)  # [n, hidden]

        # Cosine similarity with nearest stored memory
        projected_norm = torch.nn.functional.normalize(projected, dim=-1)
        stored_norm = torch.nn.functional.normalize(stored_proj, dim=-1)
        similarity = torch.matmul(projected_norm, stored_norm.T)  # [batch, n]
        max_similarity = similarity.max(dim=-1).values  # [batch]

        # Novelty = 1 - max_similarity (high when state is unlike anything stored)
        novelty = 1.0 - max_similarity
        return novelty.clamp(0.0, 1.0)

    def forward(
        self,
        cell_state: torch.Tensor,
        cell_state_before: torch.Tensor,
        memory_bank: MemoryBank,
    ) -> torch.Tensor:
        """Score cell states for storage importance.

        Args:
            cell_state: Composed cell state (after GGC) [batch, hidden]
            cell_state_before: Original cell state (before GGC) [batch, hidden]
            memory_bank: Current memory bank (for novelty computation)

        Returns:
            importance: Importance scores [batch]
        """
        # Compute base importance from state change
        score_input = torch.cat([cell_state, cell_state_before], dim=-1)
        base_importance = self.score_net(score_input).squeeze(-1)  # [batch]

        # Compute novelty
        novelty = self.compute_novelty(cell_state, memory_bank)  # [batch]

        # Combined importance: geometric mean of base and novelty
        importance = (base_importance * novelty).sqrt()

        return importance
