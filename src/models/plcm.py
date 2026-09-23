"""
Persistent Latent Cell Memory (PLCM) — Full Model.

Combines all components into a single model:
    LSTM base → Read Controller → Composition ⊕_T → Write Controller

The LSTM processes the input sequence normally. After each sequence,
the final cell state is composed with retrieved memories from the
thought space via the Gated Geometric Composition operator. The
composed state is then evaluated for storage by the write controller.

This module handles the full forward pass including memory read/write
and provides diagnostic information for analysis.
"""

import torch
import torch.nn as nn
from typing import Optional

from .lstm_base import LSTMBaseline
from .memory_bank import MemoryBank
from .controllers import ReadController, WriteController
from .composition import build_composition
from .consolidation import MemoryConsolidation


class LowRankAdapter(nn.Module):
    """Residual low-rank input adapter: A(x) = x + U(V(x)), rank r.

    Storage is O(2*d*r) per task instead of O(d^2).

    Initialization is EXACTLY the identity, not merely close to it: V is small
    random, U is zeroed, so U(V(x)) = 0 at step 0 and A(x) = x. This matters —
    the full-rank adapters relied on identity init, and a naive random low-rank
    product would start far from identity, so a failure could be blamed on
    initialization rather than capacity (which would make the pre-registered
    branch reading uninterpretable).
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.V = nn.Linear(dim, rank, bias=False)   # d -> r
        self.U = nn.Linear(rank, dim, bias=False)   # r -> d
        nn.init.normal_(self.V.weight, std=0.01)
        nn.init.zeros_(self.U.weight)               # => A(x) == x exactly at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.U(self.V(x))


class PLCM(nn.Module):
    """
    Persistent Latent Cell Memory model.

    An LSTM augmented with a persistent memory bank of cell state vectors,
    composed via a gated geometric operator in the thought space.

    Args:
        input_size: Input dimension per timestep
        hidden_size: LSTM hidden/cell state dimension
        num_classes: Number of output classes
        num_layers: Number of LSTM layers
        memory_capacity: Max stored cell state vectors
        key_dim: Projected key dimension for memory retrieval
        top_k: Number of memories to retrieve
        num_heads: Attention heads in read controller
        composition_mode: "ggc" | "additive" | "mobius"
        write_threshold: Min importance score for storage
        consolidation_interval: Steps between consolidation runs
        consolidation_clusters: Target clusters after consolidation
        dropout: LSTM dropout
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        memory_capacity: int = 512,
        key_dim: int = 128,
        top_k: int = 8,
        num_heads: int = 4,
        composition_mode: str = "ggc",
        write_threshold: float = 0.5,
        consolidation_interval: int = 50,
        consolidation_clusters: int = 64,
        dropout: float = 0.0,
        use_gate_filter: bool = True,
        use_coord_align: bool = True,
        use_context_heads: bool = True,
        freeze_encoder_after_first_task: bool = False,
        use_input_adapters: bool = False,
        adapter_dim: int = 784,
        adapter_rank: int = 0,
        adapter_mode: str = "flat",
        backbone: str = "lstm",
        # E23 arm B (2026-09-19), OPT-IN: the MLP's flattened input width when it differs
        # from the adapter's dim. On MNIST the two coincide (784, flat adapter), so every
        # existing MLP checkpoint reloads unchanged; on HAR the adapter is per-step 9-wide
        # and the MLP must read the flattened 128 x 9 window. None = the pre-E23 behaviour.
        mlp_input_dim: Optional[int] = None,
        use_memory: bool = True,
        use_task_heads: bool = True,
        use_router: bool = False,
        router_num_tasks: int = 5,
        head_routing_by_hint: bool = False,
        vit_model: str = "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        vit_pretrained: bool = True,
        resnet_model: str = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.write_threshold = write_threshold
        self.consolidation_interval = consolidation_interval
        self.step_count = 0

        # Feature toggles (for ablation / attribution studies)
        self.use_gate_filter = use_gate_filter      # gate pre-filter of retrieved memory
        self.use_coord_align = use_coord_align      # source->current distribution alignment
        self.use_context_heads = use_context_heads  # blend current vs memory logits
        # Option A: freeze the LSTM encoder after task 0. OFF by default so
        # plcm / plcm_ewc stay unfrozen (reproducible + the A/B baseline);
        # only plcm_frozen enables it.
        self.freeze_encoder_after_first_task = freeze_encoder_after_first_task

        # Option B: per-task linear input adapters. A_k maps this task's input
        # back toward the frozen encoder's native (task-0) layout.
        #
        # DESIGN PRINCIPLE (made explicit once a second modality forced the
        # choice): match the adapter to the space where the SHIFT LIVES.
        #   "flat"     — the shift mixes coordinates across timesteps, so the
        #                adapter must span the flattened input. Permuted MNIST's
        #                permutation mixes pixels across rows: a per-row map
        #                cannot represent it (A_k = P_k^T needs all 784 dims).
        #   "per_step" — the shift is a channel-space map applied identically at
        #                every timestep, so a d_in x d_in map applied per step
        #                represents it EXACTLY. UCI HAR's hardware-revision
        #                shifts are 9x9 channel maps: 81 params/task, not
        #                1152x1152 = 1.33M. nn.Linear already acts on the last
        #                dim, so [batch, T, C] needs no reshape at all.
        #
        # This is why the O(T*d^2) storage objection is regime-dependent rather
        # than fundamental: adapter cost tracks the shift's intrinsic
        # dimensionality, not the input's.
        self.input_size = input_size
        self.adapter_mode = adapter_mode
        assert adapter_mode in ("flat", "per_step"), f"bad adapter_mode {adapter_mode}"
        if adapter_mode == "per_step":
            # The adapter acts on one token's feature vector — and WHICH space
            # that is depends on where the adapter is placed. For the recurrent
            # path it is a timestep's channel vector (input_size). For the ViT
            # the registered placement is PATCH EMBEDDINGS, so it is the
            # encoder's embed dim; taking input_size there would build a 3x3 map
            # for a 768-dim token and fail at the first matmul.
            from .vit_base import VIT_EMBED_DIM
            adapter_dim = VIT_EMBED_DIM if backbone == "vit" else input_size
        self.adapter_dim = adapter_dim
        # rank 0 = full-rank d x d adapter; rank r > 0 = residual low-rank
        # A(x) = x + U V x, storage O(2*d*r) per task instead of O(d^2).
        self.adapter_rank = adapter_rank
        self.use_input_adapters = use_input_adapters
        self.task_adapters = nn.ModuleDict()

        # MAFC: with per-task heads OFF the readout phi is shared and trainable,
        # so it can drift — which is what the bank-anchored readout term pins.
        # With heads ON (default) old heads are frozen and that term is vacuous.
        self.use_task_heads = use_task_heads

        # E12/P3b — AMENDMENT 2 of 2, default off.
        # `task_hint` has always routed ONLY the adapter; the readout was either
        # the retrieval-weighted blend or the LATEST head — never head k. E7's
        # result was read as "frozen-early heads worsen forgetting" when no task
        # was ever scored by its own frozen head alone (alpha_own = 0.165, own
        # head top-weighted in 0/24 cells). "Task-IL" was a capability the model
        # did not have, written as a setting.
        #
        # DEFAULT OFF IS LOAD-BEARING: turning this on unconditionally would
        # alter the deployed path of every prior arm and destroy E11's proof that
        # forward() reproduces every recorded accuracy matrix exactly. A fix to
        # enable the next experiment must not spend the proof the last one
        # bought (CLAUDE.md, blast-radius rule).
        self.head_routing_by_hint = head_routing_by_hint

        # Ablation toggle: when False, nothing is ever written to the bank, so it
        # stays empty and every memory path (read/compose/gate/align/context) is
        # a no-op — leaving frozen encoder + adapters + heads only.
        self.use_memory = use_memory

        # Task-free routing: a small classifier over cell states -> task id, used
        # at eval to pick the per-task adapter/head when task identity is unknown.
        # Trained continually with balanced bank replay (in the trainer); the bank
        # replay is what lets the router itself resist catastrophic forgetting of
        # old-task routing boundaries. Fixed output size = max tasks.
        self.use_router = use_router
        self.router_num_tasks = router_num_tasks
        self.task_router = nn.Linear(hidden_size, router_num_tasks)

        # Backbone. "lstm" is the default; "mlp" is the architecture
        # brittleness control (docs/BRITTLENESS_prereg.md). The attribute keeps
        # the name `lstm` so every downstream path (freezing, EWC param_filter
        # "lstm.", functional_forward, checkpointing) is untouched — the only
        # thing that changes is the backbone itself.
        # E12/B4 — AMENDMENT 1 of 2, default off (backbone defaults to "lstm").
        # The ViT path bypasses memory/composition/output-gate and uses the
        # NATIVE readout (post-norm CLS), because P3a registers that tensor as
        # the deployed feature. Kept as a separate amendment from
        # head_routing_by_hint so that if a bit-identity regression ever fails,
        # it is diagnosable to one change rather than to a pair of them.
        self.backbone = backbone
        if backbone == "vit":
            from .vit_base import ViTEncoder
            self.lstm = ViTEncoder(model_name=vit_model, pretrained=vit_pretrained)
            self.vit_model = vit_model
        elif backbone == "resnet":
            # E14/C1 — AMENDMENT, opt-in, default is "lstm". Parallel to the ViT
            # branch by design, not refactored with it: unifying the backbones
            # now would be an amendment with maximal blast radius, and the
            # regression proving the shared path bit-identical to E11's matrices
            # is what it would spend. Paper 2 pays that debt down.
            from .resnet_base import ResNetEncoder, RESNET_MODEL
            self.lstm = ResNetEncoder(model_name=resnet_model or RESNET_MODEL,
                                      pretrained=vit_pretrained)
            self.resnet_model = resnet_model or RESNET_MODEL
        elif backbone == "mlp":
            from .mlp_base import MLPEncoder
            self.lstm = MLPEncoder(
                input_dim=mlp_input_dim or adapter_dim, hidden_size=hidden_size,   # E23 arm B: HAR = 128 x 9; MNIST unchanged (784)
            )
        else:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        # Memory bank (persistent across tasks)
        self.memory_bank = MemoryBank(
            capacity=memory_capacity,
            key_dim=key_dim,
            value_dim=hidden_size,
            num_classes=num_classes,
        )

        # Controllers
        self.read_controller = ReadController(
            hidden_size=hidden_size,
            num_heads=num_heads,
            top_k=top_k,
        )
        self.write_controller = WriteController(hidden_size=hidden_size)

        # Composition operator
        self.composition = build_composition(composition_mode, hidden_size)

        # Consolidation (not a nn.Module, stateless utility)
        self.consolidation = MemoryConsolidation(
            num_clusters=consolidation_clusters,
        )

        # Learned output gate: produces h' from c' by gating through
        # both the LSTM's last hidden output and the composed cell state.
        self.output_gate = nn.Linear(hidden_size * 2, hidden_size)

        # Coordinate alignment: track the current cell-state distribution and
        # per-task snapshots so retrieved old-task states can be transformed
        # into the current era before composition.
        self.register_buffer("running_mean", torch.zeros(hidden_size))
        self.register_buffer("running_var", torch.ones(hidden_size))
        self.register_buffer("stats_initialized", torch.tensor(False))
        self.task_stats = {}  # task_id -> (mean, var) tensors

        # Context heads: routing gate blends current vs memory predictions.
        self.routing_gate = nn.Linear(hidden_size * 2, 1)

        # Classification head (shared; also warm-starts each task-specific head)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
        # Per-task classifier heads, created lazily at task boundaries.
        self.task_classifiers = nn.ModuleDict()

        # Track current task for memory tagging
        self.current_task_id = 0
        self.num_tasks_seen = 0

    def set_task(self, task_id: int) -> None:
        """Set the current task.

        At a task boundary: snapshots the distribution stats, rebalances memory,
        freezes the outgoing task's classifier head, then creates (warm-started
        and trainable) the incoming task's head. Old heads are frozen so their
        parameters are never updated by later tasks.

        NOTE: heads are created lazily here, so the training loop MUST rebuild
        its optimizer after calling set_task (the trainer does this) — otherwise
        the new head's parameters are not optimized.
        """
        if self.num_tasks_seen > 0 and task_id != self.current_task_id:
            # Snapshot the finishing task's cell-state distribution for alignment.
            if self.stats_initialized:
                self.task_stats[self.current_task_id] = (
                    self.running_mean.clone(),
                    self.running_var.clone(),
                )
            # Rebalance memory into per-task protected slots.
            num_tasks_total = self.num_tasks_seen + 1
            slots_per_task = self.memory_bank.capacity // num_tasks_total
            self.memory_bank.rebalance(slots_per_task)

            # Option A: freeze the LSTM encoder after Task 0. All subsequent
            # tasks train only heads + PLCM components. This eliminates
            # representation drift — the root cause of retrieval failure and
            # classifier misalignment. (Composition, controllers, routing gate,
            # and task heads stay trainable.) Gated so only plcm_frozen freezes.
            if self.freeze_encoder_after_first_task and self.current_task_id == 0:
                for param in self.lstm.parameters():
                    param.requires_grad = False
                # The output gate is part of the encoder pathway (h' from c').
                for param in self.output_gate.parameters():
                    param.requires_grad = False

            # Freeze the outgoing task's head (never updated again).
            old_key = str(self.current_task_id)
            if old_key in self.task_classifiers:
                for param in self.task_classifiers[old_key].parameters():
                    param.requires_grad = False

            # Freeze the outgoing task's input adapter too (retention by
            # construction: nothing that scores an old task ever changes again).
            if self.use_input_adapters and old_key in self.task_adapters:
                for param in self.task_adapters[old_key].parameters():
                    param.requires_grad = False

        # Create the incoming task's head if needed (warm-start from shared head).
        # MAFC disables per-task heads so the readout phi is SHARED and can
        # drift — otherwise the bank-anchored readout term is vacuous.
        task_key = str(task_id)
        if self.use_task_heads and task_key not in self.task_classifiers:
            new_head = nn.Linear(self.hidden_size, self.num_classes)
            new_head.weight.data.copy_(self.classifier.weight.data)
            new_head.bias.data.copy_(self.classifier.bias.data)
            device = next(self.parameters()).device
            self.task_classifiers[task_key] = new_head.to(device)

        # Create the incoming task's input adapter. Both variants start as an
        # EXACT identity so task 0 (and near-identity shifts) are unharmed; the
        # adapter learns its transform via the classification loss.
        if self.use_input_adapters and task_key not in self.task_adapters:
            if self.adapter_rank and self.adapter_rank > 0:
                adapter = LowRankAdapter(self.adapter_dim, self.adapter_rank)
            else:
                adapter = nn.Linear(self.adapter_dim, self.adapter_dim, bias=False)
                nn.init.eye_(adapter.weight)
            device = next(self.parameters()).device
            self.task_adapters[task_key] = adapter.to(device)

        self.current_task_id = task_id
        self.num_tasks_seen += 1

    def _apply_adapter(self, x: torch.Tensor, adapter_key: str) -> torch.Tensor:
        """Apply task `adapter_key`'s input adapter, respecting adapter_mode.

        Single place both the training forward and the MAFC functional_forward
        route through, so the two can never disagree about adapter geometry.

        Args:
            x: input batch [batch, seq_len, input_size]
            adapter_key: str(task_id); a missing key is a no-op (task 0 before
                set_task, and the M4-LwF raw-probe arm, both rely on this)

        Returns:
            adapted input, same shape as x
        """
        if adapter_key not in self.task_adapters:
            return x
        adapter = self.task_adapters[adapter_key]
        if self.adapter_mode == "per_step":
            # nn.Linear maps the LAST dim, so [batch, T, C] -> [batch, T, C]
            # with the SAME map at every timestep. That weight sharing is the
            # point: the shift is constant over time, so the adapter is too.
            return adapter(x)
        shape = x.shape                                  # [batch, T, C]
        return adapter(x.reshape(shape[0], -1)).reshape(shape)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        store_memories: bool = True,
        return_diagnostics: bool = False,
        task_hint: Optional[int] = None,
        apply_adapter: bool = True,
    ) -> dict:
        """
        Full PLCM forward pass.

        1. LSTM processes input → cₜ, hₜ
        2. Read controller queries memory bank with cₜ → retrieves c̃
        3. Composition: cₜ' = ⊕_T(cₜ, c̃)
        4. Write controller evaluates cₜ' for storage
        5. Classifier uses hₜ' derived from cₜ'

        Args:
            x: Input sequence [batch, seq_len, input_size]
            hidden: Optional initial LSTM state
            store_memories: Whether to write to memory bank (disable for eval)
            return_diagnostics: Whether to collect diagnostic info
            apply_adapter: When False, the per-task input adapter is skipped even
                on an adapter-carrying model. This exists so an instrument that
                needs the adapter SUPPRESSED (C0 substitutes an analytic one) can
                express that THROUGH the deployed path instead of hand-rebuilding
                a bypass around it — a rebuild is what catch 28 was.

        Returns:
            Dictionary with:
                logits: [batch, num_classes]
                cell_state: cₜ' after composition [batch, hidden]
                readout_feature: h' = o_t ⊙ tanh(cₜ') — the EXACT tensor every
                    readout head consumes in this call. Instruments must read
                    this rather than reconstruct it (catch 28).
                hidden: (h_n, c_n) from LSTM
                diagnostics: Optional dict of diagnostic info
        """
        # Option B: per-task input adapter. Map this task's permuted input back
        # toward the frozen encoder's native (task-0) layout, on the FLATTENED
        # image (the permutation mixes pixels across rows). At eval on an OLD
        # task, task_hint selects that task's adapter.
        # NOTE: using task_hint at eval assumes task identity is known at test
        # time — the TASK-INCREMENTAL setting (weaker than task-free). A follow-up
        # can instead infer the adapter via memory retrieval to stay task-free.
        if self.backbone == "vit":
            return self._forward_vit(x, task_hint, apply_adapter)
        if self.backbone == "resnet":
            return self._forward_resnet(x, task_hint)

        if self.use_input_adapters and apply_adapter:
            adapter_key = str(task_hint if task_hint is not None else self.current_task_id)
            x = self._apply_adapter(x, adapter_key)

        # Step 1: LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        # lstm_out: [batch, seq_len, hidden]
        # h_n, c_n: [num_layers, batch, hidden]

        c_t = c_n[-1]  # [batch, hidden] — cell state from final layer

        # Coordinate alignment: track the running cell-state distribution so
        # retrieved old-task states can later be mapped into the current era.
        # Only track while the encoder is still trainable — once frozen (Option A)
        # the distribution stops moving, so later-task batch stats must not
        # pollute the running estimates.
        encoder_trainable = next(self.lstm.parameters()).requires_grad
        if self.training and encoder_trainable:
            with torch.no_grad():
                batch_mean = c_t.mean(dim=0)
                batch_var = c_t.var(dim=0).clamp(min=1e-6)
                if not self.stats_initialized:
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                    self.stats_initialized.fill_(True)
                else:
                    mom = 0.1
                    self.running_mean.mul_(1 - mom).add_(batch_mean * mom)
                    self.running_var.mul_(1 - mom).add_(batch_var * mom)

        # Recover the LSTM's output gate exactly: h = o * tanh(c) => o = h / tanh(c).
        # Stored with each memory so retrieval can re-mask it through the gate that
        # was active when it was created. Detached — it is a stored constant, never
        # part of the loss (keeps the write path graph-free, like the rest of it).
        h_t = h_n[-1]  # [batch, hidden]
        tanh_ct = torch.tanh(c_t)
        gate_t = torch.where(
            tanh_ct.abs() > 1e-6,
            h_t / tanh_ct,
            torch.ones_like(h_t),  # default to "open" when tanh(c) ≈ 0
        ).clamp(0.0, 1.0).detach()

        # Step 2: Read from memory bank
        c_retrieved, gate_retrieved, retrieval_info, attn_weights = self.read_controller(
            c_t, self.memory_bank, return_attention=return_diagnostics
        )
        # c_retrieved: [batch, hidden]

        # Step 3: Compose current state with retrieved memories
        if not self.memory_bank.is_empty:
            # Gate pre-filter: keep only the dimensions that were "active" in the
            # memory's original context (addresses cross-era weight mismatch).
            if self.use_gate_filter:
                c_retrieved = c_retrieved * gate_retrieved

            # Coordinate alignment: map retrieved states from their source-task
            # distribution into the current one (whiten by source, re-color by
            # current). Weighted per-sample by which tasks were retrieved.
            if self.use_coord_align and self.task_stats and retrieval_info["task_ids"] is not None:
                bank_weights = retrieval_info["bank_weights"]
                ret_task_ids = retrieval_info["task_ids"]
                src_mean = torch.zeros_like(c_t)
                src_var = torch.zeros_like(c_t)
                total_w = torch.zeros(c_t.shape[0], 1, device=c_t.device)
                for tid, (t_mean, t_var) in self.task_stats.items():
                    m_mask = (ret_task_ids == tid).float()
                    w = (m_mask * bank_weights).sum(dim=-1, keepdim=True)
                    src_mean = src_mean + w * t_mean.unsqueeze(0)
                    src_var = src_var + w * t_var.unsqueeze(0)
                    total_w = total_w + w
                if total_w.sum() > 1e-8:
                    src_mean = src_mean / (total_w + 1e-8)
                    src_var = src_var / (total_w + 1e-8)
                    src_std = src_var.sqrt().clamp(min=1e-6)
                    cur_std = self.running_var.sqrt().clamp(min=1e-6)
                    c_retrieved = (
                        (c_retrieved - src_mean) / src_std * cur_std + self.running_mean
                    )

            c_prime, comp_diag = self.composition(
                c_t, c_retrieved, return_diagnostics=return_diagnostics
            )
        else:
            # No memories yet — pass through unchanged
            c_prime = c_t
            comp_diag = None

        # Step 4: Learned output gate mirrors the LSTM's own o_t * tanh(c_t)
        # but sees both the raw hidden output and the composed cell state.
        gate_input = torch.cat([lstm_out[:, -1, :], c_prime], dim=-1)  # [batch, 2*hidden]
        o_t = torch.sigmoid(self.output_gate(gate_input))  # [batch, hidden]
        h_prime = o_t * torch.tanh(c_prime)  # [batch, hidden]

        # Context-head routing: blend the current parametric prediction with the
        # memory-based prediction (stored logits of retrieved memories). The
        # routing gate lambda is learned from the current and retrieved states.
        # Task-specific classifier heads with per-task routing from memory.
        current_key = str(self.current_task_id)

        # Logits from every available head. Old heads are detached during
        # training so gradients flow only through the current task's head.
        task_logits = {}
        for key, head in self.task_classifiers.items():
            head_out = head(h_prime)
            if self.training and key != current_key:
                head_out = head_out.detach()
            task_logits[key] = head_out

        if (self.head_routing_by_hint and task_hint is not None
                and str(task_hint) in self.task_classifiers):
            # E12/P3b, flag ON: task k is scored by head k, FIRST — ahead of the
            # routed blend, or the flag would be inert in exactly the case E7
            # exposed (non-empty bank, where the blend wins and alpha_own was
            # 0.165 with the own head top-weighted in 0/24 cells).
            # Flag OFF: this branch does not exist and the path below is
            # byte-for-byte the pre-E12 one.
            logits = task_logits[str(task_hint)]
        elif (
            task_logits
            and not self.memory_bank.is_empty
            and retrieval_info is not None
            and retrieval_info.get("bank_weights") is not None
            and retrieval_info.get("task_ids") is not None
        ):
            bank_weights = retrieval_info["bank_weights"]  # [batch, top_k]
            ret_task_ids = retrieval_info["task_ids"]       # [batch, top_k]

            # Route by per-task retrieval attention mass. Per-task weights are
            # non-negative and sum to the total bank weight -> a convex blend.
            logits = torch.zeros_like(next(iter(task_logits.values())))
            for key, head_out in task_logits.items():
                m_mask = (ret_task_ids == int(key)).float()               # [batch, top_k]
                alpha = (m_mask * bank_weights).sum(dim=-1, keepdim=True)  # [batch, 1]
                logits = logits + alpha * head_out

            # Optionally also blend the stored memory predictions (context heads).
            logits_memory = retrieval_info.get("logits")
            if self.use_context_heads and logits_memory is not None:
                route_input = torch.cat([h_prime, c_retrieved], dim=-1)
                lam = torch.sigmoid(self.routing_gate(route_input))
                logits = lam * logits + (1.0 - lam) * logits_memory
        else:
            # No memories yet (or no task heads created) — current head / fallback.
            if current_key in self.task_classifiers:
                logits = task_logits[current_key]
            else:
                logits = self.classifier(h_prime)

        # Current-task head output — used to warm-start future heads and as the
        # parametric prediction written to memory (not the routed blend).
        logits_current = (
            task_logits[current_key] if current_key in self.task_classifiers
            else self.classifier(h_prime)
        )

        # Step 5: Write controller — evaluate for storage.
        # This whole block is pure memory bookkeeping: the importance score is
        # only used (detached) to decide what to store, and never reaches the
        # loss — so the write controller receives no gradient regardless. Run
        # it under no_grad so we don't build a throwaway autograd graph over the
        # near-full bank every step (that graph was the MPS OOM leak).
        if self.use_memory and store_memories and self.training:
            with torch.no_grad():
                importance = self.write_controller(
                    c_prime, c_t, self.memory_bank
                )  # [batch]

                # Store entries above threshold
                mask = importance > self.write_threshold
                if mask.any():
                    self.memory_bank.write(
                        c_prime[mask].detach(),
                        importance[mask].detach(),
                        task_id=self.current_task_id,
                        gate_vectors=gate_t[mask],
                        logit_vectors=logits_current[mask],
                    )

                # Consolidate only when bank exceeds 90% capacity
                self.step_count += 1
                bank_utilization = self.memory_bank.size / self.memory_bank.capacity
                if (
                    self.step_count % self.consolidation_interval == 0
                    and bank_utilization > 0.9
                ):
                    self.consolidation.consolidate(self.memory_bank)

                # Importance decay (now bounded, see memory_bank.py)
                self.memory_bank.decay_importance()

        # Build output dict
        output = {
            "logits": logits,
            "cell_state": c_prime,
            "cell_state_original": c_t,
            # The deployed readout's actual argument. Exposed (not reconstructed)
            # so scripts/channel_decomp.py::features_and_logits reads the same
            # tensor the heads above just read — see assert_path_identity().
            "readout_feature": h_prime,
            "hidden": (h_n, c_n),
        }

        if return_diagnostics:
            diag = {
                "memory_bank": self.memory_bank.get_stats(),
                "composition": comp_diag,
            }
            if attn_weights is not None:
                diag["attention_weights"] = attn_weights
            output["diagnostics"] = diag

        return output

    def _forward_vit(
        self,
        x: torch.Tensor,
        task_hint: Optional[int],
        apply_adapter: bool,
    ) -> dict:
        """E12's deployed path: ViT -> post-norm CLS -> per-task head.

        Memory read, GGC composition and the learned output gate are BYPASSED —
        P3a registers the deployed feature as the post-norm CLS token, and
        wrapping it in o_t*tanh(c') would measure a hybrid whose readout is not
        a ViT's. See src/models/vit_base.py for the full reasoning.

        The adapter is applied to PATCH EMBEDDINGS (the registered placement),
        inside the encoder, via the same `_apply_adapter` the LSTM path uses so
        the two can never disagree about adapter geometry.

        Args:
            x: images [batch, 3, H, W]
            task_hint: which task's adapter and head to use
            apply_adapter: False suppresses the adapter through the deployed
                path rather than around it (catch 28's corollary)

        Returns:
            dict with logits, readout_feature (the tensor the head reads),
            cell_state (alias, for instrument compatibility), hidden=None
        """
        # TWO KEYS, DELIBERATELY SEPARATE. The adapter has always been routed by
        # task_hint — that is legacy behaviour and correct. The HEAD is routed by
        # hint only when the amendment is on; otherwise it falls back to the
        # current (latest) head, mirroring the recurrent path's semantics.
        #
        # Deriving both from task_hint made the flag inert: the fallback selected
        # head k anyway, so the amendment controlled nothing and P3b's positive
        # control could not fire. The control caught it on first use, which is
        # the entire reason a gate ships with a case it must fail.
        adapter_key = str(task_hint if task_hint is not None else self.current_task_id)
        head_key = (str(task_hint) if (self.head_routing_by_hint and task_hint is not None)
                    else str(self.current_task_id))

        adapter = None
        if self.use_input_adapters and apply_adapter and adapter_key in self.task_adapters:
            mod = self.task_adapters[adapter_key]
            adapter = lambda t: mod(t)          # per-token linear on [B, N, C]

        feature = self.lstm(x, adapter=adapter)          # [batch, embed_dim]

        if self.use_task_heads and head_key in self.task_classifiers:
            head = self.task_classifiers[head_key]
        else:
            head = self.classifier

        return {
            "logits": head(feature),
            "cell_state": feature,
            "cell_state_original": feature,
            "readout_feature": feature,
            "hidden": None,
        }

    def _forward_resnet(self, x: torch.Tensor, task_hint: Optional[int]) -> dict:
        """E14's deployed path: ResNet-50 -> global avg pool -> per-task head.

        Memory read, GGC composition and the learned output gate are BYPASSED —
        P3a registers the deployed feature as the post-pool, pre-fc 2048-d
        vector. Deliberately parallel to `_forward_vit`; see resnet_base.py for
        why the two are not refactored together.

        No adapter path: E14 registers no adapt arm (contract sec 1).
        """
        head_key = (str(task_hint) if (self.head_routing_by_hint and task_hint is not None)
                    else str(self.current_task_id))
        feature = self.lstm(x)                       # [batch, 2048]
        if self.use_task_heads and head_key in self.task_classifiers:
            head = self.task_classifiers[head_key]
        else:
            head = self.classifier
        return {
            "logits": head(feature),
            "cell_state": feature,
            "cell_state_original": feature,
            "readout_feature": feature,
            "hidden": None,
        }

    def functional_forward(
        self,
        x: torch.Tensor,
        adapter_key: Optional[str] = None,
    ) -> torch.Tensor:
        """The composed function (g_phi o f_theta o A_k)(x) — MAFC's encoder term.

        Deliberately BYPASSES the memory read/compose path: the anchor is defined
        on the parametric function only, so the target stays stationary while the
        bank changes during training. Gradients flow to A_k (if trainable), the
        LSTM (theta), the output gate, and the classifier (phi).

        Args:
            x: input batch [batch, seq_len, input_size]
            adapter_key: which task adapter to apply first; None = no adapter
                (the M4-LwF "raw probe input" arm)

        Returns:
            logits [batch, num_classes]
        """
        if adapter_key is not None:
            x = self._apply_adapter(x, adapter_key)
        lstm_out, (_, c_n) = self.lstm(x)
        c_t = c_n[-1]  # [batch, hidden]
        o_t = torch.sigmoid(
            self.output_gate(torch.cat([lstm_out[:, -1, :], c_t], dim=-1))
        )
        return self.classifier(o_t * torch.tanh(c_t))  # [batch, num_classes]

    def readout_from_state(
        self,
        cell_states: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """g_phi applied to STORED bank states — MAFC's bank-anchored readout term.

        Uses the bank's stored output gate so the readout matches how the state
        was originally read. Inputs are bank buffers (constants), so gradients
        reach ONLY the classifier phi — never theta. This is the structural
        reason the bank cannot constrain the encoder (prereg v2 §1.1).

        Args:
            cell_states: stored cell states [n, hidden]
            gates: stored output-gate vectors [n, hidden]

        Returns:
            logits [n, num_classes]
        """
        return self.classifier(gates * torch.tanh(cell_states))

    @torch.no_grad()
    def infer_task(self, x: torch.Tensor, mode: str = "self_consistency") -> torch.Tensor:
        """Infer each sample's task id via the trained router (task-free eval).

        mode:
          "self_consistency": apply EACH task's adapter, pick the adapter whose
              resulting cell state the router most confidently assigns to that
              adapter's OWN task — argmax_k softmax(router(c_k))[k]. Resolves the
              adapter/task circularity but costs K forward passes (one per adapter).
          "single_pass": apply only the current (latest) adapter and take the
              router's argmax. Cheap (1 pass); tests whether routing survives
              applying a possibly-mismatched adapter to old-task inputs.

        Returns: [batch] inferred task ids. Call with the model in eval().
        """
        keys = sorted(int(k) for k in self.task_adapters)
        if not keys:  # no adapters yet -> only the current task exists
            return torch.full(
                (x.shape[0],), self.current_task_id, dtype=torch.long, device=x.device
            )
        if mode == "single_pass":
            out = self(x, store_memories=False, task_hint=self.current_task_id)
            # restrict argmax to seen tasks
            return self.task_router(out["cell_state"])[:, : max(keys) + 1].argmax(dim=1)

        best_p = best_k = None
        for k in keys:
            out = self(x, store_memories=False, task_hint=k)
            p = torch.softmax(self.task_router(out["cell_state"]), dim=1)[:, k]  # own-task prob
            if best_p is None:
                best_p, best_k = p.clone(), torch.full_like(p, float(k))
            else:
                better = p > best_p
                best_p = torch.where(better, p, best_p)
                best_k = torch.where(better, torch.full_like(best_k, float(k)), best_k)
        return best_k.long()

    def get_memory_stats(self) -> dict:
        """Get current memory bank statistics."""
        return self.memory_bank.get_stats()

    def clear_memory(self) -> None:
        """Reset the memory bank and task tracking."""
        self.memory_bank.clear()
        self.step_count = 0
        self.num_tasks_seen = 0
        self.task_stats = {}
        self.task_classifiers = nn.ModuleDict()
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.stats_initialized.fill_(False)

    @classmethod
    def load_from_checkpoint(cls, checkpoint: dict) -> "PLCM":
        """Rebuild a PLCM from a trainer checkpoint dict.

        Task heads/adapters are created lazily by set_task, so a fresh model
        cannot load_state_dict a checkpoint that contains them. This replays
        set_task for every task up to the checkpoint's, then loads weights.

        E11/step 3 — `task_stats` is a plain dict of per-task cell-state moments,
        NOT a buffer, so it is absent from state_dict(). A model reloaded without
        it skips forward()'s coordinate-alignment block entirely (that block is
        gated on `self.task_stats` being non-empty) and therefore computes a
        DIFFERENT composed state from the model that was trained. Measured on
        E4/seed42: a bare reload missed its own accuracy matrix by up to 3.4pp,
        systematically high; restoring the stats reproduced all eight cells
        exactly. Checkpoints written after this change carry them; older ones are
        reconstructed by load_era() from the checkpoint chain.

        Args:
            checkpoint: dict with 'config', 'task_id', 'model_state', and
                (post-E11) 'task_stats'
        """
        model = cls.from_config(checkpoint["config"])
        for t in range(checkpoint["task_id"] + 1):
            model.set_task(t)
        model.load_state_dict(checkpoint["model_state"])
        stats = checkpoint.get("task_stats")
        if stats:
            model.task_stats = {int(k): (m.clone(), v.clone())
                                for k, (m, v) in stats.items()}
        return model

    @classmethod
    def load_era(cls, ckpt_dir, task: int, epoch: Optional[int] = None) -> "PLCM":
        """Load the end-of-task-`task` model AS IT WAS DEPLOYED, stats included.

        THE ERA MATTERS. A model at the end of task k held task_stats for tasks
        0..k-1 and no more. Restoring task 4's stats onto a task-1 checkpoint
        would leak future state into a ceiling measurement, which is a different
        error in the same family — so exactly `task` entries are restored, and
        the count is asserted rather than assumed.

        task_stats[k] is the running (mean, var) at the k -> k+1 boundary, which
        is precisely what the end-of-task-k checkpoint's buffers hold: set_task
        snapshots them before any task-(k+1) update. This is reconstruction from
        recorded state, not estimation.

        Args:
            ckpt_dir: directory of task{t}_epoch{e}.pt files
            task: which task's end-of-training model to load
            epoch: last epoch index; defaults to the config's epochs - 1
        """
        from pathlib import Path

        ckpt_dir = Path(ckpt_dir)

        def _read(t: int, e: Optional[int]):
            if e is None:
                cands = sorted(ckpt_dir.glob(f"task{t}_epoch*.pt"),
                               key=lambda p: int(p.stem.split("epoch")[1]))
                if not cands:
                    raise FileNotFoundError(f"no task{t}_epoch*.pt in {ckpt_dir}")
                p = cands[-1]
            else:
                p = ckpt_dir / f"task{t}_epoch{e}.pt"
            return torch.load(p, weights_only=True, map_location="cpu")

        ckpt = _read(task, epoch)
        model = cls.load_from_checkpoint(ckpt)
        # task_stats exists to feed coordinate alignment. When alignment is off
        # the dict is never populated during training either, so reconstructing
        # it would invent state the trained model never had — and asserting on
        # its length would fail every such checkpoint. E12's ViT arms are the
        # first: `_forward_vit` bypasses the alignment block entirely, so
        # `stats_initialized` stays False and `set_task` never snapshots.
        # Gated on use_coord_align, which is True for every pre-E12 arm, so this
        # branch cannot change how any existing checkpoint loads.
        if not getattr(model, "use_coord_align", True):
            return model
        if not model.task_stats:                      # pre-E11 checkpoint
            for t in range(task):
                sd = _read(t, epoch)["model_state"]
                model.task_stats[t] = (sd["running_mean"].clone(),
                                       sd["running_var"].clone())
        assert len(model.task_stats) == task, (
            f"{ckpt_dir}: era mismatch — task {task} model must carry {task} "
            f"task_stats entries, has {len(model.task_stats)}")
        return model

    @classmethod
    def from_config(cls, config: dict) -> "PLCM":
        """Build PLCM from a config dictionary.

        Args:
            config: Parsed YAML config with model, memory, composition sections

        Returns:
            Configured PLCM instance
        """
        model_cfg = config["model"]
        mem_cfg = config["memory"]
        comp_cfg = config["composition"]
        feat_cfg = config.get("plcm_features", {})  # feature toggles (default on)
        adapters_cfg = config.get("adapters", {})   # per-task input adapters
        router_cfg = config.get("router", {})       # task-free routing
        train_cfg = config.get("training", {})

        return cls(
            input_size=model_cfg["input_size"],
            hidden_size=model_cfg["hidden_size"],
            num_classes=model_cfg["num_classes"],
            num_layers=model_cfg["num_layers"],
            memory_capacity=mem_cfg["capacity"],
            key_dim=mem_cfg["key_dim"],
            top_k=mem_cfg["top_k"],
            num_heads=mem_cfg["num_heads"],
            composition_mode=comp_cfg["mode"],
            write_threshold=mem_cfg["write_threshold"],
            consolidation_interval=mem_cfg["consolidation_interval"],
            consolidation_clusters=mem_cfg["consolidation_clusters"],
            dropout=model_cfg.get("dropout", 0.0),
            use_gate_filter=feat_cfg.get("gate_filter", True),
            use_coord_align=feat_cfg.get("coord_align", True),
            use_context_heads=feat_cfg.get("context_heads", True),
            freeze_encoder_after_first_task=model_cfg.get(
                "freeze_encoder_after_first_task", False
            ),
            use_input_adapters=adapters_cfg.get("enabled", False),
            adapter_dim=adapters_cfg.get("dim", 784),
            adapter_rank=adapters_cfg.get("rank", 0),
            # "flat" | "per_step" — round-trips through checkpoints via
            # from_config, so a reloaded model rebuilds the same geometry.
            adapter_mode=adapters_cfg.get("mode", "flat"),
            backbone=model_cfg.get("backbone", "lstm"),
            mlp_input_dim=model_cfg.get("mlp_input_dim"),      # E23 arm B; absent in every earlier checkpoint config
            use_memory=mem_cfg.get("enabled", True),
            use_task_heads=model_cfg.get("use_task_heads", True),
            use_router=router_cfg.get("enabled", False),
            router_num_tasks=router_cfg.get("num_tasks", train_cfg.get("num_tasks", 5)),
            # E12's two amendments, both default-off so every pre-E12 config
            # round-trips to the pre-E12 behaviour unchanged.
            head_routing_by_hint=model_cfg.get("head_routing_by_hint", False),
            vit_model=model_cfg.get("vit_model",
                                    "vit_base_patch16_224.augreg2_in21k_ft_in1k"),
            vit_pretrained=model_cfg.get("vit_pretrained", True),
            resnet_model=model_cfg.get("resnet_model", None),
        )
