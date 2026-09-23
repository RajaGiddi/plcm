"""
Continual Learning Trainer.

Orchestrates sequential task training, evaluation, and metric collection.
Supports training PLCM, vanilla LSTM, and LSTM+EWC with the same
evaluation protocol for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import time
import os
import json
from typing import Optional

from .metrics import ContinualMetrics, evaluate_task, evaluate_task_free
from .ewc import EWC


class ContinualTrainer:
    """
    Trains models on sequential tasks and measures continual learning performance.

    Args:
        model: Model to train
        model_type: "plcm", "lstm", or "lstm_ewc"
        device: Compute device
        config: Training configuration dict
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: torch.device,
        config: dict,
    ):
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device
        self.config = config

        train_cfg = config["training"]
        self.epochs_per_task = train_cfg["epochs_per_task"]
        self.lr = train_cfg["learning_rate"]
        self.grad_clip = train_cfg["grad_clip"]
        self.num_tasks = train_cfg["num_tasks"]

        # E5d method v2 — adapter-first warmup (docs/E5D_prereg.md).
        # For tasks 1..T-1 only, the first `warmup_epochs` epochs train exactly
        # ONE component while everything else is frozen:
        #   "adapter" (v2-ON)      -> only the new task's input adapter
        #   "control" (v2-control) -> only the readout
        # Both freeze the encoder for the same number of epochs, so the arms are
        # matched on encoder training and differ ONLY in whether recruitment can
        # happen. Without that control, reduced encoder drift is indistinguishable
        # from adapter recruitment (prereg §3a).
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        self.warmup_mode = train_cfg.get("warmup_mode", "adapter")
        assert self.warmup_mode in ("adapter", "control"), \
            f"bad warmup_mode {self.warmup_mode!r}"

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # EWC (for lstm_ewc and the plcm_ewc hybrid)
        self.ewc: Optional[EWC] = None
        if model_type in ("lstm_ewc", "plcm_ewc"):
            if model_type == "plcm_ewc":
                # Selective EWC: constrain only the LSTM representation so keys
                # stay stable for retrieval, while PLCM components stay free.
                ewc_cfg = config.get("plcm_ewc", config.get("ewc", {}))
                self.ewc = EWC(
                    model, ewc_lambda=ewc_cfg.get("lambda", 200.0),
                    param_filter="lstm.",
                )
            else:
                ewc_cfg = config.get("ewc", {})
                self.ewc = EWC(model, ewc_lambda=ewc_cfg.get("lambda", 400.0))
            self.fisher_samples = ewc_cfg.get("fisher_samples", 200)

        # ---- LwF (Li & Hoiem 2017), arm `lwf_lh17` -------------------------
        # Contract: docs/E16_lwf_prereg.md. NOT `--mafc-arm lwf`, which is this
        # repo's H3 raw-probe ablation and a different arm entirely (catch 30
        # pre-loaded; the citation lives in the name so the collision cannot
        # re-form silently).
        lwf_cfg = config.get("lwf", {})
        self.lwf_enabled = bool(lwf_cfg.get("enabled", False))
        self.lwf_lambda = float(lwf_cfg.get("lambda", 0.0))
        self.lwf_T = float(lwf_cfg.get("temperature", 2.0))
        self.lwf_warmup_epochs = int(lwf_cfg.get("warmup_epochs", 0))
        self.lwf_teacher = None          # theta_{k-1}, re-snapshotted per boundary
        self._lwf_loss_sum = 0.0

        # Metrics tracker
        self.metrics = ContinualMetrics(self.num_tasks)

        # Logging
        self.log_dir = config.get("logging", {}).get("log_dir", "runs/")
        self.verbose = config.get("logging", {}).get("verbose", True)
        # Checkpointing: save the full model state_dict at every epoch end so
        # post-hoc analyses (drift maps, probes) can reconstruct f_theta_t. The
        # benchmark is fully seeded, so data is reconstructible from the config.
        self.save_checkpoints = config.get("logging", {}).get("save_checkpoints", False)
        # E12 storage plan. Separate from save_checkpoints (which writes EVERY
        # epoch): era_checkpoints writes ONE per task boundary, in fp16, and
        # audits it by loading. Default off, so no prior experiment changes.
        _log = config.get("logging", {})
        self.era_checkpoints = _log.get("era_checkpoints", False)
        self.checkpoint_fp16 = _log.get("checkpoint_fp16", False)
        # Smoke-only: also write the SAME boundary state at full precision into a
        # sibling dir, so the fp16 reload delta can be split into "rounding" and
        # "everything else serialization drops" (catch 29). Off in the sweep — it
        # doubles the 35GB.
        self.checkpoint_fp32_shadow = _log.get("checkpoint_fp32_shadow", False)
        self.checkpoint_dir = config.get("logging", {}).get("checkpoint_dir", "checkpoints/")
        # Both write task{k}_epoch{e}.pt into the same run dir, so at the last
        # epoch the fp16 era file would silently replace the fp32 per-epoch one
        # and every later analysis would read a precision it was not told about.
        assert not (self.save_checkpoints and self.era_checkpoints), (
            "save_checkpoints and era_checkpoints are both on: they collide on "
            "the same filename. Pick one.")
        self.history: list[dict] = []
        # Keys already registered as optimizer param groups (lazily-created task
        # heads / input adapters), to avoid adding the same param group twice.
        self._optimized_keys: set = set()
        # Task-free router: whether it replays old-task cell states from the bank
        # during training. This is the condition the load-bearing ablation toggles
        # (replay off -> the router should catastrophically forget old-task routing).
        self.router_replay = config.get("router", {}).get("replay", True)
        # All-adapter-view training: per batch the router also sees the input
        # through up to (max_views - 1) OTHER adapters, labeled with the true
        # task. Closes the self-consistency OOD blind spot (router must judge
        # wrong-adapter states at eval) and breaks recency gradient dominance.
        # Sampled (not all K) to bound the per-batch cost at max_views forwards.
        self.router_max_views = config.get("router", {}).get("max_views", 3)

        # ---- MAFC (prereg v2): two-term functional-consistency loss ----
        mafc_cfg = config.get("mafc", {})
        self.mafc_lambda = mafc_cfg.get("lambda", 0.0)
        self.mafc_temp = mafc_cfg.get("temperature", 2.0)
        # Arm switches (see prereg §4): full | enc-only (H4 control) |
        # raw-probe LwF (H3 control) | lambda=0 floor.
        self.mafc_encoder_term = mafc_cfg.get("encoder_term", True)
        self.mafc_readout_term = mafc_cfg.get("readout_term", True)
        self.mafc_adapter_probes = mafc_cfg.get("adapter_probes", True)
        self.mafc_bank_batch = mafc_cfg.get("bank_batch", 128)
        # Frozen end-of-task snapshots that generate p*_k on the fly.
        self.mafc_snapshots: dict = {}

        # ---- Experience Replay baseline (prereg docs/ER_prereg.md) ----
        er_cfg = config.get("er", {})
        self.er_enabled = er_cfg.get("enabled", False)
        self.er_buffer_per_task = er_cfg.get("buffer_per_task", 100)
        self.er_replay_batch = er_cfg.get("replay_batch", 128)
        self.er_weight = er_cfg.get("weight", 1.0)
        # Raw (x, y) rehearsal buffer — NOTE: unlike the adapter method, ER
        # stores raw data. Stated in the prereg as a regime asymmetry.
        self.er_buffer_x: list = []
        self.er_buffer_y: list = []

        # ---- E28: channel-permutation augmentation (docs/E28_prereg.md) ----
        # THE FIRST INPUT-SIDE INTERVENTION IN THIS TRAINER. ER, LwF, MAFC and the
        # router all attach to the LOSS; this one changes x. Everything below is
        # inert unless `aug.perm_p > 0`, and NOTHING here touches an RNG when the
        # flag is off -- a stray `torch.rand` in the default path would advance the
        # global stream and break the bit-identity the regression depends on.
        aug_cfg = config.get("aug", {})
        self.aug_perm_p = float(aug_cfg.get("perm_p", 0.0))
        self.aug_perm_seed = int(aug_cfg.get("perm_seed", 0))
        self.aug_heldout = [tuple(p) for p in aug_cfg.get("heldout", [])]
        self.aug_heldout_fingerprint = aug_cfg.get("heldout_fingerprint")
        self.aux_perm_weight = float(aug_cfg.get("aux_perm_weight", 0.0))
        self._aug_g = None
        self._aug_H = None
        self.aux_perm_head = None
        self._aux_loss_sum = 0.0
        self._aux_acc_sum = 0.0

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        test_loaders: list[DataLoader],
    ) -> dict:
        """Train on a single task and evaluate on all seen tasks.

        Args:
            task_id: Current task index
            train_loader: Training data for this task
            test_loaders: Test loaders for ALL tasks (for cross-eval)

        Returns:
            Task training results dict
        """
        if self.model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter", "mafc"):
            self.model.set_task(task_id)
            # Task heads (and, for plcm_adapter, input adapters) are created
            # lazily in set_task, so their params are NOT in the optimizer built
            # at __init__. Register each newly-created trainable module. Dedup via
            # _optimized_keys so a repeated set_task can't add a group twice.
            task_key = str(task_id)
            for attr in ("task_classifiers", "task_adapters"):
                registry = getattr(self.model, attr, {})
                if task_key not in registry:
                    continue
                add_key = f"{attr}:{task_key}"
                mod = registry[task_key]
                if add_key not in self._optimized_keys and any(
                    p.requires_grad for p in mod.parameters()
                ):
                    self.optimizer.add_param_group({"params": mod.parameters()})
                    self._optimized_keys.add(add_key)
            # Report the trainable fraction — after the Option A encoder freeze
            # (task 1+) this drops as the LSTM + output gate stop training.
            if self.verbose:
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"  Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")

        task_results = {
            "task_id": task_id,
            "epochs": [],
        }

        if self.warmup_epochs > 0 and task_id >= 1 and self.verbose:
            print(f"  E5d warmup: {self.warmup_epochs} epoch(s), mode={self.warmup_mode}")

        # Task 0's features are frozen after it (Option A), so give it extra
        # epochs — its representation quality is the ceiling for every task.
        epochs = (
            self.config["training"].get("epochs_first_task", self.epochs_per_task)
            if task_id == 0 else self.epochs_per_task
        )

        # E5d warmup bookkeeping — see _enter_warmup for the contract.
        saved_rg, warmup_active, in_warmup = None, False, False

        for epoch in range(epochs):
            in_warmup = (
                self.warmup_epochs > 0
                and task_id >= 1              # NEVER task 0: prereg §7 — an adapter
                and epoch < self.warmup_epochs  # fitted to a random encoder bakes
            )                                   # noise into the foundation
            if in_warmup and not warmup_active:
                saved_rg = self._enter_warmup(task_id)
                warmup_active = True
            elif warmup_active and not in_warmup:
                self._exit_warmup(saved_rg)
                warmup_active = False

            # Training epoch
            train_loss = self._train_epoch(train_loader, task_id)

            # Evaluate on all seen tasks
            eval_results = {}
            for eval_task_id in range(task_id + 1):
                acc = evaluate_task(
                    self.model,
                    test_loaders[eval_task_id],
                    self.device,
                    model_type="plcm" if self.model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter", "mafc") else "lstm",
                    task_id=eval_task_id,
                )
                eval_results[f"task_{eval_task_id}"] = acc

            epoch_result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval": eval_results,
            }

            # ARM-IDENTITY WITNESS for the EWC arms (Wave 1, precondition 2).
            # Verified from the run's own output, never from the launcher's
            # intent: a nonzero mean penalty in every epoch dict is what
            # distinguishes "EWC was configured" from "EWC actually acted".
            # Absent on task 0 by construction -- there is no prior Fisher yet.
            if self.ewc is not None:
                epoch_result["ewc_penalty_mean"] = float(
                    getattr(self, "_ewc_penalty_mean", 0.0))

            # ARM-IDENTITY WITNESS for lwf_lh17 (contract sec 4b): nonzero on
            # tasks >= 1, absent on task 0 by construction (no teacher yet).
            if self.lwf_enabled:
                epoch_result["lwf_distill_loss"] = float(
                    getattr(self, "_lwf_loss_mean", 0.0))

            # Add memory stats for PLCM
            if self.model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter", "mafc"):
                epoch_result["memory"] = self.model.get_memory_stats()
                # Current task-head weight norm — if this does not change across
                # epochs, the head is not actually training (guards the lazy-head
                # optimizer bug: warm-start values would look plausible but static).
                ck = str(task_id)
                tc = getattr(self.model, "task_classifiers", {})
                if ck in tc:
                    epoch_result["head_weight_norm"] = float(
                        tc[ck].weight.norm().item()
                    )
                # Adapter drift from identity — confirms A_k is learning the
                # inverse permutation (~0 for task 0, grows for later tasks).
                ta = getattr(self.model, "task_adapters", {})
                if ck in ta:
                    mod = ta[ck]
                    with torch.no_grad():
                        if hasattr(mod, "weight"):
                            # Full-rank nn.Linear: A - I directly.
                            A = mod.weight
                            eye = torch.eye(A.shape[0], device=A.device)
                            dist = (A - eye).norm()
                        else:
                            # Residual low-rank A = I + U V, so A - I = U V.
                            dist = (mod.U.weight @ mod.V.weight).norm()
                    epoch_result["adapter_dist_from_identity"] = float(dist.item())

            epoch_result["warmup"] = bool(in_warmup)
            task_results["epochs"].append(epoch_result)

            if self.save_checkpoints:
                run_dir = os.path.join(
                    self.checkpoint_dir,
                    f"{self.model_type}_seed{self.config.get('seed', 42)}",
                )
                os.makedirs(run_dir, exist_ok=True)
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "model_type": self.model_type,
                        "task_id": task_id,
                        "epoch": epoch,
                        "seed": self.config.get("seed", 42),
                        "config": self.config,
                        # E11/step 3: task_stats is a plain dict, so state_dict()
                        # does not carry it — and a model reloaded without it
                        # skips coordinate alignment and stops being the model
                        # that was trained. Saved explicitly so no later analysis
                        # has to reconstruct it (PLCM.load_era does that for the
                        # pre-E11 corpus).
                        "task_stats": {
                            int(k): (m.detach().cpu(), v.detach().cpu())
                            for k, (m, v) in getattr(self.model, "task_stats", {}).items()
                        },
                    },
                    os.path.join(run_dir, f"task{task_id}_epoch{epoch}.pt"),
                )

            if self.verbose:
                eval_str = " | ".join(
                    f"T{k.split('_')[1]}: {v:.3f}"
                    for k, v in eval_results.items()
                )
                print(
                    f"  Task {task_id} Epoch {epoch+1}/{epochs} "
                    f"— Loss: {train_loss:.4f} | {eval_str}"
                )

        # Safety net: if warmup_epochs >= epochs the loop can end mid-warmup, and
        # leaving the encoder frozen would silently turn every later task into a
        # different experiment.
        if warmup_active:
            self._exit_warmup(saved_rg)

        # Record final accuracies for this task in metrics matrix
        acc_boundary = None
        for eval_task_id in range(task_id + 1):
            final_acc = evaluate_task(
                self.model,
                test_loaders[eval_task_id],
                self.device,
                model_type="plcm" if self.model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter", "mafc") else "lstm",
                task_id=eval_task_id,
            )
            self.metrics.record(task_id, eval_task_id, final_acc)
            if eval_task_id == task_id:
                # The task's own accuracy at its own boundary — acc_ceiling,
                # computed in-process at full precision. Authoritative.
                acc_boundary = final_acc

        if self.era_checkpoints and acc_boundary is not None:
            task_results["era_checkpoint"] = self._save_era_checkpoint(
                task_id, train_loader, test_loaders[task_id], epochs - 1,
                acc_boundary)

        # Also evaluate on future tasks (for forward transfer)
        for eval_task_id in range(task_id + 1, self.num_tasks):
            fwd_acc = evaluate_task(
                self.model,
                test_loaders[eval_task_id],
                self.device,
                model_type="plcm" if self.model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter", "mafc") else "lstm",
                task_id=eval_task_id,
            )
            self.metrics.record(task_id, eval_task_id, fwd_acc)

        # ER: buffer this task's examples so later tasks can rehearse them.
        if self.er_enabled:
            self._fill_er_buffer(train_loader, task_id)

        # MAFC: freeze this task's snapshot so it can anchor all later tasks.
        if self.mafc_lambda > 0 and self.mafc_encoder_term:
            self._take_mafc_snapshot(task_id)

        # LwF: re-record the teacher at EVERY boundary (Y_o per boundary, not
        # cached from task 0). theta_{k-1} is the model as it stands now, frozen.
        if self.lwf_enabled:
            self._take_lwf_teacher()

        # EWC: compute Fisher after finishing this task
        if self.ewc is not None:
            base_type = "plcm" if self.model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter", "mafc") else "lstm"
            self.ewc.compute_fisher(
                train_loader,
                self.device,
                num_samples=self.fisher_samples,
                model_type=base_type,
            )

        self.history.append(task_results)
        return task_results

    # ---- E5d method v2: adapter-first warmup -------------------------------
    #
    # The diagnosis (E5 + E5b + e5diag): HAR's forgetting is real and
    # shift-caused, the adapter CAN represent the exact inverse for tasks 2/3,
    # and it never moved (travel 0.1042, residual 0.9927 — no better than the
    # identity). Proposed cause: encoder accommodation is the locally cheaper
    # gradient path, so the adapter is never recruited.
    #
    # The fix makes the adapter the ONLY loss-reducing path for W epochs at each
    # task boundary. The control arm runs the identical freeze schedule with the
    # adapter frozen instead, so the two arms are matched on encoder training and
    # differ only in whether recruitment is possible.

    READOUT_PREFIXES = ("classifier.", "output_gate.", "task_classifiers.")

    def _enter_warmup(self, task_id: int) -> dict:
        """Freeze everything except one component; return the prior grad state.

        v2-ON ("adapter"): only `task_adapters.<task_id>` trains. The shared
            classifier is frozen too — a trainable readout is a bypass that
            would partially defeat the design and blur every downstream number
            (prereg §7, catch 4).
        v2-control ("control"): only the readout trains; the adapter is frozen.
            Same encoder-epoch count, recruitment impossible.

        Args:
            task_id: the incoming task, whose adapter is the warmup target

        Returns:
            {param_name: requires_grad} captured before freezing
        """
        saved = {n: p.requires_grad for n, p in self.model.named_parameters()}
        adapter_prefix = f"task_adapters.{task_id}."
        for n, p in self.model.named_parameters():
            if self.warmup_mode == "adapter":
                keep = n.startswith(adapter_prefix)
            else:
                keep = n.startswith(self.READOUT_PREFIXES)
            # Never RESURRECT a parameter the arm had already frozen (e.g. an
            # old task's adapter, or the encoder under Option A) — warmup may
            # only ever restrict the trainable set, never widen it.
            p.requires_grad_(keep and saved[n])
        return saved

    def _exit_warmup(self, saved: dict) -> None:
        """Restore the exact grad state captured by _enter_warmup."""
        for n, p in self.model.named_parameters():
            if n in saved:
                p.requires_grad_(saved[n])

    def _train_epoch(self, data_loader: DataLoader, task_id: int) -> float:
        """Run one training epoch.

        Args:
            data_loader: Training data
            task_id: Current task id

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        # Wave-1 fix 1. The EWC penalty was already computed and added to the
        # loss; it was never RECORDED, so "this arm is EWC" had no witness in
        # the run's own artifacts and the contract's arm-identity precondition
        # could not be satisfied. Accumulate-and-report only -- nothing here
        # enters the loss or the graph (`.item()` on an already-built scalar).
        self._ewc_penalty_sum = 0.0
        self._lwf_loss_sum = 0.0

        # Live per-batch progress bar (tqdm writes to stderr and flushes as it
        # goes, so progress is visible even when stdout is redirected). Shows a
        # running average loss. Falls back to the plain loader when not verbose.
        iterator = data_loader
        if self.verbose:
            iterator = tqdm(
                data_loader,
                desc=f"  Task {task_id} train",
                leave=False,
                unit="batch",
            )

        for x, y in iterator:
            x, y = x.to(self.device), y.to(self.device)

            # E28 sec 4: the ONLY input-side intervention in this loop. Gated on
            # perm_p > 0 so the default path neither permutes nor draws from any
            # RNG -- the flag-off bit-identity regression depends on that.
            perm_targets = None
            if self.aug_perm_p > 0:
                x, perm_targets = self._augment_perms(x)

            self.optimizer.zero_grad()

            # Forward pass
            if self.model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter", "mafc"):
                output = self.model(x, store_memories=True)
                logits = output["logits"]
            else:
                logits, _ = self.model(x)

            # Task loss
            loss = self.criterion(logits, y)

            # EWC penalty
            if self.ewc is not None:
                ewc_penalty = self.ewc.penalty()
                loss = loss + ewc_penalty
                self._ewc_penalty_sum += float(ewc_penalty.item())

            # LwF distillation against theta_{k-1} on the CURRENT task's data.
            #
            # Computed even when lambda == 0, DELIBERATELY: the lambda=0 control
            # must exercise the teacher path, or it degenerates into "run the
            # vanilla arm" and becomes tautological (catch 25). With the term
            # computed and scaled by zero, the control still tests (a) that the
            # teacher's forward perturbs nothing, and (b) that the term is ADDED
            # rather than multiplied into the loss.
            if self.lwf_teacher is not None:
                lwf_term = self._lwf_loss(x, logits)
                loss = loss + self.lwf_lambda * lwf_term
                self._lwf_loss_sum += float(lwf_term.item())

            # MAFC functional-consistency penalty (prereg v2 §1)
            if self.mafc_lambda > 0:
                loss = loss + self.mafc_lambda * self._mafc_loss(x)

            # Experience Replay rehearsal loss (prereg docs/ER_prereg.md)
            if self.er_enabled and self.er_buffer_x:
                loss = loss + self.er_weight * self._er_loss()

            # E28 B2: the auxiliary head that stops the encoder discarding channel
            # identity. Reads the DEPLOYED readout feature, so what it constrains
            # is the tensor the classifier sees.
            if self.aux_perm_weight > 0 and perm_targets is not None:
                aux, aux_acc = self._aux_perm_loss(output["readout_feature"], perm_targets)
                loss = loss + self.aux_perm_weight * aux
                self._aux_loss_sum += float(aux.item()); self._aux_acc_sum += aux_acc

            # Task-free router: train it on cell states -> task id, balanced across
            # seen tasks via bank replay. Cell state is detached, so this only
            # trains the router head, not the (frozen) encoder or the classifier.
            if getattr(self.model, "use_router", False) and self.model_type in (
                "plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter"
            ):
                loss = loss + self._router_loss(x, output["cell_state"].detach())

            # Backward
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if self.verbose:
                iterator.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        self._ewc_penalty_mean = self._ewc_penalty_sum / max(num_batches, 1)
        self._lwf_loss_mean = self._lwf_loss_sum / max(num_batches, 1)
        return total_loss / max(num_batches, 1)

    def arm_provenance(self) -> dict:
        """What this run's arm ACTUALLY is, read off the built model.

        Catch 21 in its launcher form. `configs/e12_vit.yaml` sets
        `use_task_heads: true` and `adapters.enabled: false`, and
        `scripts/train.py` overwrites both from CLI flags — so the config field
        cited as evidence for "this is the base arm" described a model that
        never ran. The audit covered artifacts and the premise rule covered
        contracts; nothing covered the object in memory. Now every run carries
        it, and "which arm was this?" is answered from the result file rather
        than from the launch command that produced it.
        """
        m = self.model
        return {
            "model_type": self.model_type,
            "backbone": getattr(m, "backbone", "lstm"),
            "use_task_heads": bool(getattr(m, "use_task_heads", False)),
            "head_routing_by_hint": bool(getattr(m, "head_routing_by_hint", False)),
            "use_input_adapters": bool(getattr(m, "use_input_adapters", False)),
            "adapter_mode": getattr(m, "adapter_mode", None),
            "use_memory": bool(getattr(m, "use_memory", True)),
            "num_classes": int(getattr(m, "num_classes", 0)),
            # Wave-1 fix 2. Without these, an artifact can prove "this is an
            # EWC model" but not "this is EWC at lambda=200" -- one level
            # coarser than the cited configuration, which is the two-vanillas
            # problem wearing a lambda.
            # W2 diagnosis. The floor rule always said (seed x config x
            # platform x BLAS); the plain_lstm pair failing 0/15 while the mafc
            # pair passed 15/15 on the same path showed that "platform"
            # includes THREADING. Recorded so every future floor pair carries
            # its thread config as part of the tuple it is a property of.
            "torch_num_threads": int(__import__("torch").get_num_threads()),
            "omp_num_threads": __import__("os").environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": __import__("os").environ.get("MKL_NUM_THREADS"),
            "lwf_enabled": bool(getattr(self, "lwf_enabled", False)),
            "lwf_lambda": (float(self.lwf_lambda)
                           if getattr(self, "lwf_enabled", False) else None),
            "lwf_temperature": (float(self.lwf_T)
                                if getattr(self, "lwf_enabled", False) else None),
            "lwf_warmup_epochs": (int(self.lwf_warmup_epochs)
                                  if getattr(self, "lwf_enabled", False) else None),
            "ewc_lambda": (float(self.ewc.ewc_lambda)
                           if self.ewc is not None else None),
            "ewc_param_filter": (getattr(self.ewc, "param_filter", None)
                                 if self.ewc is not None else None),
            "mafc_lambda": float(getattr(self, "mafc_lambda", 0.0)),
            # E20-B: the E5d warmup arms (v2-ON / v2-control) differ from v1-ON in
            # nothing the arm dict recorded. The per-epoch `warmup` flag and the
            # warmup-epoch `adapter_dist_from_identity` (0.0 for control) were
            # the witnesses; now the configuration is recorded too.
            "warmup_epochs": int(getattr(self, "warmup_epochs", 0)),
            "warmup_mode": (getattr(self, "warmup_mode", None)
                            if int(getattr(self, "warmup_epochs", 0)) > 0 else None),
            "n_task_heads": len(getattr(m, "task_classifiers", {})),
            "n_task_adapters": len(getattr(m, "task_adapters", {})),
            "num_tasks": self.num_tasks,
            "epochs_per_task": self.epochs_per_task,
            "learning_rate": self.lr,
            "seed": self.config.get("seed", 42),
            "benchmark": self.config.get("benchmark", {}).get("name"),
            # E18 / S72 rule: a construction is an arm-identity field.
            "disjoint_content": bool(self.config.get("benchmark", {}).get("disjoint_content", False)),
            "content_fingerprint": self.config.get("benchmark", {}).get("content_fingerprint"),
            # E23-B (2026-09-20): `content_fingerprint` is BLIND to the chunk boundaries --
            # T=5 and T=20 collide on it (src/data/permuted_mnist.py). The construction hash
            # and the chunk count are what separate them, and a field that is set in the config
            # but not copied here never reaches the artifact, which is how the first E23-B
            # launch produced runs whose own record could not say which construction they were.
            "construction_fingerprint": self.config.get("benchmark", {}).get("construction_fingerprint"),
            "content_chunks": self.config.get("benchmark", {}).get("content_chunks"),
            "angles": self.config.get("benchmark", {}).get("angles"),
            "remap_labels": self.config.get("benchmark", {}).get("remap_labels", True),
            "era_checkpoints": bool(self.era_checkpoints),
            "checkpoint_fp16": bool(self.checkpoint_fp16),
            # E28 (2026-09-22). An augmented run and a baseline run differ in
            # NOTHING the 28 previous fields record, which is exactly how E23-B
            # lost `content_chunks`. `batch_size` joins them because it was never
            # recorded either and it is what sets ER's replay ratio (E25).
            "batch_size": int(self.config.get("training", {}).get("batch_size", 0)),
            "aug_family": ("channel_permutation" if self.aug_perm_p > 0 else None),
            "aug_space": ("standardized" if self.aug_perm_p > 0 else None),
            "aug_p": float(self.aug_perm_p),
            "aug_seed": (int(self.aug_perm_seed) if self.aug_perm_p > 0 else None),
            "aug_heldout_n": len(self.aug_heldout),
            "aug_heldout_fingerprint": self.aug_heldout_fingerprint,
            "aux_perm_weight": float(self.aux_perm_weight),
        }

    @torch.no_grad()
    def _deployed_features(self, model, loader: DataLoader, task_id: int,
                           cap: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(feature, label) off the DEPLOYED path — the tensor the head reads.

        The same forward the accuracy matrix goes through, so the boundary refit
        and the post-hoc decomposition read the same object. Rebuilding the
        readout outside forward() is what catch 28 was.
        """
        was_training = model.training
        model.eval()
        feats, ys, n = [], [], 0
        for x, y in loader:
            out = model(x.to(self.device), store_memories=False, task_hint=task_id)
            feats.append(out["readout_feature"].float().cpu())
            ys.append(y.cpu())
            n += y.shape[0]
            if n >= cap:
                break
        if was_training:
            model.train()
        return torch.cat(feats)[:cap], torch.cat(ys)[:cap]

    def _refit_ceiling(self, model, task_id: int, train_loader: DataLoader,
                       test_loader: DataLoader) -> float:
        """acc_refit_ceiling(k) — the convex probe on task k's own end-of-task model.

        Recipe and sample caps are IMPORTED from the analysis instrument rather
        than re-declared, so the belt-and-braces value and the one the
        decomposition computes later differ in nothing but when they were taken.
        """
        from scripts.channel_decomp import N_TEST, N_TRAIN, refit_probe

        f_tr, y_tr = self._deployed_features(model, train_loader, task_id, N_TRAIN)
        f_te, y_te = self._deployed_features(model, test_loader, task_id, N_TEST)
        n_classes = int(getattr(model, "num_classes", 0)) or int(y_tr.max()) + 1
        return refit_probe(f_tr, y_tr, f_te, y_te, n_classes)

    def _p3_gate(self, model, loader: DataLoader, task_id: int, label: str) -> dict:
        """P3a/P3b on the deployed path (docs/E12_prereg.md sec 2), at the boundary.

        Hard failure, by design. With per-task heads off, P3b raises "no head
        exists for task k" — which is the arm-identity check the launcher needs:
        a run labelled task-IL that routes all 20 tasks through one shared head
        stops at the first boundary instead of finishing and being mislabelled.

        Skipped, with the reason recorded, where the asserts do not apply:
        class-IL's shared readout claims no head-per-task, and `assert_p3`'s
        "the last hooked module is the readout" holds on the ViT path, which
        calls exactly one head, not on the recurrent path, which evaluates all
        of them (the LSTM arms are gated by channel_decomp.assert_path_identity
        in their own analyses instead).
        """
        if getattr(model, "backbone", "lstm") != "vit":
            return {"skipped": f"backbone={getattr(model, 'backbone', 'lstm')} — "
                               "P3 is the ViT readout's gate"}
        if not getattr(model, "head_routing_by_hint", False):
            return {"skipped": "head_routing_by_hint off — no per-task readout claimed"}
        from scripts.e12_p3 import assert_p3

        x = next(iter(loader))[0][:32].to(self.device)
        was_training = model.training
        model.eval()
        try:
            return assert_p3(model, x, task_id, label=label)
        finally:
            if was_training:
                model.train()

    def _save_era_checkpoint(self, task_id: int, train_loader: DataLoader,
                             test_loader: DataLoader, epoch_idx: int,
                             acc_boundary: float) -> dict:
        """E12's storage plan: ONE fp16 checkpoint per task, audited by loading.

        Contract: docs/E12_prereg.md sec 3. Four rules, all of them reactions to
        things that have already gone wrong in this program:

          * fp16, and only at TASK boundaries. Per-epoch fp32 would be 100
            checkpoints x 173MB x 10 runs = ~170GB; era-only fp16 is ~35GB.
          * AUDITED BY LOADING, at save time. "A checkpoint exists" is not "a
            checkpoint loads" — 14 of 33 dirs were unusable when that was last
            checked, discovered weeks later.
          * BELT-AND-BRACES: acc_ceiling AND acc_refit_ceiling are both computed
            here, in-process, at full precision. Both are authoritative; the
            reload is the reproduction path, and its disagreement with them is
            MEASURED, never assumed negligible. A reloaded checkpoint is not the
            model that was trained until proven (catch 29), and fp16 rounding is
            exactly the kind of difference that assumption hides.
          * The deployed-path gate runs on BOTH models. A ceiling and its reload
            are different objects and either can be hooked wrong.

        Returns the boundary values, the reload values, their deltas and the
        file's size on disk — in the results json, not in a log line.
        """
        run_dir = os.path.join(
            self.checkpoint_dir,
            f"{self.model_type}_seed{self.config.get('seed', 42)}")
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"task{task_id}_epoch{epoch_idx}.pt")

        # ---- boundary-time, in-process, full precision. AUTHORITATIVE. --------
        p3_live = self._p3_gate(self.model, test_loader, task_id, f"live/task{task_id}")
        acc_refit_boundary = self._refit_ceiling(
            self.model, task_id, train_loader, test_loader)

        state32 = self.model.state_dict()
        state = ({k: (v.half() if v.is_floating_point() else v)
                  for k, v in state32.items()} if self.checkpoint_fp16 else state32)
        payload = {
            "model_type": self.model_type,
            "task_id": task_id,
            "epoch": epoch_idx,
            "seed": self.config.get("seed", 42),
            "config": self.config,
            "task_stats": {int(k): (m.detach().cpu(), v.detach().cpu())
                           for k, (m, v) in getattr(self.model, "task_stats", {}).items()},
            "acc_boundary": float(acc_boundary),
            "acc_refit_ceiling_boundary": float(acc_refit_boundary),
        }
        # E28 defect: the auxiliary permutation head lives on the TRAINER, not on
        # the model, so `model.state_dict()` never held it and E28's third
        # contracted number -- "the auxiliary head's accuracy on held-out swaps" --
        # was unmeasurable after the fact. Save it beside the model state.
        # ADDED ONLY WHEN IT EXISTS: an unconditional key would change the payload
        # of every flag-off checkpoint too, spending the bit-identity that admits
        # the existing S72 OFF checkpoints as B0 (E28 sec 4).
        if self.aux_perm_head is not None:
            payload["aux_perm_head_state"] = {
                k: v.detach().cpu() for k, v in self.aux_perm_head.state_dict().items()}
            payload["aux_perm_weight"] = float(self.aux_perm_weight)
        torch.save({**payload, "model_state": state,
                    "fp16": bool(self.checkpoint_fp16)}, path)

        rec = {"task": task_id, "path": path, "fp16": bool(self.checkpoint_fp16),
               "bytes": os.path.getsize(path),
               "acc_boundary": float(acc_boundary),
               "acc_refit_ceiling_boundary": float(acc_refit_boundary),
               "p3_live": p3_live, "loads": False,
               "acc_reload": None, "acc_refit_ceiling_reload": None,
               "delta": None, "delta_refit": None}
        try:
            from src.models.plcm import PLCM
            reloaded = PLCM.load_era(run_dir, task_id, epoch=epoch_idx)
            reloaded.to(self.device).eval()
            rec["loads"] = True
            rec["p3_reload"] = self._p3_gate(
                reloaded, test_loader, task_id, f"reload/task{task_id}")
            acc_reload = evaluate_task(
                reloaded, test_loader, self.device,
                model_type="plcm", task_id=task_id)
            rec["acc_reload"] = float(acc_reload)
            rec["delta"] = float(acc_reload - acc_boundary)
            rec["acc_refit_ceiling_reload"] = float(self._refit_ceiling(
                reloaded, task_id, train_loader, test_loader))
            rec["delta_refit"] = float(
                rec["acc_refit_ceiling_reload"] - acc_refit_boundary)
            del reloaded
        except Exception as e:                     # loud, not silent
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            print(f"    ERA CHECKPOINT AUDIT FAILED task {task_id}: {rec['error']}")

        # ---- full-precision shadow: the control that splits the delta ---------
        # An fp16 reload that misses the boundary could be rounding OR something
        # state_dict() drops. Only the fp32 save of the same state separates
        # them, and it must land at exactly 0.0000.
        if self.checkpoint_fp32_shadow:
            sdir = run_dir + "_fp32"
            os.makedirs(sdir, exist_ok=True)
            spath = os.path.join(sdir, f"task{task_id}_epoch{epoch_idx}.pt")
            torch.save({**payload, "model_state": state32, "fp16": False}, spath)
            rec["fp32_shadow"] = {"path": spath, "bytes": os.path.getsize(spath)}
            try:
                from src.models.plcm import PLCM
                m32 = PLCM.load_era(sdir, task_id, epoch=epoch_idx)
                m32.to(self.device).eval()
                a32 = evaluate_task(m32, test_loader, self.device,
                                    model_type="plcm", task_id=task_id)
                rec["fp32_shadow"]["acc_reload"] = float(a32)
                rec["fp32_shadow"]["delta"] = float(a32 - acc_boundary)
                rec["fp32_shadow"]["acc_refit_ceiling_reload"] = float(
                    self._refit_ceiling(m32, task_id, train_loader, test_loader))
                rec["fp32_shadow"]["delta_refit"] = float(
                    rec["fp32_shadow"]["acc_refit_ceiling_reload"] - acc_refit_boundary)
                del m32
            except Exception as e:
                rec["fp32_shadow"]["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                print(f"    FP32 SHADOW AUDIT FAILED task {task_id}: "
                      f"{rec['fp32_shadow']['error']}")
        # `loads` is set the moment load_era returns; the audit can still raise
        # AFTER that (device mismatch, refit failure), leaving acc_reload et al.
        # None. Printing them then raised TypeError and took the training run
        # down with it -- the audit's "loud, not silent" branch turned fatal one
        # line later. The record already carries `error`; the summary line is
        # for the success path only. Found by a local smoke of the S72 wave.
        if rec["loads"] and "error" not in rec:
            print(f"    era ckpt task {task_id} "
                  f"({rec['bytes'] / 1e6:.0f}MB{', fp16' if self.checkpoint_fp16 else ''})"
                  f" | ceiling {acc_boundary:.4f} -> {rec['acc_reload']:.4f} "
                  f"({rec['delta']:+.4f}) | refit-ceiling "
                  f"{acc_refit_boundary:.4f} -> {rec['acc_refit_ceiling_reload']:.4f} "
                  f"({rec['delta_refit']:+.4f})")
        return rec

    def _augment_perms(self, x: torch.Tensor):
        """E28 sec 1: per-sample channel permutation in STANDARDIZED space.

        `x` is [B, T, 9] and has already been standardized with task-0 statistics
        and mapped into the task's frame by `har_subject._apply` (:126-129), so a
        permutation of its channel axis is a swap in standardized space -- the
        space the contract specifies.

        Convention matches `channel_affine`'s P: output channel i holds input
        channel perm[i], so `x[..., perm]`.

        The returned target is the AUGMENTATION permutation alone, relative to
        the task's own frame. A target relative to the raw sensor would compose
        with `PERM` and hand the auxiliary head task identity on exactly the two
        tasks that carry a permutation (sec 2).
        """
        B, d = x.shape[0], x.shape[-1]
        if self._aug_g is None:
            self._aug_g = torch.Generator(device="cpu").manual_seed(self.aug_perm_seed)
            self._aug_H = (torch.tensor(self.aug_heldout, dtype=torch.long)
                           if self.aug_heldout else None)
        perms = torch.rand(B, d, generator=self._aug_g).argsort(dim=1)
        keep = torch.rand(B, generator=self._aug_g) >= self.aug_perm_p
        perms[keep] = torch.arange(d)                      # identity where unaugmented
        if self._aug_H is not None and self._aug_H.numel():
            # REJECTION SAMPLING, then the assert. The contract says the support is
            # "all 9! - |H| permutations not in H, sampled uniformly by rejection",
            # and the first implementation asserted without rejecting. With |H| = 20
            # of 362,880 and ~90k draws per run, about five hits are EXPECTED, so the
            # assert fired on a legitimate draw and killed six runs at task 2. The
            # assert stays -- it is the guard that caught this -- but it now sits
            # after the resample, where it can only fire if rejection failed.
            # Rows held at identity are never resampled: identity is excluded from H
            # by construction, so it is never flagged, and p's semantics are intact.
            for _ in range(64):
                bad = (perms[:, None, :] == self._aug_H[None, :, :]).all(-1).any(1)
                if not bool(bad.any()):
                    break
                n_bad = int(bad.sum())
                perms[bad] = torch.rand(n_bad, d, generator=self._aug_g).argsort(dim=1)
            hit = (perms[:, None, :] == self._aug_H[None, :, :]).all(-1).any()
            assert not bool(hit), "rejection failed to clear a held-out permutation"
        idx = perms.to(x.device)[:, None, :].expand(-1, x.shape[1], -1)
        return torch.gather(x, 2, idx), perms.to(x.device)

    def _aux_perm_loss(self, feature: torch.Tensor, targets: torch.Tensor):
        """Nine 9-way classifications from the deployed readout feature."""
        d = targets.shape[1]
        if self.aux_perm_head is None:
            self.aux_perm_head = torch.nn.Linear(feature.shape[1], d * d).to(self.device)
            self.optimizer.add_param_group({"params": self.aux_perm_head.parameters()})
        logits = self.aux_perm_head(feature).view(-1, d, d)
        loss = F.cross_entropy(logits.reshape(-1, d), targets.reshape(-1))
        acc = float((logits.argmax(-1) == targets).float().mean().item())
        return loss, acc

    def _fill_er_buffer(self, train_loader: DataLoader, task_id: int) -> None:
        """Store `buffer_per_task` raw examples from the finished task.

        Uniform sample taken at task end (prereg §2). Slightly favours ER over a
        reservoir — biases against our own method, the safe direction (R3).
        """
        xs, ys = [], []
        need = self.er_buffer_per_task
        for x, y in train_loader:
            take = min(need - sum(t.shape[0] for t in xs), x.shape[0])
            if take <= 0:
                break
            xs.append(x[:take].cpu())
            ys.append(y[:take].cpu())
            if sum(t.shape[0] for t in xs) >= need:
                break
        if xs:
            self.er_buffer_x.append(torch.cat(xs))
            self.er_buffer_y.append(torch.cat(ys))

    def _er_loss(self) -> torch.Tensor:
        """Cross-entropy on a replayed batch drawn uniformly over buffered tasks."""
        if not self.er_buffer_x:
            return torch.zeros((), device=self.device)
        allx = torch.cat(self.er_buffer_x)
        ally = torch.cat(self.er_buffer_y)
        m = min(self.er_replay_batch, allx.shape[0])
        idx = torch.randint(0, allx.shape[0], (m,))
        x, y = allx[idx].to(self.device), ally[idx].to(self.device)
        if self.model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter", "mafc"):
            logits = self.model(x, store_memories=False)["logits"]
        else:
            logits, _ = self.model(x)
        return self.criterion(logits, y)

    def _take_lwf_teacher(self) -> None:
        """Freeze theta_{k-1}. Re-taken at every boundary, never cached."""
        self.lwf_teacher = copy.deepcopy(self.model).to(self.device).eval()
        for p in self.lwf_teacher.parameters():
            p.requires_grad_(False)

    def _lwf_logits(self, model, x):
        """Logits from either backbone family, matching the deployed path."""
        if self.model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter",
                               "mafc"):
            return model(x, store_memories=False)["logits"]
        out, _ = model(x)
        return out

    def _lwf_loss(self, x, student_logits):
        """LwF's modified cross-entropy: T^2 * KL(teacher || student) at temp T.

        FORWARD KL, teacher first. This program has already shipped a reversed
        KL once; the direction is stated here and asserted by the unit test
        rather than left to the reader. `F.kl_div(input=log q, target=p)`
        computes sum p log(p/q) = KL(p || q), so passing the STUDENT as `input`
        (log-probs) and the TEACHER as `target` gives KL(teacher || student).

        The T^2 factor is Hinton's: distillation gradients scale as 1/T^2, so
        without it the term's weight would change with temperature.

        SEPARATE RNG. The teacher's forward runs inside `fork_rng`, so it cannot
        consume draws from the global generator and shift the student's
        trajectory. Without this the lambda=0 control would fail for an entirely
        benign reason and be waved through as "expected nondeterminism" -- the
        caveat is part of the control, not a footnote to it.
        """
        T = self.lwf_T
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            teacher_logits = self._lwf_logits(self.lwf_teacher, x)
        p_teacher = F.softmax(teacher_logits / T, dim=1)
        log_q_student = F.log_softmax(student_logits / T, dim=1)
        return (T * T) * F.kl_div(log_q_student, p_teacher, reduction="batchmean")

    def _take_mafc_snapshot(self, task_id: int) -> None:
        """Freeze an end-of-task-k copy that generates p*_k on the fly.

        Stores (theta_k, phi_k) together with the frozen adapter A_k, so
        snapshot.functional_forward(x, adapter_key=k) is exactly
        (g_phi(k) o f_theta(k) o A_k)(x). Parameter snapshots are standard in CL
        (LwF/EWC) and store no raw data (prereg §1, R2).
        """
        snap = copy.deepcopy(self.model)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self.mafc_snapshots[task_id] = snap

    def _mafc_loss(self, x: torch.Tensor) -> torch.Tensor:
        """L_enc (adapter-anchored, constrains theta+phi) + L_read (bank-anchored,
        constrains phi only). Forward KL at temperature T, scaled by T^2 so the
        soft-target gradient magnitude is comparable to the task loss.
        """
        T = self.mafc_temp
        device = x.device
        terms = []

        # --- Encoder term: probe inputs through OLD adapters, snapshot targets ---
        if self.mafc_encoder_term and self.mafc_snapshots:
            keys = sorted(self.mafc_snapshots)
            enc = x.new_zeros(())
            for k in keys:
                akey = str(k) if self.mafc_adapter_probes else None
                with torch.no_grad():
                    teacher = self.mafc_snapshots[k].functional_forward(x, adapter_key=akey)
                student = self.model.functional_forward(x, adapter_key=akey)
                enc = enc + nn.functional.kl_div(
                    nn.functional.log_softmax(student / T, dim=-1),
                    nn.functional.softmax(teacher / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
            terms.append(enc / len(keys))

        # --- Readout term: the bank's own committed (state, gate, logits) ---
        if self.mafc_readout_term:
            bank = self.model.memory_bank
            n = bank.size
            if n > 0:
                # task_ids >= 0 excludes slots freed by rebalance() (CLAUDE.md gotcha).
                valid = (bank.task_ids[:n] >= 0).nonzero(as_tuple=True)[0]
                if len(valid) > 0:
                    m = min(len(valid), self.mafc_bank_batch)
                    sel = valid[torch.randint(0, len(valid), (m,), device=device)]
                    pred = self.model.readout_from_state(
                        bank.values[sel], bank.gate_values[sel]
                    )
                    terms.append(nn.functional.kl_div(
                        nn.functional.log_softmax(pred / T, dim=-1),
                        nn.functional.softmax(bank.logits[sel] / T, dim=-1),
                        reduction="batchmean",
                    ) * (T * T))

        if not terms:
            return x.new_zeros(())
        return sum(terms)

    def _router_loss(self, x: torch.Tensor, live_state: torch.Tensor) -> torch.Tensor:
        """Router cross-entropy (cell state -> task id), balanced across tasks AND
        adapter views.

        Live batch: seen through the current adapter (free — already computed) plus
        up to (router_max_views - 1) sampled OTHER adapters, all labeled with the
        true (current) task. Wrong-adapter views teach the router that an adapter's
        signature is not the task — required for self-consistency eval, which asks
        it to judge wrong-adapter states. Views are averaged so the current task
        carries total weight 1 regardless of view count.

        Older tasks: equal-count replay of stored cell states from the bank (when
        router_replay is on); `tids == t` with t >= 0 already excludes freed (-1)
        slots. The final loss is the mean over task terms, so no task dominates.
        """
        device = live_state.device
        cur = self.model.current_task_id
        m = live_state.shape[0]

        # Live views: current adapter + sampled others (true label = cur).
        keys = sorted(int(k) for k in self.model.task_adapters)
        others = [k for k in keys if k != cur]
        n_extra = max(0, self.router_max_views - 1)
        if len(others) > n_extra:
            pick = torch.randperm(len(others))[:n_extra]
            others = [others[i] for i in pick.tolist()]
        view_states = [live_state]
        with torch.no_grad():
            for k in others:
                view_states.append(
                    self.model(x, store_memories=False, task_hint=k)["cell_state"]
                )
        cur_label = torch.full((m,), cur, dtype=torch.long, device=device)
        loss = sum(
            nn.functional.cross_entropy(self.model.task_router(s), cur_label)
            for s in view_states
        ) / len(view_states)
        n_terms = 1

        # Balanced bank replay for old tasks.
        if self.router_replay and cur > 0:
            bank = self.model.memory_bank
            n = bank.size
            tids = bank.task_ids[:n]
            for t in range(cur):
                idx = (tids == t).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    sel = idx[torch.randint(0, len(idx), (m,), device=device)]
                    loss = loss + nn.functional.cross_entropy(
                        self.model.task_router(bank.values[sel]),
                        torch.full((m,), t, dtype=torch.long, device=device),
                    )
                    n_terms += 1
        return loss / n_terms

    def _evaluate_task_free(self, test_loaders: list) -> dict:
        """Task-free eval via the router (no task_hint): routing accuracy and
        end-to-end accuracy per task, for both routing modes."""
        result = {"router_replay": self.router_replay, "modes": {}}
        for mode in ("self_consistency", "single_pass"):
            per_task = []
            for t in range(self.num_tasks):
                ra, ea = evaluate_task_free(
                    self.model, test_loaders[t], self.device, t, mode=mode
                )
                per_task.append({"task": t, "routing_acc": ra, "end_acc": ea})
            result["modes"][mode] = {
                "per_task": per_task,
                "mean_routing_acc": sum(p["routing_acc"] for p in per_task) / len(per_task),
                "mean_end_acc": sum(p["end_acc"] for p in per_task) / len(per_task),
            }
        return result

    def _joint_router_ceiling(self, benchmark, num_workers: int = 0) -> dict:
        """Upper bound for task-free routing: retrain a FRESH router jointly
        (iid over all tasks, correct + wrong adapter views), then task-free eval.

        Non-continual by construction — it answers "how well could ANY linear
        router on these cell states route?" (the 0.99-probe ceiling), which is
        what the continual router is measured against. The continually-trained
        router's weights are restored afterward, so summary['task_free'] still
        reflects the continual result.
        """
        model = self.model
        device = self.device
        backup = {k: v.clone() for k, v in model.task_router.state_dict().items()}
        nn.init.xavier_uniform_(model.task_router.weight)
        nn.init.zeros_(model.task_router.bias)
        opt = torch.optim.Adam(model.task_router.parameters(), lr=3e-3)
        keys = sorted(int(k) for k in model.task_adapters)
        steps = self.config.get("router", {}).get("joint_steps", 300)

        model.eval()
        loaders = [
            benchmark.get_task_loaders(t, num_workers)[0]
            for t in range(self.num_tasks)
        ]
        iters = [iter(dl) for dl in loaders]
        for step in range(steps):
            t = step % self.num_tasks  # round-robin => iid across tasks
            try:
                x, _ = next(iters[t])
            except StopIteration:
                iters[t] = iter(loaders[t])
                x, _ = next(iters[t])
            x = x.to(device)
            # Correct-adapter view + one random wrong-adapter view.
            views = [t]
            wrong = [k for k in keys if k != t]
            if wrong:
                views.append(wrong[torch.randint(0, len(wrong), (1,)).item()])
            loss = 0.0
            for k in views:
                with torch.no_grad():
                    s = model(x, store_memories=False, task_hint=k)["cell_state"]
                lbl = torch.full((s.shape[0],), t, dtype=torch.long, device=device)
                loss = loss + nn.functional.cross_entropy(model.task_router(s), lbl)
            opt.zero_grad()
            (loss / len(views)).backward()
            opt.step()

        result = self._evaluate_task_free(benchmark.get_all_test_loaders(num_workers))
        model.task_router.load_state_dict(backup)  # restore continual router
        return result

    def run(
        self,
        benchmark,
        num_workers: int = 0,
    ) -> dict:
        """Run the full continual learning experiment.

        Args:
            benchmark: Benchmark object with get_task_loaders() and get_all_test_loaders()
            num_workers: DataLoader workers

        Returns:
            Full experiment results
        """
        test_loaders = benchmark.get_all_test_loaders(num_workers)

        # Set baselines (random chance for MNIST = 0.1)
        for t in range(self.num_tasks):
            self.metrics.set_baseline(t, 0.1)

        arm = self.arm_provenance()
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training: {self.model_type.upper()}")
            print(f"Tasks: {self.num_tasks} | Epochs/task: {self.epochs_per_task}")
            # Printed BEFORE the first task: the arm the object is, not the arm
            # the launch command meant.
            print("ARM: " + " ".join(
                f"{k}={arm[k]}" for k in
                ("backbone", "use_task_heads", "head_routing_by_hint",
                 "use_input_adapters", "use_memory", "remap_labels", "seed")))
            print(f"{'='*60}")

        start_time = time.time()

        for task_id in range(self.num_tasks):
            if self.verbose:
                print(f"\n--- Task {task_id} ---")

            train_loader, _ = benchmark.get_task_loaders(task_id, num_workers)
            self.train_task(task_id, train_loader, test_loaders)

        elapsed = time.time() - start_time

        # Final summary
        summary = self.metrics.summary()
        summary["model_type"] = self.model_type
        summary["training_time"] = elapsed
        summary["arm"] = self.arm_provenance()
        summary["task_history"] = self.history

        # Task-free evaluation via the router (routing acc + end-to-end acc), if
        # the model has one. This is the number that, compared with replay on vs
        # off, shows whether the memory (as router rehearsal) is load-bearing.
        if getattr(self.model, "use_router", False):
            summary["task_free"] = self._evaluate_task_free(test_loaders)
            if self.verbose:
                tf = summary["task_free"]
                print(f"\n{'=' * 50}")
                print(f"Task-free eval (router, replay={tf['router_replay']})")
                print(f"{'=' * 50}")
                for mode, r in tf["modes"].items():
                    print(
                        f"  {mode:16s}: routing {r['mean_routing_acc']:.4f} | "
                        f"end-to-end {r['mean_end_acc']:.4f}"
                    )
                    print(
                        "    per-task routing:",
                        [round(p["routing_acc"], 3) for p in r["per_task"]],
                    )
            # Ceiling: fresh router retrained jointly (non-continual upper bound).
            if self.config.get("router", {}).get("joint_ceiling", True):
                summary["task_free_joint"] = self._joint_router_ceiling(
                    benchmark, num_workers
                )
                if self.verbose:
                    tj = summary["task_free_joint"]
                    print(f"\n  JOINT-router ceiling (non-continual upper bound):")
                    for mode, r in tj["modes"].items():
                        print(
                            f"    {mode:16s}: routing {r['mean_routing_acc']:.4f} | "
                            f"end-to-end {r['mean_end_acc']:.4f}"
                        )

        if self.verbose:
            self.metrics.print_summary()
            print(f"\nTotal training time: {elapsed:.1f}s")

        return summary

    def save_results(self, filepath: str, results: dict) -> None:
        """Save experiment results to JSON.

        Args:
            filepath: Output path
            results: Results dict from run()
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
