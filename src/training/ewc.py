"""
Elastic Weight Consolidation (EWC) — Baseline Method.

EWC prevents catastrophic forgetting by adding a regularization term
that penalizes changes to parameters important for previous tasks.
Importance is estimated via the diagonal of the Fisher information matrix.

Reference: Kirkpatrick et al. "Overcoming catastrophic forgetting in
neural networks" (PNAS, 2017).

This serves as the primary baseline comparison for PLCM. It represents
the standard weight-constraint approach to continual learning, which
PLCM aims to improve upon by using an external memory instead.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from copy import deepcopy


class EWC:
    """
    Elastic Weight Consolidation regularizer.

    After each task, computes the Fisher information matrix diagonal and
    stores a snapshot of the parameters. During training on subsequent tasks,
    adds a penalty for deviating from the stored parameters, weighted by
    the Fisher diagonal.

    Args:
        model: The model being trained
        ewc_lambda: Regularization strength
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0, param_filter: str = None):
        self.model = model
        self.ewc_lambda = ewc_lambda
        # If set (e.g. "lstm."), only parameters whose name starts with this
        # prefix are constrained — "selective" EWC. None constrains everything.
        self.param_filter = param_filter

        # Storage for each completed task
        self.fisher_diags: list[dict[str, torch.Tensor]] = []
        self.param_snapshots: list[dict[str, torch.Tensor]] = []

    def compute_fisher(
        self,
        data_loader: DataLoader,
        device: torch.device,
        num_samples: int = 200,
        model_type: str = "lstm",
    ) -> None:
        """Compute Fisher information diagonal and store parameter snapshot.

        Call this after finishing training on each task.

        Note: this method must run with autograd ENABLED — the Fisher diagonal
        is estimated from squared gradients of the log-likelihood, so it calls
        loss.backward(). eval() only disables dropout/batchnorm; it does not
        disable gradient tracking (do not wrap this in torch.no_grad()).

        Args:
            data_loader: Training data for the just-completed task
            device: Compute device
            num_samples: Number of samples for Fisher estimation
            model_type: "lstm" or "plcm" (affects forward call)
        """
        self.model.eval()

        # Initialize Fisher diagonal accumulators
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad and (
                self.param_filter is None or name.startswith(self.param_filter)
            )
        }

        count = 0
        for x, y in data_loader:
            if count >= num_samples:
                break
            x, y = x.to(device), y.to(device)

            self.model.zero_grad()

            if model_type == "plcm":
                output = self.model(x, store_memories=False)
                logits = output["logits"]
            else:
                logits, _ = self.model(x)

            # Use log-likelihood of true labels
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.nll_loss(log_probs, y)
            loss.backward()

            # Accumulate squared gradients (Fisher diagonal estimate). Guard on
            # `name in fisher` so a selective param_filter (LSTM-only) doesn't
            # KeyError on the non-constrained params.
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and name in fisher:
                    fisher[name] += param.grad.data ** 2

            count += x.shape[0]

        # Normalize
        for name in fisher:
            fisher[name] /= max(count, 1)

        self.fisher_diags.append(fisher)

        # Store parameter snapshot
        snapshot = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.param_snapshots.append(snapshot)

    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty term to add to the loss.

        Returns:
            Scalar penalty: λ/2 Σ_task Σ_param F_param * (θ - θ*_task)²
        """
        if not self.fisher_diags:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        total_penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for fisher, snapshot in zip(self.fisher_diags, self.param_snapshots):
            for name, param in self.model.named_parameters():
                if name in fisher:
                    diff = param - snapshot[name]
                    total_penalty += (fisher[name] * diff ** 2).sum()

        return (self.ewc_lambda / 2.0) * total_penalty
