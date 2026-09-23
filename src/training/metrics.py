"""
Continual Learning Metrics.

Measures catastrophic forgetting, forward transfer, and backward transfer
across sequential tasks. These are the standard metrics from the continual
learning literature.

Definitions:
    R[i,j] = accuracy on task j after training on task i

    Average Accuracy: (1/T) Σⱼ R[T,j] after all T tasks

    Forgetting: (1/(T-1)) Σⱼ (max_{i≥j} R[i,j] - R[T,j])
        How much performance dropped from peak on each task

    Backward Transfer: (1/(T-1)) Σⱼ (R[T,j] - R[j,j])
        How much learning new tasks helped/hurt old task performance

    Forward Transfer: (1/(T-1)) Σⱼ (R[j-1,j] - baseline_j)
        How much previous learning helps on new unseen tasks
"""

import torch
import numpy as np
from typing import Optional


class ContinualMetrics:
    """
    Tracks accuracy across tasks over time and computes continual learning metrics.

    Usage:
        metrics = ContinualMetrics(num_tasks=5)
        # After training on task 0, evaluate on all seen tasks:
        metrics.record(trained_task=0, eval_task=0, accuracy=0.95)
        # After training on task 1:
        metrics.record(trained_task=1, eval_task=0, accuracy=0.82)
        metrics.record(trained_task=1, eval_task=1, accuracy=0.94)
        # Get summary:
        metrics.summary()
    """

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        # R[i,j] = accuracy on task j after training through task i
        self.accuracy_matrix = np.full((num_tasks, num_tasks), np.nan)
        # Track baseline (random) performance per task
        self.baselines = np.zeros(num_tasks)

    def record(self, trained_task: int, eval_task: int, accuracy: float) -> None:
        """Record accuracy on eval_task after training through trained_task.

        Args:
            trained_task: Index of the last task the model was trained on
            eval_task: Index of the task being evaluated
            accuracy: Accuracy on eval_task
        """
        self.accuracy_matrix[trained_task, eval_task] = accuracy

    def set_baseline(self, task_id: int, accuracy: float) -> None:
        """Set random/untrained baseline for a task.

        Args:
            task_id: Task index
            accuracy: Random chance accuracy (usually 0.1 for MNIST)
        """
        self.baselines[task_id] = accuracy

    def average_accuracy(self) -> float:
        """Average accuracy across all tasks after training on all tasks.

        Returns:
            (1/T) Σⱼ R[T-1, j]
        """
        T = self.num_tasks
        final_row = self.accuracy_matrix[T - 1, :]
        valid = ~np.isnan(final_row)
        if not valid.any():
            return 0.0
        return float(final_row[valid].mean())

    def forgetting(self) -> float:
        """Average forgetting across tasks.

        Forgetting for task j = max peak accuracy on j minus final accuracy on j.
        Higher = more forgetting (bad).

        Returns:
            (1/(T-1)) Σⱼ<T (max_{i≥j} R[i,j] - R[T-1,j])
        """
        T = self.num_tasks
        if T < 2:
            return 0.0

        total_forgetting = 0.0
        count = 0
        for j in range(T - 1):  # exclude last task (no chance to forget yet)
            col = self.accuracy_matrix[:, j]
            valid = ~np.isnan(col)
            if not valid.any():
                continue

            peak = np.nanmax(col)
            final = self.accuracy_matrix[T - 1, j]
            if not np.isnan(final):
                total_forgetting += max(0.0, peak - final)
                count += 1

        return float(total_forgetting / max(count, 1))

    def backward_transfer(self) -> float:
        """Average backward transfer.

        BWT for task j = final accuracy on j minus accuracy right after training j.
        Positive = learning new tasks helped old ones.
        Negative = catastrophic forgetting.

        Returns:
            (1/(T-1)) Σⱼ<T (R[T-1,j] - R[j,j])
        """
        T = self.num_tasks
        if T < 2:
            return 0.0

        total_bwt = 0.0
        count = 0
        for j in range(T - 1):
            r_final = self.accuracy_matrix[T - 1, j]
            r_after_training = self.accuracy_matrix[j, j]

            if not np.isnan(r_final) and not np.isnan(r_after_training):
                total_bwt += r_final - r_after_training
                count += 1

        return float(total_bwt / max(count, 1))

    def forward_transfer(self) -> float:
        """Average forward transfer.

        FWT for task j = zero-shot accuracy on j (before training on j)
        minus random baseline.
        Positive = previous learning helps on new tasks.

        Returns:
            (1/(T-1)) Σⱼ>0 (R[j-1,j] - baseline_j)
        """
        T = self.num_tasks
        if T < 2:
            return 0.0

        total_fwt = 0.0
        count = 0
        for j in range(1, T):
            r_before = self.accuracy_matrix[j - 1, j]
            baseline = self.baselines[j]

            if not np.isnan(r_before):
                total_fwt += r_before - baseline
                count += 1

        return float(total_fwt / max(count, 1))

    def summary(self) -> dict:
        """Compute all metrics.

        Returns:
            Dictionary with all continual learning metrics
        """
        return {
            "average_accuracy": self.average_accuracy(),
            "forgetting": self.forgetting(),
            "backward_transfer": self.backward_transfer(),
            "forward_transfer": self.forward_transfer(),
            "accuracy_matrix": self.accuracy_matrix.tolist(),
        }

    def print_summary(self) -> None:
        """Print a formatted summary of all metrics."""
        s = self.summary()
        print("\n" + "=" * 50)
        print("Continual Learning Metrics")
        print("=" * 50)
        print(f"  Average accuracy:   {s['average_accuracy']:.4f}")
        print(f"  Forgetting:         {s['forgetting']:.4f}")
        print(f"  Backward transfer:  {s['backward_transfer']:.4f}")
        print(f"  Forward transfer:   {s['forward_transfer']:.4f}")
        print("\nAccuracy matrix R[i,j] (row=trained through task i, col=eval on task j):")
        T = self.num_tasks
        header = "      " + "  ".join(f"T{j:d}" for j in range(T))
        print(header)
        for i in range(T):
            row_str = f"  T{i:d}  "
            for j in range(T):
                val = self.accuracy_matrix[i, j]
                if np.isnan(val):
                    row_str += "  —  "
                else:
                    row_str += f"{val:.2f} "
            print(row_str)
        print("=" * 50)


@torch.no_grad()
def evaluate_task(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_type: str = "plcm",
    task_id: int = None,
) -> float:
    """Evaluate model accuracy on a single task.

    Args:
        model: Model to evaluate
        data_loader: Test data loader for the task
        device: Compute device
        model_type: "plcm" or "lstm" (affects forward call)
        task_id: Id of the task being evaluated, forwarded as the model's
            task_hint so a PLCM with per-task input adapters selects the right
            adapter (task-incremental eval). Ignored by models without adapters.

    Returns:
        Accuracy (0 to 1)
    """
    model.eval()
    correct = 0
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        if model_type == "plcm":
            output = model(x, store_memories=False, task_hint=task_id)
            logits = output["logits"]
        else:
            logits, _ = model(x)

        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.shape[0]

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_task_free(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    true_task: int,
    mode: str = "self_consistency",
) -> tuple[float, float]:
    """Task-free evaluation: infer each sample's task via the router (no task_hint),
    then classify with the inferred task's adapter + head.

    Args:
        model: PLCM with a trained task_router and per-task adapters/heads
        data_loader: test loader for a single (known-to-us) task
        device: compute device
        true_task: the actual task id, to score routing accuracy against
        mode: "self_consistency" (K passes) or "single_pass" (1 pass)

    Returns:
        (routing_accuracy, end_to_end_accuracy)
    """
    model.eval()
    route_correct = end_correct = total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        inferred = model.infer_task(x, mode=mode)  # [batch]
        route_correct += (inferred == true_task).sum().item()

        # Classify each sample through its inferred task's adapter + head.
        logits = torch.zeros(x.shape[0], model.num_classes, device=device)
        for k in inferred.unique().tolist():
            sel = inferred == k
            out = model(x[sel], store_memories=False, task_hint=int(k))
            logits[sel] = out["logits"]
        end_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.shape[0]

    if total == 0:
        return 0.0, 0.0
    return route_correct / total, end_correct / total
