"""
Ablation studies for PLCM.

Trains PLCM repeatedly with individual components toggled off, holding
everything else fixed, to isolate each component's contribution to continual
learning performance. All variants share the trainer, benchmark, seed, and
evaluation protocol of ``scripts/train.py`` for a fair comparison.

Ablations:
    full             — full PLCM (GGC composition, memory + consolidation on)
    additive         — GGC replaced by additive blending (converges to centroid)
    no_memory        — write threshold raised so nothing is stored; the memory
                       read/compose pathway is effectively disabled
    no_consolidation — periodic k-means consolidation disabled

Usage:
    # Run every ablation
    python scripts/ablation.py --config configs/default.yaml

    # Run a subset, quickly (fewer epochs)
    python scripts/ablation.py --config configs/default.yaml \
        --ablations full additive no_memory --epochs 2
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import PLCM
from src.data.permuted_mnist import PermutedMNISTBenchmark
from src.training.trainer import ContinualTrainer


# Each ablation is a set of dotted-path overrides applied to the base config.
# An empty dict means "the unmodified full model".
ABLATIONS: dict[str, dict[str, object]] = {
    "full": {},
    "additive": {"composition.mode": "additive"},
    # importance scores are in [0, 1]; a threshold above 1 stores nothing, so
    # retrieval stays empty and the composition pathway is bypassed.
    "no_memory": {"memory.write_threshold": 2.0},
    # an interval far larger than any run's step count disables consolidation.
    "no_consolidation": {"memory.consolidation_interval": 10 ** 9},
}


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: dict[str, object]) -> dict:
    """Return a deep copy of config with dotted-path overrides applied.

    Args:
        config: Base configuration
        overrides: Mapping of "section.key" -> value

    Returns:
        A new config dict with the overrides set
    """
    cfg = copy.deepcopy(config)
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        cfg[section][key] = value
    return cfg


def run_ablation(
    name: str,
    config: dict,
    device: torch.device,
) -> dict:
    """Train one ablation variant and return its summary metrics.

    Args:
        name: Ablation name (for logging)
        config: Config already mutated for this ablation
        device: Compute device

    Returns:
        Summary metrics dict from the trainer
    """
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    model = PLCM.from_config(config)

    print(f"\n{'=' * 60}")
    print(f"ABLATION: {name}")
    print(f"  composition={config['composition']['mode']} "
          f"write_threshold={config['memory']['write_threshold']} "
          f"consolidation_interval={config['memory']['consolidation_interval']}")
    print(f"  parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"{'=' * 60}")

    benchmark = PermutedMNISTBenchmark(
        num_tasks=config["training"]["num_tasks"],
        batch_size=config["training"]["batch_size"],
        seed=seed,
    )

    trainer = ContinualTrainer(
        model=model,
        model_type="plcm",
        device=device,
        config=config,
    )
    return trainer.run(benchmark)


def select_device(requested: str | None) -> torch.device:
    """Pick a compute device, mirroring scripts/train.py."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="PLCM ablation studies")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--ablations", type=str, nargs="+", default=None,
        choices=list(ABLATIONS.keys()),
        help="Which ablations to run (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epochs per task (useful for quick ablation sweeps)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    args = parser.parse_args()

    base_config = load_config(args.config)
    if args.epochs is not None:
        base_config["training"]["epochs_per_task"] = args.epochs

    device = select_device(args.device)
    print(f"Device: {device}")

    names = args.ablations if args.ablations else list(ABLATIONS.keys())

    results = {}
    for name in names:
        cfg = apply_overrides(base_config, ABLATIONS[name])
        results[name] = run_ablation(name, cfg, device)

    # Comparison table
    print("\n" + "=" * 64)
    print("ABLATION SUMMARY")
    print("=" * 64)
    print(f"{'Ablation':<20} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>10} {'FWT':>10}")
    print("-" * 64)
    for name, res in results.items():
        print(
            f"{name:<20} "
            f"{res['average_accuracy']:>10.4f} "
            f"{res['forgetting']:>12.4f} "
            f"{res['backward_transfer']:>10.4f} "
            f"{res['forward_transfer']:>10.4f}"
        )
    print("=" * 64)

    # Save
    log_dir = base_config.get("logging", {}).get("log_dir", "runs/")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "ablation.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                name: {
                    "average_accuracy": res["average_accuracy"],
                    "forgetting": res["forgetting"],
                    "backward_transfer": res["backward_transfer"],
                    "forward_transfer": res["forward_transfer"],
                }
                for name, res in results.items()
            },
            f, indent=2,
        )
    print(f"\nAblation results saved to {out_path}")


if __name__ == "__main__":
    main()
