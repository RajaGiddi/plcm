"""
Main training script for PLCM continual learning experiments.

Usage:
    # Train PLCM only
    python scripts/train.py --config configs/default.yaml --model plcm

    # Train all models for comparison
    python scripts/train.py --config configs/default.yaml --model all

    # Train vanilla LSTM baseline
    python scripts/train.py --config configs/default.yaml --model lstm

    # Train with EWC baseline
    python scripts/train.py --config configs/default.yaml --model lstm_ewc

    # Override config values
    python scripts/train.py --config configs/default.yaml --model plcm \
        --epochs 20 --memory-capacity 1024
"""

import argparse
import yaml
import torch
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import PLCM, LSTMBaseline
from src.data.permuted_mnist import PermutedMNISTBenchmark
from src.data.rotated_mnist import RotatedMNISTBenchmark
from src.data.har_shift import HARShiftBenchmark, spec_fingerprint
from src.data.har_subject import HARSubjectBenchmark
from src.data.shifted_cifar import ShiftedCIFAR10Benchmark
from src.data.split_cifar100 import SplitCIFAR100Benchmark
from src.training.trainer import ContinualTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_type: str, config: dict) -> torch.nn.Module:
    """Build model from config.

    Args:
        model_type: "plcm", "lstm", or "lstm_ewc"
        config: Full config dict

    Returns:
        Model instance
    """
    if model_type in ("plcm", "plcm_ewc", "plcm_frozen", "plcm_adapter", "mafc"):
        return PLCM.from_config(config)
    elif model_type in ("lstm", "lstm_ewc"):
        m = config["model"]
        return LSTMBaseline(
            # MNIST row width (28) by default; HAR overrides to 9 channels.
            input_size=m.get("input_size", 28),
            hidden_size=m["hidden_size"],
            num_classes=m["num_classes"],
            num_layers=m["num_layers"],
            dropout=m.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(
    model_type: str,
    config: dict,
    device: torch.device,
    save_name: str | None = None,
) -> dict:
    """Run a single experiment.

    Args:
        model_type: Architecture/trainer type to build ("plcm", "lstm", "lstm_ewc")
        config: Configuration
        device: Compute device
        save_name: Label for logging and the results filename. Defaults to
            model_type; set separately so ablations that reuse an architecture
            (e.g. "plcm_additive" built as "plcm") get their own results file.

    Returns:
        Experiment results
    """
    save_name = save_name or model_type

    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Build benchmark FIRST — HAR has a different input width (9 channels),
    # class count (6), and adapter geometry than the MNIST-family benchmarks,
    # so the model's shape is derived from the benchmark rather than assumed.
    # "rotated" tests whether the per-task-adapter result survives a shift that
    # is linear but NOT exactly invertible (prereg R1).
    bench_cfg = config.get("benchmark", {})
    bench_name = bench_cfg.get("name", "permuted")
    ds = bench_cfg.get("dataset", "mnist")   # "mnist" | "fashion"
    if bench_name in ("har_subject", "har_subject_noshift"):
        # E10: tasks differ in CONTENT (disjoint subjects) as well as
        # PRESENTATION (the frozen shifts), so generative repair is falsifiable.
        benchmark = HARSubjectBenchmark(
            num_tasks=config["training"]["num_tasks"],
            batch_size=config["training"]["batch_size"],
            root=bench_cfg.get("root", "."), seed=seed,
            no_shift=(bench_name == "har_subject_noshift"),
        )
        config["model"]["input_size"] = benchmark.input_size      # 9
        config["model"]["num_classes"] = benchmark.num_classes    # 6
        config.setdefault("adapters", {})["mode"] = "per_step"
        config["adapters"]["dim"] = benchmark.input_size
        if config.get("model", {}).get("backbone") == "mlp":
            # E23 arm B (2026-09-19): the MLP reads the flattened 128 x 9 window; the adapter
            # stays per-step 9-wide. Recorded in the checkpoint config, so from_config rebuilds it.
            from src.data.har_shift import N_STEPS
            config["model"]["mlp_input_dim"] = N_STEPS * benchmark.input_size    # 1152
        print(f"\nBenchmark: subject-disjoint HAR  "
              f"(partition {benchmark.partition_fingerprint()}, shift {spec_fingerprint()})")
    elif bench_name in ("har", "har_noshift"):
        benchmark = HARShiftBenchmark(
            num_tasks=config["training"]["num_tasks"],
            batch_size=config["training"]["batch_size"],
            root=bench_cfg.get("root", "."),
            seed=seed,
            # E5b control: identity map on every task, so any forgetting
            # measured cannot be attributed to input shift.
            no_shift=(bench_name == "har_noshift"),
        )
        # The HAR shifts are 9x9 channel maps applied identically at every
        # timestep, so the adapter lives in channel space: 81 params/task.
        config["model"]["input_size"] = benchmark.input_size      # 9
        config["model"]["num_classes"] = benchmark.num_classes    # 6
        config.setdefault("adapters", {})["mode"] = "per_step"
        config["adapters"]["dim"] = benchmark.input_size
        print(f"\nBenchmark: UCI HAR shifts  (spec {spec_fingerprint()})")
    elif bench_name in ("cifar100", "cifar100_permuted"):
        # E12. Task-IL is the primary setting, so labels are remapped to 0..4;
        # class-IL keeps global ids and is selected by benchmark.remap_labels.
        # "cifar100_permuted" is the EFFICACY CONTROL's data (B6): the shift is
        # patch-consistent, i.e. inside the class the registered adapter
        # placement can represent, so a flat control means the placement is dead
        # rather than the shift being unrepresentable.
        benchmark = SplitCIFAR100Benchmark(
            num_tasks=config["training"]["num_tasks"],
            batch_size=config["training"]["batch_size"],
            root=bench_cfg.get("root", "./data"), seed=seed,
            remap_labels=bench_cfg.get("remap_labels", True),
            repeat_task=bench_cfg.get("repeat_task", False),
            shift_mode="patch" if bench_name == "cifar100_permuted" else None,
            download=False,          # baked into the image at build time (B3)
        )
        config["model"]["input_size"] = 3                          # unused by ViT
        config["model"]["num_classes"] = benchmark.num_classes     # 5 task-IL / 100 class-IL
        config.setdefault("adapters", {})["mode"] = "per_step"     # per-token on patches
        config["adapters"]["dim"] = 768                            # set by placement, not input
        co_fp, sh_fp = (benchmark.class_order_fingerprint(),
                        benchmark.shift_fingerprint())
        print(f"\nBenchmark: Split CIFAR-100 {bench_name}  "
              f"(class order {co_fp}, shift {sh_fp})")
        # GATE, not a print. E10's partition fingerprint was printed by every run
        # and checked by none, so the registered split and the executed one
        # diverged for a whole experiment without anyone seeing it. Only the
        # REGISTERED shape is gated: a smoke run with --num-tasks 2 legitimately
        # produces a different split and must not be silently blessed either.
        registered = (benchmark.num_tasks == 20 and benchmark.classes_per_task == 5
                      and not benchmark.repeat_task)
        exp_co = (bench_cfg.get("expected_class_order") or {}).get(seed)
        exp_sh = (bench_cfg.get("expected_shift") or {}).get(seed)
        if registered and exp_co:
            assert co_fp == exp_co, (
                f"class-order fingerprint {co_fp} != registered {exp_co} for seed "
                f"{seed} — the executed split is not the registered one")
            if bench_name == "cifar100_permuted" and exp_sh:
                assert sh_fp == exp_sh, (
                    f"shift fingerprint {sh_fp} != registered {exp_sh} for seed {seed}")
            print(f"  fingerprints GATED against the registered values -> MATCH")
        elif registered:
            print(f"  WARNING: no registered fingerprint for seed {seed}; ungated")
        else:
            print(f"  non-registered shape ({benchmark.num_tasks}x"
                  f"{benchmark.classes_per_task}"
                  f"{', repeat-task' if benchmark.repeat_task else ''}) — "
                  f"fingerprint gate does not apply")
    elif bench_name in ("cifar_permuted", "cifar_rotated"):
        benchmark = ShiftedCIFAR10Benchmark(
            num_tasks=config["training"]["num_tasks"],
            batch_size=config["training"]["batch_size"],
            shift="permuted" if bench_name == "cifar_permuted" else "rotated",
            root=bench_cfg.get("root", "./data"),
            max_angle=bench_cfg.get("max_angle", 90.0),
            seed=seed,
        )
        # 32 rows x 96 features per row; the shift lives in the flattened
        # 3072-dim image (a spatial permutation mixes pixels across rows), so
        # the adapter is flat — the MNIST case, not HAR's per_step case.
        config["model"]["input_size"] = benchmark.input_size        # 96
        config["model"]["num_classes"] = benchmark.num_classes      # 10
        config.setdefault("adapters", {})["mode"] = "flat"
        config["adapters"]["dim"] = benchmark.adapter_dim           # 3072
        print(f"\nBenchmark: CIFAR-10 {bench_name}  (spec {benchmark.shift_fingerprint()})")
    elif bench_name == "rotated":
        benchmark = RotatedMNISTBenchmark(
            num_tasks=config["training"]["num_tasks"],
            batch_size=config["training"]["batch_size"],
            max_angle=bench_cfg.get("max_angle", 90.0),
            seed=seed, dataset=ds,
            disjoint_content=bool(bench_cfg.get("disjoint_content", False)),
        )
    else:
        benchmark = PermutedMNISTBenchmark(
            num_tasks=config["training"]["num_tasks"],
            batch_size=config["training"]["batch_size"],
            seed=seed, dataset=ds,
            disjoint_content=bool(bench_cfg.get("disjoint_content", False)),
            content_chunks=bench_cfg.get("content_chunks"),   # E23 seq-length control; None = chunk by num_tasks
        )
    # E18: the construction is an arm-identity field. Recorded into the config
    # the trainer writes into `arm`, from the BENCHMARK OBJECT, so the artifact
    # says what ran rather than what was asked for.
    if bench_name in ("rotated", "permuted") or bench_name is None:
        bench_cfg["disjoint_content"] = bool(getattr(benchmark, "disjoint_content", False))
        bench_cfg["content_fingerprint"] = getattr(benchmark, "content_fingerprint", lambda: None)()
        # 2026-09-20: content_fingerprint is blind to the chunk boundaries (src/data/permuted_mnist.py),
        # so the construction's own identity is recorded beside it, with the chunk count.
        bench_cfg["construction_fingerprint"] = getattr(benchmark, "construction_fingerprint", lambda: None)()
        bench_cfg["content_chunks"] = getattr(benchmark, "content_chunks", None)
        if bench_name == "rotated":
            bench_cfg["angles"] = list(benchmark.angles)
        config["benchmark"] = bench_cfg

    # Build model (after the benchmark, so HAR's shape overrides are in config)
    model = build_model(model_type, config)
    print(f"\nModel: {save_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build trainer
    trainer = ContinualTrainer(
        model=model,
        model_type=model_type,
        device=device,
        config=config,
    )

    # Run
    results = trainer.run(benchmark)

    # Save
    log_dir = config.get("logging", {}).get("log_dir", "runs/")
    results_path = os.path.join(log_dir, f"{save_name}_results.json")
    trainer.save_results(results_path, results)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PLCM Continual Learning")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--model", type=str, default="plcm",
        choices=["plcm", "lstm", "lstm_ewc", "plcm_additive", "plcm_ewc", "plcm_frozen",
                 "plcm_adapter", "mafc", "all"],
        help="Model to train",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per task")
    parser.add_argument(
        "--epochs-first", type=int, default=None,
        help="Override epochs for TASK 0. The trainer honours `epochs_first_task` "
             "separately (trainer.py:243), so --epochs alone leaves task 0 at the "
             "config's value and the budget is NOT matched across tasks. E28b's "
             "comparison is budget-matched, so it sets both.")
    parser.add_argument("--memory-capacity", type=int, default=None, help="Override memory capacity")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed (for multi-seed runs)")
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Disable the memory bank (ablation: frozen encoder + adapters + heads only)",
    )
    parser.add_argument(
        "--no-router-replay", action="store_true",
        help="Train the task-free router WITHOUT bank replay (load-bearing-memory ablation)",
    )
    parser.add_argument(
        "--baseline", type=str, default=None, choices=["lwf_lh17"],
        help="External baseline arm. `lwf_lh17` = Learning without Forgetting "
             "(Li & Hoiem 2017), contract docs/E16_lwf_prereg.md. This is NOT "
             "`--mafc-arm lwf`, which is this repo's H3 raw-probe ablation and a "
             "different arm entirely; the citation is in the name so the two "
             "cannot be confused again.",
    )
    parser.add_argument(
        "--lwf-lambda", type=float, default=None,
        help="LwF distillation weight lambda_o (registered grid: 0.25/1/4/16). "
             "REQUIRED with --baseline lwf_lh17: no default is supplied, because "
             "a launcher default would select the reported configuration outside "
             "the measurement (the EWC lambda=200 lesson, made structural).",
    )
    parser.add_argument(
        "--lwf-warmup-epochs", type=int, default=1,
        help="LwF warm-up: readout-only epochs at each task boundary (tasks>=1). "
             "Reuses the audited `warmup_mode=control` path rather than a "
             "parallel implementation (catch 32).",
    )
    parser.add_argument(
        "--ewc-lambda", type=float, default=None,
        help="Override the EWC regularization strength (plcm_ewc.lambda and ewc.lambda)",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Override logging.log_dir (per-run output directory, e.g. for sweeps)",
    )
    parser.add_argument(
        "--mafc-lambda", type=float, default=None,
        help="MAFC functional-consistency strength (prereg sweep: 0.1/0.5/1/2/5)",
    )
    parser.add_argument(
        "--mafc-arm", type=str, default=None,
        choices=["full", "enc", "lwf", "lambda0"],
        help="MAFC arm (prereg §4): full = both terms, adapter probes; "
             "enc = encoder term only (H4 control); "
             "lwf = both terms but RAW probe inputs (H3 control); "
             "lambda0 = no MAFC loss (floor control)",
    )
    parser.add_argument(
        "--no-adapters", action="store_true",
        help="Force per-task input adapters OFF (attribution control: isolates "
             "the adapters as the single changed variable)",
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        choices=["permuted", "rotated", "har", "har_noshift",
                 "har_subject", "har_subject_noshift",
                 "cifar_permuted", "cifar_rotated",
                 # E12. The dispatch branch in run_experiment was added without
                 # these, so `--benchmark cifar100_permuted` was rejected by
                 # argparse and all three efficacy-control jobs died in under a
                 # second. The headline arms were unaffected because they select
                 # the benchmark through the config file, not this flag — so the
                 # gap only showed on the one arm that overrides it.
                 "cifar100", "cifar100_permuted"],
        help="permuted = exactly-invertible pixel shift; rotated = linear but "
             "LOSSY shift (tests whether the adapter result generalizes, R1)",
    )
    # ---- E28: channel-permutation augmentation (docs/E28_prereg.md sec 4) ----
    parser.add_argument(
        "--aug-perm-p", type=float, default=0.0,
        help="E28: per-sample probability of a channel permutation, applied in "
             "STANDARDIZED space in the trainer's batch loop. 0 disables the whole "
             "path, including its RNG, so the default is bit-identical to before.",
    )
    parser.add_argument(
        "--aug-perm-seed", type=int, default=0,
        help="E28: seed for the augmentation's own generator (never the global stream)",
    )
    parser.add_argument(
        "--aug-heldout", type=str, default=None,
        help="E28: path to runs/e28/heldout.json. Its permutations are EXCLUDED from "
             "the augmentation support and asserted against at every draw.",
    )
    parser.add_argument(
        "--aux-perm-weight", type=float, default=0.0,
        help="E28 B2: weight on the auxiliary head predicting which channel of the "
             "TASK'S OWN FRAME each input position holds. 0 = B1 (augmentation only).",
    )
    parser.add_argument(
        "--er", action="store_true",
        help="Enable the Experience Replay baseline (stores raw examples)",
    )
    parser.add_argument(
        "--er-buffer", type=int, default=None,
        help="ER buffer size per task (prereg default 100)",
    )
    parser.add_argument(
        "--adapter-rank", type=int, default=None,
        help="Low-rank residual adapters A(x)=x+UVx with this rank "
             "(0 = full-rank d x d). Storage O(2*d*r) per task.",
    )
    parser.add_argument(
        "--task-heads", action="store_true",
        help="E7: per-task classifier heads, frozen after their task. Overrides "
             "the mafc default of a shared readout. Only sound at --mafc-arm "
             "lambda0, where the bank-anchored readout term is off.",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=None,
        help="E5d method v2: adapter-first warmup epochs at each task boundary "
             "(tasks 1..T-1 only; 0 = v1 behaviour)",
    )
    parser.add_argument(
        "--warmup-mode", type=str, default=None, choices=["adapter", "control"],
        help="'adapter' = v2-ON (only the new adapter trains during warmup); "
             "'control' = v2-control (only the readout trains, adapter frozen) — "
             "the arm that separates recruitment from encoder-rest",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="Override encoder hidden size (E2 capacity calibration searches "
             "{256, 512, 1024}; both arms then share the winner)",
    )
    parser.add_argument(
        "--repeat-task", action="store_true",
        help="P2's control: every task is the SAME data with a fresh head, so "
             "any drop is optimization drift rather than sequence forgetting.",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Override number of tasks (capacity calibration uses 1 = "
             "single-task clean accuracy)",
    )
    parser.add_argument(
        "--save-checkpoints", action="store_true",
        help="Save per-epoch model state (needed to compare learned adapters "
             "across ranks post-hoc)",
    )
    parser.add_argument(
        "--era-checkpoints", action="store_true",
        help="E12: save ONE fp16 checkpoint per task boundary (not per epoch), "
             "audit it by loading at save time, and record the boundary-vs-"
             "reload accuracy delta in the results json.",
    )
    parser.add_argument(
        "--fp32-shadow", action="store_true",
        help="With --era-checkpoints, also write the same boundary state at full "
             "precision into a sibling dir, so the fp16 reload delta splits into "
             "rounding vs anything serialization drops. Smoke only — it doubles "
             "the storage the plan is costed against.",
    )
    parser.add_argument(
        "--disjoint-content", action="store_true",
        help="E18: per-task DISJOINT image subsets for Permuted/Rotated MNIST "
             "(train and test). Opt-in; default off is the construction every "
             "existing MNIST artifact was produced on (docs/E18_prereg.md sec 1).")
    parser.add_argument(
        "--content-chunks", type=int, default=None,
        help="E23 seq-length control: split disjoint content into N chunks and use the first "
             "--num-tasks of them, so images-per-task can be held fixed while tasks-per-sequence "
             "varies. Default None = chunk by num_tasks (every pre-2026-09-20 construction).")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Where to write checkpoints")
    parser.add_argument(
        "--dataset", type=str, default=None, choices=["mnist", "fashion"],
        help="Base dataset for the shift benchmarks (E1 uses fashion)",
    )
    parser.add_argument(
        "--backbone", type=str, default=None, choices=["lstm", "mlp", "vit", "resnet"],
        help="Encoder backbone; mlp = architecture brittleness control",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.epochs is not None:
        config["training"]["epochs_per_task"] = args.epochs
    if args.epochs_first is not None:
        config["training"]["epochs_first_task"] = args.epochs_first
    if args.memory_capacity is not None:
        config["memory"]["capacity"] = args.memory_capacity
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_memory:
        config.setdefault("memory", {})["enabled"] = False
    if args.no_router_replay:
        config.setdefault("router", {})["replay"] = False
    if args.baseline == "lwf_lh17":
        if args.lwf_lambda is None:
            raise SystemExit(
                "--baseline lwf_lh17 requires --lwf-lambda: the reported "
                "configuration is selected from runs/e16_sweep.json on AVG, "
                "not from a launcher default.")
        lwf = config.setdefault("lwf", {})
        lwf["enabled"] = True
        lwf["lambda"] = args.lwf_lambda
        lwf["temperature"] = float(
            config.get("mafc", {}).get("temperature", 2.0))
        lwf["warmup_epochs"] = args.lwf_warmup_epochs
        # LwF's warm-up trains the NEW head first, then joint. That is exactly
        # what `warmup_mode=control` already does (readout-only, audited path),
        # so it is reused rather than re-implemented (catch 32).
        #
        # ASYMMETRY, stated for the caption rather than buried: warm-up gives
        # the incoming task readout-only epochs that no EARLIER task received at
        # its own boundary in the arms LwF is compared against. It is part of
        # Li & Hoiem's protocol and is kept for fidelity, but it is a head start
        # the comparison arms do not get, and the caption says so.
        tr = config.setdefault("training", {})
        tr["warmup_epochs"] = args.lwf_warmup_epochs
        tr["warmup_mode"] = "control"
    if args.ewc_lambda is not None:
        config.setdefault("plcm_ewc", {})["lambda"] = args.ewc_lambda
        config.setdefault("ewc", {})["lambda"] = args.ewc_lambda
    if args.log_dir is not None:
        config.setdefault("logging", {})["log_dir"] = args.log_dir
    if args.benchmark is not None:
        config.setdefault("benchmark", {})["name"] = args.benchmark
    if args.dataset is not None:
        config.setdefault("benchmark", {})["dataset"] = args.dataset
    if args.warmup_epochs is not None:
        config["training"]["warmup_epochs"] = args.warmup_epochs
    if args.warmup_mode is not None:
        config["training"]["warmup_mode"] = args.warmup_mode
    if args.hidden_size is not None:
        config["model"]["hidden_size"] = args.hidden_size
    if getattr(args, "repeat_task", False):
        config.setdefault("benchmark", {})["repeat_task"] = True
    if args.num_tasks is not None:
        config["training"]["num_tasks"] = args.num_tasks
    if args.aug_perm_p > 0 or args.aux_perm_weight > 0:
        aug = config.setdefault("aug", {})
        aug["perm_p"] = args.aug_perm_p
        aug["perm_seed"] = args.aug_perm_seed
        aug["aux_perm_weight"] = args.aux_perm_weight
        assert args.aug_heldout, "E28: --aug-heldout is required whenever augmentation is on"
        ho = json.load(open(args.aug_heldout))
        aug["heldout"] = ho["heldout"]
        aug["heldout_fingerprint"] = ho["fingerprint"]
        print(f"  E28 augmentation: p={args.aug_perm_p} seed={args.aug_perm_seed} "
              f"aux={args.aux_perm_weight} heldout={ho['fingerprint']} ({len(ho['heldout'])} perms)")
    if args.er:
        config.setdefault("er", {})["enabled"] = True
    if args.er_buffer is not None:
        config.setdefault("er", {})["buffer_per_task"] = args.er_buffer
    if args.backbone is not None:
        config.setdefault("model", {})["backbone"] = args.backbone
    if args.adapter_rank is not None:
        config.setdefault("adapters", {})["rank"] = args.adapter_rank
    if args.disjoint_content:
        config.setdefault("benchmark", {})["disjoint_content"] = True
    if args.content_chunks is not None:
        config.setdefault("benchmark", {})["content_chunks"] = int(args.content_chunks)
    if args.save_checkpoints:
        config.setdefault("logging", {})["save_checkpoints"] = True
    if getattr(args, "era_checkpoints", False):
        config.setdefault("logging", {})["era_checkpoints"] = True
        config["logging"]["checkpoint_fp16"] = True
        config["logging"]["checkpoint_fp32_shadow"] = bool(args.fp32_shadow)
    # Applies to BOTH checkpoint modes. Nesting it under --era-checkpoints made
    # `--save-checkpoints --checkpoint-dir /runs/ckpt_...` — the exact form every
    # E10 job uses — write to the default dir instead, silently.
    if args.checkpoint_dir:
        config.setdefault("logging", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.mafc_lambda is not None:
        config.setdefault("mafc", {})["lambda"] = args.mafc_lambda
    if args.mafc_arm is not None:
        # Arm = which terms are active and whether probes go through old adapters.
        arm = {
            "full":    dict(encoder_term=True,  readout_term=True,  adapter_probes=True),
            "enc":     dict(encoder_term=True,  readout_term=False, adapter_probes=True),
            "lwf":     dict(encoder_term=True,  readout_term=True,  adapter_probes=False),
            "lambda0": dict(encoder_term=False, readout_term=False, adapter_probes=True),
        }[args.mafc_arm]
        config.setdefault("mafc", {}).update(arm)
        if args.mafc_arm == "lambda0":
            config["mafc"]["lambda"] = 0.0

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Run experiments
    all_results = {}

    # display name -> (architecture/trainer type to build, composition mode)
    model_specs = {
        "lstm": ("lstm", "ggc"),            # composition mode unused for LSTM
        "lstm_ewc": ("lstm_ewc", "ggc"),    # composition mode unused for LSTM
        "plcm": ("plcm", "ggc"),
        "plcm_additive": ("plcm", "additive"),
        "plcm_ewc": ("plcm_ewc", "ggc"),
        "plcm_frozen": ("plcm_frozen", "ggc"),
        "plcm_adapter": ("plcm_adapter", "ggc"),
        "mafc": ("mafc", "ggc"),
    }

    if args.model == "all":
        models_to_run = ["lstm", "lstm_ewc", "plcm_additive", "plcm", "plcm_ewc"]
    else:
        models_to_run = [args.model]

    for name in models_to_run:
        build_type, mode = model_specs[name]
        config["composition"]["mode"] = mode
        # Encoder freeze: plcm_frozen AND plcm_adapter (adapters need a frozen
        # encoder). plcm / plcm_ewc stay unfrozen (reproducibility + A/B baseline).
        config["model"]["freeze_encoder_after_first_task"] = build_type in (
            "plcm_frozen", "plcm_adapter"
        )
        # Per-task input adapters: plcm_adapter AND mafc (MAFC's encoder term
        # anchors probe inputs through the old adapters).
        config.setdefault("adapters", {})["enabled"] = build_type in (
            "plcm_adapter", "mafc"
        ) and not args.no_adapters
        config.setdefault("router", {})["enabled"] = (build_type == "plcm_adapter")
        # MAFC (prereg v2): unfrozen encoder + adapters + SHARED readout.
        # Per-task heads are disabled so phi can drift — otherwise the
        # bank-anchored readout term is vacuous (prereg §1).
        #
        # E7 (docs/E7_prereg.md) overrides this. The suppression exists only to
        # keep MAFC's bank-anchored readout term non-vacuous, and E7 runs at
        # --mafc-arm lambda0 where that term is OFF, so the two do not conflict.
        # E6b measured the shared reader carrying 75-90% of HAR's forgetting; a
        # frozen per-task head cannot walk, which is what E7 tests.
        if build_type == "mafc":
            config["model"]["use_task_heads"] = bool(args.task_heads)
        results = run_experiment(build_type, config, device, save_name=name)
        all_results[name] = results

    # Print comparison if multiple models
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<18} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>10} {'FWT':>10}")
        print("-" * 60)
        for name, res in all_results.items():
            print(
                f"{name:<18} "
                f"{res['average_accuracy']:>10.4f} "
                f"{res['forgetting']:>12.4f} "
                f"{res['backward_transfer']:>10.4f} "
                f"{res['forward_transfer']:>10.4f}"
            )
        print("=" * 60)

    # Save combined results
    log_dir = config.get("logging", {}).get("log_dir", "runs/")
    os.makedirs(log_dir, exist_ok=True)
    combined_path = os.path.join(log_dir, "comparison.json")
    with open(combined_path, "w") as f:
        json.dump(
            {k: {"average_accuracy": v["average_accuracy"],
                  "forgetting": v["forgetting"],
                  "backward_transfer": v["backward_transfer"],
                  "forward_transfer": v["forward_transfer"]}
             for k, v in all_results.items()},
            f, indent=2,
        )


if __name__ == "__main__":
    main()
