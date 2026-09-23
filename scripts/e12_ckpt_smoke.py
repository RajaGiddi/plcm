"""
E12 — the era-checkpoint smoke: one task, one seed, the whole storage path.

Contract: docs/E12_prereg.md sec 3 (storage plan). Ruling: run this BEFORE the
headline sweep, not after.

WHY IT EXISTS. The boundary-time-authoritative rule means a bad fp16 reload
cannot corrupt a headline number — but it would mean the reproduction path is
broken while 35GB accumulates, and the whole value of the check is knowing that
on day one rather than 60 task-boundaries into a 5-seed sweep. Catch 29's
discipline applied prospectively: the fidelity question gets its first measured
answer on a throwaway run.

IT LAUNCHES THE REAL COMMAND. `scripts/train.py` is invoked as a subprocess with
the headline base-arm flags — only the scale is reduced. A smoke that rebuilt the
model itself would verify a construction path the sweep does not use, which is
the defect it is partly here to catch.

WHAT IS GATED vs WHAT IS MEASURED. The reproduction floor for this GPU path has
not been measured yet (that pair runs with the sweep), so the fp16 delta has no
bar to clear and is REPORTED, not gated. What is decidable now:

  G1  the checkpoint loads through the loader the analysis uses
  G2  the stored floating tensors really are fp16 — otherwise "35GB" is a label
  G3  the FULL-PRECISION shadow of the same state reloads EXACTLY (0.0000).
      This is the gate that matters: it separates fp16 rounding from anything
      state_dict() drops, and only the second kind is a defect.
  G4  P3a/P3b hold on the live model AND on the reload
  G5  the audit FIRES on a truncated checkpoint — a gate that has only ever
      passed is indistinguishable from one that cannot fail (catch 25)
  G6  the arm that ran is the contracted base arm, read off the model

Usage:
    modal run modal_runner.py::analysis_gpu --argv "scripts/e12_ckpt_smoke.py"
    python scripts/e12_ckpt_smoke.py --device cpu     # mechanism only, very slow
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM

# The contracted base arm, as flags. Identical to what jobs_e12() launches;
# only --num-tasks/--epochs differ.
BASE_ARM = ["--config", "configs/e12_vit.yaml", "--model", "mafc",
            "--mafc-arm", "lambda0", "--task-heads", "--no-adapters"]

# docs/E12_prereg.md sec 3: fp32 x 20 tasks x 2 arms x 5 seeds ~= 69GB, fp16 ~= 35GB.
SWEEP_CELLS = 20 * 2 * 5
CONTRACT_FP16_GB = 35.0


def run_training(root: Path, seed: int, epochs: int, device: str | None) -> Path:
    """One task, one seed, through the launcher the sweep uses."""
    log_dir = root / "run"
    ckpt_dir = root / "ckpt"
    cmd = [sys.executable, "scripts/train.py", *BASE_ARM,
           "--seed", str(seed), "--num-tasks", "1", "--epochs", str(epochs),
           "--era-checkpoints", "--fp32-shadow",
           "--checkpoint-dir", str(ckpt_dir), "--log-dir", str(log_dir)]
    if device:
        cmd += ["--device", device]
    print("RUN:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(f"training exited {p.returncode} — smoke stops here")
    return log_dir / "mafc_results.json"


def stored_dtypes(path: str) -> dict:
    ck = torch.load(path, weights_only=True, map_location="cpu")
    kinds = {}
    for k, v in ck["model_state"].items():
        if torch.is_tensor(v) and v.is_floating_point():
            kinds[str(v.dtype)] = kinds.get(str(v.dtype), 0) + 1
    return kinds


def rounding_error(fp16_path: str, fp32_path: str) -> dict:
    """How far fp16 moved the weights — the diagnosis an accuracy delta needs."""
    a = torch.load(fp16_path, weights_only=True, map_location="cpu")["model_state"]
    b = torch.load(fp32_path, weights_only=True, map_location="cpu")["model_state"]
    worst, worst_name, denom = 0.0, "", 0.0
    for k, v in b.items():
        if not (torch.is_tensor(v) and v.is_floating_point()):
            continue
        d = float((a[k].float() - v.float()).abs().max())
        if d > worst:
            worst, worst_name, denom = d, k, float(v.float().abs().max())
    return {"max_abs": worst, "tensor": worst_name,
            "max_abs_relative": (worst / denom) if denom else 0.0}


def truncation_control(path: str, scratch: Path) -> dict:
    """The audit must FAIL on a checkpoint that exists but does not load."""
    d = scratch / "truncated"
    d.mkdir(parents=True, exist_ok=True)
    name = Path(path).name
    dst = d / name
    raw = Path(path).read_bytes()
    dst.write_bytes(raw[: len(raw) // 2])
    task = int(name.split("task")[1].split("_")[0])
    epoch = int(name.split("epoch")[1].split(".")[0])
    try:
        PLCM.load_era(str(d), task, epoch=epoch)
        return {"fired": False, "error": None}
    except Exception as e:
        return {"fired": True, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="E12 era-checkpoint smoke")
    ap.add_argument("--root", default="runs/e12_ckpt_smoke")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/e12_ckpt_smoke.json")
    args = ap.parse_args()

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    print("=" * 96)
    print("E12 ERA-CHECKPOINT SMOKE — one task, one seed, the real launch path")
    print("=" * 96)

    results = json.load(open(run_training(root, args.seed, args.epochs, args.device)))
    arm = results.get("arm", {})
    era = results["task_history"][0].get("era_checkpoint")
    if era is None:
        raise SystemExit("no era_checkpoint in the results — --era-checkpoints "
                         "did not reach the trainer")
    shadow = era.get("fp32_shadow", {})

    print("\n" + "-" * 96)
    print("ARM (read off the built model, not the launch command)")
    print("-" * 96)
    for k in ("backbone", "use_task_heads", "head_routing_by_hint",
              "use_input_adapters", "use_memory", "num_classes", "n_task_heads",
              "n_task_adapters", "remap_labels", "seed"):
        print(f"  {k:<24} {arm.get(k)}")

    print("\n" + "-" * 96)
    print("BOUNDARY (authoritative, in-process, full precision) vs RELOAD")
    print("-" * 96)
    print(f"  acc_ceiling          boundary {era['acc_boundary']:.4f}  "
          f"fp16-reload {era['acc_reload']}  delta {era['delta']}")
    print(f"  acc_refit_ceiling    boundary {era['acc_refit_ceiling_boundary']:.4f}  "
          f"fp16-reload {era['acc_refit_ceiling_reload']}  delta {era['delta_refit']}")
    if shadow:
        print(f"  acc_ceiling          fp32-shadow reload {shadow.get('acc_reload')}  "
              f"delta {shadow.get('delta')}")
        print(f"  acc_refit_ceiling    fp32-shadow reload "
              f"{shadow.get('acc_refit_ceiling_reload')}  delta {shadow.get('delta_refit')}")

    dtypes = stored_dtypes(era["path"])
    rounding = (rounding_error(era["path"], shadow["path"])
                if shadow.get("path") else None)
    control = truncation_control(era["path"], root)
    gb = era["bytes"] * SWEEP_CELLS / 1e9

    print("\n" + "-" * 96)
    print("STORAGE (measured, not projected from parameter counts)")
    print("-" * 96)
    print(f"  stored float dtypes  {dtypes}")
    print(f"  bytes per era ckpt   {era['bytes'] / 1e6:.1f} MB")
    print(f"  x {SWEEP_CELLS} cells (20 tasks x 2 arms x 5 seeds) = {gb:.1f} GB "
          f"vs the contract's {CONTRACT_FP16_GB} GB")
    if rounding:
        print(f"  worst fp16 weight shift  {rounding['max_abs']:.3e} abs / "
              f"{rounding['max_abs_relative']:.3e} rel  ({rounding['tensor']})")

    # ---- verdicts, each computed from the values printed above (catch 22) ----
    g1 = bool(era.get("loads"))
    g2 = dtypes == {"torch.float16": sum(dtypes.values())} and bool(dtypes)
    g3 = bool(shadow) and shadow.get("delta") == 0.0 and shadow.get("delta_refit") == 0.0
    p3_live, p3_reload = era.get("p3_live", {}), era.get("p3_reload", {})
    g4 = (p3_live.get("p3a_delta") == 0.0 and "p3b_module" in p3_live
          and p3_reload.get("p3a_delta") == 0.0 and "p3b_module" in p3_reload)
    g5 = control["fired"]
    g6 = (arm.get("backbone") == "vit" and arm.get("use_task_heads") is True
          and arm.get("head_routing_by_hint") is True
          and arm.get("use_input_adapters") is False)

    gates = [
        ("G1 checkpoint loads through PLCM.load_era", g1, era.get("error") or "audited at save time"),
        ("G2 stored floats are fp16", g2, str(dtypes)),
        ("G3 fp32 shadow reloads EXACTLY", g3,
         f"delta {shadow.get('delta')} / refit {shadow.get('delta_refit')}"),
        ("G4 P3a+P3b hold on live AND reload", g4,
         f"live {p3_live} | reload {p3_reload}"),
        ("G5 audit fires on a truncated ckpt", g5, control["error"]),
        ("G6 arm is the contracted base arm", g6,
         f"heads={arm.get('use_task_heads')} adapters={arm.get('use_input_adapters')}"),
    ]

    print("\n" + "=" * 96)
    print("GATES")
    print("=" * 96)
    for name, ok, detail in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} {detail}")

    passed = all(ok for _, ok, _ in gates)
    print("\n" + "=" * 96)
    print(f"SMOKE: {'PASS' if passed else 'FAIL'} ({sum(ok for _, ok, _ in gates)}/{len(gates)})")
    print("=" * 96)
    print(f"  fp16 reload delta acc_ceiling {era['delta']} / acc_refit_ceiling "
          f"{era['delta_refit']} — REPORTED, NOT GATED: the reproduction floor on "
          f"this GPU path is not measured yet (the floor pair runs with the "
          f"sweep). The contract's rule — the delta must sit below that floor, or "
          f"the storage precision is the finding and fp32 is re-costed — is read "
          f"against it then, not here.")

    out = {"arm": arm, "era": era, "dtypes": dtypes, "rounding": rounding,
           "truncation_control": control, "projected_sweep_gb": gb,
           "gates": {n: bool(ok) for n, ok, _ in gates}, "pass": bool(passed)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print(f"  wrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
