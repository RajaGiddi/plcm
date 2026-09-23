"""Pre-draft probe for the class-geometry check. Answers two questions with
measurements instead of readings, before the contract is written.

  Q1  Is `scipy.optimize.quadratic_assignment` available on the Modal image, at
      the sizes the ten-class arms need? `scipy` is in the image's pip_install
      UNPINNED (modal_runner.py:41), so the requirements file answers nothing
      about which version a container actually holds -- it has to be asked.

  Q2  Do the B1 pretrained checkpoints (unpermuted ViT and ResNet, `e12_base`
      and `e14_base`) extract through the SAME path as B6 (`e23_vit`/`e23_rn`)?

ON Q2, WHAT READING THE CODE ALREADY SHOWS, and why it is not enough. B6's
decomposition (`e23_decompose.py`) does not own an extraction path: it
`importlib`s `e12_decompose` / `e14_decompose` and calls their `load_era` and
`features_and_logits`. Those are B1's OWN modules -- B1 is the original
consumer and B6 the borrower. Neither function asserts a benchmark; the only
benchmark gate is `e23_decompose`'s arm check at its line 81. So on inspection
the answer is yes.

That is a premise, and a premise is not a property (catch 21/28). This probe
converts it into three printed facts per backbone: the checkpoint LOADS through
that loader (catch 20's criterion, not existence), the extraction runs on a B1
task, and `head(captured_feature) == deployed_logits` EXACTLY on live inputs, so
the features the class-geometry check would read are the deployed ones.

The arm fields are printed beside it because B1 and B6 differ in exactly ONE
field -- `benchmark`, `cifar100` vs `cifar100_permuted` -- and both are built
from `SplitCIFAR100Benchmark` with `shift_mode` None vs "patch". If any other
field differs, the class-geometry check is comparing two things and the
contract has to say so.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARMS = {                       # (run dir stem, ckpt stem, decomposition module)
    # The ckpt path carries the NESTED `mafc_seed{s}` component that
    # `e23_decompose.py:78` builds (`{root}/ckpt_{arm}_seed{s}/mafc_seed{s}`).
    # Omitting it made the first run of this probe report every B1 checkpoint
    # missing, which would have been read as "B1 has no checkpoints" -- the
    # inverse of catch 20's error, a live artifact declared absent by a wrong
    # path rather than a dead one declared present by a cheap check.
    "B1_vit": ("e12_base_seed{s}", "ckpt_e12_base_seed{s}/mafc_seed{s}", "scripts.e12_decompose"),
    "B1_rn":  ("e14_base_seed{s}", "ckpt_e14_base_seed{s}/mafc_seed{s}", "scripts.e14_decompose"),
    "B6_vit": ("e23_vit_seed{s}",  "ckpt_e23_vit_seed{s}/mafc_seed{s}",  "scripts.e12_decompose"),
    "B6_rn":  ("e23_rn_seed{s}",   "ckpt_e23_rn_seed{s}/mafc_seed{s}",   "scripts.e14_decompose"),
}
FIELDS = ["benchmark", "backbone", "model_type", "use_task_heads", "use_input_adapters",
          "era_checkpoints", "num_tasks", "num_classes", "epochs_per_task",
          "checkpoint_fp16", "remap_labels", "head_routing_by_hint"]


def q1(out):
    print("=" * 100 + "\nQ1  scipy on THIS container\n" + "=" * 100)
    import scipy
    rec = {"scipy_version": scipy.__version__, "numpy_version": np.__version__}
    print(f"  scipy {scipy.__version__}   numpy {np.__version__}   torch {torch.__version__}")
    try:
        from scipy.optimize import quadratic_assignment
        rec["quadratic_assignment_importable"] = True
    except Exception as e:
        rec["quadratic_assignment_importable"] = False
        rec["import_error"] = f"{type(e).__name__}: {e}"
        print(f"  quadratic_assignment NOT IMPORTABLE: {rec['import_error']}")
        out["q1"] = rec
        return
    # Exercise it at the sizes that matter, with a planted permutation it must
    # recover: an availability check that never runs the solver would not detect
    # a version whose signature or defaults moved.
    rec["solved"] = {}
    for n in (5, 10, 20):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(n, n)); A = A @ A.T
        P = rng.permutation(n)
        B = A[P][:, P]
        # SIGN CONVENTION, and it is a trap worth stating in the contract.
        # `quadratic_assignment` MINIMISES trace(A^T P B P^T). Matching two
        # matrices means MAXIMISING that, so the call passes -B. Called with +B
        # it asks the solver to ANTI-align them and the first run of this probe
        # duly reported "NOT exact" at every size -- an availability check that
        # would have condemned a perfectly available solver.
        per = {}
        for meth in ("faq", "2opt"):
            r = quadratic_assignment(A, -B, method=meth)
            got = float(np.abs(A - B[np.ix_(r.col_ind, r.col_ind)]).max())
            per[meth] = {"max_abs_reconstruction_err": got, "exact": bool(got < 1e-8)}
            print(f"  n={n:>3} {meth:<5}: planted permutation recovered to {got:.2e} "
                  f"-> {'EXACT' if got < 1e-8 else 'NOT exact'}")
        rec["solved"][n] = per
    out["q1"] = rec


def q2(out, seed, root, device):
    print("\n" + "=" * 100 + "\nQ2  do the B1 checkpoints extract through B6's path?\n" + "=" * 100)
    rec = {}
    print(f"  {'arm':<9}" + "".join(f"{f[:12]:>14}" for f in FIELDS[:5]))
    arms = {}
    for name, (rundir, _, _) in ARMS.items():
        p = f"{root}/{rundir.format(s=seed)}/mafc_results.json"
        if not os.path.exists(p):
            print(f"  {name:<9} run artifact ABSENT: {p}")
            arms[name] = None
            continue
        a = json.load(open(p))["arm"]
        arms[name] = a
        print(f"  {name:<9}" + "".join(f"{str(a.get(f)):>14}" for f in FIELDS[:5]))
    # which fields differ between B1 and B6, per backbone
    rec["field_diffs"] = {}
    for bk in ("vit", "rn"):
        a, b = arms.get(f"B1_{bk}"), arms.get(f"B6_{bk}")
        if not (a and b):
            continue
        d = [f for f in FIELDS if a.get(f) != b.get(f)]
        rec["field_diffs"][bk] = {f: [a.get(f), b.get(f)] for f in d}
        print(f"\n  {bk}: fields differing between B1 and B6 -> {d or 'NONE'}")
        for f in d:
            print(f"      {f}: B1={a.get(f)!r}  B6={b.get(f)!r}")

    # the three facts, per B1 arm
    rec["arms"] = {}
    from src.data.split_cifar100 import SplitCIFAR100Benchmark
    for name in ("B1_vit", "B1_rn"):
        rundir, ckpt, modname = ARMS[name]
        a = arms.get(name)
        if a is None:
            rec["arms"][name] = {"skipped": "run artifact absent"}
            continue
        r = {"module": modname}
        d = f"{root}/{ckpt.format(s=seed)}"
        print(f"\n  {name} via {modname}   ckpt {d}")
        dec = importlib.import_module(modname)
        T = int(a["num_tasks"]) - 1
        # (1) LOADS -- catch 20's criterion, not existence
        t0 = time.time()
        try:
            m = dec.load_era(d, T, device)
            r["loads"] = True
            print(f"    (1) load_era(task={T}) -> LOADS  ({time.time()-t0:.1f}s)")
        except Exception as e:
            r["loads"] = False
            r["load_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"    (1) load_era(task={T}) -> FAILED: {r['load_error']}")
            rec["arms"][name] = r
            continue
        # (2) extraction runs on a B1 task through the SAME function
        bench = SplitCIFAR100Benchmark(
            num_tasks=int(a["num_tasks"]), batch_size=64, root="/data", seed=seed,
            remap_labels=bool(a.get("remap_labels", True)),
            shift_mode="patch" if a["benchmark"] == "cifar100_permuted" else None,
            download=False)
        r["class_order_fingerprint"] = bench.class_order_fingerprint()
        r["shift_fingerprint"] = bench.shift_fingerprint()
        k = 0
        _, te = bench.get_task_loaders(k)
        f_te, l_te, y_te = dec.features_and_logits(m, te, k, False, device)
        r["feature_shape"] = list(f_te.shape)
        r["n_test"] = int(len(y_te))
        r["acc"] = float((l_te.argmax(1) == y_te).float().mean())
        print(f"    (2) features_and_logits(task {k}) -> features {list(f_te.shape)}, "
              f"n={len(y_te)}, deployed acc {r['acc']:.4f}")
        # (3) PATH IDENTITY, non-tautologically: the captured feature pushed
        #     through the deployed head must reproduce the deployed logits.
        head = None
        for getter in ("era_head_of",):
            try:
                from scripts.cure_screen import era_head_of
                head = era_head_of(m, k)
            except Exception as e:
                r["head_lookup_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        if head is not None:
            with torch.no_grad():
                re_l = head(f_te.to(device)).cpu()
            gap = float((re_l - l_te).abs().max())
            r["path_identity_max_abs_gap"] = gap
            r["path_identity"] = bool(gap < 1e-4)
            print(f"    (3) head(feature) vs deployed logits: max|gap| {gap:.3e} -> "
                  f"{'IDENTICAL (path proven)' if gap < 1e-4 else 'DIFFERS — the feature is not the deployed one'}")
        else:
            print(f"    (3) head lookup unavailable: {r.get('head_lookup_error')}")
        rec["arms"][name] = r
    out["q2"] = rec


def main():
    ap = argparse.ArgumentParser(description="class-geometry pre-draft probe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-root", default="/runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/runs/classgeom/probe.json")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = {"seed": args.seed, "device": str(device)}
    q1(out)
    q2(out, args.seed, args.ckpt_root.rstrip("/"), device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
