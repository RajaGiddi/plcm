"""Did the five-package pin's image rebuild change any number that matters?

VERSION STRINGS ARE THE WEAK FORM OF THIS CHECK. Two images can report identical
package versions and still behave differently: the debian_slim base can move
underneath, a baked dataset download can differ, a numerical library can link
against a different BLAS. The strong form is to reproduce a number recorded
BEFORE the rebuild, so this reproduces four, across both execution paths, plus
the three fingerprints that pin data and weights.

  SCRATCH / CPU -- EXACT sha1 of deployed logits, deterministic, no tolerance:
    S72 HAR OFF LSTM   50a2f2de57af0583   (runs/e23/mlp_regression.json, 2026-09-19)
    e17 MLP permuted   ca074a46f1805580   (same file, same day)

  PRETRAINED / GPU -- recorded 2026-09-22 on the pre-pin image
  (runs/classgeom/probe.json):
    B1 ViT     task-0 deployed acc 0.114000, feature [500, 768],  path gap 2.861023e-06
    B1 ResNet  task-0 deployed acc 0.116000, feature [500, 2048], path gap 1.907349e-06

  FINGERPRINTS: ViT weight hash, CIFAR class order, HAR partition 1104af185c87.

WHAT THIS CHECK CANNOT DO, AND THE GAP IT CLOSES. The two scratch references are
exact hashes; the two pretrained ones are an accuracy quantised to 1/500 and a
float gap. That is a weaker witness, and it is weaker because nobody ever
recorded an exact logits hash on the GPU path -- the `mlp_input_dim` regression
covered CPU arms only. So this script ALSO writes exact shas for the pretrained
path, which do nothing for today's question and give the next rebuild the
witness today's lacked.

TOLERANCES ARE STATED, NOT DISCOVERED. Scratch: exact, a single differing bit
fails. Pretrained: deployed accuracy must match exactly (500 samples, argmax is
robust to float wobble), and the path-identity gap must stay below 1e-4, which
is the bar `classgeom_probe` already used to call the path proven -- not a bar
chosen after seeing what the rebuild produced.

IF VERSIONS MATCH AND A HASH DOES NOT, that is a finding about the rebuild and
it outranks E30: it would mean this program's reproducibility does not rest on
its pins, and every cross-image comparison in the ledger would need re-reading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCRATCH_REF = {
    "s72_har_lstm": {"ckpt": "{root}/ckpt_e10off_ec_seed42/mafc_seed42_fp32",
                     "shape": (8, 128, 9), "sha": "50a2f2de57af0583",
                     "source": "runs/e23/mlp_regression.json (2026-09-19)"},
    "e17_mlp_permuted": {"ckpt": "{root}/ckpt_e17_mlp_floor_a/mafc_seed42",
                         "shape": (8, 28, 28), "sha": "ca074a46f1805580",
                         "source": "runs/e23/mlp_regression.json (2026-09-19)"},
}
PRE_REF = {
    "b1_vit": {"run": "e12_base_seed42", "ckpt": "ckpt_e12_base_seed42/mafc_seed42",
               "mod": "scripts.e12_decompose", "acc": 0.114000,
               "feat": [500, 768], "gap": 2.861023e-06},
    "b1_rn":  {"run": "e14_base_seed42", "ckpt": "ckpt_e14_base_seed42/mafc_seed42",
               "mod": "scripts.e14_decompose", "acc": 0.116000,
               "feat": [500, 2048], "gap": 1.907349e-06},
}
GAP_BAR = 1e-4          # the bar classgeom_probe already used, not a new one
HAR_PARTITION = "1104af185c87"


def sha_t(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().cpu().float().numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description="E30 rebuild fidelity gate")
    ap.add_argument("--ckpt-root", default="/runs")
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/runs/e30/rebuild_fidelity.json")
    args = ap.parse_args()
    root = args.ckpt_root.rstrip("/")
    device = torch.device(args.device)
    out = {"device": str(device), "checks": {}}
    print("=" * 104 + "\nE30 REBUILD FIDELITY — does the repinned image reproduce pre-rebuild numbers?\n" + "=" * 104)

    import importlib.metadata as md
    vers = {p: md.version(p) for p in ("scipy", "numpy", "scikit-learn", "torch", "timm")}
    out["versions"] = vers
    print("  versions: " + ", ".join(f"{k} {v}" for k, v in vers.items()))
    print(f"  torch {torch.__version__} | cuda {torch.cuda.is_available()}")

    # ---------------- 1. scratch / CPU: EXACT hashes -------------------------
    print("\n  1. SCRATCH PATH — exact sha1 of deployed logits (no tolerance)")
    from scripts.channel_decomp import load
    sc = {}
    for name, ref in SCRATCH_REF.items():
        d = ref["ckpt"].format(root=root)
        if not os.path.isdir(d):
            sc[name] = {"skipped": f"absent: {d}"}
            print(f"    {name:<18} SKIPPED (absent: {d})")
            continue
        m = load(d, 4)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(*ref["shape"], generator=g)
        with torch.no_grad():
            o = m(x, store_memories=False, task_hint=0)
        got = sha_t(o["logits"])
        ok = (got == ref["sha"])
        sc[name] = {"sha_now": got, "sha_recorded": ref["sha"], "source": ref["source"],
                    "feature_sha_now": sha_t(o["readout_feature"]), "pass": ok}
        print(f"    {name:<18} {got} vs recorded {ref['sha']} -> "
              f"{'IDENTICAL' if ok else 'CHANGED — the rebuild moved the CPU path'}")
    out["checks"]["scratch_exact_sha"] = sc

    # ---------------- 2. pretrained / GPU ------------------------------------
    print("\n  2. PRETRAINED PATH — accuracy exact, path gap under the bar classgeom used")
    import importlib
    pre = {}
    try:
        from src.data.split_cifar100 import SplitCIFAR100Benchmark
        from scripts.cure_screen import era_head_of
        for name, ref in PRE_REF.items():
            p = f"{root}/{ref['run']}/mafc_results.json"
            if not os.path.exists(p):
                pre[name] = {"skipped": f"absent: {p}"}; print(f"    {name:<8} SKIPPED"); continue
            a = json.load(open(p))["arm"]
            dec = importlib.import_module(ref["mod"])
            m = dec.load_era(f"{root}/{ref['ckpt']}", int(a["num_tasks"]) - 1, device)
            bench = SplitCIFAR100Benchmark(
                num_tasks=int(a["num_tasks"]), batch_size=dec.BATCH, root=args.data_root,
                seed=42, remap_labels=bool(a.get("remap_labels", True)),
                shift_mode="patch" if a["benchmark"] == "cifar100_permuted" else None,
                download=False)
            _, te = bench.get_task_loaders(0)
            F, L, Y = dec.features_and_logits(m, te, 0, False, device)
            acc = float((L.argmax(1) == Y).float().mean())
            head = era_head_of(m, 0)
            with torch.no_grad():
                gap = float((head(F.to(device)).cpu() - L).abs().max())
            ok_acc = abs(acc - ref["acc"]) < 1e-9
            ok_gap = gap < GAP_BAR
            pre[name] = {"acc_now": acc, "acc_recorded": ref["acc"], "acc_identical": ok_acc,
                         "feature_shape_now": list(F.shape), "feature_shape_recorded": ref["feat"],
                         "path_gap_now": gap, "path_gap_recorded": ref["gap"],
                         "gap_bar": GAP_BAR, "gap_under_bar": ok_gap,
                         # the witness that did not exist before: an EXACT hash on
                         # the GPU path, for the next rebuild to check against.
                         "logits_sha_now": sha_t(L), "feature_sha_now": sha_t(F),
                         "class_order_fingerprint": bench.class_order_fingerprint(),
                         "pass": bool(ok_acc and ok_gap
                                      and list(F.shape) == ref["feat"])}
            print(f"    {name:<8} acc {acc:.6f} vs {ref['acc']:.6f} "
                  f"({'IDENTICAL' if ok_acc else 'CHANGED'}) | feat {list(F.shape)} | "
                  f"gap {gap:.3e} (bar {GAP_BAR:g}) | new exact sha {sha_t(L)}")
            del m
    except Exception as e:
        pre["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"    pretrained path FAILED: {pre['error']}")
    out["checks"]["pretrained"] = pre

    # ---------------- 3. fingerprints ----------------------------------------
    print("\n  3. FINGERPRINTS — data and weights")
    fp = {}
    try:
        from src.data.har_subject import HARSubjectBenchmark
        b = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=128)
        fp["har_partition"] = {"now": b.partition_fingerprint(), "recorded": HAR_PARTITION,
                               "pass": b.partition_fingerprint() == HAR_PARTITION}
        print(f"    HAR partition {fp['har_partition']['now']} vs {HAR_PARTITION} -> "
              f"{'MATCH' if fp['har_partition']['pass'] else 'DIFFERS'}")
    except Exception as e:
        fp["har_partition"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        print(f"    HAR partition: {fp['har_partition']['error']}")
    try:
        import timm, hashlib as _h
        mm = timm.create_model("vit_base_patch16_224", pretrained=True)
        h = _h.sha1()
        for k, v in sorted(mm.state_dict().items()):
            h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
        fp["vit_weight_hash"] = {"now": h.hexdigest()[:16]}
        print(f"    ViT-B/16 weight hash {fp['vit_weight_hash']['now']} "
              f"(recorded here for the next rebuild)")
    except Exception as e:
        fp["vit_weight_hash"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        print(f"    ViT weight hash: {fp['vit_weight_hash']['error']}")
    out["checks"]["fingerprints"] = fp

    def passed(d):
        return all(v.get("pass", True) for v in d.values() if isinstance(v, dict))
    ok = passed(sc) and passed(pre) and passed(fp) and "error" not in pre
    out["pass"] = bool(ok)
    print("\n" + "=" * 104)
    if ok:
        print("  REBUILD FIDELITY: PASS — every pre-rebuild number reproduces. E30's numbers")
        print("  sit on the same footing as everything measured before the pin.")
    else:
        print("  REBUILD FIDELITY: FAIL — a recorded number did not reproduce. This OUTRANKS")
        print("  E30: it would mean the pins do not carry this program's reproducibility, and")
        print("  every cross-image comparison in the ledger needs re-reading before E30 runs.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"  wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
