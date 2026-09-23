"""
E12 build gate — B1..B6, each row a verification, run in the EXECUTION environment.

Contract: docs/E12_prereg.md sec 1b. No training run enters the record until all
six are green. Every item is a fresh premise under catch 21 and is verified here,
not assumed anywhere.

The fingerprint set this prints (class order, shift spec, ViT weight hash) is
P0's record, and it is computed WHERE EXECUTED — E10's lesson: a fingerprint
certifies executed data only in the environment that builds it, and "the
registered value reproduces on my machine" is a statement about my machine.

Usage:
    python scripts/e12_build_gate.py                      # local (B2/B3 may be red)
    modal run modal_runner.py::analysis --argv "scripts/e12_build_gate.py"
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VIT_MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
DATA_ROOT = os.environ.get("E12_DATA_ROOT", "./data")

rows: list[dict] = []


def record(item, desc, ok, detail):
    rows.append({"item": item, "desc": desc, "ok": bool(ok), "detail": detail})
    return ok


# ------------------------------------------------------------------- B1/B6 --
def check_b1_b6():
    from src.data.split_cifar100 import SplitCIFAR100Benchmark, N_CLASSES_TOTAL

    try:
        b = SplitCIFAR100Benchmark(root=DATA_ROOT, download=False)
    except Exception as e:
        record("B1", "CIFAR-100 20x5 split", False, f"{type(e).__name__}: {e}")
        record("B6", "permuted-pixel shift", False, "blocked on B1")
        return

    seen: set[int] = set()
    disjoint = True
    for cs in b.task_classes:
        if len(cs) != 5 or (set(cs) & seen):
            disjoint = False
        seen |= set(cs)
    complete = len(seen) == N_CLASSES_TOTAL and len(b.task_classes) == 20
    counts_ok = True
    detail_counts = []
    for k in (0, 19):
        tr, te = b.get_task_loaders(k)
        n_tr, n_te = len(tr.dataset), len(te.dataset)
        counts_ok &= (n_tr == 2500 and n_te == 500)
        detail_counts.append(f"t{k}: {n_tr}/{n_te}")
    record("B1", "CIFAR-100 20x5 split",
           disjoint and complete and counts_ok,
           f"fingerprint {b.class_order_fingerprint()} | disjoint={disjoint} "
           f"complete={complete} | counts(train/test) {', '.join(detail_counts)}")

    # B6: the registered shift, verified by apply/invert/compare — exact.
    import torch
    from src.data.split_cifar100 import (apply_perm, invert_perm, IMAGE_SIZE,
                                         build_patch_consistent_perm)
    bs = SplitCIFAR100Benchmark(root=DATA_ROOT, download=False, shift_mode="patch")
    p = build_patch_consistent_perm(IMAGE_SIZE, seed=42 * 1000 + 1)
    x = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
    shifted = apply_perm(x, p)
    exact = torch.equal(apply_perm(shifted, invert_perm(p)), x)
    differs = not torch.equal(shifted, x)
    record("B6", "permuted-pixel shift (efficacy control data)", exact and differs,
           f"shift fingerprint {bs.shift_fingerprint()} | round-trip exact={exact} "
           f"| shift is non-identity={differs}")


# --------------------------------------------------------------------- B2 --
def check_b2():
    try:
        import timm
    except Exception as e:
        return record("B2", "timm + ViT-B/16 weights, hash-pinned", False,
                      f"timm not importable: {type(e).__name__}")
    try:
        m = timm.create_model(VIT_MODEL, pretrained=True)
    except Exception as e:
        return record("B2", "timm + ViT-B/16 weights, hash-pinned", False,
                      f"weights unavailable offline: {type(e).__name__}: {str(e)[:80]}")
    # The hash IS the pin: file presence proves nothing about which weights.
    h = hashlib.sha1()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    n_params = sum(p.numel() for p in m.parameters())
    record("B2", "timm + ViT-B/16 weights, hash-pinned", True,
           f"timm {timm.__version__} | {VIT_MODEL} | params {n_params/1e6:.1f}M "
           f"| WEIGHT HASH {h.hexdigest()[:12]}")


# --------------------------------------------------------------------- B3 --
def check_b3():
    p = Path(DATA_ROOT) / "cifar-100-python"
    if not p.exists():
        return record("B3", "CIFAR-100 present at start (never fetched at run time)",
                      False, f"absent at {p}")
    try:
        from torchvision import datasets
        tr = datasets.CIFAR100(root=DATA_ROOT, train=True, download=False)
        te = datasets.CIFAR100(root=DATA_ROOT, train=False, download=False)
        ok = len(tr) == 50000 and len(te) == 10000
        record("B3", "CIFAR-100 present at start (never fetched at run time)", ok,
               f"train {len(tr)}, test {len(te)} at {p}")
    except Exception as e:
        record("B3", "CIFAR-100 present at start", False, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------- B4 --
def _build_vit(n_tasks=2, num_classes=5):
    """A minimal E12-shaped model: ViT trunk, per-task heads, adapters on."""
    from src.models.plcm import PLCM
    m = PLCM(input_size=3, hidden_size=768, num_classes=num_classes,
             backbone="vit", use_memory=False, use_task_heads=True,
             head_routing_by_hint=True, use_input_adapters=True,
             adapter_dim=768, adapter_mode="per_step")
    for t in range(n_tasks):
        m.set_task(t)
    m.eval()
    return m


def check_b4():
    """Parity with timm, output shapes, trainability, and the blast-radius
    regression's own precondition (both amendments default OFF)."""
    import torch
    from src.models.plcm import PLCM

    # The amendments must be opt-in. Assert the DEFAULTS, never set them.
    probe = PLCM(input_size=28, hidden_size=32, num_classes=5)
    defaults_ok = (probe.backbone == "lstm"
                   and getattr(probe, "head_routing_by_hint", None) is False)
    if not defaults_ok:
        return record("B4", "ViT backbone path in PLCM", False,
                      "amendments are not default-off — blast-radius violation")
    try:
        m = _build_vit()
    except Exception as e:
        return record("B4", "ViT backbone path in PLCM", False,
                      f"{type(e).__name__}: {str(e)[:90]}")

    x = torch.randn(2, 3, 224, 224)
    # (a) the reimplemented feature path IS timm's, bitwise
    d_parity = m.lstm.assert_matches_timm(x)
    # (b) shapes
    with torch.no_grad():
        out = m(x, store_memories=False, task_hint=1)
    shapes_ok = (tuple(out["logits"].shape) == (2, 5)
                 and tuple(out["readout_feature"].shape) == (2, 768))
    # (c) trainable: gradients reach the trunk and the loss moves
    m.train()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    y = torch.tensor([0, 1])
    l0 = torch.nn.functional.cross_entropy(m(x, store_memories=False, task_hint=1)["logits"], y)
    l0.backward()
    g = m.lstm.vit.blocks[0].attn.qkv.weight.grad
    grads_ok = g is not None and float(g.abs().sum()) > 0
    opt.step(); opt.zero_grad()
    with torch.no_grad():
        l1 = torch.nn.functional.cross_entropy(
            m(x, store_memories=False, task_hint=1)["logits"], y)
    m.eval()
    record("B4", "ViT backbone path in PLCM",
           bool(d_parity == 0.0 and shapes_ok and grads_ok),
           f"timm parity max|d|={d_parity:.1e} | logits{tuple(out['logits'].shape)} "
           f"feature{tuple(out['readout_feature'].shape)} | trunk grads={grads_ok} "
           f"| loss {float(l0):.4f}->{float(l1):.4f} | amendments default-off=True")


# --------------------------------------------------------------------- B5 --
def check_b5():
    ok_b4 = any(r["item"] == "B4" and r["ok"] for r in rows)
    if not ok_b4:
        return record("B5", "instrument binding + P3a/P3b asserts, both controls fired",
                      False, "blocked on B4 (no ViT forward to bind to)")
    import torch
    from scripts.e12_p3 import assert_p3, positive_controls
    m = _build_vit()
    x = torch.randn(2, 3, 224, 224)
    try:
        cells = [assert_p3(m, x, k, label=f"task{k}") for k in (0, 1)]
    except AssertionError as e:
        return record("B5", "instrument binding + P3a/P3b asserts, both controls fired",
                      False, f"live assert failed: {str(e)[:90]}")
    ctrl = positive_controls(m, x, task_k=0, verbose=False)
    ok = ctrl["p3a_control"] and ctrl["p3b_control"]
    record("B5", "instrument binding + P3a/P3b asserts, both controls fired", ok,
           f"P3a max|d|={max(c['p3a_delta'] for c in cells):.1e} on 2 cells | "
           f"P3b module={cells[1]['p3b_module']} | "
           f"controls fired: P3a={ctrl['p3a_control']} P3b={ctrl['p3b_control']}")


def main():
    ap = argparse.ArgumentParser(description="E12 build gate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 104)
    print("E12 BUILD GATE — six items, six verifications, computed where executed")
    print("=" * 104)
    print(f"  {platform.python_version()} on {platform.machine()} / "
          f"{platform.system()}   data root {DATA_ROOT}")
    print()

    check_b3()          # B1's loaders need the data, so B3 is probed first
    check_b1_b6()
    check_b2()
    check_b4()
    check_b5()

    order = {"B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5}
    rows.sort(key=lambda r: order[r["item"]])
    print(f"  {'#':<4}{'item':<52}{'status':<10}detail")
    for r in rows:
        print(f"  {r['item']:<4}{r['desc']:<52}{'GREEN' if r['ok'] else 'BLOCKED':<10}"
              f"{r['detail']}")

    n_green = sum(r["ok"] for r in rows)
    print()
    print(f"  {n_green}/6 green")
    print(f"  P0 build gate: {'PASS — training may start' if n_green == 6 else 'HELD — no training run enters the record'}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")
    return 0 if n_green == 6 else 1


if __name__ == "__main__":
    sys.exit(main())
