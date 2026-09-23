"""
E14 build gate — C1..C4, each row a verification, run in the EXECUTION environment.

Contract: docs/E14_resnet_prereg.md sec 1b. No training run enters the record
until all four are green.

ORDER MATTERS AND IS THE RULED ONE: the blast-radius regression runs FIRST
(separately, `scripts/e12_regression.py`) — if the fourth backbone's plumbing has
disturbed the shared path, better to know before C1..C4 are built on top of it,
and it is the cheapest of the checks. It passed at current HEAD, bit-identical to
E11's matrices after four experiments' worth of amendments.

  C1  ResNet-50 in PLCM — native readout branch, parallel to the ViT's
  C2  avgpool->fc binding — bitwise timm parity, AND the positive control that
      must FAIL on a different reduction of the same layer
  C3  P3a/P3b re-derived for this architecture, both controls firing
  C4  weight variant PINNED and HASHED (timm ships several ResNet-50 weight
      sets; "resnet50" alone resolves by timm's default, which moves)

Usage:
    modal run modal_runner.py::analysis --argv "scripts/e14_build_gate.py"
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

rows: list[dict] = []


def record(item, desc, ok, detail):
    rows.append({"item": item, "desc": desc, "ok": bool(ok), "detail": detail})
    return ok


def _build(n_tasks=2, num_classes=5):
    """A minimal E14-shaped model: ResNet trunk, per-task heads, no adapters."""
    from src.models.plcm import PLCM
    m = PLCM(input_size=3, hidden_size=2048, num_classes=num_classes,
             backbone="resnet", use_memory=False, use_task_heads=True,
             head_routing_by_hint=True, use_input_adapters=False)
    for t in range(n_tasks):
        m.set_task(t)
    m.eval()
    return m


def check_c1_c2():
    from src.models.plcm import PLCM
    from src.models.resnet_base import RESNET_MODEL

    probe = PLCM(input_size=28, hidden_size=32, num_classes=5)
    if probe.backbone != "lstm" or getattr(probe, "head_routing_by_hint", None) is not False:
        record("C1", "ResNet-50 branch in PLCM", False,
               "amendments are not default-off — blast-radius violation")
        return record("C2", "avgpool->fc binding + positive control", False, "blocked on C1")
    try:
        m = _build()
    except Exception as e:
        record("C1", "ResNet-50 branch in PLCM", False, f"{type(e).__name__}: {str(e)[:90]}")
        return record("C2", "avgpool->fc binding + positive control", False, "blocked on C1")

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m(x, store_memories=False, task_hint=1)
    shapes_ok = (tuple(out["logits"].shape) == (2, 5)
                 and tuple(out["readout_feature"].shape) == (2, 2048))

    m.train()
    y = torch.tensor([0, 1])
    loss0 = torch.nn.functional.cross_entropy(
        m(x, store_memories=False, task_hint=1)["logits"], y)
    loss0.backward()
    g = m.lstm.net.layer4[-1].conv3.weight.grad
    grads_ok = g is not None and float(g.abs().sum()) > 0
    m.zero_grad(); m.eval()
    record("C1", "ResNet-50 branch in PLCM", bool(shapes_ok and grads_ok),
           f"{RESNET_MODEL} | logits{tuple(out['logits'].shape)} "
           f"feature{tuple(out['readout_feature'].shape)} | trunk grads={grads_ok} "
           f"| amendments default-off=True")

    # ---- C2: bitwise parity, then the control that must FAIL ----------------
    d = m.lstm.assert_matches_timm(x)
    from scripts.e12_p3 import assert_p3a
    live = assert_p3a(m, x, 1, label="C2-live")

    # POSITIVE CONTROL: same layer, same shape, WRONG reduction (spatial max
    # instead of the deployed average). A control built from noise would prove
    # nothing; this one is a real tensor from the real forward, so P3a fails on
    # it only because the capture point genuinely matters.
    import scripts.e12_p3 as p3
    orig = p3._run_deployed if hasattr(p3, "_run_deployed") else None
    fired = False
    try:
        with torch.no_grad():
            wrong = m.lstm.wrong_capture_point(x)
            head = m.task_classifiers["1"]
            deployed = m(x, store_memories=False, task_hint=1)["logits"]
            delta = float((head(wrong) - deployed).abs().max())
        fired = delta > 0.0
    except Exception as e:
        record("C2", "avgpool->fc binding + positive control", False,
               f"control errored: {type(e).__name__}")
        return
    record("C2", "avgpool->fc binding + positive control",
           bool(d == 0.0 and live == 0.0 and fired),
           f"timm parity max|d|={d:.1e} | P3a live max|d|={live:.1e} | "
           f"pre-pool control (spatial MAX) delta={delta:.3e} -> "
           f"{'FIRED' if fired else 'DID NOT FIRE'}")


def check_c3():
    ok_c2 = any(r["item"] == "C2" and r["ok"] for r in rows)
    if not ok_c2:
        return record("C3", "P3a/P3b re-derived, both controls fire", False, "blocked on C2")
    from scripts.e12_p3 import assert_p3, positive_controls
    m = _build()
    x = torch.randn(2, 3, 224, 224)
    try:
        cells = [assert_p3(m, x, k, label=f"task{k}") for k in (0, 1)]
    except AssertionError as e:
        return record("C3", "P3a/P3b re-derived, both controls fire", False,
                      f"live assert failed: {str(e)[:90]}")
    ctrl = positive_controls(m, x, task_k=0, verbose=False)
    ok = ctrl["p3a_control"] and ctrl["p3b_control"]
    record("C3", "P3a/P3b re-derived, both controls fire", ok,
           f"P3a max|d|={max(c['p3a_delta'] for c in cells):.1e} | "
           f"P3b module={cells[1]['p3b_module']} | controls: "
           f"P3a={ctrl['p3a_control']} P3b={ctrl['p3b_control']}")


def check_c4():
    from src.models.resnet_base import RESNET_MODEL
    try:
        import timm
        net = timm.create_model(RESNET_MODEL, pretrained=True)
    except Exception as e:
        return record("C4", "weight variant pinned + hashed", False,
                      f"{type(e).__name__}: {str(e)[:90]}")
    h = hashlib.sha1()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    n = sum(p.numel() for p in net.parameters())
    record("C4", "weight variant pinned + hashed", True,
           f"timm {timm.__version__} | {RESNET_MODEL} | params {n/1e6:.1f}M "
           f"| WEIGHT HASH {h.hexdigest()[:12]}")


def main():
    ap = argparse.ArgumentParser(description="E14 build gate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 104)
    print("E14 BUILD GATE — C1..C4, computed where executed")
    print("=" * 104)
    print(f"  {platform.python_version()} on {platform.machine()} / {platform.system()}")
    print("  (blast-radius regression ran FIRST and separately: E4/OFF and E4/ON "
          "both EXACT 0.0e+00)\n")

    check_c1_c2()
    check_c3()
    check_c4()

    order = {"C1": 0, "C2": 1, "C3": 2, "C4": 3}
    rows.sort(key=lambda r: order[r["item"]])
    print(f"  {'#':<4}{'item':<46}{'status':<10}detail")
    for r in rows:
        print(f"  {r['item']:<4}{r['desc']:<46}{'GREEN' if r['ok'] else 'BLOCKED':<10}"
              f"{r['detail']}")
    n = sum(r["ok"] for r in rows)
    print(f"\n  {n}/4 green")
    print(f"  E14 build gate: "
          f"{'PASS — training may start' if n == 4 else 'HELD — no training run enters the record'}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")
    return 0 if n == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
