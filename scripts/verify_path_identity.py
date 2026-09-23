"""
E11 Step 1 — prove the instrument reads the deployed path (catch 28).

Contract: docs/E11_instrument_correction.md

Runs assert_path_identity() on EVERY arm type in the program -- OFF (no adapters,
no task heads), ON (adapters), and task-heads -- on both HAR benchmarks and
permuted MNIST. Nothing downstream may be recomputed until this is green.

It also runs the POSITIVE CONTROL the gate requires (catch 25): a gate that has
only ever passed is indistinguishable from a gate that cannot fail. So P-A is
fired here on v3's own feature -- the exact tensor catch 28 was about -- where it
MUST fail. If the control passes silently, the gate is worthless and this script
exits non-zero.

Usage:
    python scripts/verify_path_identity.py
    python scripts/verify_path_identity.py --skip-mnist
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.channel_decomp as cd
from scripts.channel_decomp import load, load_task_data, assert_path_identity
from src.data.har_shift import HARShiftBenchmark
from src.data.har_subject import HARSubjectBenchmark
from src.data.permuted_mnist import PermutedMNISTBenchmark

SEED = 42
N_PROBE = 64          # samples per identity check
TASKS = [0, 1, 2, 3]  # the old tasks every downstream number is computed on

# (label, checkpoint dir, use_adapter, benchmark key)
ARMS = [
    ("E10/OFF",      "runs/ckpt_e10_off_seed42/mafc_seed42",       False, "har_subject"),
    ("E10/ON",       "runs/ckpt_e10_on_seed42/mafc_seed42",        True,  "har_subject"),
    ("E5/OFF",       "runs/ckpt_e5_off_seed42/mafc_seed42",        False, "har_shift"),
    ("E5/v1-ON",     "runs/ckpt_e5_seed42/mafc_seed42",            True,  "har_shift"),
    ("E7/heads",     "runs/ckpt_e7_heads_seed42/mafc_seed42",      False, "har_shift"),
    ("E7/heads+ad",  "runs/ckpt_e7_heads_adapt_seed42/mafc_seed42", True, "har_shift"),
    ("E4/OFF-mnist", "checkpoints/e4_off_seed42/mafc_seed42",      False, "mnist"),
    ("E4/ON-mnist",  "checkpoints/fullrank_ref/mafc_seed42",       True,  "mnist"),
]


def get_data(key, cache):
    if key not in cache:
        if key == "har_subject":
            b = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=256)
        elif key == "har_shift":
            b = HARShiftBenchmark(num_tasks=5, root=".", batch_size=256)
        else:
            b = PermutedMNISTBenchmark(num_tasks=5, batch_size=256, seed=SEED)
        cache[key] = load_task_data(b, n_tasks=5)
    return cache[key]


def positive_control(model, x, task_k, use_adapter, device) -> bool:
    """Fire P-A on v3's feature (o_t*tanh(c_t), raw cell state). It MUST fail.

    The corruption is injected at the module boundary rather than through a knob
    on the gate, so the gate itself is the unmodified one used everywhere else.
    Returns True iff the gate fired.
    """
    orig = cd._run_deployed

    def wrong(m, xb, k, ua):
        out, caps = orig(m, xb, k, ua)
        key = str(k)
        xin = (m._apply_adapter(xb, key)
               if (ua and key in m.task_adapters) else xb)
        lstm_out, (_, c_n) = m.lstm(xin)
        c_t = c_n[-1]
        o_t = torch.sigmoid(m.output_gate(torch.cat([lstm_out[:, -1, :], c_t], -1)))
        out = dict(out)
        out["readout_feature"] = o_t * torch.tanh(c_t)   # <- v3's asserted feature
        return out, caps

    cd._run_deployed = wrong
    try:
        assert_path_identity(model, x, task_k, use_adapter, device,
                             label="POSITIVE CONTROL", n=N_PROBE, verbose=False)
        return False                      # gate did NOT fire -> gate is worthless
    except AssertionError as e:
        print(f"    gate fired as required: {str(e)[:110]}...")
        return True
    finally:
        cd._run_deployed = orig


def main():
    ap = argparse.ArgumentParser(description="E11 step 1: deployed-path identity")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-mnist", action="store_true")
    ap.add_argument("--only", default=None,
                    help="comma-separated benchmark keys to run "
                         "(har_subject, har_shift, mnist); default = all")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    device = torch.device(args.device)
    cache = {}

    print("=" * 100)
    print("E11 STEP 1 — DEPLOYED-PATH IDENTITY   (P-A: instrument reads the "
          "deployed tensor; P-B: one head reproduces deployed logits)")
    print("=" * 100)

    rows, missing, control_ok = [], [], None
    for label, d, use_ad, bench in ARMS:
        if bench == "mnist" and args.skip_mnist:
            continue
        if only and bench not in only:
            continue
        if not Path(d).exists():
            missing.append((label, d))
            continue
        data = get_data(bench, cache)
        model = load(d, 4)
        n_heads = len(model.task_classifiers)
        print(f"\n{label}   heads={getattr(model, 'use_task_heads', False)}"
              f"({n_heads})  adapters={model.use_input_adapters}"
              f"({len(model.task_adapters)})  bank={model.memory_bank.size}"
              f"  use_memory={model.use_memory}")
        for k in TASKS:
            xte = data[k][2]
            rows.append(dict(arm=label, task=k,
                             **assert_path_identity(model, xte, k, use_ad, device,
                                                    label=f"task {k}", n=N_PROBE)))
        if control_ok is None:                      # once is enough; it is algebraic
            print("  POSITIVE CONTROL (catch 25) — P-A on v3's feature, must FAIL:")
            control_ok = positive_control(model, data[0][2], 0, use_ad, device)

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    if missing:
        for label, d in missing:
            print(f"  MISSING CHECKPOINT  {label:<14} {d}")
    print(f"  P-A  passed on {len(rows)}/{len(rows)} arm-task cells "
          f"(any failure would have raised)")
    print(f"  POSITIVE CONTROL  {'PASS — the gate can fail' if control_ok else 'FAIL — GATE IS INERT'}")

    bad = [r for r in rows if not r["p_b"]]
    if bad:
        print(f"\n  P-B FAILED on {len(bad)}/{len(rows)} cells — the deployed "
              f"prediction is not any single head's argmax on these arms:")
        for arm in sorted({r["arm"] for r in bad}):
            rs = [r for r in bad if r["arm"] == arm]
            print(f"    {arm:<14} tasks {[r['task'] for r in rs]}  "
                  f"max|Δlogit| up to {max(r['d_logit'] for r in rs):.3f}  "
                  f"pred-disagree up to {max(r['d_pred'] for r in rs)*100:.1f}%")
        print("  -> per E11 contract: STOP AND REPORT. Do not special-case.")
    else:
        print(f"  P-B  passed on all {len(rows)} cells")

    worst_v3 = max((r["d_v3_feature"] for r in rows), default=0.0)
    print(f"\n  v3-vs-deployed feature drift (the catch-28 defect, sized): "
          f"max {worst_v3:.3f} over {len(rows)} cells")
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        print(f"    {arm:<14} max {max(r['d_v3_feature'] for r in rs):.4f}")

    ok = control_ok and not missing
    print(f"\n  STEP 1 {'GREEN' if ok and not bad else 'NOT GREEN'} — "
          f"{'proceed to step 2' if ok and not bad else 'see above'}")
    return 0 if (ok and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
