"""
E11 Step 4b — E4b/E6 geometry recomputed in the space the classifier reads.

Ruling (2026-08-02): the lemma W(I - P_S) = 0 is about the feature the classifier
READS. E4b's `subspace_drift.py` and E6's `drift_anatomy.py` both compute drift on
the RAW cell state c_t (`encode()` returns `c[-1]`), then project it onto the row
space of `classifier.weight` -- a matrix that acts on h' = o_t * tanh(c_t'). Right
subspace, wrong space. So the program's two spine numbers (MNIST ON/OFF drift_S
ratio 0.214, HAR 0.970) and the in-S energy "aim" fractions are exposed.

SCOPE: headline ratios and aim fractions only. The full hypothesis apparatus is
not re-run -- those branches closed on comparative logic that likely survives --
but these NUMBERS are quoted in the outline and ledger-adjacent text.

Both spaces are computed and printed SIDE BY SIDE:
    v3  raw c_t                     -- as published
    v4  deployed h' = o*tanh(c')    -- what the classifier actually reads

BLAST RADIUS, recorded before the numbers are read: E6's H-B1 bar is
`H_B1_BAR = 0.43  # 2 x MNIST's measured ON/OFF drift_S ratio of 0.214`
(drift_anatomy.py:45). A pre-registered threshold in one experiment was DERIVED
from a measured quantity in another, and this script recomputes that quantity.
The bar is NOT moved here -- re-baring after seeing data is the error this
program exists to avoid -- but its derivation input has changed and that is a
fact about the bar's provenance, not a licence to adjust it.

Usage:
    python scripts/subspace_recompute.py --dataset mnist        # local corpus
    modal run modal_runner.py::analysis --argv "scripts/subspace_recompute.py --dataset har"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.channel_decomp import load, load_task_data, _run_deployed, assert_path_identity
from scripts.subspace_drift import (readout_projector, random_projector, drift_in,
                                    RANK, N_RANDOM, RANDOM_SEED)
from src.data.har_shift import HARShiftBenchmark
from src.data.permuted_mnist import PermutedMNISTBenchmark

SEEDS = [42, 1337, 2024]
N_EVAL = 2000

MNIST_ARMS = {
    "ON":  ({42: "checkpoints/fullrank_ref/mafc_seed42",
             1337: "checkpoints/e4_on_seed1337/mafc_seed1337",
             2024: "checkpoints/e4_on_seed2024/mafc_seed2024"}, True),
    "OFF": ({s: f"checkpoints/e4_off_seed{s}/mafc_seed{s}" for s in SEEDS}, False),
}
HAR_ARMS = {
    "v1-ON":      ("runs/ckpt_e5_seed{s}/mafc_seed{s}", True),
    "OFF":        ("runs/ckpt_e5_off_seed{s}/mafc_seed{s}", False),
    "v2-ON":      ("runs/ckpt_e5d_v2on_seed{s}/mafc_seed{s}", True),
    "v2-control": ("runs/ckpt_e5d_v2ctl_seed{s}/mafc_seed{s}", True),
}
# Published values this recompute supersedes, for the side-by-side.
PUBLISHED = {"mnist_ratio": 0.214, "har_ratio": 0.970}


@torch.no_grad()
def encode_v3(model, x, use_adapter, device, bs=512):
    """The PUBLISHED representation: raw final-layer cell state c_t."""
    out = []
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].to(device)
        if use_adapter and "0" in model.task_adapters:
            xb = model._apply_adapter(xb, "0")
        _, (_, c) = model.lstm(xb)
        out.append(c[-1].cpu())
    return torch.cat(out)


@torch.no_grad()
def encode_v4(model, x, use_adapter, device, bs=512):
    """The DEPLOYED representation: h' = o_t * tanh(c_t'), read off forward()."""
    out = []
    for i in range(0, x.shape[0], bs):
        o, _ = _run_deployed(model, x[i:i + bs].to(device), 0, use_adapter)
        out.append(o["readout_feature"].cpu())
    return torch.cat(out)


def geometry(f0, f4, W, seed):
    """drift_full / drift_S / in-S energy fraction, in whatever space f lives."""
    P_S = readout_projector(W)
    gen = torch.Generator().manual_seed(RANDOM_SEED + seed)
    rand = [drift_in(random_projector(f0.shape[1], RANK, gen), f0, f4)
            for _ in range(N_RANDOM)]
    delta = f4 - f0
    return {
        "drift_full": float(1.0 - nn.functional.cosine_similarity(f0, f4, dim=1).mean()),
        "drift_S": drift_in(P_S, f0, f4),
        "drift_random_mean": float(np.mean(rand)),
        "inS_energy_fraction": float((delta @ P_S).pow(2).sum() / delta.pow(2).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="E11 step 4b: geometry in the read space")
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "har"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/e11_subspace/")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    arms = MNIST_ARMS if args.dataset == "mnist" else HAR_ARMS
    hdata = (load_task_data(HARShiftBenchmark(num_tasks=5, root=".", batch_size=256), 1)
             if args.dataset == "har" else None)

    rows = []
    for arm, (dirs, use_ad) in arms.items():
        for seed in SEEDS:
            d = dirs[seed] if isinstance(dirs, dict) else dirs.format(s=seed)
            if not Path(d).exists():
                print(f"  MISSING {arm}/{seed}: {d}")
                continue
            if args.dataset == "har":
                x = hdata[0][2][:N_EVAL]
            else:
                x = load_task_data(PermutedMNISTBenchmark(
                    num_tasks=5, batch_size=256, seed=seed), 1)[0][2][:N_EVAL]
            m0, m4 = load(d, 0), load(d, 4)
            # The permanent gate, on both eras, before either is measured.
            assert_path_identity(m0, x, 0, use_ad, device, label=f"{arm}/s{seed}/m0",
                                 verbose=False)
            assert_path_identity(m4, x, 0, use_ad, device, label=f"{arm}/s{seed}/m4",
                                 verbose=False)
            W = m0.classifier.weight.detach()      # the era's readout, as published
            r = {"arm": arm, "seed": seed}
            for tag, enc in (("v3", encode_v3), ("v4", encode_v4)):
                f0, f4 = enc(m0, x, use_ad, device), enc(m4, x, use_ad, device)
                for k, v in geometry(f0, f4, W, seed).items():
                    r[f"{tag}_{k}"] = v
            rows.append(r)

    agg = lambda arm, k: float(np.mean([r[k] for r in rows if r["arm"] == arm]))

    print("=" * 100)
    print(f"E11 STEP 4b — SUBSPACE GEOMETRY, PUBLISHED SPACE vs READ SPACE "
          f"({args.dataset}, n={len(SEEDS)})")
    print("=" * 100)
    print(f"  {'arm':<14}{'drift_S v3':>13}{'drift_S v4':>13}{'in-S energy v3':>17}"
          f"{'in-S energy v4':>17}{'drift_full v4':>15}")
    for arm in arms:
        if not any(r["arm"] == arm for r in rows):
            continue
        print(f"  {arm:<14}{agg(arm,'v3_drift_S'):>13.4f}{agg(arm,'v4_drift_S'):>13.4f}"
              f"{agg(arm,'v3_inS_energy_fraction')*100:>16.2f}%"
              f"{agg(arm,'v4_inS_energy_fraction')*100:>16.2f}%"
              f"{agg(arm,'v4_drift_full'):>15.4f}")

    print("\n" + "=" * 100)
    print("HEADLINE RATIO — ON/OFF drift_S  (the spine number)")
    print("=" * 100)
    on_key = "ON" if args.dataset == "mnist" else "v1-ON"
    pub = PUBLISHED["mnist_ratio" if args.dataset == "mnist" else "har_ratio"]
    for tag in ("v3", "v4"):
        on, off = agg(on_key, f"{tag}_drift_S"), agg("OFF", f"{tag}_drift_S")
        ratio = on / off if off else float("nan")
        print(f"  {tag}: {on_key} {on:.4f} / OFF {off:.4f} = {ratio:.4f}"
              f"{'   <- as published: ' + str(pub) if tag == 'v3' else ''}")
    on4, off4 = agg(on_key, "v4_drift_S"), agg("OFF", "v4_drift_S")
    r4 = on4 / off4 if off4 else float("nan")
    on3, off3 = agg(on_key, "v3_drift_S"), agg("OFF", "v3_drift_S")
    r3 = on3 / off3 if off3 else float("nan")
    # Verdict computed from r3/r4 printed above, not from an expectation.
    moved = abs(r4 - r3)
    print(f"\n  ratio moves {r3:.4f} -> {r4:.4f}  (|delta| = {moved:.4f})")
    print(f"  reproduction of the published value: v3 ratio {r3:.4f} vs published "
          f"{pub} -> {'reproduces' if abs(r3 - pub) < 0.02 else 'DOES NOT reproduce'}")
    if args.dataset == "mnist":
        print(f"  qualitative split (MNIST steered = ratio well below 1): "
              f"v3 {'steered' if r3 < 0.6 else 'unsteered'} -> "
              f"v4 {'steered' if r4 < 0.6 else 'unsteered'}"
              f"   [{'SURVIVES' if (r3 < 0.6) == (r4 < 0.6) else 'FLIPS'}]")
    else:
        print(f"  qualitative split (HAR unsteered = ratio near 1): "
              f"v3 {'unsteered' if r3 > 0.6 else 'steered'} -> "
              f"v4 {'unsteered' if r4 > 0.6 else 'steered'}"
              f"   [{'SURVIVES' if (r3 > 0.6) == (r4 > 0.6) else 'FLIPS'}]")
    print(f"\n  NOTE (blast radius): E6's H_B1_BAR = 0.43 was frozen as 2 x the "
          f"MNIST v3 ratio 0.214.\n  It is NOT moved by this recompute; its "
          f"derivation input having changed is recorded, not acted on.")

    json.dump(rows, open(os.path.join(args.out, f"subspace_{args.dataset}.json"), "w"),
              indent=2)
    print(f"\n  wrote {args.out}subspace_{args.dataset}.json")


if __name__ == "__main__":
    main()
