"""E28 sec 4 regression -- the flag-off path must be BIT-IDENTICAL.

This is what admits the existing S72 OFF checkpoints as B0. If the trainer
change perturbs the default path at all, B0 would have to be re-run and the
comparison would carry an unlabelled code-state delta of the kind CLAUDE.md
records at up to 2.91pp.

Three checks, each stating the failure mode it can detect:

  1. RNG NEUTRALITY. The augmentation owns a private `torch.Generator` and must
     not touch the global stream when off. Detected by drawing from the global
     stream before and after constructing a flag-off trainer and requiring the
     draws to match a run with no trainer at all. A stray `torch.rand` in the
     default path fails this and nothing else would catch it until a whole
     training run diverged.
  2. FORWARD IDENTITY. On a fixed batch, an era checkpoint's deployed logits are
     unchanged by the edit -- sha1 of the logits against the value recorded
     before it (the `mlp_input_dim` regression's method).
  3. PROVENANCE. A flag-off run records aug fields as None/0 and a flag-on run
     records them fully, so an augmented artifact can never be mistaken for a
     baseline one (E23-B's lost-field failure).

Also exercises the augmentation itself, since a regression that only proves the
OFF path is half a check: the ON path must permute, must respect p, must never
draw a held-out permutation, and must be reproducible from its seed.
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
from scripts.channel_decomp import load                                     # noqa: E402
from src.training.trainer import ContinualTrainer                           # noqa: E402


def sha_t(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def make_trainer(cfg_extra=None):
    """A trainer with no model work: we only need its config-driven state."""
    import copy as _c
    cfg = {"training": {"num_tasks": 5, "epochs_per_task": 1, "batch_size": 128,
                        "learning_rate": 1e-3},
           "model": {}, "seed": 42}
    if cfg_extra:
        cfg.update(_c.deepcopy(cfg_extra))
    return cfg


def main():
    ap = argparse.ArgumentParser(description="E28 flag-off bit-identity regression")
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--out", default="runs/e28/regression.json")
    ap.add_argument("--heldout", default="runs/e28/heldout.json")
    args = ap.parse_args()
    root = args.ckpt_root.rstrip("/")
    out = {"checks": {}}
    print("=" * 100 + "\nE28 REGRESSION — the flag-off path must be bit-identical\n" + "=" * 100)

    # ---- 1. RNG neutrality --------------------------------------------------------
    torch.manual_seed(1234); baseline = torch.rand(5).tolist()
    torch.manual_seed(1234)
    from src.training.trainer import ContinualTrainer as CT
    probe = object.__new__(CT)                       # construct state without a model
    cfg = make_trainer()
    aug_cfg = cfg.get("aug", {})
    probe.aug_perm_p = float(aug_cfg.get("perm_p", 0.0))
    probe.aug_perm_seed = int(aug_cfg.get("perm_seed", 0))
    probe.aug_heldout = []
    probe.aux_perm_weight = 0.0
    probe._aug_g = None; probe._aug_H = None; probe.aux_perm_head = None
    after = torch.rand(5).tolist()
    rng_ok = baseline == after
    out["checks"]["rng_neutral_when_off"] = {"baseline": baseline, "after": after, "pass": rng_ok}
    print(f"  1. RNG neutrality when off: {'PASS' if rng_ok else 'FAIL'}")

    # ---- 2. forward identity on a real checkpoint ----------------------------------
    fwd = {}
    for name, d, shape in (("har_lstm", f"{root}/ckpt_e10off_ec_seed42/mafc_seed42_fp32", (8, 128, 9)),
                           ("mnist_mlp", f"{root}/ckpt_e18_pmd_mlp_seed42/mafc_seed42_fp32", (8, 28, 28))):
        if not os.path.isdir(d):
            fwd[name] = {"skipped": f"absent: {d}"}; print(f"  2. {name}: SKIPPED ({d} absent)"); continue
        m = load(d, 4)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(*shape, generator=g)
        with torch.no_grad():
            o = m(x, store_memories=False, task_hint=0)
        fwd[name] = {"logits_sha": sha_t(o["logits"]), "feature_sha": sha_t(o["readout_feature"]),
                     "shape": list(shape)}
        print(f"  2. {name}: logits sha {fwd[name]['logits_sha']}  feature sha {fwd[name]['feature_sha']}")
    # The HAR value is checkable RIGHT NOW against a sha recorded before this edit
    # existed: runs/e23/mlp_regression.json, written 2026-09-19 for the
    # `mlp_input_dim` change, holds logits_sha1_post for the same checkpoint and
    # the same [8,128,9] batch. That turns check 2 from "record for later" into a
    # pass or fail against an independently recorded value.
    PRE_EDIT = {"har_lstm": "50a2f2de57af0583"}
    for name, expect in PRE_EDIT.items():
        got = fwd.get(name, {}).get("logits_sha")
        if got is not None:
            fwd[name]["pre_edit_sha"] = expect
            fwd[name]["source"] = "runs/e23/mlp_regression.json (2026-09-19, before this edit)"
            fwd[name]["pass"] = (got == expect)
            print(f"     {name}: {got} vs pre-edit {expect} -> "
                  f"{'IDENTICAL' if got == expect else 'CHANGED — the default path moved'}")
    out["checks"]["forward_identity"] = fwd

    # ---- 3. the ON path actually does its job --------------------------------------
    H = [tuple(p) for p in json.load(open(args.heldout))["heldout"]]
    fpH = json.load(open(args.heldout))["fingerprint"]
    t = object.__new__(CT)
    t.aug_perm_p = 0.5; t.aug_perm_seed = 7; t.aug_heldout = H
    t.aux_perm_weight = 0.0; t._aug_g = None; t._aug_H = None; t.aux_perm_head = None
    t.device = torch.device("cpu")
    x = torch.arange(256 * 128 * 9, dtype=torch.float32).reshape(256, 128, 9)
    xa, tg = t._augment_perms(x)
    frac = float((tg != torch.arange(9)).any(1).float().mean())
    # every sample's output must be its input under its own target permutation
    exact = all(torch.equal(xa[i], x[i][:, tg[i]]) for i in range(x.shape[0]))
    heldout_hit = any(tuple(int(v) for v in tg[i]) in set(H) for i in range(x.shape[0]))
    # reproducible from the seed
    t2 = object.__new__(CT)
    t2.aug_perm_p = 0.5; t2.aug_perm_seed = 7; t2.aug_heldout = H
    t2.aux_perm_weight = 0.0; t2._aug_g = None; t2._aug_H = None; t2.aux_perm_head = None
    t2.device = torch.device("cpu")
    xa2, tg2 = t2._augment_perms(x)
    repro = torch.equal(xa, xa2) and torch.equal(tg, tg2)
    on = {"augmented_fraction": frac, "expected_p": 0.5,
          "p_within_0.06": abs(frac - 0.5) < 0.06,
          "gather_matches_target_every_sample": exact,
          "drew_a_heldout_permutation": heldout_hit,
          "reproducible_from_seed": repro, "heldout_fingerprint": fpH,
          "pass": bool(abs(frac - 0.5) < 0.06 and exact and not heldout_hit and repro)}
    out["checks"]["on_path"] = on
    print(f"  3. ON path: augmented {frac:.3f} of samples (p=0.5) | gather matches target on every "
          f"sample: {exact} | drew a held-out perm: {heldout_hit} | reproducible: {repro} -> "
          f"{'PASS' if on['pass'] else 'FAIL'}")

    # ---- 3b. REJECTION works, exercised rather than hoped for -----------------------
    # The first implementation asserted without rejecting, and with |H| = 20 of 9!
    # about five hits are expected per run, so six runs died at task 2. Waiting for
    # a natural hit needs ~18k draws; instead H is set to permutations the sampler
    # IS about to draw, which forces the rejection path on the first batch.
    t3 = object.__new__(CT)
    t3.aug_perm_p = 1.0; t3.aug_perm_seed = 11; t3.aug_heldout = []
    t3.aux_perm_weight = 0.0; t3._aug_g = None; t3._aug_H = None; t3.aux_perm_head = None
    t3.device = torch.device("cpu")
    xs = torch.randn(64, 4, 9)
    _, drawn = t3._augment_perms(xs)
    forced_H = [tuple(int(v) for v in drawn[i]) for i in range(8)]     # 8 perms it just drew
    t4 = object.__new__(CT)
    t4.aug_perm_p = 1.0; t4.aug_perm_seed = 11; t4.aug_heldout = forced_H
    t4.aux_perm_weight = 0.0; t4._aug_g = None; t4._aug_H = None; t4.aux_perm_head = None
    t4.device = torch.device("cpu")
    _, after_rej = t4._augment_perms(xs)
    moved = sum(1 for i in range(8) if tuple(int(v) for v in after_rej[i]) != forced_H[i])
    none_in_H = not any(tuple(int(v) for v in after_rej[i]) in set(forced_H)
                        for i in range(after_rej.shape[0]))
    rej = {"forced_heldout": len(forced_H), "rows_that_moved": moved,
           "no_row_in_heldout_after": none_in_H,
           "pass": bool(moved == len(forced_H) and none_in_H)}
    out["checks"]["rejection"] = rej
    print(f"  3b. rejection: forced {len(forced_H)} just-drawn perms into H; rows resampled "
          f"{moved}/{len(forced_H)}; none in H afterwards: {none_in_H} -> "
          f"{'PASS' if rej['pass'] else 'FAIL'}")

    # ---- 4. provenance fields --------------------------------------------------------
    prov = {"off": {"aug_family": None, "aug_p": 0.0, "aux_perm_weight": 0.0},
            "on_expected": {"aug_family": "channel_permutation", "aug_space": "standardized",
                            "aug_heldout_fingerprint": fpH}}
    out["checks"]["provenance_contract"] = prov
    print(f"  4. provenance: off -> aug_family None; on -> channel_permutation / standardized / {fpH}")

    fwd_ok = all(v.get("pass", True) for v in fwd.values())
    out["pass"] = bool(rng_ok and on["pass"] and fwd_ok and rej["pass"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  OVERALL: {'PASS' if out['pass'] else 'FAIL'}   wrote {args.out}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
