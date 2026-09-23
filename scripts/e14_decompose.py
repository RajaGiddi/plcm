"""
E14 — the three-channel decomposition on a ResNet-50. One (arm, seed) per invocation.

Contract: docs/E14_resnet_prereg.md sec 3-4. Measures, per old task k at final theta_T:

    acc_ceiling(k)          task k at its OWN end-of-task checkpoint
    acc_orig(k)             deployed accuracy at theta_T (the collapse)
    acc_refit(k)            convex probe on the FROZEN theta_T deployed feature
    acc_refit_ceiling(k)    same recipe on the era checkpoint (recipe control)

    R      = acc_refit_ceiling - acc_ceiling      instrument bias
    F_enc  = acc_refit_ceiling - acc_refit        encoder channel
    F_read = acc_refit         - acc_orig         reader channel
    F_total= acc_ceiling       - acc_orig         deployed forgetting
    identity: F_enc + F_read - R == F_total       exact; printed per cell

THE COLUMN THIS EXPERIMENT EXISTS FOR is `acc_refit` at task 0. The frozen-probe
arm proved the PRETRAINED features support 0.974 probes; task 0 collapses to
~0.16 through a head that is frozen, correct, and verifiably selected. If a probe
refit on the FINE-TUNED trunk's features recovers task 0 near its ceiling, the
reader channel swallows the collapse and H-V1 fires. If the refit also fails, the
trunk genuinely overwrote and E12 is branch (C) with the boundary measured.

WHY IT RUNS ON MODAL. The 38GB of era checkpoints live on the volume, and the
data environment that produced them is Modal's — the E10/E11 rule is that a
number is recomputed where its inputs were built, not where it is convenient.

GATES, re-fired here rather than inherited (asserts are per-script, not
per-session — B5 certified a model constructed in the build gate, which is not
the model train.py writes):
  * P3a/P3b on every live cell, plus BOTH positive controls before any of them
  * fp16 reload fidelity: the checkpoint's stored boundary accuracy vs the
    accuracy recomputed from the reload. Boundary-time is authoritative; the
    delta is measured and reported, never assumed (catch 29).

Usage:
    modal run --detach modal_runner.py::spawn_experiment --experiment e12dec
    python scripts/e12_decompose.py --arm base --seed 42 --out runs/e12_decomp/base_42.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.channel_decomp import refit_probe, _SKLEARN_VERSION
from scripts.e12_p3 import assert_p3, positive_controls
from src.data.split_cifar100 import SplitCIFAR100Benchmark
from src.models.plcm import PLCM

NUM_TASKS = 20
FINAL = NUM_TASKS - 1          # theta_T
OLD_TASKS = list(range(NUM_TASKS - 1))     # 0..18
EPOCH_IDX = 4                  # era checkpoints written at epochs_per_task - 1
BATCH = 128                    # 224x224 ViT-B/16 inference on an L4
R_BAR = 0.05                   # per-arm pooled instrument-bias gate


def load_era(ckpt_dir: str, task: int, device):
    m = PLCM.load_era(ckpt_dir, task, epoch=EPOCH_IDX)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def features_and_logits(model, loader, task_k: int, use_adapter: bool, device):
    """Deployed feature and deployed logits over a FULL split, from forward()."""
    F, L, Y = [], [], []
    for x, y in loader:
        out = model(x.to(device), store_memories=False, task_hint=task_k,
                    apply_adapter=use_adapter)
        F.append(out["readout_feature"].float().cpu())
        L.append(out["logits"].float().cpu())
        Y.append(y)
    return torch.cat(F), torch.cat(L), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser(description="E12 ViT decomposition (one arm/seed)")
    ap.add_argument("--arm", choices=["base"], default="base")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    use_adapter = (args.arm == "adapt")
    class_il = (args.arm == "cil")
    # Class-IL keeps GLOBAL class ids and scores them with ONE shared head, so
    # the probe's class count and the checkpoint path both differ.
    n_classes = 100 if class_il else 5
    ckpt_dir = os.path.join(args.ckpt_root,
                            f"ckpt_e14_{args.arm}_seed{args.seed}",
                            f"mafc_seed{args.seed}")
    out_path = args.out or f"runs/e14_decomp/{args.arm}_{args.seed}.json"

    print("=" * 100)
    print(f"E14 DECOMPOSITION — arm={args.arm} seed={args.seed} device={device}")
    print("=" * 100)
    print(f"  checkpoints: {ckpt_dir}")

    bench = SplitCIFAR100Benchmark(num_tasks=NUM_TASKS, batch_size=BATCH,
                                   root=args.root, seed=args.seed,
                                   remap_labels=not class_il, download=False)
    print(f"  class order fingerprint {bench.class_order_fingerprint()}")

    m_final = load_era(ckpt_dir, FINAL, device)

    # ---- gates, before any live cell ---------------------------------------
    print("\n" + "-" * 100)
    print("P3 GATES (re-fired on the trained ViT checkpoints, not inherited)")
    print("-" * 100)
    _, te0 = bench.get_task_loaders(0)
    xprobe = next(iter(te0))[0][:16].to(device)
    if class_il:
        # P3b asks "was task k scored by head k". Class-IL has ONE head by
        # definition, so the question has no referent — this is inapplicability
        # by construction, not a skipped check, and the failure mode P3b guards
        # (wrong head selected) cannot arise where there is nothing to select.
        # Declared here rather than silently omitted.
        from scripts.e12_p3 import assert_p3a, _capture
        try:
            assert_p3a(m_final, xprobe, 0, label="CONTROL", pre_norm=True)
            fired = False
        except AssertionError as e:
            fired = True
            print(f"    P3a control fired: {str(e)[:96]}...")
        assert fired, "P3a positive control did not fire — the gate is inert"
        print("    P3b: N/A — class-IL has a single shared readout "
              "(use_task_heads=false), so 'which head' has no referent")
    else:
        ctrl = positive_controls(m_final, xprobe, task_k=0, verbose=True)
        assert ctrl["p3a_control"] and ctrl["p3b_control"], (
            f"positive controls did not both fire: {ctrl} — a gate that cannot "
            f"fail is not evidence, so no cell below is readable")
        print(f"  both positive controls FIRED on the ViT path")

    # ---- PRECONDITION 2 (catch 32): label alignment, proven not assumed -------
    # The probe's features and labels must come from the same ordering. Checked
    # bitwise against the dataset's own indexing on a shared input: loader
    # position i must be dataset[i]. Non-tautological — a shuffled loader fails
    # it immediately, which is precisely the defect that inverted this
    # experiment's conclusion once already.
    _tr0, _ = bench.get_task_loaders(0)
    _seq = DataLoader(_tr0.dataset, batch_size=BATCH, shuffle=False)
    _xs, _ys = [], []
    for _x, _y in _seq:
        _xs.append(_x); _ys.append(_y)
        if sum(t.shape[0] for t in _xs) >= 64:
            break
    _xl, _yl = torch.cat(_xs)[:64], torch.cat(_ys)[:64]
    _ds = _tr0.dataset
    _xd = torch.stack([_ds[i][0] for i in range(64)])
    _yd = torch.tensor([_ds[i][1] for i in range(64)])
    assert torch.equal(_xl, _xd) and torch.equal(_yl, _yd), (
        "LABEL ALIGNMENT FAILED — loader order does not match dataset indexing; "
        "every probe below would train on misaligned labels (catch 32)")
    print(f"  catch-32 label alignment: loader order == dataset indexing, "
          f"bitwise on 64 samples -> PASS")

    # ---- class-IL only: the all-seen-classes probe (sec 1, unmasked column) ---
    # Fit ONCE over every task's train split in the 100-class label space, then
    # scored per task. This measures recoverability PLUS cross-task
    # discriminability — the extra burden class-IL imposes — and is reported
    # beside the masked column, never in place of it.
    #
    # WHAT THIS QUANTITY IS, AND IS NOT. Fitting jointly over all 20 splits
    # grants the probe a resource the deployed class-IL system never had:
    # simultaneous access to every task's data. So this is a JOINT-ACCESS READER
    # CEILING — it upper-bounds what ANY reader could recover from these
    # features, which is exactly what makes it the clean measure of
    # recoverability-plus-discriminability, and exactly why it must never be
    # quoted as "what a cure would achieve". A storage-honest repair has no such
    # access. The masked column remains the like-for-like decomposition
    # quantity; this one is the class-IL burden's ceiling.
    unmasked_probe = None
    if class_il:
        print("  fitting the all-seen-classes probe (100-way, all 20 train splits)")
        Xs, Ys = [], []
        for j in range(NUM_TASKS):
            trj, _ = bench.get_task_loaders(j)
            trj = DataLoader(trj.dataset, batch_size=BATCH, shuffle=False)
            fj, _, yj = features_and_logits(m_final, trj, j, use_adapter, device)
            Xs.append(fj); Ys.append(yj)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xtr, Ytr = torch.cat(Xs).numpy(), torch.cat(Ys).numpy()
        sc100 = StandardScaler().fit(Xtr)
        clf100 = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)
        clf100.fit(sc100.transform(Xtr), Ytr)
        unmasked_probe = (sc100, clf100)
        print(f"    fit on {Xtr.shape[0]} samples, {len(set(Ytr.tolist()))} classes "
              f"(chance 0.0100)")

    rows = []
    t0 = time.time()
    for k in OLD_TASKS:
        tr, te = bench.get_task_loaders(k)
        # DETERMINISTIC ORDER FOR FEATURE EXTRACTION. get_task_loaders returns a
        # shuffle=True train loader, so two passes yield two different
        # permutations. Extracting features in one pass and reusing labels from
        # another silently pairs row i of A with label i of B — the probe then
        # trains on randomly relabelled data and reads chance. That is the exact
        # defect channel_decomp.load_task_data documents ("produced R ~ -0.63 and
        # was misread as optimizer underfitting"), and it is why acc_refit_ceiling
        # looked healthy while acc_refit read chance: the ceiling's features and
        # labels came from the SAME pass.
        # Fixed structurally rather than by careful pairing: a sequential loader
        # makes every pass identically ordered, so misalignment is impossible
        # rather than merely avoided.
        tr = DataLoader(tr.dataset, batch_size=BATCH, shuffle=False)
        m_era = load_era(ckpt_dir, k, device)

        # P3a/P3b on BOTH eras for this cell.
        if class_il:
            from scripts.e12_p3 import assert_p3a
            g_era = {"p3a_delta": assert_p3a(m_era, xprobe, k, f"era{k}")}
            g_fin = {"p3a_delta": assert_p3a(m_final, xprobe, k, f"final/T{k}")}
        else:
            g_era = assert_p3(m_era, xprobe, k, label=f"era{k}")
            g_fin = assert_p3(m_final, xprobe, k, label=f"final/T{k}")

        # fp16 reload fidelity: stored boundary value vs the reload's own number.
        ck = torch.load(Path(ckpt_dir) / f"task{k}_epoch{EPOCH_IDX}.pt",
                        weights_only=True, map_location="cpu")
        acc_boundary = ck.get("acc_boundary")

        f_e_tr, _, y_tr = features_and_logits(m_era, tr, k, use_adapter, device)
        f_e_te, l_e_te, y_te = features_and_logits(m_era, te, k, use_adapter, device)
        f_4_tr, _, y_tr4 = features_and_logits(m_final, tr, k, use_adapter, device)
        # Non-tautological: with a sequential loader the two passes MUST agree
        # element-wise. If this ever fires, the loader is not deterministic and
        # every refit below is training on shuffled labels.
        assert torch.equal(y_tr, y_tr4), (
            "train labels differ between extraction passes — the feature loader "
            "is not deterministic and the refit would train on misaligned labels")
        f_4_te, l_4_te, _ = features_and_logits(m_final, te, k, use_adapter, device)

        acc_ceiling = float((l_e_te.argmax(1) == y_te).float().mean())
        acc_orig = float((l_4_te.argmax(1) == y_te).float().mean())
        i_rc, i_r4 = {}, {}
        acc_rc = refit_probe(f_e_tr, y_tr, f_e_te, y_te, n_classes, info=i_rc)
        acc_r4 = refit_probe(f_4_tr, y_tr, f_4_te, y_te, n_classes, info=i_r4)

        # CLASS-IL CONFOUND, measured rather than carried silently. The deployed
        # head chooses among 100 classes; the refit probe is fit on task k's own
        # train split and so only ever discriminates that task's 5. F_read would
        # then absorb the TASK-IDENTITY information the probe gets for free, and
        # report it as reader mismatch. `acc_orig_masked` restricts the deployed
        # logits to the task's own classes, matching the probe's problem: the gap
        # between the two IS the task-identity component the recency-bias
        # literature is about, so it is reported instead of being hidden inside
        # F_read.
        acc_refit_unmasked = None
        if class_il and unmasked_probe is not None:
            sc100, clf100 = unmasked_probe
            pred100 = clf100.predict(sc100.transform(f_4_te.numpy()))
            acc_refit_unmasked = float((pred100 == y_te.numpy()).mean())

        acc_orig_masked = None
        if class_il:
            cols = torch.tensor(sorted(set(int(v) for v in y_te)))
            sub = l_4_te[:, cols]
            pred = cols[sub.argmax(1)]
            acc_orig_masked = float((pred == y_te).float().mean())

        # ---- H-R4 SPAN DIAGNOSTIC (descriptive, NO BAR) --------------------
        # E15 measured span rotation under stable features on the ViT; a second
        # backbone family costs nothing now the machinery is here. The era
        # control prints BEFORE any span statement, same 0.05 instrument bar as
        # E15's — and the rank ratio is even more favourable (5/2048 vs 5/768),
        # so an era control that passes while theta_T projection recovers
        # nothing is the mechanism measured twice across two backbone families.
        from scripts.transport_estimate import span_of as _span_of
        _cls = sorted(set(int(v) for v in y_tr))
        _B = torch.stack([f_e_tr[y_tr == c].mean(0) for c in _cls]).numpy().astype(np.float64)
        _Pb = _span_of(_B)
        _P = torch.from_numpy((_Pb @ _Pb.T).astype(np.float32)).to(device)
        _h4 = (m_final.task_classifiers[str(k)]
               if str(k) in m_final.task_classifiers else m_final.classifier)
        _he = (m_era.task_classifiers[str(k)]
               if str(k) in m_era.task_classifiers else m_era.classifier)
        with torch.no_grad():
            span_proj_t4 = float((_h4(f_4_te.to(device) @ _P).argmax(1).cpu()
                                  == y_te).float().mean())
            _proj_era = float((_he(f_e_te.to(device) @ _P).argmax(1).cpu()
                               == y_te).float().mean())
        span_rank = int(_Pb.shape[1])
        span_ctrl_gap = acc_ceiling - _proj_era

        R = acc_rc - acc_ceiling
        F_enc = acc_rc - acc_r4
        F_read = acc_r4 - acc_orig
        F_total = acc_ceiling - acc_orig
        resid = abs(F_enc + F_read - R - F_total)
        fp16_delta = (None if acc_boundary is None
                      else float(acc_ceiling - acc_boundary))

        rows.append(dict(arm=args.arm, seed=args.seed, task=k,
                         acc_ceiling=acc_ceiling, acc_orig=acc_orig,
                         acc_refit=acc_r4, acc_refit_ceiling=acc_rc,
                         R=R, F_enc=F_enc, F_read=F_read, F_total=F_total,
                         identity_resid=resid, n_test=int(y_te.shape[0]),
                         acc_boundary=acc_boundary, fp16_delta=fp16_delta,
                         acc_orig_masked=acc_orig_masked,
                         acc_refit_unmasked=acc_refit_unmasked,
                         F_read_masked=(None if acc_orig_masked is None
                                        else acc_r4 - acc_orig_masked),
                         p3a=max(g_era["p3a_delta"], g_fin["p3a_delta"]),
                         span_rank=span_rank, span_ctrl_gap=span_ctrl_gap,
                         span_proj_t4=span_proj_t4,
                         probe_n_iter={"ceiling": i_rc.get("probe_n_iter"),
                                       "t4": i_r4.get("probe_n_iter")},
                         sklearn_version=_SKLEARN_VERSION))
        share = F_read / (F_enc + F_read) if (F_enc + F_read) else float("nan")
        print(f"  T{k:<2} ceil {acc_ceiling:.4f} orig {acc_orig:.4f} "
              f"refit {acc_r4:.4f} refit_ceil {acc_rc:.4f} | "
              f"R {R:+.4f} F_enc {F_enc:+.4f} F_read {F_read:+.4f} "
              f"share {share*100:5.1f}% | id {resid:.1e}"
              f"{'' if fp16_delta is None else f' | fp16 d {fp16_delta:+.4f}'}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        del m_era

    # ---- summary, computed from the rows above -----------------------------
    m = lambda key: float(np.mean([r[key] for r in rows]))
    fe, fr = m("F_enc"), m("F_read")
    share = fr / (fe + fr) if (fe + fr) else float("nan")
    worst_id = max(r["identity_resid"] for r in rows)
    worst_p3a = max(r["p3a"] for r in rows)
    fp16 = [r["fp16_delta"] for r in rows if r["fp16_delta"] is not None]

    print("\n" + "=" * 100)
    print(f"  identity   max |F_enc + F_read - R - F_total| = {worst_id:.2e} "
          f"-> {'OK' if worst_id < 1e-9 else 'COMPUTATION ERROR'}")
    print(f"  P3a        max |head(feature) - logits| = {worst_p3a:.2e}")
    if fp16:
        print(f"  fp16 reload delta vs stored boundary: mean "
              f"{np.mean(fp16):+.5f}  max |d| {max(abs(x) for x in fp16):.5f}  "
              f"(boundary-time value is authoritative)")
    print(f"  R (pooled) {m('R'):+.4f} vs bar {R_BAR} -> "
          f"{'PASS' if abs(m('R')) <= R_BAR else 'FAIL'}")
    print(f"  F_total {m('F_total'):.4f}  F_enc {fe:.4f}  F_read {fr:.4f}  "
          f"reader share {share*100:.1f}%")
    _g = [r["span_ctrl_gap"] for r in rows]
    print(f"\n  H-R4 span diagnostic (descriptive, no bar), rank "
          f"{rows[0]['span_rank']}/2048:")
    print(f"    era control  mean (ceiling - proj@era) {np.mean(_g):+.4f} "
          f"(max {max(_g):+.4f}) vs 0.05 -> "
          f"{'PASS - readable' if np.mean(_g) <= 0.05 else 'FAIL - unreadable at this rank'}")
    print(f"    theta_T span-projection {np.mean([r['span_proj_t4'] for r in rows]):.4f} "
          f"vs deployed {m('acc_orig'):.4f} vs refit {m('acc_refit'):.4f}")
    print(f"  task-0: ceiling {rows[0]['acc_ceiling']:.4f} -> deployed "
          f"{rows[0]['acc_orig']:.4f} -> REFIT {rows[0]['acc_refit']:.4f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
