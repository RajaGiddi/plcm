"""C0deg controls -- the two rulings that gate the row's entry into the ledger, plus
one must-fail the rulings did not ask for.

WHY THIS EXISTS. `cure_screen.py` labels C0deg (A_k := M_4 . M_k^-1, old-task
inputs re-laid-out into the CURRENT frame, scored by the CURRENT model) a
"CEILING ARTIFACT, never a cure", on a justification written for E8's
shared-window HAR: there every task was the same 2947 windows, so relayout into
task 4's frame reproduced task 4's test set exactly. E10 rebuilt the benchmark
around subject-disjoint content -- the property that justification depends on --
and the label was carried across unexamined. Catch 21 ("validated is
context-bound") applied to an EXCLUSION rather than an inclusion.

On E10 the column was computed and never read. Read through the ledger's own
H-X2 instrument (hx2_forgetting.py --include-c0deg) it is the best storage-honest
method in the program: HAR/OFF forgetting -0.0247 [-0.0346, -0.0148] against
bridging's 0.0886, intervals separated, and it uses a strict SUBSET of bridging's
resources (the maps only; no era snapshot, no refit, no pseudo-labels). Catch 24:
the trivial use of the resource dominates the method.

A number that changes the paper's center enters the ledger through controls, not
through a recompute. Ruled sequence: controls -> ledger row -> E16 §7.2 closes ->
bridging's regime decided -> E18 arm table. This script is the first step.

CONTROLS, each with a verdict DERIVED from the values printed beside it:

  G1  SUBJECT DISJOINTNESS, from the benchmark object, citing `test_subjects`.
      The premise C0deg's non-tautology rests on: task k's test subject must not
      appear in ANY task's training group, so the current model has never seen
      the windows it is scored on. Fails if TEST_SUBJECTS_PER_GROUP were 0 or a
      subject were reused across groups.
  G2  MAP ALGEBRA. channel_affine(0) must be the identity (the no-shift benchmark
      routes every task through it -- src/data/har_subject.py:127), and the E10
      round-trip gate is re-fired unchanged.
  CA  NO-SHIFT POSITIVE CONTROL (ruling a). On `har_subject_noshift` every task
      shares one frame, so M_4 . M_k^-1 = I and C0deg must equal the deployed
      accuracy EXACTLY -- bit-identical logits, not "close". This bounds what the
      map can recover: presentation drift only. The arm's own forgetting (content
      change with no shift) is printed beside it because it is the part of E10's
      forgetting that no map can touch, and the scope statement needs the number.
  CB  ARTIFACT CONSISTENCY. C0deg and acc_orig re-scored live on the OFF and ON
      checkpoints must agree with `runs/e11_e10/cures_e10.json` to float32
      representation (1e-6, hx2_forgetting's rationale). Two scripts, one run:
      if they disagree, the stored column does not describe these checkpoints.
  CC  WRONG-MAP MUST-FAIL (not in the ruling; catch 25 requires it). Re-lay task k
      out as if it were task j = (k+1) mod 4. If the IDENTITY of the map is
      load-bearing, the wrong map must score below the right one in EVERY cell.
      A gate that has only ever passed is indistinguishable from one that cannot
      fail; this is the case where it must.

ENVIRONMENT. The E10 partition is architecture-dependent (subjects 2/5 are an
exact tie in the greedy balancer; runs used 1104af185c87, the laptop produces
5d047e4213d1 -- runs/MEMO_catch28_supersession.md). Scoring against the
checkpoints is therefore GATED on the as-executed fingerprint, not merely
printed. `--algebra-only` runs G1/G2 anywhere; everything else needs Modal.

ON THE ON ARM. C0deg suppresses the learned adapter through the deployed path
(forward(apply_adapter=False)) and substitutes the analytic map -- the same
semantics cure_screen.py:195-200 uses. It is a different arm from OFF and is
reported as such; the OFF arm is the bridging benchmark's.

Usage:
    python scripts/c0deg_controls.py --algebra-only
    modal run modal_runner.py::analysis --argv "scripts/c0deg_controls.py"
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.har_shift import channel_affine, spec_fingerprint, N_CHANNELS
from src.data.har_subject import HARSubjectBenchmark
from scripts.channel_decomp import (load, features_and_logits, load_task_data,
                                    assert_path_identity)
from scripts.cure_screen import (har_maps, har_relayout, generator_gate_e10,
                                 E10_HAR_ARMS, CUR, SEEDS, BATCH)

REGISTERED_SHIFT = "3de66e205eb7"       # docs/E10_prereg.md:17
AS_EXECUTED_PARTITION = "1104af185c87"  # runs/MEMO_catch28_supersession.md, x86
REGISTERED_PARTITION = "5d047e4213d1"   # docs/E10_prereg.md:18, the laptop's
NOSHIFT_ARM = "runs/ckpt_e10_noshift_seed{s}/mafc_seed{s}"
MATRIX = {"OFF": "runs/e10_off_seed{s}/mafc_results.json",
          "ON": "runs/e10_on_seed{s}/mafc_results.json",
          "NOSHIFT": "runs/e10_noshift_seed{s}/mafc_results.json"}
CONSISTENCY_TOL = 1e-6                  # float32 representation, per hx2_forgetting
OLD = list(range(CUR))                  # tasks 0..3


def banner(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


def acc(logits, y):
    return float((logits.argmax(1) == y).float().mean())


# ------------------------------------------------------------------- G1 / G2 --
def gate_disjointness(har) -> dict:
    banner("G1  SUBJECT DISJOINTNESS  (premise cited to HARSubjectBenchmark.test_subjects)")
    all_train = set()
    for tr in har.train_subjects:
        all_train |= set(tr)
    rows, ok = [], True
    for k in range(har.num_tasks):
        te = list(har.test_subjects[k])
        overlap = sorted(set(te) & all_train)
        rows.append({"task": k, "train": list(har.train_subjects[k]),
                     "test": te, "overlap_with_any_train": overlap})
        ok &= (len(te) == 1 and not overlap)
        print(f"  task {k}: train {list(har.train_subjects[k])}  test {te}"
              f"  overlap-with-any-train {overlap or 'none'}")
    tests = [s for te in har.test_subjects for s in te]
    distinct = len(set(tests)) == len(tests)
    ok &= distinct
    print(f"  test subjects {tests}: {'pairwise distinct' if distinct else 'REUSED'}")
    print(f"  G1 -> {'PASS' if ok else 'FAIL'}")
    return {"pass": bool(ok), "rows": rows, "test_subjects": tests}


def gate_map_algebra(hdata, maps) -> dict:
    banner("G2  MAP ALGEBRA")
    M0, b0 = channel_affine(0)
    ident = bool(np.array_equal(M0, np.eye(N_CHANNELS)) and np.all(b0 == 0))
    print(f"  channel_affine(0) == (I, 0): {ident}   "
          f"(no-shift benchmark routes every task through it, har_subject.py:127)")
    generator_gate_e10(hdata, maps)     # sys.exit(2) on failure, unchanged
    print(f"  G2 -> {'PASS' if ident else 'FAIL'}")
    return {"base_map_is_identity": ident}


# ------------------------------------------------------------------ scoring --
def matrix_final_row(arm, seed):
    m = np.array(json.load(open(MATRIX[arm].format(s=seed)))["accuracy_matrix"], float)
    return m


def score_arm(tag, tmpl, use_adapter, data, maps_right, maps_wrong, device, cures_ref):
    """Per (seed, task): deployed acc, C0deg with the right maps, C0deg with the
    wrong maps, ceiling from the era model. Consistency against the matrix and,
    where given, against cures_e10.json."""
    rows = []
    for s in SEEDS:
        d = tmpl.format(s=s)
        m4 = load(d, CUR)
        mat = matrix_final_row(tag, s)
        for k in OLD:
            xtr, ytr, xte, yte = data[k]
            g4 = assert_path_identity(m4, xte, k, use_adapter, device,
                                      label=f"{tag}/t4/T{k}", verbose=False)
            m_era = load(d, k)
            _, l_era = features_and_logits(m_era, xte, yte, k, use_adapter, device)
            _, l_orig = features_and_logits(m4, xte, yte, k, use_adapter, device)
            x_cur = har_relayout(xte, k, CUR, maps_right)
            _, l_cd = features_and_logits(m4, x_cur, yte, k, False, device)
            row = {"arm": tag, "seed": s, "task": k, "p_b": bool(g4["p_b"]),
                   "acc_ceiling": acc(l_era, yte), "acc_orig": acc(l_orig, yte),
                   "C0deg": acc(l_cd, yte),
                   "max_abs_dlogit_vs_orig": float((l_cd - l_orig).abs().max()),
                   "matrix_final_row": float(mat[CUR, k]),
                   "matrix_diag": float(mat[k, k])}
            if maps_wrong is not None:
                j = (k + 1) % CUR
                x_wrong = har_relayout(xte, j, CUR, maps_wrong)   # treat task k as task j
                _, l_w = features_and_logits(m4, x_wrong, yte, k, False, device)
                row["wrong_map_task"] = j
                row["C0deg_wrong"] = acc(l_w, yte)
            if cures_ref is not None:
                ref = cures_ref[(s, k)]
                row["ref_C0deg"] = ref["C0deg"]
                row["ref_acc_orig"] = ref["acc_orig"]
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="C0deg controls (E10)")
    ap.add_argument("--algebra-only", action="store_true",
                    help="G1/G2 only; no checkpoints, runs on any partition")
    ap.add_argument("--cures", default="runs/e11_e10/cures_e10.json")
    ap.add_argument("--out", default="runs/e10/c0deg_controls.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    banner("C0deg CONTROLS")
    print(f"  platform {platform.machine()}  torch {torch.__version__}  numpy {np.__version__}")
    fp = spec_fingerprint()
    print(f"  shift fingerprint {fp} -> {'MATCH' if fp == REGISTERED_SHIFT else 'MISMATCH - STOP'}")
    if fp != REGISTERED_SHIFT:
        sys.exit(1)

    har = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=BATCH)
    pfp = har.partition_fingerprint()
    executed = pfp == AS_EXECUTED_PARTITION
    print(f"  partition fingerprint {pfp} -> "
          f"{'AS-EXECUTED (x86)' if executed else ('REGISTERED/laptop' if pfp == REGISTERED_PARTITION else 'UNKNOWN')}")
    maps = har_maps(har._sd)
    hdata = load_task_data(har, n_tasks=5)

    out = {"platform": platform.machine(), "shift_fingerprint": fp,
           "partition_fingerprint": pfp, "partition_as_executed": executed,
           "G1": gate_disjointness(har), "G2": gate_map_algebra(hdata, maps)}
    if args.algebra_only:
        print("\n  --algebra-only: stopping before any checkpoint is scored.")
        return
    if not executed:
        print(f"\n  partition {pfp} is not the one the checkpoints were trained on "
              f"({AS_EXECUTED_PARTITION}); scoring here would evaluate different "
              f"tensors (runs/MEMO_catch28_supersession.md). STOP -- run on Modal.")
        sys.exit(1)
    if not (out["G1"]["pass"] and out["G2"]["base_map_is_identity"]):
        print("\n  G1/G2 failed; nothing below is readable. STOP")
        sys.exit(2)

    # ---- CB: live re-score on OFF and ON, consistency with cures_e10.json -------
    cures = json.load(open(args.cures))
    ref = {arm: {(r["seed"], r["task"]): r for r in rows}
           for arm, rows in cures.items()}
    banner("CB  ARTIFACT CONSISTENCY + CC WRONG-MAP MUST-FAIL   (OFF and ON, live)")
    out["arms"] = {}
    for tag, (tmpl, use_ad) in E10_HAR_ARMS.items():
        rows = score_arm(tag, tmpl, use_ad, hdata, maps, maps, device, ref[f"HAR/{tag}"])
        worst_cd = max(abs(r["C0deg"] - r["ref_C0deg"]) for r in rows)
        worst_or = max(abs(r["acc_orig"] - r["ref_acc_orig"]) for r in rows)
        worst_mx = max(abs(r["acc_orig"] - r["matrix_final_row"]) for r in rows)
        consistent = worst_cd < CONSISTENCY_TOL and worst_or < CONSISTENCY_TOL \
            and worst_mx < CONSISTENCY_TOL
        wrong_below = [r["C0deg_wrong"] < r["C0deg"] for r in rows]
        gap = float(np.mean([r["C0deg"] - r["C0deg_wrong"] for r in rows]))
        print(f"\n  HAR/{tag}   (adapter {'suppressed, analytic map substituted' if use_ad else 'absent'})")
        print(f"  {'cell':<10}{'diag':>8}{'orig':>8}{'C0deg':>8}{'ref':>8}{'wrong':>8}{'p_b':>6}")
        for r in rows:
            print(f"  s{r['seed']}/t{r['task']:<5}{r['matrix_diag']:8.4f}{r['acc_orig']:8.4f}"
                  f"{r['C0deg']:8.4f}{r['ref_C0deg']:8.4f}{r['C0deg_wrong']:8.4f}"
                  f"{'ok' if r['p_b'] else 'FAIL':>6}")
        print(f"  CB  max|C0deg - stored| {worst_cd:.1e}  max|orig - stored| {worst_or:.1e}"
              f"  max|orig - matrix| {worst_mx:.1e}  (tol {CONSISTENCY_TOL:.0e})"
              f" -> {'PASS' if consistent else 'FAIL'}")
        print(f"  CC  wrong map below right map in {sum(wrong_below)}/{len(rows)} cells,"
              f" mean gap {gap:+.4f} -> {'PASS' if all(wrong_below) else 'FAIL'}")
        out["arms"][tag] = {"rows": rows, "CB_pass": bool(consistent),
                            "CB_worst": {"C0deg": worst_cd, "acc_orig": worst_or, "matrix": worst_mx},
                            "CC_pass": bool(all(wrong_below)),
                            "CC_cells_below": int(sum(wrong_below)), "CC_mean_gap": gap}

    # ---- CA: no-shift positive control --------------------------------------
    banner("CA  NO-SHIFT POSITIVE CONTROL   (M_4 . M_k^-1 = I  =>  C0deg == deployed, exactly)")
    har0 = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=BATCH, no_shift=True)
    assert har0.test_subjects == har.test_subjects and har0.train_subjects == har.train_subjects, \
        "no-shift partition differs from the shifted one; the control would compare different subjects"
    data0 = load_task_data(har0, n_tasks=5)
    I = np.eye(N_CHANNELS); z = np.zeros(N_CHANNELS)
    maps_id = {k: (I, z) for k in range(5)}
    rows = score_arm("NOSHIFT", NOSHIFT_ARM, False, data0, maps_id, None, device, None)
    worst_dl = max(r["max_abs_dlogit_vs_orig"] for r in rows)
    worst_da = max(abs(r["C0deg"] - r["acc_orig"]) for r in rows)
    worst_mx = max(abs(r["acc_orig"] - r["matrix_final_row"]) for r in rows)
    print(f"  {'cell':<10}{'diag':>8}{'orig':>8}{'C0deg':>8}{'|dlogit|':>10}{'p_b':>6}")
    for r in rows:
        print(f"  s{r['seed']}/t{r['task']:<5}{r['matrix_diag']:8.4f}{r['acc_orig']:8.4f}"
              f"{r['C0deg']:8.4f}{r['max_abs_dlogit_vs_orig']:10.1e}{'ok' if r['p_b'] else 'FAIL':>6}")
    exact = worst_dl == 0.0 and worst_da == 0.0
    content_forget = float(np.mean([r["matrix_diag"] - r["acc_orig"] for r in rows]))
    print(f"  max|dlogit| {worst_dl:.1e}  max|C0deg - orig| {worst_da:.1e}"
          f"  max|orig - matrix| {worst_mx:.1e} -> {'PASS (bit-identical)' if exact else 'FAIL'}")
    print(f"  no-shift arm forgetting mean_j(diag - orig) = {content_forget:.4f}"
          f"   <- content-only forgetting; the map recovers none of it BY CONSTRUCTION")
    out["arms"]["NOSHIFT"] = {"rows": rows, "CA_pass": bool(exact),
                              "CA_worst_dlogit": worst_dl, "CA_worst_dacc": worst_da,
                              "matrix_consistency_worst": worst_mx,
                              "content_only_forgetting": content_forget}

    # ---- verdict, from the flags set above ------------------------------------
    banner("VERDICT  (derived from the printed values)")
    checks = {"G1 disjointness": out["G1"]["pass"],
              "G2 base map identity": out["G2"]["base_map_is_identity"],
              "CA no-shift exact": out["arms"]["NOSHIFT"]["CA_pass"],
              "CB consistency OFF": out["arms"]["OFF"]["CB_pass"],
              "CB consistency ON": out["arms"]["ON"]["CB_pass"],
              "CC wrong-map fails OFF": out["arms"]["OFF"]["CC_pass"],
              "CC wrong-map fails ON": out["arms"]["ON"]["CC_pass"]}
    for name, ok in checks.items():
        print(f"  {name:<26} {'PASS' if ok else 'FAIL'}")
    out["all_pass"] = bool(all(checks.values()))
    print(f"\n  ALL {'PASS -- the C0deg row may enter the ledger' if out['all_pass'] else 'NOT PASS -- do not cite'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
