"""CC' -- the permutation-only must-fail for C0deg, and CD, the component ablation.

RULING (runs/MEMO_c0deg.md sec 2c/2d, 2026-09-15). CC -- "the wrong map scores
below the right map in every cell" with the wrong map being the NEIGHBOURING
task's -- FAILED as pre-committed (9/12 OFF, 11/12 ON). Every failing cell was
one where right and wrong maps differed by a gain/offset or a 30-degree
z-rotation only; every cell where they differed by a channel permutation
collapsed. The bar was set before the run and does not move after it: CC stands
as FAIL. This script is CC', the control that tests what CC was meant to test,
with its bar stated in the memo BEFORE launch.

CC'  Two PERMUTATION-ONLY wrong maps, each applied on top of the RIGHT relayout
     x_cur = M_4 . M_k^-1 (x - b_k) + b_4, so the only thing wrong is a wiring:
       within:  Q = PERM, the benchmark's own wiring revision, applied once more
                (axes relabelled within each sensor triple)
       across:  Q = channel i -> (i+3) mod 9  (acc -> gyro -> total_acc)
     BAR, per cell, both variants, both arms:  acc(Q . x_cur) < C0deg in 12/12
     cells per arm. Verdict derived from the printed values. If any cell fails,
     CC' fails as CC did and is reported so.

CD   COMPONENT ABLATION -- descriptive, NO BAR, no registered prediction. The
     ruled scope finding ("recovery comes from the permutation components") rests
     on CC's four failing cells and one rotation cell contradicts it. For each
     old task k and each subset S of {gain, offset, rot, perm}, the input is
     corrected in the components of S only: task-4's values for S, task-k's for
     the rest, composed exactly as `channel_affine` composes them (M = P R G,
     b_eff = P R b) and then divided by sd as the loader does. Two self-checks
     are ASSERTED so the implementation cannot silently drift from the loader's
     algebra: S = {} must reproduce the deployed input and S = all must reproduce
     x_cur, both to < 1e-5. Reported per (cell x S); {perm} alone and all-{perm}
     are printed beside C0deg and none.

Same checkpoints the row cites (ckpt_e10_{off,on}_seed*), scoring gated on the
as-executed partition (1104af185c87) exactly as c0deg_controls.py gates it.

Usage:
    modal run modal_runner.py::analysis --argv "scripts/c0deg_cc2.py"
"""

import argparse
import itertools
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.har_shift import (channel_affine, spec_fingerprint, N_CHANNELS,
                                SHIFT_SPEC, PERM, _rot_z)
from src.data.har_subject import HARSubjectBenchmark
from scripts.channel_decomp import load, features_and_logits, load_task_data
from scripts.cure_screen import (har_maps, har_relayout, E10_HAR_ARMS, E10EC_HAR_ARMS,
                                 CUR, SEEDS, BATCH)
import scripts.c0deg_controls as ctl
from scripts.c0deg_controls import REGISTERED_SHIFT, AS_EXECUTED_PARTITION, banner, acc

# S72: the same control on the relaunched (era + fp32 shadow) checkpoints the
# three-way row cites. An amendment re-verifies what it did not touch; the row
# moved to new checkpoints, so CC' moves with it. Default `e10` = the ledger row.
ARM_SETS = {"e10": (E10_HAR_ARMS, ctl.MATRIX),
            "e10ec": (E10EC_HAR_ARMS, {"OFF": "runs/e10off_ec_seed{s}/mafc_results.json"})}

OLD = list(range(CUR))
COMPONENTS = ("gain", "offset", "rot", "perm")
Q_WITHIN = list(PERM)                                   # the benchmark's own wiring revision
Q_ACROSS = [(i + 3) % N_CHANNELS for i in range(N_CHANNELS)]   # triples swapped
SELF_CHECK_TOL = 1e-5


def perm_matrix(perm):
    P = np.zeros((N_CHANNELS, N_CHANNELS))
    for i, j in enumerate(perm):
        P[i, j] = 1.0
    return P


def apply_perm(x, perm):
    """Q . x on the channel axis: out[..., i] = x[..., perm[i]] -- the same
    convention channel_affine uses for P."""
    return x[..., list(perm)]


def affine_from_components(gain, offset, rot_deg, perm):
    """channel_affine's composition, with the components supplied instead of
    read from SHIFT_SPEC. Mirrors src/data/har_shift.py:154-181 line for line so
    that a subset built here and the loader's own map agree exactly (asserted in
    partial_input)."""
    M = np.eye(N_CHANNELS)
    b = np.zeros(N_CHANNELS)
    if gain is not None:
        M = np.diag(gain) @ M
    if offset is not None:
        b = np.asarray(offset, dtype=np.float64)
    if rot_deg is not None:
        R = np.zeros((N_CHANNELS, N_CHANNELS))
        r = _rot_z(rot_deg)
        for blk in range(3):
            R[3 * blk:3 * blk + 3, 3 * blk:3 * blk + 3] = r
        M, b = R @ M, R @ b
    if perm is not None:
        P = perm_matrix(perm)
        M, b = P @ M, P @ b
    return M, b


def components_of(task):
    s = SHIFT_SPEC[task]
    return {"gain": s.get("gain"), "offset": s.get("offset"),
            "rot": s.get("rot_deg"), "perm": s.get("perm")}


def partial_input(x, k, S, maps, sd):
    """Task-k input corrected in the components of S only.

    Recover Z from the full (exactly invertible) task-k map, then re-apply the
    MIXED composition: task-4's component where it is in S, task-k's otherwise.
    """
    Mk, ck = maps[k]
    Z = (x.numpy() - ck) @ np.linalg.inv(Mk).T
    src, dst = components_of(k), components_of(CUR)
    mixed = {c: (dst[c] if c in S else src[c]) for c in COMPONENTS}
    M, b = affine_from_components(mixed["gain"], mixed["offset"], mixed["rot"], mixed["perm"])
    return torch.from_numpy((Z @ M.T + b / sd).astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description="C0deg CC' + CD")
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default="e10")
    ap.add_argument("--out", default=None, help="default runs/e10/c0deg_cc2.json; runs/e10ec/c0deg_cc2.json for --arms e10ec")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    arm_table, matrix_map = ARM_SETS[args.arms]
    ctl.MATRIX = matrix_map                      # matrix_final_row reads the module global
    matrix_final_row = ctl.matrix_final_row
    if args.out is None:
        args.out = f"runs/{args.arms}/c0deg_cc2.json"
    device = torch.device(args.device)

    banner("C0deg CC' (permutation-only must-fail) + CD (component ablation)")
    print(f"  platform {platform.machine()}  torch {torch.__version__}  numpy {np.__version__}")
    fp = spec_fingerprint()
    print(f"  shift fingerprint {fp} -> {'MATCH' if fp == REGISTERED_SHIFT else 'MISMATCH - STOP'}")
    if fp != REGISTERED_SHIFT:
        sys.exit(1)
    har = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=BATCH)
    pfp = har.partition_fingerprint()
    print(f"  partition fingerprint {pfp} -> {'AS-EXECUTED' if pfp == AS_EXECUTED_PARTITION else 'NOT the executed partition - STOP'}")
    if pfp != AS_EXECUTED_PARTITION:
        sys.exit(1)
    maps = har_maps(har._sd)
    sd = har._sd
    hdata = load_task_data(har, n_tasks=5)

    # ---- the two wirings are what the docstring says they are ---------------
    assert Q_WITHIN != list(range(N_CHANNELS)) and Q_ACROSS != list(range(N_CHANNELS))
    assert Q_WITHIN != Q_ACROSS
    print(f"  Q_within {Q_WITHIN}   Q_across {Q_ACROSS}")

    # ---- CD self-checks, on task-0..3 test inputs, before anything is scored --
    banner("CD SELF-CHECKS  (S = {} reproduces the input; S = all reproduces x_cur)")
    worst_empty = worst_all = 0.0
    for k in OLD:
        xte = hdata[k][2]
        e0 = float((partial_input(xte, k, set(), maps, sd) - xte).abs().max())
        ea = float((partial_input(xte, k, set(COMPONENTS), maps, sd)
                    - har_relayout(xte, k, CUR, maps)).abs().max())
        worst_empty, worst_all = max(worst_empty, e0), max(worst_all, ea)
        print(f"  task {k}: |S={{}} - x| {e0:.2e}   |S=all - x_cur| {ea:.2e}")
    assert worst_empty < SELF_CHECK_TOL and worst_all < SELF_CHECK_TOL, \
        "partial_input does not reproduce the loader's algebra at its endpoints"
    print(f"  worst {max(worst_empty, worst_all):.2e} < {SELF_CHECK_TOL:.0e} -> PASS")

    subsets = [frozenset(c) for n in range(len(COMPONENTS) + 1)
               for c in itertools.combinations(COMPONENTS, n)]
    label = lambda S: "{" + ",".join(c for c in COMPONENTS if c in S) + "}"

    out = {"platform": platform.machine(), "shift_fingerprint": fp,
           "partition_fingerprint": pfp, "Q_within": Q_WITHIN, "Q_across": Q_ACROSS,
           "self_check": {"empty": worst_empty, "all": worst_all}, "arms": {}}

    for tag, (tmpl, use_ad) in arm_table.items():
        banner(f"HAR/{tag}   (adapter {'suppressed, analytic map substituted' if use_ad else 'absent'})")
        rows = []
        for s in SEEDS:
            m4 = load(tmpl.format(s=s), CUR)
            mat = matrix_final_row(tag, s)
            for k in OLD:
                xtr, ytr, xte, yte = hdata[k]
                _, l_orig = features_and_logits(m4, xte, yte, k, use_ad, device)
                x_cur = har_relayout(xte, k, CUR, maps)
                _, l_cd = features_and_logits(m4, x_cur, yte, k, False, device)
                row = {"arm": tag, "seed": s, "task": k, "matrix_diag": float(mat[k, k]),
                       "acc_orig": acc(l_orig, yte), "C0deg": acc(l_cd, yte)}
                # ---- CC' -------------------------------------------------------
                for name, Q in (("within", Q_WITHIN), ("across", Q_ACROSS)):
                    _, l_q = features_and_logits(m4, apply_perm(x_cur, Q), yte, k, False, device)
                    row[f"CCp_{name}"] = acc(l_q, yte)
                # ---- CD --------------------------------------------------------
                row["CD"] = {}
                for S in subsets:
                    _, l_s = features_and_logits(m4, partial_input(xte, k, S, maps, sd),
                                                 yte, k, False, device)
                    row["CD"][label(S)] = acc(l_s, yte)
                rows.append(row)

        # ---- CC' verdict, per cell, from the values printed -------------------
        print(f"\n  CC'  {'cell':<10}{'orig':>8}{'C0deg':>8}{'within':>8}{'across':>8}")
        ok_w = ok_a = 0
        for r in rows:
            bw, ba = r["CCp_within"] < r["C0deg"], r["CCp_across"] < r["C0deg"]
            ok_w += bw; ok_a += ba
            print(f"       s{r['seed']}/t{r['task']:<5}{r['acc_orig']:8.4f}{r['C0deg']:8.4f}"
                  f"{r['CCp_within']:8.4f}{'' if bw else '!'}{r['CCp_across']:8.4f}{'' if ba else '!'}")
        n = len(rows)
        pass_w, pass_a = ok_w == n, ok_a == n
        print(f"  CC'-within: below C0deg in {ok_w}/{n} cells -> {'PASS' if pass_w else 'FAIL'}")
        print(f"  CC'-across: below C0deg in {ok_a}/{n} cells -> {'PASS' if pass_a else 'FAIL'}")

        # ---- CD table, descriptive ----------------------------------------------
        keys = [label(S) for S in subsets]
        print(f"\n  CD   mean over {n} cells (no bar; reading written after)")
        print(f"       {'S':<26}{'acc':>8}   {'S':<26}{'acc':>8}")
        means = {kk: float(np.mean([r["CD"][kk] for r in rows])) for kk in keys}
        half = (len(keys) + 1) // 2
        for i in range(half):
            left = f"{keys[i]:<26}{means[keys[i]]:8.4f}"
            right = (f"{keys[i+half]:<26}{means[keys[i+half]]:8.4f}" if i + half < len(keys) else "")
            print(f"       {left}   {right}")
        allbut = label(frozenset(COMPONENTS) - {"perm"})
        print(f"\n       none {means['{}']:.4f} | {{perm}} {means['{perm}']:.4f} | "
              f"{allbut} {means[allbut]:.4f} | all (=C0deg) {means[label(frozenset(COMPONENTS))]:.4f}")
        print(f"       per-task, {{perm}} vs all:")
        for k in OLD:
            cells = [r for r in rows if r["task"] == k]
            p = np.mean([r["CD"]["{perm}"] for r in cells]); a = np.mean([r["C0deg"] for r in cells])
            o = np.mean([r["acc_orig"] for r in cells]); nb = np.mean([r["CD"][allbut] for r in cells])
            print(f"         task {k}: none {o:.4f}  {{perm}} {p:.4f}  all-{{perm}} {nb:.4f}  all {a:.4f}"
                  f"   (task-{k} spec: {SHIFT_SPEC[k]['desc']})")
        out["arms"][tag] = {"rows": rows, "CCp_within_pass": bool(pass_w),
                            "CCp_across_pass": bool(pass_a),
                            "CCp_within_cells_below": ok_w, "CCp_across_cells_below": ok_a,
                            "CD_means": means}

    banner("VERDICT (CC' only; CD carries no verdict)")
    checks = {f"CC'-{v} {arm}": out["arms"][arm][f"CCp_{v}_pass"]
              for arm in arm_table for v in ("within", "across")}
    for name, ok in checks.items():
        print(f"  {name:<22} {'PASS' if ok else 'FAIL'}")
    out["CCp_all_pass"] = bool(all(checks.values()))
    print(f"\n  CC' {'PASS' if out['CCp_all_pass'] else 'FAIL'}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
