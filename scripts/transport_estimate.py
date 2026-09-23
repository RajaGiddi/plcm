"""
E9 — Estimating current-era geometry without old data.

Pre-registration: docs/E9_prereg.md (v2)

Read-time pipeline for old task k, with NO era encoder and NO old-task data:
    1. encode x_k with theta_4                      -> f
    2. transport f BACK to the era frame via chained stored inverses
    3. project onto the stored era-prototype span
    4. apply the stored era head

Write-time transport (catch 24). v1 assumed a stored era ENCODER -- but simply
running task k through that snapshot scores rho = 1.237, above every ceiling in
the program, because a snapshot has neither encoder damage nor reader mismatch.
The estimator would have approximated what its own stored object delivers exactly.
So: at each boundary, while the previous encoder is STILL IN MEMORY, estimate the
per-step Procrustes on that era's own data (in-distribution FOR THAT PAIR), store
the 256x256 map, discard the encoder. Rolling snapshot, permanent maps.

That relocates the risk: each link is in-distribution, so the exposure is
COMPOSITION ERROR GROWTH, not transport-OOD. Chaining (T2) is therefore primary
and the accumulation column is load-bearing.

Guard (catch 2). v1's split-half test estimated from two halves of the SAME
distribution -- they agree by construction, so it could pass with the OOD problem
intact. Primary guard is now estimated-vs-oracle map comparison:
    ||Q_est - Q_oracle||_F / ||Q_oracle - I||_F <= 0.5
Low rho WITH high agreement -> the alignment component was smaller than C2-oracle
suggested. Low rho WITH low agreement -> drift is not input-independent, failure
mode 6's second instance.

Usage:
    python scripts/transport_estimate.py
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.har_shift import HARShiftBenchmark, spec_fingerprint
from scripts.channel_decomp import load, features_and_logits, load_task_data
from scripts.cure_screen import era_head_of

SEEDS = [42, 1337, 2024]
CUR = 4
FLOOR_REF, DENOM = 0.5087, 0.2311          # E8 recorded, HAR/OFF bias-adjusted
TIER0_STACK, ORACLE_SPAN, ORACLE_FULL = 0.347, 0.477, 0.572
GUARD_BAR = 0.5                             # H-T2
H_T1_BAR = 0.41
ARMS = {"OFF": ("runs/ckpt_e5_off_seed{s}/mafc_seed{s}", False),
        "v1-ON": ("runs/ckpt_e5_seed{s}/mafc_seed{s}", True)}


def procrustes(X, Y):
    """Orthogonal Q minimising ||X Q - Y||_F. Full-rank here (N >> d)."""
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return U @ Vt


def span_of(P):
    return (np.linalg.svd(P.T, full_matrices=False)[0])[:, :P.shape[0]]


def run_arm(tmpl, use_ad, data, device, n_classes=6):
    rows = []
    for seed in SEEDS:
        d = tmpl.format(s=seed)
        m4 = load(d, CUR)

        # ---- WRITE TIME: per-step maps, encoder discarded after each -------
        # Q_step[j] maps theta_{j+1}-frame features -> theta_j-frame, estimated
        # on task-(j+1) data: the data live at that boundary.
        Q_step, spectra = {}, {}
        for j in range(CUR):
            m_lo, m_hi = load(d, j), load(d, j + 1)
            x = data[j + 1][0]
            F_lo, _ = features_and_logits(m_lo, x, None, j + 1, use_ad, device)
            F_hi, _ = features_and_logits(m_hi, x, None, j + 1, use_ad, device)
            Q = procrustes(F_hi.numpy().astype(np.float64), F_lo.numpy().astype(np.float64))
            Q_step[j] = Q
            # how far from identity is this step? (ruled in: report regardless)
            ang = np.rad2deg(np.arccos(np.clip(np.linalg.svd(Q)[1], -1, 1)))
            spectra[j] = {"dev_from_I": float(np.linalg.norm(Q - np.eye(Q.shape[0]))),
                          "mean_rot_deg": float(np.rad2deg(
                              np.arccos(np.clip((np.trace(Q) - (Q.shape[0] - 2)) / 2, -1, 1))))}

        for k in range(CUR):
            xtr, ytr, xte, yte = data[k]
            m_era = load(d, k)
            eh = era_head_of(m_era, k)

            f_e_tr, _ = features_and_logits(m_era, xtr, ytr, k, use_ad, device)
            f_4_tr, _ = features_and_logits(m4, xtr, ytr, k, use_ad, device)
            f_4_te, l4 = features_and_logits(m4, xte, yte, k, use_ad, device)
            floor = float((l4.argmax(1) == yte).float().mean())

            B = torch.stack([f_e_tr[ytr == c].mean(0) for c in range(n_classes)]).numpy().astype(np.float64)
            A = torch.stack([f_4_tr[ytr == c].mean(0) for c in range(n_classes)]).numpy().astype(np.float64)
            Pb = span_of(B)                                   # stored era-proto span
            Q_oracle = procrustes(A, B)                       # tier-2 DIAGNOSTIC only

            # ---- T2 chained: theta_4 -> theta_k, composing stored steps ----
            Q_chain = np.eye(A.shape[1])
            for j in range(CUR - 1, k - 1, -1):
                Q_chain = Q_chain @ Q_step[j]

            # ---- T1 direct: one map, estimated on CURRENT (task-4) data ----
            xc = data[CUR][0]
            F_cur_4, _ = features_and_logits(m4, xc, None, CUR, use_ad, device)
            F_cur_e, _ = features_and_logits(m_era, xc, None, CUR, use_ad, device)
            Q_direct = procrustes(F_cur_4.numpy().astype(np.float64),
                                  F_cur_e.numpy().astype(np.float64))

            def pipeline(Q):
                """transport -> project onto era-proto span -> era head."""
                M = (Q @ (Pb @ Pb.T)).astype(np.float32)
                with torch.no_grad():
                    return float((eh(f_4_te @ torch.from_numpy(M)).argmax(1) == yte).float().mean())

            dev_o = np.linalg.norm(Q_oracle - np.eye(Q_oracle.shape[0]))
            rows.append({
                "seed": seed, "task": k, "floor": floor,
                "T2_chained": pipeline(Q_chain),
                "T1_direct": pipeline(Q_direct),
                "tier0_stack": pipeline(np.eye(A.shape[1])),
                "oracle_span": pipeline(Q_oracle),
                "guard_T2": float(np.linalg.norm(Q_chain - Q_oracle) / dev_o),
                "guard_T1": float(np.linalg.norm(Q_direct - Q_oracle) / dev_o),
                "step_spectra": spectra,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="E9 transport estimation")
    ap.add_argument("--out", default="runs/e9/")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    fp = spec_fingerprint()
    print(f"HAR fingerprint {fp} -> {'OK' if fp == '3de66e205eb7' else 'MISMATCH'}")
    if fp != "3de66e205eb7":
        sys.exit(1)

    har = HARShiftBenchmark(num_tasks=5, root=".", batch_size=256)
    data = load_task_data(har, n_tasks=5)
    res = {a: run_arm(t, u, data, device) for a, (t, u) in ARMS.items()}

    m = lambda rr, k: float(np.mean([r[k] for r in rr]))
    rho = lambda a: (a - FLOOR_REF) / DENOM

    print("\n" + "=" * 76)
    print("0. PER-STEP MAP GEOMETRY (reported regardless of branch)")
    print("=" * 76)
    sp = res["OFF"][0]["step_spectra"]
    print(f"  {'boundary':<14}{'||Q - I||_F':>14}{'mean rotation':>16}")
    for j in sorted(sp):
        print(f"  theta_{j+1}->theta_{j}{'':<3}{sp[j]['dev_from_I']:>14.3f}"
              f"{sp[j]['mean_rot_deg']:>15.1f} deg")

    print("\n" + "=" * 76)
    print("1. H-T2 GATE — relative deviation capture <= 0.5 (evaluated FIRST)")
    print("=" * 76)
    gates = {}
    for arm, rr in res.items():
        g2, g1 = m(rr, "guard_T2"), m(rr, "guard_T1")
        gates[arm] = g2 <= GUARD_BAR
        print(f"  {arm:<8} T2 {g2:.3f}  T1 {g1:.3f}   -> "
              f"{'PASS' if gates[arm] else 'FAIL'}")

    print("\n" + "=" * 76)
    print("2. PIPELINE ACCURACY (rho vs E8's bias-adjusted denominator)")
    print("=" * 76)
    print(f"  {'arm':<8}{'tier0':>16}{'T2 chained':>16}{'T1 direct':>16}{'oracle span':>16}")
    for arm, rr in res.items():
        print(f"  {arm:<8}"
              + "".join(f"{m(rr,k):>9.4f}/{rho(m(rr,k)):+.3f}"
                        for k in ("tier0_stack", "T2_chained", "T1_direct", "oracle_span")))

    off = res["OFF"]
    r2, r1 = rho(m(off, "T2_chained")), rho(m(off, "T1_direct"))
    print("\n" + "=" * 76)
    print("3. HYPOTHESES")
    print("=" * 76)
    if not gates["OFF"]:
        print("  H-T2 FAILED -> H-T1 not read (prereg sec-4, gates before grades)")
        branch = "(D) estimated map does not capture the oracle's deviation -> no repair claims"
    else:
        best = max(r2, r1)
        print(f"  H-T1  best transport rho {best:+.3f} >= {H_T1_BAR}"
              f"  -> {'PASS' if best >= H_T1_BAR else 'FAIL'}")
        if best >= H_T1_BAR:
            branch = "(A) current-era geometry IS estimable from stored maps"
        elif best > TIER0_STACK:
            branch = "(B) transport adds something, under half the gap -> tier-0 stack deployable"
        else:
            branch = ("(C) transport adds nothing or hurts -> RBST lesson generalises "
                      "to closed-form low-DOF maps")
    print(f"  H-T3  accumulation: T2 {r2:+.3f} - T1 {r1:+.3f} = {r2-r1:+.3f}"
          f"   ({'chained >= direct' if r2 >= r1 else 'direct > chained: error compounds'})")
    print(f"\n  BRANCH: {branch}")
    print(f"\n  reference anchors: tier0 {TIER0_STACK:.3f} | oracle-span {ORACLE_SPAN:.3f} "
          f"| oracle-full {ORACLE_FULL:.3f} | snapshot 1.237")

    json.dump(res, open(os.path.join(args.out, "transport.json"), "w"),
              indent=2, default=float)
    print(f"\n  wrote {args.out}transport.json")


if __name__ == "__main__":
    main()
