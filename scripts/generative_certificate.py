"""
E10 P3 — generative-honesty certificate, v3.

Pre-registration: docs/E10_prereg.md (v2, assembled post-training; see its header)

WHAT THIS DECIDES. E8's pseudo-refit reached ceiling but was unfalsifiable,
because that benchmark's tasks were the same windows in different coordinates:
"generated" old data was the literal old data (err 1e-6). This certificate
decides whether a given construction lets generation be told apart from replay.

HISTORY, carried because it is what makes the verdict trustworthy:
  P3-v1 normalised pseudo->real NN distance by PER-TASK within-group real->real
  NN distance. It fired gate (G) on the live E10 data, failing tasks 0 and 2.
  Diagnosis: the numerator was near-constant across tasks (9.25-10.79) while the
  denominator varied 2.8x (5.27-14.98) purely with how tightly each subject group
  clusters -- so v1 scored GROUP TIGHTNESS, not honesty. v1's failure STANDS as
  recorded.

  P3-v2 pooled the denominator across tasks, fixing that. Its DISTRIBUTION bar
  (per-task median pseudo->real NN >= 0.5 * D) survives unchanged into v3 and is
  not re-litigated here. Its CONTAMINATION bar did not survive: a per-point
  threshold at 0.5*D cannot discriminate, because honest pseudo data already puts
  26-32% of its points below 0.5*D while a 10%-injected set puts 34-39% there.
  The bar could not separate the two, so it could not fire on contamination --
  a bar that cannot fail (catch 25).

  P3-v3 changes ONLY the contamination threshold, from a round multiple of D to
  the GEOMETRIC MIDPOINT of two control-measured scales:

      tau = sqrt(recon_scale * honest_scale) * D

  where recon_scale is measured on Control A (reconstruction) and honest_scale on
  Control C (known-honest generation). Deriving tau from the controls rather than
  from the live data is deliberate: a threshold fitted to the thing it certifies
  is not a certificate. The recorded prior-session values (recon 0.00065, honest
  0.966, tau ~= 0.0251*D) are REPRODUCTION TARGETS ONLY -- they have no artifact
  in this repo, so they cannot serve as the pass. If the controls' separation
  here differs materially from those numbers, that difference is the finding.

CONTROLS (all three must behave before any live reading):
  A. full reconstruction   -- E8's shared-window construction. Must FAIL.
  B. partial contamination -- 10% real task-k windows injected into an otherwise
     honest pseudo set. The CONTAMINATION bar specifically must fire.
  C. known-honest negative -- real windows from a DIFFERENT subject group, mapped
     into task k's layout. Genuinely novel content, so it must PASS. C also
     anchors honest_scale, which is why it is not optional.

Usage:
    python scripts/generative_certificate.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.har_shift import HARShiftBenchmark
from src.data.har_subject import HARSubjectBenchmark
from scripts.cure_screen import har_maps, har_relayout

DIST_BAR = 0.5          # distribution bar, as multiple of pooled D (v2, unchanged)
CONTAM_FRAC = 0.01      # contamination bar: fraction below tau must be < 1.00%
OLD_TASKS = (0, 1, 2, 3)


def nn_to(A: torch.Tensor, B: torch.Tensor, exclude_self: bool = False) -> torch.Tensor:
    """Nearest-neighbour distance from each row of A to the set B."""  # [n_A]
    A2, B2 = A.reshape(A.shape[0], -1), B.reshape(B.shape[0], -1)
    D = torch.cdist(A2, B2)
    if exclude_self:
        D.fill_diagonal_(float("inf"))
    return D.min(1).values


def pooled_denominator(real_by_task: dict[int, torch.Tensor]) -> float:
    """Median within-task real->real NN distance, POOLED across tasks.

    Pooling is v1's fix: a per-task denominator varies with how tightly that
    task's subjects cluster, which is nuisance variation, not honesty.
    """
    return float(torch.cat([nn_to(r, r, exclude_self=True)
                            for r in real_by_task.values()]).median())


def scale_of(pseudo_by_task: dict[int, torch.Tensor],
             real_by_task: dict[int, torch.Tensor], D: float) -> float:
    """Median pseudo->real NN distance in units of D, pooled over tasks."""
    return float(torch.cat([nn_to(pseudo_by_task[k], real_by_task[k])
                            for k in sorted(pseudo_by_task)]).median()) / D


def certify(pseudo_by_task: dict[int, torch.Tensor],
            real_by_task: dict[int, torch.Tensor], label: str,
            D: float, tau_frac: float) -> tuple[bool, bool, bool]:
    """Both bars, per task. Verdicts are computed from the printed values."""
    bar, tau = DIST_BAR * D, tau_frac * D
    print(f"\n  {label}")
    print(f"    pooled D = {D:.4f}   dist bar = {DIST_BAR}*D = {bar:.4f}"
          f"   tau = {tau_frac:.5f}*D = {tau:.4f}")
    print(f"    {'task':<6}{'median p->r':>14}{'/D':>9}{'frac < tau':>12}  verdict")
    dist_ok, contam_ok = True, True
    for k in sorted(pseudo_by_task):
        d = nn_to(pseudo_by_task[k], real_by_task[k])
        med, frac = float(d.median()), float((d < tau).float().mean())
        dpass, cpass = med >= bar, frac < CONTAM_FRAC
        dist_ok &= dpass
        contam_ok &= cpass
        print(f"    {k:<6}{med:>14.4f}{med / D:>9.4f}{frac * 100:>11.2f}%  "
              f"dist {'OK' if dpass else 'FAIL'}, "
              f"contam {'OK' if cpass else 'FAIL'}")
    print(f"    -> certificate {'PASSES' if (dist_ok and contam_ok) else 'FAILS'}"
          f"  (distribution {'ok' if dist_ok else 'FAIL'}, "
          f"contamination {'ok' if contam_ok else 'FAIL'})")
    return dist_ok and contam_ok, dist_ok, contam_ok


def main() -> None:
    print("=" * 78)
    print("P3-v3 CERTIFICATE — controls measure tau, then controls test it, live last")
    print("=" * 78)

    # ---- constructions -----------------------------------------------------
    h8 = HARShiftBenchmark(num_tasks=5, root=".", batch_size=256)
    m8 = har_maps(h8._sd)
    x4_8 = h8.get_task_loaders(4)[1].dataset.tensors[0]
    real8 = {k: h8.get_task_loaders(k)[1].dataset.tensors[0] for k in OLD_TASKS}
    pseudo8 = {k: har_relayout(x4_8, 4, k, m8) for k in OLD_TASKS}
    D8 = pooled_denominator(real8)

    h10 = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=256)
    m10 = har_maps(h10._sd)
    x4_10 = h10.raw_windows(4, train=True)[0]
    real10 = {k: h10.raw_windows(k, train=True)[0] for k in OLD_TASKS}
    pseudo10 = {k: har_relayout(x4_10, 4, k, m10) for k in OLD_TASKS}
    D10 = pooled_denominator(real10)

    # Control C: real windows from a DIFFERENT old subject group, mapped into
    # task k's layout. Novel content, honest by construction -> must PASS, and
    # anchors honest_scale independently of the live data being certified.
    pseudo_C = {k: har_relayout(real10[(k + 1) % 4], (k + 1) % 4, k, m10)
                for k in OLD_TASKS}

    # ---- tau, derived from the controls -----------------------------------
    recon_scale = scale_of(pseudo8, real8, D8)
    honest_scale = scale_of(pseudo_C, real10, D10)
    tau_frac = math.sqrt(recon_scale * honest_scale)
    print("\n" + "=" * 78)
    print("TAU DERIVATION (from control-measured scales, not from live data)")
    print("=" * 78)
    print(f"  recon_scale  (Control A, reconstruction)   {recon_scale:.6f} * D")
    print(f"  honest_scale (Control C, novel content)    {honest_scale:.6f} * D")
    print(f"  tau = sqrt(recon * honest)                 {tau_frac:.6f} * D")
    print(f"  recorded targets: recon 0.00065, honest 0.966, tau 0.0251"
          f"   (reproduction reference only)")
    print(f"  control separation {honest_scale / max(recon_scale, 1e-12):,.0f}x")

    # ---- controls ----------------------------------------------------------
    okA, _, _ = certify(pseudo8, real8, "CONTROL A — E8 shared-window (MUST FAIL)",
                        D8, tau_frac)
    okC, _, _ = certify(pseudo_C, real10,
                        "CONTROL C — foreign subject group (MUST PASS)", D10, tau_frac)

    g = torch.Generator().manual_seed(0)
    pseudo_B = {}
    for k in OLD_TASKS:
        p, r = pseudo10[k], real10[k]
        n_inj = int(0.10 * p.shape[0])
        idx = torch.randperm(r.shape[0], generator=g)[:n_inj]
        keep = torch.randperm(p.shape[0], generator=g)[:p.shape[0] - n_inj]
        pseudo_B[k] = torch.cat([p[keep], r[idx]])
    okB, _, contamB = certify(pseudo_B, real10,
                              "CONTROL B — 10% real injected (CONTAMINATION bar MUST fire)",
                              D10, tau_frac)

    print("\n" + "=" * 78)
    print("CONTROL VERDICTS")
    print("=" * 78)
    a_ok, b_ok, c_ok = (not okA), (not contamB), okC
    print(f"  A: reconstruction control FAILED certificate   -> {a_ok}   (want True)")
    print(f"  B: contamination bar fired on 10% injection    -> {b_ok}   (want True)")
    print(f"  C: known-honest control PASSED certificate     -> {c_ok}   (want True)")
    if not (a_ok and b_ok and c_ok):
        print("\n  v3 FAILED its own controls -> STOP, do not read live (prereg).")
        return
    print("  All three controls behaved as required. v3 has earned its standing.")

    # ---- LIVE --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("LIVE — E10 subject-disjoint construction")
    print("=" * 78)
    live, dist_ok, contam_ok = certify(pseudo10, real10, "E10 subject-disjoint",
                                       D10, tau_frac)
    print("\n" + "=" * 78)
    print(f"  P3-v3 LIVE VERDICT: {'PASS' if live else 'FAIL -> gate (G)'}"
          f"   (distribution {'ok' if dist_ok else 'FAIL'}, "
          f"contamination {'ok' if contam_ok else 'FAIL'})")
    print("  P3-v1 fired gate (G) on this same data; that failure stands as recorded.")
    print("=" * 78)


if __name__ == "__main__":
    main()
