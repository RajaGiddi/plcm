"""
E8 — Reader-repair cure screen (analysis-only).

Pre-registration: docs/E8_prereg.md (v2)

E6b proved a matched-late reader recovers old-task accuracy to within F_enc of
ceiling -- but did it WITH old-task labels. This screens storage-honest read-time
corrections against that measured ceiling.

Cures, in the contracted read order:
  C0     analytic adapter A_k := M_k^-1 (AFFINE -- tasks 1/4 carry offsets, so this
         is a different adapter class than the locked bias-free spec; declared, not
         quietly upgraded). Three experiments asked why SGD will not find M_k^-1;
         none wrote it in by hand. This does.
  C0-deg A_k := M_4 . M_k^-1, mapping old tasks into the CURRENT layout. On E8's
         HAR all five tasks share identical windows and labels, so this reproduces
         task 4's test set exactly and hits DIAG by construction -- a LABELLED
         CEILING ARTIFACT there, never a cure.
         CORRECTION (2026-09-15, runs/MEMO_c0deg.md). That label was earned on
         the SHARED-WINDOW construction and was carried onto E10's subject-disjoint
         benchmark, where its premise does not hold: task k's test subject is held
         out from every training group, so re-layout into the current frame feeds
         the current model windows it has never seen. On E10 C0deg is a legitimate
         storage-honest method -- the maps only, no snapshot, no refit -- and the
         best one measured (OFF forgetting -0.0247 vs C3's 0.0886, intervals
         separate). Controls: scripts/c0deg_controls.py, scripts/c0deg_cc2.py.
         The label below is printed per arm set: artifact on e8, method on the
         subject-disjoint sets. An inherited exclusion hides a result the way an
         inherited inclusion fakes one (catch 21 applied to an exclusion).
  C1     moment matching: standardize the eval batch's theta_4 features to stored
         era moments, read with the stored era head.
  C0+C1  both measured channels at once -- the closest this screen comes to "the
         full cure, storage-honest".
  C2     Procrustes-in-S, closed-form SVD, ORACLE theta_4 prototypes (tier 2).
  C3     pseudo-refit: pseudo-old inputs via the sec-0 generator, labelled by the era
         snapshot's own predictions, convex head fit on their theta_4 features.

Scoring: rho = (acc_cure - acc_orig) / (acc_refit - R - acc_orig), the BIAS-ADJUSTED
ceiling. Charging the instrument's bias against the cure keeps the arm rather than
discarding it (v1's gate would have excluded HAR/OFF, the arm H-R1/H-R3 name).
Raw-ceiling rho prints alongside.

Usage:
    python scripts/cure_screen.py
    python scripts/cure_screen.py --skip-mnist
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM
from src.data.har_shift import HARShiftBenchmark, channel_affine, spec_fingerprint
from src.data.har_subject import HARSubjectBenchmark
from src.data.permuted_mnist import PermutedMNISTBenchmark
from src.data.rotated_mnist import RotatedMNISTBenchmark
from scripts.channel_decomp import (load, features_and_logits, refit_probe,
                                    load_task_data, assert_path_identity,
                                    N_TRAIN, N_TEST, PROBE_SUBSET_SEED)

SEEDS = [42, 1337, 2024]
CUR = 4                      # the current task; old tasks are 0..3
BATCH = 256                  # C1 protocol parameter (prereg sec-1)
RHO_MIN_DENOM = 0.02
ROUNDTRIP_TOL = 1e-4         # E10 generator gate: relative round-trip error

HAR_ARMS = {"OFF":   ("runs/ckpt_e5_off_seed{s}/mafc_seed{s}", False),
            "v1-ON": ("runs/ckpt_e5_seed{s}/mafc_seed{s}", True)}
MNIST_ARMS = {"OFF": ({s: f"checkpoints/e4_off_seed{s}/mafc_seed{s}" for s in SEEDS}, False)}

# E10 -- subject-disjoint HAR. Same shift set (fingerprint 3de66e205eb7, reused
# verbatim per prereg sec-1), but tasks differ in CONTENT as well as presentation,
# so the E8 reconstruction pre-gate is INAPPLICABLE here by construction -- see
# generator_gate_e10().
E10_HAR_ARMS = {"OFF": ("runs/ckpt_e10_off_seed{s}/mafc_seed{s}", False),
                "ON":  ("runs/ckpt_e10_on_seed{s}/mafc_seed{s}", True)}

# S72 (runs/MEMO_c0deg.md sec 6) -- the bridging arm relaunched with era
# checkpoints + fp32 shadow, uniform with the LwF arm it is compared against.
# The screen reads the SHADOW directory: era checkpoints are fp16 and reload one
# window off the recorded matrix (e16_har_lam0.25_seed42 task 0: +0.00034), which
# hx2_forgetting's 1e-6 consistency gate would reject; the fp32 shadow reloads at
# exactly 0.0. OFF only -- the ON arm was never part of the sec 7.2 comparison.
# The floor pair's "seeds" are the replicate labels a/b, so `screen_arm` takes
# the seed list as a parameter instead of reading the module constant.
E10EC_HAR_ARMS = {"OFF": ("runs/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32", False)}
E10EC_FLOOR_ARMS = {"OFF": ({t: f"runs/ckpt_e10off_ec_floor4tec_{t}/mafc_seed42_fp32"
                             for t in ("a", "b")}, False)}
# E23 arm B (2026-09-20): the MLP backbone on the S72 construction (era + shadow), screened with the
# same instrument and draw as the LSTM arm so its Table-1-style row carries the re-layout column.
E23_MLP_ARMS = {"OFF": ("runs/ckpt_e23_har_mlp_seed{s}/mafc_seed{s}_fp32", False)}
ARM_SETS = {"e8": (HAR_ARMS, SEEDS), "e10": (E10_HAR_ARMS, SEEDS),
            "e10ec": (E10EC_HAR_ARMS, SEEDS), "e10ecfloor": (E10EC_FLOOR_ARMS, ["a", "b"]),
            "e23mlp": (E23_MLP_ARMS, SEEDS)}

# E18 (docs/E18_prereg.md sec 1, sec 8): the disjoint-content MNIST constructions.
# Each set: (checkpoint template or {tag: dir}, use_adapter, construction, seeds).
# The screen reads the fp32 SHADOW directories, and every run here records
# `disjoint_content: True` and its `content_fingerprint` in `arm` -- the
# benchmark is rebuilt from the run's seed with the same flag, and the
# fingerprint is asserted equal before any cell is scored (construction is an
# arm-identity field). Floor sets are replicates a/b of seed 42.
def _e18(tag, seeds):
    if seeds == SEEDS:
        return (f"runs/ckpt_{tag}_seed{{s}}/mafc_seed{{s}}_fp32", seeds)
    return ({t: f"runs/ckpt_{tag}_floor4tec_{t}/mafc_seed42_fp32" for t in seeds}, seeds)

E18_SETS = {}
for _tag, _kind in (("e18_pmd_mlp", "permuted"), ("e18_pmd_lstm", "permuted"), ("e18_rmd_mlp", "rotated")):
    d, sd = _e18(_tag, SEEDS);        E18_SETS[_tag] = (d, False, _kind, sd)
    d, sd = _e18(_tag, ["a", "b"]);   E18_SETS[_tag + "_floor"] = (d, False, _kind, sd)
E18_FACTORY = {"permuted": lambda seed: PermutedMNISTBenchmark(num_tasks=5, batch_size=BATCH, seed=seed, disjoint_content=True),
               "rotated":  lambda seed: RotatedMNISTBenchmark(num_tasks=5, batch_size=BATCH, seed=seed, disjoint_content=True)}
CURRENT_PROBE_SEED = None      # recorded into every row the screen writes


# ---------------------------------------------------------------- generators --
def har_maps(sd_vec):
    """Affine (M, b) per task in CALIBRATED units, matching the loader's algebra.

    The loader computes  x_k = Z @ M_k.T + b_k/sd, so the calibrated offset is
    b_k/sd and the inverse of the whole affine map is what C0 needs.
    """
    out = {}
    for k in range(5):
        M, b = channel_affine(k)
        out[k] = (M, b / sd_vec)
    return out


def har_to_base(x, k, maps):
    """C0: A_k(x) = M_k^-1 (x - b_k)  -- map task-k input back to the base layout."""
    M, b = maps[k]
    return torch.from_numpy(
        ((x.numpy() - b) @ np.linalg.inv(M).T).astype(np.float32))


def har_relayout(x, src, dst, maps):
    """Generator: task-src input -> task-dst layout, via M_dst . M_src^-1 (affine).

    v1's spec mapped current inputs THROUGH M_k, giving M_k M_4 Z instead of M_k Z
    (max err 5.1985 vs true task-1 data). This composition gives 0.0000.
    """
    Ms, bs = maps[src]
    Md, bd = maps[dst]
    Z = (x.numpy() - bs) @ np.linalg.inv(Ms).T
    return torch.from_numpy((Z @ Md.T + bd).astype(np.float32))


def rotated_relayout(x, src, dst, angles):
    """Rotated-MNIST analogue: rotate by (angle_dst - angle_src), bilinear, on
    [N, 28, 28] sequence-format images. Exact when the difference is a multiple
    of 90 degrees (a pixel permutation), LOSSY otherwise -- E18's k=1..3 pairs.
    The contract states which pairs are which; this function does not hide it."""
    import torchvision.transforms.functional as TF
    delta = float(angles[dst]) - float(angles[src])
    if delta == 0.0:
        return x
    return TF.rotate(x.unsqueeze(1), delta, interpolation=TF.InterpolationMode.BILINEAR).squeeze(1)


def mnist_relayout(x, src, dst, perms):
    """Permuted-MNIST analogue: undo perm_src, apply perm_dst. Exactly invertible."""
    flat = x.reshape(x.shape[0], -1)
    if perms[src] is not None:
        inv = torch.argsort(perms[src])
        flat = flat[:, inv]
    if perms[dst] is not None:
        flat = flat[:, perms[dst]]
    return flat.reshape(x.shape)


# -------------------------------------------------------------------- cures --
@torch.no_grad()
def acc_with_head(model, feats, y, head):
    return float((head(feats).argmax(1) == y).float().mean())


@torch.no_grad()
def era_head_of(model, task_k):
    if getattr(model, "use_task_heads", False) and str(task_k) in model.task_classifiers:
        return model.task_classifiers[str(task_k)]
    return model.classifier


def cure_c1(f_eval_t4, era_mu, era_sd, era_head, y):
    """Moment matching: batch-standardize theta_4 features into the era frame."""
    mu, sd = f_eval_t4.mean(0), f_eval_t4.std(0) + 1e-6
    aligned = (f_eval_t4 - mu) / sd * era_sd + era_mu
    with torch.no_grad():
        return float((era_head(aligned).argmax(1) == y).float().mean())


def cure_c2(f_eval_t4, proto_era, proto_now, era_head, y):
    """Procrustes: closed-form orthogonal map aligning current protos to era protos."""
    A = proto_now.numpy().astype(np.float64)
    B = proto_era.numpy().astype(np.float64)
    U, _, Vt = np.linalg.svd(A.T @ B)
    Q = torch.from_numpy((U @ Vt).astype(np.float32))       # d x d, orthogonal
    with torch.no_grad():
        return float((era_head(f_eval_t4 @ Q).argmax(1) == y).float().mean())


# ------------------------------------------------------------------ per-cell --
def screen_arm(dirs, use_adapter, data, maps_or_perms, kind, n_classes, device,
               out_rows, seeds=SEEDS):
    for seed in seeds:
        d = dirs[seed] if isinstance(dirs, dict) else dirs.format(s=seed)
        m4 = load(d, CUR)
        for k in range(4):
            xtr, ytr, xte, yte = data[k]
            m_era = load(d, k)
            eh = era_head_of(m_era, k)

            # E11/catch 28: the permanent gate, on every cell this screen scores,
            # for BOTH eras. `acc_orig` is the floor in every rho -- numerator and
            # denominator -- so an instrument reading the wrong tensor corrupts
            # each rho twice. This raises rather than warns.
            g_era = assert_path_identity(m_era, xte, k, use_adapter, device,
                                         label=f"era/T{k}", verbose=False)
            g_4 = assert_path_identity(m4, xte, k, use_adapter, device,
                                       label=f"t4/T{k}", verbose=False)

            # ---- references -------------------------------------------------
            f_e_tr, _ = features_and_logits(m_era, xtr, ytr, k, use_adapter, device)
            f_e_te, l_e_te = features_and_logits(m_era, xte, yte, k, use_adapter, device)
            f_4_tr, _ = features_and_logits(m4, xtr, ytr, k, use_adapter, device)
            f_4_te, l_4_te = features_and_logits(m4, xte, yte, k, use_adapter, device)

            acc_ceiling = float((l_e_te.argmax(1) == yte).float().mean())
            acc_orig = float((l_4_te.argmax(1) == yte).float().mean())
            acc_rc = refit_probe(f_e_tr, ytr, f_e_te, yte, n_classes)
            acc_r4 = refit_probe(f_4_tr, ytr, f_4_te, yte, n_classes)
            R = acc_rc - acc_ceiling

            row = {"seed": seed, "task": k, "acc_orig": acc_orig,
                   "acc_ceiling": acc_ceiling, "acc_refit": acc_r4, "R": R,
                   "probe_subset_seed": CURRENT_PROBE_SEED,
                   # P-B travels with the row so a routed-blend arm can never be
                   # read as reader-walk downstream without the flag being visible.
                   "p_b": bool(g_era["p_b"] and g_4["p_b"])}

            # ---- C0: analytic adapter, base layout --------------------------
            if kind == "har":
                x_base = har_to_base(xte, k, maps_or_perms)
                x_cur = har_relayout(xte, k, CUR, maps_or_perms)
            elif kind == "rotated":
                x_base = rotated_relayout(xte, k, 0, maps_or_perms)
                x_cur = rotated_relayout(xte, k, CUR, maps_or_perms)
            else:
                x_base = mnist_relayout(xte, k, 0, maps_or_perms)
                x_cur = mnist_relayout(xte, k, CUR, maps_or_perms)
            # C0 replaces the model's adapter with the analytic one, so the model's
            # own adapter is NOT applied on top.
            _, l_c0 = features_and_logits(m4, x_base, yte, k, False, device)
            row["C0"] = float((l_c0.argmax(1) == yte).float().mean())
            _, l_cd = features_and_logits(m4, x_cur, yte, k, False, device)
            row["C0deg"] = float((l_cd.argmax(1) == yte).float().mean())

            # ---- C1: moment matching + era head -----------------------------
            era_mu, era_sd = f_e_tr.mean(0), f_e_tr.std(0) + 1e-6
            row["C1"] = cure_c1(f_4_te, era_mu, era_sd, eh, yte)

            # ---- C0+C1: both channels --------------------------------------
            f_c0_te, _ = features_and_logits(m4, x_base, yte, k, False, device)
            row["C0C1"] = cure_c1(f_c0_te, era_mu, era_sd, eh, yte)

            # ---- C2: Procrustes (oracle theta_4 prototypes) -----------------
            pe = torch.stack([f_e_tr[ytr == c].mean(0) for c in range(n_classes)])
            pn = torch.stack([f_4_tr[ytr == c].mean(0) for c in range(n_classes)])
            row["C2"] = cure_c2(f_4_te, pe, pn, eh, yte)

            # ---- C3: pseudo-refit -------------------------------------------
            xcur_tr = data[CUR][0] if CUR in data else None
            if xcur_tr is not None:
                if kind == "har":
                    x_pseudo = har_relayout(xcur_tr, CUR, k, maps_or_perms)
                elif kind == "rotated":
                    x_pseudo = rotated_relayout(xcur_tr, CUR, k, maps_or_perms)
                else:
                    x_pseudo = mnist_relayout(xcur_tr, CUR, k, maps_or_perms)
                _, l_snap = features_and_logits(m_era, x_pseudo, None, k, use_adapter, device)
                y_pseudo = l_snap.argmax(1)                      # era snapshot's own labels
                f_ps, _ = features_and_logits(m4, x_pseudo, y_pseudo, k, use_adapter, device)
                row["C3"] = refit_probe(f_ps, y_pseudo, f_4_te, yte, n_classes)
            else:
                row["C3"] = float("nan")
            out_rows.append(row)


def generator_gate_e10(hdata, maps):
    """E10 generator gate -- ROUND-TRIP identity, not reconstruction.

    E8's pre-gate compared relayout(task4 -> k) against REAL task-k windows and
    demanded 0.0000. That was only checkable because every E8 task was the same
    2947 windows in different coordinates. On subject-disjoint HAR, task 4's
    subjects are different people from task k's, so that comparison is large BY
    CONSTRUCTION -- it is the property E10 exists to create. Carrying the gate
    over unchanged would sys.exit before a single cure was scored, and loosening
    its threshold would delete the only algebraic check on the generator.

    What survives the loss of shared windows is the AFFINE ALGEBRA. Mapping
    task-4 windows into task-k layout and back must return the originals:

        relayout(relayout(x, 4, k), k, 4) == x

    This fails loudly when M_k . M_4^-1 is composed wrongly (v1's x@M_k.T spec
    fails it), and it is scale-free -- error is normalised by the data's own
    magnitude rather than an absolute tolerance that assumes a feature scale.

    Falsifiability is deliberately NOT decided here -- that is P3's job. Keeping
    this gate narrow is the point: a gate broad enough to certify honesty would
    be a gate that cannot fail (catch 25).
    """
    print("\n" + "=" * 78)
    print("0. GENERATOR GATE -- round-trip identity")
    print("   (E8's reconstruction pre-gate is UNDEFINED here: disjoint subject")
    print("    groups have no row correspondence to compare against.)")
    print("=" * 78)
    worst = 0.0
    x0 = hdata[CUR][2]
    scale = float(np.abs(x0.numpy()).max())
    for k in range(4):
        back = har_relayout(har_relayout(x0, CUR, k, maps), k, CUR, maps).numpy()
        rel = float(np.abs(back - x0.numpy()).max()) / scale
        worst = max(worst, rel)
        print(f"  task {CUR} -> {k} -> {CUR}:  round-trip rel err {rel:.3e}")
    if worst > ROUNDTRIP_TOL:
        print(f"\n  worst {worst:.3e} > {ROUNDTRIP_TOL:.0e} -> GATE FAILED, STOP")
        sys.exit(2)
    print(f"\n  worst {worst:.3e} <= {ROUNDTRIP_TOL:.0e} -> GATE PASSED"
          f"  (honesty NOT asserted here; P3 decides)")


def screen_e18(args, device):
    """E18: one MNIST construction, one arm, all seeds; probe subset SEEDED and
    recorded; construction fingerprint asserted from each run's own artifact."""
    global CURRENT_PROBE_SEED
    dirs, use_ad, kind, seeds = E18_SETS[args.arms]
    out_dir = args.out or f"runs/{args.arms}/"
    os.makedirs(out_dir, exist_ok=True)
    CURRENT_PROBE_SEED = PROBE_SUBSET_SEED
    print("=" * 78)
    print(f"E18 SCREEN  {args.arms}  construction={kind}, disjoint content, probe subset seed {CURRENT_PROBE_SEED}")
    print("=" * 78)
    rows = {("MNIST", "OFF"): []}
    for seed in seeds:
        bench_seed = seed if isinstance(seed, int) else 42
        bench = E18_FACTORY[kind](bench_seed)
        run_dir = f"runs/{args.arms.replace('_floor', '')}_{'seed' + str(seed) if isinstance(seed, int) else 'floor4tec_' + seed}"
        arm = json.load(open(os.path.join(run_dir, "mafc_results.json")))["arm"]
        assert arm.get("disjoint_content") is True and arm.get("content_fingerprint") == bench.content_fingerprint(), \
            f"{run_dir}: artifact construction {arm.get('content_fingerprint')} != rebuilt {bench.content_fingerprint()}"
        # 2026-09-20: content_fingerprint is BLIND to the chunk boundaries -- T=5 and T=20 collide on it
        # (src/data/permuted_mnist.py). num_tasks and the chunk count are what separate them, so they are
        # asserted here; older artifacts have no `content_chunks` key and rebuild with None, which matches.
        assert arm.get("num_tasks") == bench.num_tasks and arm.get("content_chunks") == bench.content_chunks, \
            f"{run_dir}: construction shape (num_tasks {arm.get('num_tasks')}, chunks {arm.get('content_chunks')}) != rebuilt ({bench.num_tasks}, {bench.content_chunks})"
        if kind == "rotated":
            assert list(arm.get("angles")) == list(bench.angles), (arm.get("angles"), bench.angles)
        print(f"  {run_dir}: construction fingerprint {bench.content_fingerprint()} matches the artifact"
              + (f"; angles {bench.angles}" if kind == "rotated" else ""))
        mdata = load_task_data(bench, n_tasks=5, probe_subset_seed=CURRENT_PROBE_SEED)
        maps = bench.angles if kind == "rotated" else bench.permutations
        d = dirs if isinstance(dirs, str) else {seed: dirs[seed]}
        screen_arm(d, use_ad, mdata, maps, kind, 10, device, rows[("MNIST", "OFF")], seeds=[seed])
    m = lambda rr, key: float(np.nanmean([r[key] for r in rr]))
    rr = rows[("MNIST", "OFF")]
    print(f"\n  cells {len(rr)}  floor {m(rr,'acc_orig'):.4f}  refit {m(rr,'acc_refit'):.4f}  ceiling {m(rr,'acc_ceiling'):.4f}  R {m(rr,'R'):+.4f}")
    for c in ("C0", "C0deg", "C1", "C0C1", "C2", "C3"):
        print(f"    {c:<6} acc {m(rr, c):.4f}")
    json.dump({f"{k[0]}/{k[1]}": v for k, v in rows.items()},
              open(os.path.join(out_dir, "cures.json"), "w"), indent=2, default=float)
    print(f"\n  wrote {out_dir}cures.json  (hypotheses: docs/E18_prereg.md; read by hx2_forgetting / rho_percell)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Reader-repair cure screen (E8 / E10 / E18)")
    ap.add_argument("--arms", choices=sorted(ARM_SETS) + sorted(E18_SETS), default="e8",
                    help="e8: shared-window HAR (E5 ckpts). e10: subject-disjoint HAR. "
                         "e10ec / e10ecfloor: the S72 relaunch (era + fp32 shadow) and its floor pair.")
    ap.add_argument("--out", default=None, help="default: runs/e8/ or runs/e10/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-mnist", action="store_true")
    ap.add_argument("--probe-seed", type=int, default=None,
                    help="R2 (runs/MEMO_e20.md): seed the probe's training-subset draw and record it. "
                         "Default None = the pre-fix, unrecorded draw; E18 sets always seed.")
    args = ap.parse_args()
    global CURRENT_PROBE_SEED
    CURRENT_PROBE_SEED = args.probe_seed
    device = torch.device(args.device)
    if args.arms in E18_SETS:
        return screen_e18(args, device)
    is_e10 = args.arms != "e8"          # every non-E8 set is the subject-disjoint benchmark
    arm_table, seeds = ARM_SETS[args.arms]
    if args.out is None:
        args.out = f"runs/{args.arms}/"
    if is_e10:
        args.skip_mnist = True          # E10 is a HAR-only benchmark repair
    os.makedirs(args.out, exist_ok=True)

    fp = spec_fingerprint()
    print(f"HAR shift fingerprint {fp} -> "
          f"{'MATCHES frozen config' if fp == '3de66e205eb7' else 'MISMATCH - STOP'}")
    if fp != "3de66e205eb7":
        sys.exit(1)

    if is_e10:
        har = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=BATCH)
        print(f"E10 subject partition fingerprint {har.partition_fingerprint()}")
        print(f"  groups (train|test): "
              + "  ".join(f"{tr}|{te}" for tr, te in
                          zip(har.train_subjects, har.test_subjects)))
    else:
        har = HARShiftBenchmark(num_tasks=5, root=".", batch_size=BATCH)
    maps = har_maps(har._sd)
    hdata = load_task_data(har, n_tasks=5, probe_subset_seed=CURRENT_PROBE_SEED)
    print(f"probe subset draw: {'seeded ' + str(CURRENT_PROBE_SEED) + ' (recorded per row)' if CURRENT_PROBE_SEED is not None else 'UNSEEDED (pre-R2 path; draw unrecorded)'}")

    if is_e10:
        generator_gate_e10(hdata, maps)
    else:
        # ---- sec-0 PRE-GATE: generator correctness -----------------------------
        print("\n" + "=" * 78)
        print("0. GENERATOR PRE-GATE (must print 0.0000 before any cure is scored)")
        print("=" * 78)
        worst = 0.0
        for src in (CUR,):
            for dst in range(4):
                got = har_relayout(hdata[src][2], src, dst, maps).numpy()
                want = hdata[dst][2].numpy()
                e = np.abs(got - want).max(); worst = max(worst, e)
                naive = (hdata[src][2].numpy() @ maps[dst][0].T)
                print(f"  task {src} -> {dst}:  M_k.M_4^-1 affine err {e:.4f}"
                      f"   |  v1 spec (x@M_k.T) err {np.abs(naive - want).max():.4f}")
        if worst > 1e-4:
            print(f"\n  PRE-GATE FAILED (worst {worst:.4f}) -> STOP"); sys.exit(2)
        print(f"\n  worst {worst:.4f} -> PRE-GATE PASSED")

    rows = {}
    for arm, (tmpl, use_ad) in arm_table.items():
        rows[("HAR", arm)] = []
        screen_arm(tmpl, use_ad, hdata, maps, "har", 6, device, rows[("HAR", arm)], seeds=seeds)

    if not args.skip_mnist:
        for arm, (dirs, use_ad) in MNIST_ARMS.items():
            rows[("MNIST", arm)] = []
            for seed in SEEDS:
                mb = PermutedMNISTBenchmark(num_tasks=5, batch_size=BATCH, seed=seed)
                mdata = load_task_data(mb, n_tasks=5)
                screen_arm({seed: dirs[seed]}, use_ad, mdata, mb.permutations,
                           "mnist", 10, device, rows[("MNIST", arm)])

    CURES = ["C0", "C1", "C0C1", "C2", "C3"]
    m = lambda rr, key: float(np.nanmean([r[key] for r in rr]))

    print("\n" + "=" * 78)
    print("1. REFERENCES (floor / ceiling / instrument bias)")
    print("=" * 78)
    print(f"  {'arm':<14}{'floor':>9}{'ceiling':>9}{'R':>9}{'adj denom':>12}")
    denom = {}
    for key, rr in rows.items():
        fl, ce, R = m(rr, "acc_orig"), m(rr, "acc_refit"), m(rr, "R")
        denom[key] = (ce - R - fl, ce - fl)
        print(f"  {key[0]+'/'+key[1]:<14}{fl:>9.4f}{ce:>9.4f}{R:>+9.4f}{denom[key][0]:>12.4f}")

    print("\n" + "=" * 78)
    print("2. CURES — accuracy, then rho vs BIAS-ADJUSTED ceiling (raw in parens)")
    print("=" * 78)
    for key, rr in rows.items():
        fl = m(rr, "acc_orig")
        dadj, draw = denom[key]
        print(f"\n  {key[0]}/{key[1]}   floor {fl:.4f}   adj denom {dadj:.4f}")
        for c in CURES:
            a = m(rr, c)
            if dadj <= RHO_MIN_DENOM:
                print(f"    {c:<6} acc {a:.4f}   rho N/A (denominator {dadj:.4f})")
            else:
                print(f"    {c:<6} acc {a:.4f}   rho {(a-fl)/dadj:>+7.3f}"
                      f"  ({(a-fl)/draw:+.3f} raw)")
        c0deg_label = ("[CEILING ARTIFACT on shared windows, not a cure]" if not is_e10
                       else "[storage-honest METHOD on subject-disjoint HAR; runs/MEMO_c0deg.md]")
        print(f"    {'C0deg':<6} acc {m(rr,'C0deg'):.4f}   {c0deg_label}")

    if is_e10:
        # E10's hypotheses are H-X1..H-X4, not E8's H-R set, and their thresholds
        # live in the v2 contract. Printing E8 verdicts over E10 numbers would be
        # a verdict that agrees with the wrong contract while sitting beside these
        # values (catch 22). Per-seed rho + propagated CIs (catch 4) and the H-X1
        # reading are emitted by the E10 reporter once v2 is on disk.
        json.dump({f"{k[0]}/{k[1]}": v for k, v in rows.items()},
                  open(os.path.join(args.out, "cures_e10.json"), "w"), indent=2)
        print("\n" + "=" * 78)
        print("3. HYPOTHESES -- DEFERRED")
        print("=" * 78)
        print("  E10 evaluates H-X1..H-X4 against docs/E10_prereg.md (v2).")
        print("  E8's H-R thresholds are NOT applicable and are not printed.")
        print(f"  Per-cell rows written to {args.out}cures_e10.json")
        return

    print("\n" + "=" * 78)
    print("3. HYPOTHESES")
    print("=" * 78)
    hk = ("HAR", "OFF")
    fl = m(rows[hk], "acc_orig"); dadj = denom[hk][0]
    rho = {c: (m(rows[hk], c) - fl) / dadj for c in CURES}
    c0_forget = 1.0 - m(rows[hk], "C0")
    print(f"  H-R0  C0 'forgetting' proxy (1-acc) {c0_forget:.4f} <= 0.20"
          f"  -> {'PASS' if c0_forget <= 0.20 else 'FAIL'}")
    best_honest = max(rho["C0"], rho["C1"], rho["C3"])
    print(f"  H-R1  best non-oracle rho {best_honest:+.3f} >= 0.50"
          f"  -> {'PASS' if best_honest >= 0.50 else 'FAIL'}")
    print(f"  H-R2  C2 {rho['C2']:+.3f} - C1 {rho['C1']:+.3f} = {rho['C2']-rho['C1']:+.3f}"
          f" >= 0.15 -> {'PASS' if rho['C2']-rho['C1'] >= 0.15 else 'FAIL'}")
    print(f"  H-R3  C3 rho {rho['C3']:+.3f} >= 0.80"
          f"  -> {'PASS' if rho['C3'] >= 0.80 else 'FAIL'}")
    print(f"  H-R5  C0+C1 {rho['C0C1']:+.3f} vs best single {max(rho['C0'],rho['C1']):+.3f}"
          f" (+{rho['C0C1']-max(rho['C0'],rho['C1']):.3f}) >= 0.10"
          f" -> {'PASS' if rho['C0C1']-max(rho['C0'],rho['C1']) >= 0.10 else 'FAIL'}")
    if ("MNIST", "OFF") in rows:
        mk = ("MNIST", "OFF"); mfl = m(rows[mk], "acc_orig"); md = denom[mk][0]
        bestc = max(CURES, key=lambda c: m(rows[hk], c))
        r4 = (m(rows[mk], bestc) - mfl) / md
        print(f"  H-R4  best HAR cure ({bestc}) on MNIST/OFF rho {r4:+.3f} >= 0.30"
              f"  -> {'PASS' if r4 >= 0.30 else 'FAIL'}")

    if best_honest >= 0.50:
        br = "(A) reader channel repairable at read time -> draft E9"
    elif best_honest >= 0.20:
        br = "(B) partially repairable -> name the residual before any E9"
    elif rho["C2"] >= 0.50:
        br = "(C) rotational, and ESTIMATION is the hard part -> named open question"
    else:
        br = "(D) not linearly correctable at read time -> repairability claim dropped"
    print(f"\n  BRANCH: {br}")

    json.dump({f"{a}/{b}": rr for (a, b), rr in rows.items()},
              open(os.path.join(args.out, "cures.json"), "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}cures.json")


if __name__ == "__main__":
    main()
