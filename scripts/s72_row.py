"""S72 -- E16 sec 7.2 closes: the three-way row on the bridging benchmark.

Ruling (runs/MEMO_c0deg.md sec 6): "the E16 sec 7.2 relaunch pair, both arms
uniform era status, three-way row" -- LwF / C3 (bridging) / C0deg, on the
benchmark bridging's number lives on (`har_subject`, E10 construction, OFF arm).

WHY BOTH ARMS ARE NEW RUNS. The artifacts' `arm` field says the E16 LwF HAR arm
ran `benchmark: 'har'` (E5 shared-window) while bridging is `har_subject`; the
old E10 OFF runs have no era checkpoints and no `arm` field. So the pair was
relaunched on one benchmark with one flag set (`modal_runner.jobs_s72_*`):
era checkpoints + fp32 shadow, 4 threads. LwF's lambda was selected on the other
construction, so its sweep moved with it (the EWC lambda=200 rule).

WHAT THIS SCRIPT DOES, in order, every verdict from the values it prints:
  1. FINISH -- verify_runs.check on every run (matrix 5x5 no NaN, 5 tasks x 10
     epochs). A directory is not a run.
  2. ARM IDENTITY FROM ARTIFACTS, not from the launcher: benchmark, era status,
     adapters, threads on every run; LwF's witness is a nonzero
     `lwf_distill_loss` in every epoch dict of tasks >= 1 (E16 sec 4b); the OFF
     arm's is `lwf_enabled: False`. fp32 shadow witness: every task boundary
     carries `era_checkpoint.fp32_shadow` with delta == 0.0 and no error.
     UNIFORMITY IS CHECKED ACROSS ARMS: the non-LwF, non-seed `arm` fields must
     be identical over all 19 runs.
  3. LwF SWEEP on har_subject: selection on AVG (never forgetting), competence
     floor = mean DIAG within 5pp of the reference arm's DIAG, full table,
     exclusions counted, across-lambda spread vs across-seed sd.
  4. THE ROW: none / LwF(lambda*) / C3 / C0deg / SNAP, BOTH formulas (diag = the
     paper's sec 3.1 definition; peak-clipped = metrics.py), on the cure
     screen's intersection cell population AND on all 12 cells. C3/C0deg come
     from runs/e10ec/hx2_c0deg{,_peak}.json (the H-X2 instrument on the new
     checkpoints); LwF from its own matrices, scored on the SAME cells.
  5. FLOORS, per quantity: the e10off_ec and LwF floor pairs (seed 42 twice),
     AVG and forgetting deltas and matrix cell identity; cure-level floors from
     runs/e10ecfloor/hx2_c0deg*.json. No delta below its floor is read.
  6. E16's pending prediction "bridging > LwF on HAR/OFF, ~55%" scored under
     both formulas, and C0deg vs LwF beside it.

Usage:
    python scripts/s72_row.py            # after `modal volume get` of the run dirs
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.verify_runs import check as verify_check
from scripts.rho_percell import E10_NTEST

SEEDS = [42, 1337, 2024]
LAMBDAS = [0.25, 1.0, 4.0, 16.0]
N_TASKS, EPOCHS, CUR = 5, 10, 4
OLD = list(range(CUR))
COMPETENCE_PP = 0.05                 # E16 sec 1, unchanged
BAR = 0.10                           # H-X2, unchanged
REF = "runs/e10off_ec_seed{s}/mafc_results.json"
REF_FLOOR = "runs/e10off_ec_floor4tec_{t}/mafc_results.json"
LWF = "runs/e16s_har_lam{lam}_seed{s}/mafc_results.json"
LWF_FLOOR = "runs/e16s_har_floor4tec_lam{lam}_{t}/mafc_results.json"   # 0.25 = prior, lambda* = required
OLD_E10 = "runs/e10_off_seed{s}/mafc_results.json"    # era OFF, pre-arm-recording
CURES = {"diag": "runs/e10ec/hx2_c0deg.json", "peak": "runs/e10ec/hx2_c0deg_peak.json"}
CURES_FLOOR = {"diag": "runs/e10ecfloor/hx2_c0deg.json", "peak": "runs/e10ecfloor/hx2_c0deg_peak.json"}
LWF_INVARIANT = ("lwf_enabled", "lwf_lambda", "lwf_temperature", "lwf_warmup_epochs", "seed")


def banner(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


def load(path):
    return json.load(open(path))


def matrix(r):
    return np.array(r["accuracy_matrix"], float)


def forgetting_cells(m, cells, formula):
    """Per-cell forgetting terms for old tasks in `cells` (task indices), from a
    matrix, under the named formula -- the same two definitions
    hx2_forgetting.py implements."""
    if formula == "diag":
        return [m[k, k] - m[CUR, k] for k in cells]
    return [max(0.0, m[:, k].max() - m[CUR, k]) for k in cells]


def per_seed_mean_and_se(mats, cells_by_seed, formula):
    """Mean over seeds of the per-seed mean over cells; binomial SE of the final
    row propagated the way hx2_forgetting.py does it."""
    per_seed, ses = [], []
    for s, m in mats.items():
        cells = cells_by_seed[s]
        if not cells:
            continue
        per_seed.append(float(np.mean(forgetting_cells(m, cells, formula))))
        se = math.sqrt(sum(max(m[CUR, k] * (1 - m[CUR, k]), 0.0) / E10_NTEST[k] for k in cells)) / len(cells)
        ses.append(se)
    return float(np.mean(per_seed)), float(math.sqrt(sum(x * x for x in ses)) / len(ses)), per_seed


def main():
    ap = argparse.ArgumentParser(description="S72 three-way row")
    ap.add_argument("--out", default="runs/s72_row.json")
    args = ap.parse_args()
    out = {}

    # ---- 1. finish ------------------------------------------------------------
    banner("1. FINISH  (verify_runs.check: 5x5 matrix, no NaN, 5 tasks x 10 epochs)")
    paths = {f"e10off_ec_seed{s}": REF.format(s=s) for s in SEEDS}
    paths.update({f"e10off_ec_floor4tec_{t}": REF_FLOOR.format(t=t) for t in "ab"})
    paths.update({f"e16s_har_lam{l}_seed{s}": LWF.format(lam=l, s=s) for l in LAMBDAS for s in SEEDS})
    paths.update({f"e16s_har_floor4tec_lam0.25_{t}": LWF_FLOOR.format(lam=0.25, t=t) for t in "ab"})
    finished = {}
    for name, p in paths.items():
        v = verify_check(p, N_TASKS, EPOCHS, None) if Path(p).exists() else {"ok": False, "problems": ["missing"]}
        finished[name] = v["ok"]
        print(f"  {name:<36} {'ok' if v['ok'] else 'NOT FINISHED: ' + '; '.join(v['problems'])}")
    n_ok = sum(finished.values())
    print(f"  {n_ok}/{len(paths)} finished")
    out["finished"] = finished
    if n_ok < len(paths):
        print("  STOP -- the row is not readable until every run is finished.")
        json.dump(out, open(args.out, "w"), indent=2)
        sys.exit(1)
    runs = {name: load(p) for name, p in paths.items()}

    # ---- 2. arm identity, from artifacts ---------------------------------------
    banner("2. ARM IDENTITY FROM ARTIFACTS  (uniformity checked ACROSS arms)")
    problems = []
    inv = None
    for name, r in runs.items():
        a = r["arm"]
        core = {k: v for k, v in a.items() if k not in LWF_INVARIANT}
        if inv is None:
            inv = core
        elif core != inv:
            diff = {k: (inv.get(k), core.get(k)) for k in set(inv) | set(core) if inv.get(k) != core.get(k)}
            problems.append(f"{name}: arm fields differ from the first run: {diff}")
        if a["benchmark"] != "har_subject":
            problems.append(f"{name}: benchmark {a['benchmark']!r}")
        if not a["era_checkpoints"] or a["use_input_adapters"] or a["torch_num_threads"] != 4:
            problems.append(f"{name}: era {a['era_checkpoints']} adapters {a['use_input_adapters']} threads {a['torch_num_threads']}")
        is_lwf = name.startswith("e16s_")
        if a["lwf_enabled"] != is_lwf:
            problems.append(f"{name}: lwf_enabled {a['lwf_enabled']} for a {'LwF' if is_lwf else 'reference'} run")
        if is_lwf:
            lam = float(name.split("lam")[1].split("_")[0])
            if a["lwf_lambda"] != lam:
                problems.append(f"{name}: lwf_lambda {a['lwf_lambda']} != {lam}")
            for t in r["task_history"][1:]:
                if any(not e.get("lwf_distill_loss") for e in t["epochs"]):
                    problems.append(f"{name}: task {t['task_id']} has an epoch with no distillation loss")
        for t in r["task_history"]:
            e = t.get("era_checkpoint") or {}
            sh = e.get("fp32_shadow") or {}
            if e.get("error") or sh.get("error") or sh.get("delta") != 0.0:
                problems.append(f"{name}: task {t['task_id']} era/shadow audit: {e.get('error')} {sh.get('error')} shadow delta {sh.get('delta')}")
    print(f"  shared arm fields over {len(runs)} runs: benchmark={inv['benchmark']} era={inv['era_checkpoints']} "
          f"adapters={inv['use_input_adapters']} threads={inv['torch_num_threads']} epochs={inv['epochs_per_task']}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print(f"  arm identity -> {'PASS' if not problems else 'FAIL'}")
    out["arm_identity"] = {"pass": not problems, "problems": problems, "shared": inv}
    if problems:
        json.dump(out, open(args.out, "w"), indent=2)
        sys.exit(2)

    # ---- 3. reference arm, and the era effect on this benchmark ----------------
    banner("3. REFERENCE ARM  e10off_ec (era ON)  vs  e10_off (era OFF, pre-arm-recording)")
    ref = {s: runs[f"e10off_ec_seed{s}"] for s in SEEDS}
    old = {s: load(OLD_E10.format(s=s)) for s in SEEDS}
    print(f"  {'seed':<6}{'AVG new':>9}{'AVG old':>9}{'delta':>8}{'DIAG new':>10}{'DIAG old':>10}{'cells id':>10}")
    era_effect = []
    for s in SEEDS:
        mn, mo = matrix(ref[s]), matrix(old[s])
        ident = int((mn == mo).sum())
        era_effect.append(ref[s]["average_accuracy"] - old[s]["average_accuracy"])
        print(f"  {s:<6}{ref[s]['average_accuracy']:9.4f}{old[s]['average_accuracy']:9.4f}{era_effect[-1]:+8.4f}"
              f"{np.mean(np.diag(mn)):10.4f}{np.mean(np.diag(mo)):10.4f}{ident:>7}/25")
    print(f"  era-checkpoint effect on AVG, this benchmark: mean {np.mean(era_effect):+.4f} "
          f"(E16 measured +2.13pp MNIST / -1.67pp HAR-shared-window; this is a third value, not a correction)")
    ref_diag = float(np.mean([np.mean(np.diag(matrix(r))) for r in ref.values()]))
    out["reference"] = {"avg": [r["average_accuracy"] for r in ref.values()], "diag": ref_diag,
                        "era_effect_avg": era_effect}

    # ---- 4. LwF sweep on har_subject -------------------------------------------
    banner("4. LwF SWEEP on har_subject  (selection on AVG; competence = DIAG within 5pp of reference)")
    sweep = {}
    print(f"  {'lambda':<8}{'AVG':>8}{'sd':>8}{'DIAG':>8}{'gap':>8}{'fgt diag':>10}{'fgt peak':>10}  competent")
    for lam in LAMBDAS:
        rs = [runs[f"e16s_har_lam{lam}_seed{s}"] for s in SEEDS]
        avg = [r["average_accuracy"] for r in rs]
        diag = float(np.mean([np.mean(np.diag(matrix(r))) for r in rs]))
        fd = float(np.mean([np.mean(forgetting_cells(matrix(r), OLD, "diag")) for r in rs]))
        fp = float(np.mean([np.mean(forgetting_cells(matrix(r), OLD, "peak")) for r in rs]))
        gap = diag - ref_diag
        competent = gap >= -COMPETENCE_PP
        sweep[lam] = {"avg_mean": float(np.mean(avg)), "avg_sd": float(np.std(avg, ddof=1)), "avg": avg,
                      "diag": diag, "gap": gap, "competent": competent, "fgt_diag": fd, "fgt_peak": fp}
        print(f"  {lam:<8}{np.mean(avg):8.4f}{np.std(avg, ddof=1):8.4f}{diag:8.4f}{gap:+8.4f}{fd:10.4f}{fp:10.4f}  "
              f"{'yes' if competent else 'EXCLUDED'}")
    candidates = [l for l in LAMBDAS if sweep[l]["competent"]]
    across_lam = max(sweep[l]["avg_mean"] for l in LAMBDAS) - min(sweep[l]["avg_mean"] for l in LAMBDAS)
    across_seed = float(np.mean([sweep[l]["avg_sd"] for l in LAMBDAS]))
    print(f"  across-lambda AVG spread {across_lam:.4f} vs mean across-seed sd {across_seed:.4f} "
          f"({across_lam / max(across_seed, 1e-9):.1f}x) -> {'structure' if across_lam > across_seed else 'NOISE-DOMINATED'}")
    print(f"  exclusions: {len(LAMBDAS) - len(candidates)}/{len(LAMBDAS)}")
    if not candidates:
        print("  no competent lambda; LwF has no reportable configuration on this benchmark. STOP")
        out["sweep"] = sweep
        json.dump(out, open(args.out, "w"), indent=2)
        sys.exit(3)
    lam_star = max(candidates, key=lambda l: sweep[l]["avg_mean"])
    print(f"  lambda* = {lam_star} (AVG {sweep[lam_star]['avg_mean']:.4f})"
          f"{'' if lam_star == 0.25 else '  <- differs from the prior winner 0.25; a floor pair at lambda* is REQUIRED before the row is cited'}")
    diag_star = [float(np.mean(np.diag(matrix(runs[f"e16s_har_lam{lam_star}_seed{s}"])))) for s in SEEDS]
    print(f"  lambda* per-seed DIAG {['%.4f' % d for d in diag_star]} vs reference DIAG {ref_diag:.4f} "
          f"(competence margin {sweep[lam_star]['gap'] + COMPETENCE_PP:+.4f})")
    out["sweep"] = {"table": {str(l): v for l, v in sweep.items()}, "lambda_star": lam_star,
                    "across_lambda": across_lam, "across_seed": across_seed,
                    "floor_pair_at_lambda_star": lam_star == 0.25}

    # ---- 5. the row -------------------------------------------------------------
    banner("5. THE ROW  none / LwF(lambda*) / C3 / C0deg / SNAP  -- both formulas")
    lwf_mats = {s: matrix(runs[f"e16s_har_lam{lam_star}_seed{s}"]) for s in SEEDS}
    ref_mats = {s: matrix(ref[s]) for s in SEEDS}
    row = {}
    for formula in ("diag", "peak"):
        cpath = CURES[formula]
        if not Path(cpath).exists():
            print(f"\n  [{formula}] {cpath} missing -- run the cure screen + hx2 on e10ec first. Cure columns PENDING.")
            cures = None
            cells_by_seed = {s: OLD for s in SEEDS}
            n_inter = 12
        else:
            cures = load(cpath)["HAR/OFF"]
            cells = [(int(a) if str(a).isdigit() else a, int(b)) for a, b in cures["intersection_cells"]]
            cells_by_seed = {s: sorted(b for a, b in cells if a == s) for s in SEEDS}
            n_inter = len(cells)
        all_cells = {s: OLD for s in SEEDS}
        none_i, none_se, _ = per_seed_mean_and_se(ref_mats, cells_by_seed, formula)
        none_f, _, _ = per_seed_mean_and_se(ref_mats, all_cells, formula)
        lwf_i, lwf_se, lwf_ps = per_seed_mean_and_se(lwf_mats, cells_by_seed, formula)
        lwf_f, _, _ = per_seed_mean_and_se(lwf_mats, all_cells, formula)
        print(f"\n  formula = {formula}   intersection population {n_inter} cells")
        print(f"  {'method':<14}{'intersection':>14}{'+-95%':>8}{'full 12':>10}")
        print(f"  {'none':<14}{none_i:14.4f}{1.96*none_se:8.3f}{none_f:10.4f}")
        print(f"  {'LwF l*='+str(lam_star):<14}{lwf_i:14.4f}{1.96*lwf_se:8.3f}{lwf_f:10.4f}")
        entry = {"n_intersection": n_inter, "none": {"inter": none_i, "se": none_se, "full": none_f},
                 "lwf": {"inter": lwf_i, "se": lwf_se, "full": lwf_f, "per_seed": lwf_ps, "lambda": lam_star}}
        if cures is not None:
            for c in ("C3", "C0deg", "SNAP"):
                pm = cures["per_method"][c]
                print(f"  {c:<14}{pm['intersection']:14.4f}{1.96*pm['se']:8.3f}{pm['full_pop']:10.4f}")
                entry[c] = {"inter": pm["intersection"], "se": pm["se"], "full": pm["full_pop"]}
            print(f"  (screen's uncured baseline on these cells: {cures['uncured']:.4f}; the row's 'none' above is "
                  f"recomputed from the matrices -- they must agree)")
            entry["screen_uncured"] = cures["uncured"]
            entry["none_agrees_with_screen"] = abs(cures["uncured"] - none_i) < 1e-6
            print(f"  none vs screen uncured: |diff| {abs(cures['uncured'] - none_i):.1e} -> "
                  f"{'agree' if entry['none_agrees_with_screen'] else 'DISAGREE -- different runs or cells'}")
            # ---- comparisons, intervals ----
            def sep(a, sa, b, sb):
                return (a + 1.96 * sa) < (b - 1.96 * sb)
            c3, c0 = entry["C3"], entry["C0deg"]
            print(f"  bridging (C3) < LwF: {'yes' if c3['inter'] < lwf_i else 'no'}"
                  f"  {'RESOLVED' if sep(c3['inter'], c3['se'], lwf_i, lwf_se) or sep(lwf_i, lwf_se, c3['inter'], c3['se']) else 'unresolved (intervals overlap)'}")
            print(f"  C0deg < LwF:        {'yes' if c0['inter'] < lwf_i else 'no'}"
                  f"  {'RESOLVED' if sep(c0['inter'], c0['se'], lwf_i, lwf_se) or sep(lwf_i, lwf_se, c0['inter'], c0['se']) else 'unresolved'}")
            print(f"  H-X2 bar {BAR}: C3 {'MET' if c3['inter'] <= BAR else 'NOT MET'}, "
                  f"C0deg {'MET' if c0['inter'] <= BAR else 'NOT MET'}, LwF {'MET' if lwf_i <= BAR else 'NOT MET'}")
            entry["bridging_beats_lwf"] = c3["inter"] < lwf_i
            entry["c0deg_beats_lwf"] = c0["inter"] < lwf_i
        row[formula] = entry
    out["row"] = row

    # ---- 6. floors ----------------------------------------------------------------
    banner("6. FLOORS  (seed 42 twice, same flags, same threading -- per quantity)")
    floors = {}
    pairs = [("e10off_ec", [runs["e10off_ec_floor4tec_a"], runs["e10off_ec_floor4tec_b"]]),
             ("lwf_lam0.25 (prior; floor observation)", [runs["e16s_har_floor4tec_lam0.25_a"],
                                                          runs["e16s_har_floor4tec_lam0.25_b"]])]
    star_paths = [LWF_FLOOR.format(lam=lam_star, t=t) for t in "ab"]
    if lam_star != 0.25:
        if all(Path(q).exists() and verify_check(q, N_TASKS, EPOCHS, None)["ok"] for q in star_paths):
            pairs.append((f"lwf_lam{lam_star} (lambda*)", [load(q) for q in star_paths]))
        else:
            print(f"  lwf_lam{lam_star} (lambda*) floor pair NOT FINISHED -- the LwF cell is not citable yet")
            floors["lambda_star_floor_pending"] = True
    for label, (ra, rb) in pairs:
        ma, mb = matrix(ra), matrix(rb)
        f = {"avg": abs(ra["average_accuracy"] - rb["average_accuracy"]),
             "cells_identical": int((ma == mb).sum()),
             "fgt_diag": abs(np.mean(forgetting_cells(ma, OLD, "diag")) - np.mean(forgetting_cells(mb, OLD, "diag"))),
             "fgt_peak": abs(np.mean(forgetting_cells(ma, OLD, "peak")) - np.mean(forgetting_cells(mb, OLD, "peak")))}
        floors[label] = f
        print(f"  {label:<12} AVG |a-b| {f['avg']:.4f}   cells identical {f['cells_identical']}/25   "
              f"forgetting |a-b| diag {f['fgt_diag']:.4f} peak {f['fgt_peak']:.4f}")
    for formula in ("diag", "peak"):
        p = CURES_FLOOR[formula]
        if Path(p).exists():
            cf = load(p)["HAR/OFF"]
            # the floor artifact holds replicates a/b as "seeds"; the per-method
            # intersection value is a mean over both, so read the per-cell rows
            # of the cures file for the a-vs-b spread instead
            cures_rows = load("runs/e10ecfloor/cures_e10.json")["HAR/OFF"]
            for c in ("C3", "C0deg"):
                va = {r["task"]: r[c] for r in cures_rows if r["seed"] == "a"}
                vb = {r["task"]: r[c] for r in cures_rows if r["seed"] == "b"}
                spread = max(abs(va[k] - vb[k]) for k in va)
                floors[f"{c}_percell_{formula}"] = spread
                print(f"  {c:<12} per-cell |a-b| max {spread:.4f}  ({formula} population n={cf['n_intersection']})")
        else:
            print(f"  cure-level floors ({formula}): {p} missing -- run the screen on e10ecfloor. PENDING")
    out["floors"] = floors

    # ---- 7. E16's pending prediction --------------------------------------------
    banner("7. E16 PREDICTION ON RECORD: 'bridging > LwF on HAR/OFF' (~55%), pending since 2026-08-17")
    for formula in ("diag", "peak"):
        e = row[formula]
        if "bridging_beats_lwf" in e:
            print(f"  [{formula}] bridging (C3 {e['C3']['inter']:.4f}) vs LwF ({e['lwf']['inter']:.4f}): "
                  f"{'FIRED' if e['bridging_beats_lwf'] else 'MISS'};   C0deg ({e['C0deg']['inter']:.4f}) vs LwF: "
                  f"{'C0deg ahead' if e['c0deg_beats_lwf'] else 'LwF ahead'}")
        else:
            print(f"  [{formula}] cure columns pending")
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
