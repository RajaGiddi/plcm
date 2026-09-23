"""E23 row -- preconditions, controls, floors, the crux (E1)/(E2), the sec 6
readings and the sec 7 predictions (docs/E23_prereg.md, signed v2; option (b)).
Written 2026-09-19 BEFORE any B6 artifact was read, so the floor and the
readings below are stated first.

Per backbone (ViT-B/16 = e23_vit vs B1 e12_base; ResNet-50 = e23_rn vs e14_base):

  P1     frozen reference (runs/e23/frozen_{vit,resnet}.json, per-task PERMUTED extraction)
         mean DIAG >= 0.85 and the trainable trunk's mean DIAG within 0.05 of it    (E12/E14 form)
  P2(a)  task-0 deployed drop R[0,0] - R[-1,0] >= 0.15, mean over seeds             (E12/E14 form)
  P2(b)  repeat-task control: artifact `forgetting` <= 0.05, mean over seeds        (E12/E14 form, the permuted control)
  C-WIT  era_checkpoints True on B6 and B1 runs, fp32 shadow on neither, per-task heads, same backbone
  C-FLOOR seed-42 relaunch: matrix cells identical / max |delta| / AVG delta; and, when the floor relaunch is
         decomposed (decomp_floor42.json), |F_enc(main) - F_enc(floor)| pooled, raw and re-laid
  C-RELOAD fp16 reload floor per run (the identity gate's resolution under option (b))
  C-ID   relay identity and algebraic-vs-dataset rendering exact; wrong-permutation must-fail 19/19 per seed

  (E1)  F_enc(B6, re-laid) ~ F_enc(B1)          |delta| <= floor
  (E2)  F_enc(B6, raw)     >  F_enc(B6, re-laid) by more than floor

  floor = max( t95 half-width over the 3 seeds of the per-seed pooled difference,
               binomial: 1.96 * sqrt(p(1-p)/n_test) * sqrt(2) / sqrt(n_cells)   (two independent refit accuracies, pooled over cells),
               F_enc floor from the seed-42 relaunch pair when present )
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = [42, 1337, 2024]
ARMS = {"vit": ("e23_vit", "runs/e12_decomp_v2/base_{s}.json", "runs/e12_base_seed{s}", "runs/e23/frozen_vit.json"),
        "resnet": ("e23_rn", "runs/e14_decomp/base_{s}.json", "runs/e14_base_seed{s}", "runs/e23/frozen_resnet.json")}
P1_BAR, P1_MARGIN, P2A_BAR, P2B_BAR = 0.85, 0.05, 0.15, 0.05


def t95_half(x):
    x = np.asarray(x, float); return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/e23_row.json"); args = ap.parse_args()
    out = {}
    print("=" * 110 + "\nE23 ROW\n" + "=" * 110)
    for b, (arm, b1_t, b1_run_t, frozen_p) in ARMS.items():
        print(f"\n## {b}  ({arm} vs B1)")
        decs = {s: json.load(open(f"runs/e23/{arm}/decomp_seed{s}.json")) for s in SEEDS if Path(f"runs/e23/{arm}/decomp_seed{s}.json").exists()}
        runs = {s: json.load(open(f"runs/{arm}_seed{s}/mafc_results.json")) for s in SEEDS if Path(f"runs/{arm}_seed{s}/mafc_results.json").exists()}
        rpts = {s: json.load(open(f"runs/e23_rpt_{'vit' if b == 'vit' else 'rn'}_seed{s}/mafc_results.json")) for s in SEEDS if Path(f"runs/e23_rpt_{'vit' if b == 'vit' else 'rn'}_seed{s}/mafc_results.json").exists()}
        b1 = {s: json.load(open(b1_t.format(s=s))) for s in SEEDS if Path(b1_t.format(s=s)).exists()}
        b1_runs = {s: json.load(open(f"{b1_run_t.format(s=s)}/mafc_results.json"))["arm"] for s in SEEDS if Path(f"{b1_run_t.format(s=s)}/mafc_results.json").exists()}
        res = {"runs": len(runs), "decomps": len(decs), "rpt": len(rpts), "b1": len(b1)}
        print(f"  artifacts: runs {len(runs)}/3  decompositions {len(decs)}/3  repeat-task {len(rpts)}/3  B1 column {len(b1)}/3")
        # ---- preconditions ------------------------------------------------------------
        if Path(frozen_p).exists() and runs:
            fz = json.load(open(frozen_p)); diag_tr = float(np.mean([np.mean(np.diag(np.array(r["accuracy_matrix"]))) for r in runs.values()]))
            p1 = fz["min_diag"] >= P1_BAR and abs(fz["mean_diag"] - diag_tr) <= P1_MARGIN
            print(f"  P1  frozen mean DIAG {fz['mean_diag']:.4f} (min {fz['min_diag']:.4f}, bar {P1_BAR}); trainable mean DIAG {diag_tr:.4f}, gap {fz['mean_diag']-diag_tr:+.4f} vs {P1_MARGIN} -> {'PASS' if p1 else 'FAIL'}")
            res["P1"] = {"frozen_mean": fz["mean_diag"], "frozen_min": fz["min_diag"], "trainable_mean": diag_tr, "pass": bool(p1)}
        else:
            print("  P1  not readable yet (frozen reference or runs missing)"); res["P1"] = None
        if runs:
            drops = {s: float(np.array(r["accuracy_matrix"])[0, 0] - np.array(r["accuracy_matrix"])[-1, 0]) for s, r in runs.items()}
            p2a = float(np.mean(list(drops.values()))) >= P2A_BAR
            print(f"  P2a task-0 drop {', '.join(f's{s} {d:+.4f}' for s, d in drops.items())}  mean {np.mean(list(drops.values())):.4f} vs {P2A_BAR} -> {'PASS' if p2a else 'FAIL'}")
            res["P2a"] = {"drops": drops, "pass": bool(p2a)}
        if rpts:
            ctrl = {s: float(r["forgetting"]) for s, r in rpts.items()}; p2b = float(np.mean(list(ctrl.values()))) <= P2B_BAR
            print(f"  P2b repeat-task forgetting {', '.join(f's{s} {v:+.4f}' for s, v in ctrl.items())}  mean {np.mean(list(ctrl.values())):.4f} vs {P2B_BAR} -> {'PASS' if p2b else 'FAIL'}")
            res["P2b"] = {"ctrl": ctrl, "pass": bool(p2b)}
        else:
            print("  P2b NOT YET RUN -- P2 incomplete; no decomposition is read"); res["P2b"] = None
        # ---- C-WIT across the arms of the comparison ------------------------------------
        if runs and b1_runs:
            keys = ("backbone", "use_task_heads", "era_checkpoints", "use_input_adapters", "checkpoint_fp16")
            wit = all(runs[s]["arm"].get(k) == b1_runs[s].get(k) for s in runs if s in b1_runs for k in keys) and all("fp32_shadow" not in json.dumps(runs[s]) for s in runs)
            print(f"  C-WIT B6 vs B1 uniform on {keys}, no shadow on either: {'PASS' if wit else 'FAIL'}"); res["C_WIT"] = bool(wit)
        # ---- C-FLOOR ------------------------------------------------------------------
        fp = Path(f"runs/{arm.replace('e23_', 'e23_')}_floor_seed42/mafc_results.json")
        fl = {}
        if 42 in runs and fp.exists():
            A, B = np.array(runs[42]["accuracy_matrix"]), np.array(json.load(open(fp))["accuracy_matrix"])
            fl = {"cells_identical": int((A == B).sum()), "n_cells": int(A.size), "max_cell_delta": float(np.abs(A - B).max()), "avg_delta": float(A[-1].mean() - B[-1].mean())}
            print(f"  C-FLOOR seed 42 relaunch: {fl['cells_identical']}/{fl['n_cells']} cells identical, max |delta| {fl['max_cell_delta']:.4f}, AVG delta {fl['avg_delta']:+.4f}")
        fdec = Path(f"runs/e23/{arm}/decomp_floor42.json")
        if 42 in decs and fdec.exists():
            F = json.load(open(fdec))["pooled"]; M = decs[42]["pooled"]
            fl.update({"F_enc_floor_raw": abs(M["F_enc"] - F["F_enc"]), "F_enc_floor_relaid": abs(M["F_enc_relaid"] - F["F_enc_relaid"])})
            print(f"  C-FLOOR on F_enc (seed 42 main vs relaunch, pooled): raw {fl['F_enc_floor_raw']:.4f}  re-laid {fl['F_enc_floor_relaid']:.4f}")
        res["C_FLOOR"] = fl
        # ---- C-RELOAD, C-ID ----------------------------------------------------------------
        if decs:
            rel = max(d["fp16_reload_floor"] or 0.0 for d in decs.values()); wp = all(d["wrong_perm_must_fail"]["pass"] for d in decs.values())
            print(f"  C-RELOAD fp16 reload floor (max over seeds/cells) {rel:.4f} | C-ID relay exact on every seed: {all(d['relay_checks']['algebraic_vs_dataset_max_abs'] == 0.0 for d in decs.values())};"
                  f" wrong-permutation must-fail {'PASS 19/19 x ' + str(len(decs)) if wp else 'FAIL ' + str({s: d['wrong_perm_must_fail']['fail_tasks'] for s, d in decs.items()})}")
            res.update({"C_RELOAD": rel, "C_ID_wrong_perm": bool(wp)})
        # ---- the crux ------------------------------------------------------------------
        if len(decs) == 3 and len(b1) == 3:
            per_seed = {}
            for s in SEEDS:
                rows6 = decs[s]["rows"]; rows1 = b1[s]
                per_seed[s] = {"raw": float(np.mean([r["F_enc"] for r in rows6])), "relaid": float(np.mean([r["F_enc_relaid"] for r in rows6])), "b1": float(np.mean([r["F_enc"] for r in rows1])),
                               "p": float(np.mean([r["acc_refit_relaid"] for r in rows6])), "n_test": rows6[0]["n_test"], "n_cells": len(rows6)}
            raw, rel, b1v = (np.mean([v[k] for v in per_seed.values()]) for k in ("raw", "relaid", "b1"))
            d1 = [v["relaid"] - v["b1"] for v in per_seed.values()]; d2 = [v["raw"] - v["relaid"] for v in per_seed.values()]
            p = np.mean([v["p"] for v in per_seed.values()]); n_te = per_seed[42]["n_test"]; n_c = per_seed[42]["n_cells"]
            binom = 1.96 * np.sqrt(p * (1 - p) / n_te) * np.sqrt(2) / np.sqrt(n_c)
            floor1 = max(t95_half(d1), binom, fl.get("F_enc_floor_relaid", 0.0)); floor2 = max(t95_half(d2), binom, fl.get("F_enc_floor_raw", 0.0))
            e1 = abs(np.mean(d1)) <= floor1; e2 = np.mean(d2) > floor2
            print(f"  F_enc  B6 raw {raw:+.4f}   B6 re-laid {rel:+.4f}   B1 {b1v:+.4f}   (per seed re-laid - B1: {', '.join(f'{x:+.4f}' for x in d1)}; raw - re-laid: {', '.join(f'{x:+.4f}' for x in d2)})")
            print(f"  floors: t95 half-width E1 {t95_half(d1):.4f} E2 {t95_half(d2):.4f}; binomial {binom:.4f}; F_enc relaunch floor {fl.get('F_enc_floor_relaid', float('nan')):.4f}/{fl.get('F_enc_floor_raw', float('nan')):.4f} -> floor E1 {floor1:.4f}, E2 {floor2:.4f}")
            print(f"  (E1) |re-laid - B1| = {abs(np.mean(d1)):.4f} <= {floor1:.4f} -> {'HOLDS' if e1 else 'FAILS ' + ('ABOVE' if np.mean(d1) > 0 else 'BELOW')}")
            print(f"  (E2) raw - re-laid = {np.mean(d2):+.4f} > {floor2:.4f} -> {'HOLDS' if e2 else 'FAILS'}")
            res.update({"F_enc": {"raw": raw, "relaid": rel, "b1": b1v, "per_seed": per_seed}, "E1": {"delta": float(np.mean(d1)), "floor": floor1, "holds": bool(e1)}, "E2": {"delta": float(np.mean(d2)), "floor": floor2, "holds": bool(e2)},
                        "sign_relaid_minus_b1_positive": bool(np.mean(d1) > 0)})
            shape = ("both hold: the map is the discriminator; pretraining is not the explanation" if e1 and e2 else
                     "E2 holds, E1 fails ABOVE: re-layout recovers the presentation term but training under layout shift damaged the pretrained features -- a third term" if e2 and np.mean(d1) > 0 else
                     "E2 holds, E1 fails BELOW: the permuted regime preserved features better; unexpected, reported" if e2 else
                     "E2 fails, both small: the pretrained encoder absorbs the permutation; nothing to discriminate on this construction" if max(raw, rel) < 0.05 else
                     "E2 fails, both large: re-layout does not recover on a pretrained backbone; the discriminator is scratch-scoped")
            print(f"  reading: {shape}"); res["reading"] = shape
        out[b] = res
    # ---- arm B: MLP on the S72 construction vs the LSTM (same instrument, same draw) ------------------
    armb = {}
    pb, ps, pf = Path("runs/e20/seeded/har_e23_mlp_linear.json"), Path("runs/e20/seeded/har_s72_off_linear.json"), Path("runs/e23/frozen_mlp.json")
    if pb.exists() and ps.exists():
        print("\n## arm B  (e23_har_mlp vs s72_off; runs/e20/seeded, probe draw 20260916)")
        B, S = json.load(open(pb))["rows"], json.load(open(ps))["rows"]
        m = lambda rows, k: float(np.mean([r[k] for r in rows]))
        shareB, shareS = m(B, "F_read") / (m(B, "F_enc") + m(B, "F_read")), m(S, "F_read") / (m(S, "F_enc") + m(S, "F_read"))
        runsB = {s: json.load(open(f"runs/e23_har_mlp_seed{s}/mafc_results.json")) for s in SEEDS if Path(f"runs/e23_har_mlp_seed{s}/mafc_results.json").exists()}
        drops = {s: float(np.array(r["accuracy_matrix"])[0, 0] - np.array(r["accuracy_matrix"])[-1, 0]) for s, r in runsB.items()}
        p2a = bool(np.mean(list(drops.values())) >= P2A_BAR) if drops else None
        p1 = None
        if pf.exists() and runsB:
            fz = json.load(open(pf)); diag_tr = float(np.mean([np.mean(np.diag(np.array(r["accuracy_matrix"]))) for r in runsB.values()]))
            p1 = bool(fz["min_diag"] >= P1_BAR and abs(fz["mean_diag"] - diag_tr) <= P1_MARGIN)
            print(f"  P1  reference (linear on flattened windows) mean DIAG {fz['mean_diag']:.4f} (min {fz['min_diag']:.4f}, bar {P1_BAR}); trainable MLP mean DIAG {diag_tr:.4f}, gap {fz['mean_diag']-diag_tr:+.4f} vs {P1_MARGIN} -> {'PASS' if p1 else 'FAIL'}"
                  f"   [S72 LSTM DIAG 0.8371: gap {0.8371-diag_tr:+.4f}]")
        if drops:
            print(f"  P2a task-0 drop {', '.join(f's{s} {d:+.4f}' for s, d in drops.items())}  mean {np.mean(list(drops.values())):.4f} vs {P2A_BAR} -> {'PASS' if p2a else 'FAIL'}")
        print(f"  decomposition: F_enc {m(B,'F_enc'):+.4f}  F_read {m(B,'F_read'):.4f}  R {m(B,'R'):+.4f}  F_total {m(B,'F_total'):.4f}  share {100*shareB:.1f}   | LSTM: F_enc {m(S,'F_enc'):+.4f}  F_read {m(S,'F_read'):.4f}  share {100*shareS:.1f}")
        armb = {"P1": p1, "P2a": p2a, "F_enc": m(B, "F_enc"), "F_read": m(B, "F_read"), "share": shareB, "lstm_share": shareS, "lstm_F_enc": m(S, "F_enc")}
    out["arm_b"] = armb

    # ---- predictions (sec 7) -------------------------------------------------------
    print("\nPREDICTIONS (sec 7)")
    preds = []
    def add(stmt, odds, hit):
        v = "pending" if hit is None else ("fired" if hit else ("MISS" if odds >= 50 else "did not fire")); preds.append({"prediction": stmt, "odds": odds, "outcome": v}); print(f"  {stmt:<74} ~{odds:>2}%  {v}")
    P1 = [out[b].get("P1") for b in ("vit", "resnet")]
    add("P1 passes on both A arms", 90, None if any(p is None for p in P1) else all(p["pass"] for p in P1))
    add("(E2) holds on A-ViT", 75, out["vit"].get("E2", {}).get("holds")); add("(E2) holds on A-RN", 80, out["resnet"].get("E2", {}).get("holds"))
    add("(E1) holds on A-ViT", 45, out["vit"].get("E1", {}).get("holds")); add("(E1) holds on A-RN", 50, out["resnet"].get("E1", {}).get("holds"))
    sg = [out[b].get("sign_relaid_minus_b1_positive") for b in ("vit", "resnet")]
    add("F_enc(B6 re-laid) - F_enc(B1) positive on both backbones", 60, None if any(v is None for v in sg) else all(sg))
    ab = out.get("arm_b") or {}
    add("Arm B passes all preconditions (P1 as contracted, P2a)", 80, None if not ab or ab.get("P1") is None or ab.get("P2a") is None else (ab["P1"] and ab["P2a"]))
    add("Arm B share within 10 points of S72's 0.778", 45, None if not ab else abs(ab["share"] - ab["lstm_share"]) <= 0.10)
    add("Arm B F_enc below S72's 0.087", 65, None if not ab else ab["F_enc"] < ab["lstm_F_enc"])
    ci = [out[b].get("C_ID_wrong_perm") for b in ("vit", "resnet")]
    add("C-ID wrong-permutation must-fail 19/19 on every seed and backbone", 90, None if any(v is None for v in ci) else all(ci))
    out["predictions"] = preds
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float); print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
