"""E23-B row -- the three-point comparison and the six registered predictions
(docs/E23B_seqlen_prereg.md §5). Reads runs/e23b/{arm}/decomp_seed{s}.json.

A1 e18_pmd_lstm   T=5,  12k images/task   (exists)
A2 e23b_t5c20     T=5,   3k               A2 vs A1 isolates images-per-task
A3 e23b_t20       T=20,  3k               A3 vs A2 isolates sequence length

Floor: max(t95 half-width over the three seeds of the pooled quantity,
mean per-cell 1.96*SE_binomial, and the seed-42 replicate's own delta when the
floor arm has been decomposed).
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ["e18_pmd_lstm", "e23b_t5c20", "e23b_t20"]
LABEL = {"e18_pmd_lstm": "A1  T=5, 12k/task", "e23b_t5c20": "A2  T=5,  3k/task", "e23b_t20": "A3  T=20, 3k/task"}


def t95(x):
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def load(arm):
    fs = sorted(glob.glob(f"runs/e23b/{arm}/decomp_seed*.json"))
    return [json.load(open(f)) for f in fs]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/e23b_row.json"); args = ap.parse_args()
    print("=" * 108 + "\nE23-B ROW   is E23's crux failure about pretraining or about sequence length?\n" + "=" * 108)
    D, out = {}, {}
    for a in ARMS:
        ds = load(a)
        if not ds:
            print(f"  {LABEL[a]}: no artifacts"); continue
        D[a] = ds
        rows = [r for d in ds for r in d["rows"]]
        per_seed = {d["seed"]: d["pooled"] for d in ds}
        m = lambda k: float(np.mean([p[k] for p in per_seed.values()]))
        res = float(np.mean([r["res95"] for r in rows]))
        cid = [d.get("C_ID") for d in ds if d.get("C_ID")]
        wrong = all(d["wrong_perm_must_fail"]["pass"] for d in ds)
        ident = max(d["identity_max"] for d in ds)
        relay = all(d["relay_checks"]["identity"] == 0.0 and d["relay_checks"]["round_trip"] == 0.0 for d in ds)
        print(f"\n  {LABEL[a]}   seeds {sorted(per_seed)}  cells {len(rows)}  construction {ds[0]['construction_fingerprint']}")
        print(f"    controls: relay exact {relay} | identity {ident:.1e} | wrong-perm must-fail {'PASS' if wrong else 'FAIL ' + str({d['seed']: d['wrong_perm_must_fail']['fail_tasks'] for d in ds})}"
              + (f" | C-ID {'PASS' if all(c['pass'] for c in cid) else 'FAIL'}" if cid else ""))
        print(f"    ceiling {m('acc_ceiling'):.4f} | raw: deployed {m('acc_orig'):.4f} F_enc {m('F_enc'):+.4f} F_read {m('F_read'):.4f} F_total {m('F_total'):.4f}")
        print(f"    re-laid: deployed {m('acc_relaid'):.4f} F_enc {m('F_enc_relaid'):+.4f} [t95 ±{t95([p['F_enc_relaid'] for p in per_seed.values()]):.4f}] F_read {m('F_read_relaid'):.4f} | wrong-perm deployed {m('acc_wrong_perm'):.4f}")
        floor = max(t95([p["F_enc_relaid"] for p in per_seed.values()]), res)
        removes = abs(m("F_enc_relaid")) <= floor
        helps = m("acc_relaid") > m("acc_orig")
        print(f"    => re-laying removes the encoder term: |{m('F_enc_relaid'):+.4f}| <= floor {floor:.4f} -> {'YES' if removes else 'NO'}"
              f" | re-laid deployed above raw: {'YES' if helps else 'NO'} ({m('acc_relaid')-m('acc_orig'):+.4f})")
        out[a] = {"n_cells": len(rows), "seeds": sorted(per_seed), "pooled": {k: m(k) for k in ds[0]["pooled"]}, "res95": res, "floor": floor,
                  "removes_encoder_term": bool(removes), "relaid_helps": bool(helps), "wrong_perm_pass": bool(wrong), "identity_max": ident,
                  "construction_fingerprint": ds[0]["construction_fingerprint"], "per_seed_F_enc_relaid": {s: p["F_enc_relaid"] for s, p in per_seed.items()}}
    # ---- predictions ---------------------------------------------------------------
    print("\nPREDICTIONS (§5)")
    A1, A2, A3 = (out.get(a) for a in ARMS)
    preds = []
    def add(stmt, odds, hit):
        v = "pending" if hit is None else ("fired" if hit else ("MISS" if odds >= 50 else "did not fire"))
        preds.append({"prediction": stmt, "odds": odds, "outcome": v}); print(f"  {stmt:<86} ~{odds:>2}%  {v}")
    add("A3 (T=20): re-laying still removes the encoder term — sequence length does not explain E23", 55, None if not A3 else A3["removes_encoder_term"])
    add("A2 (T=5, 3k): re-laying removes it — images-per-task does not explain E23", 85, None if not A2 else A2["removes_encoder_term"])
    add("A3: re-laid deployed accuracy above raw (the scratch direction)", 70, None if not A3 else A3["relaid_helps"])
    add("total forgetting higher at T=20 than at T=5 with the same images/task", 75, None if not (A2 and A3) else A3["pooled"]["F_total"] > A2["pooled"]["F_total"])
    add("A2's F_enc within 2 pp of A1's", 50, None if not (A1 and A2) else abs(A2["pooled"]["F_enc"] - A1["pooled"]["F_enc"]) <= 0.02)
    add("the wrong-permutation must-fail passes on both new arms", 80, None if not (A2 and A3) else (A2["wrong_perm_pass"] and A3["wrong_perm_pass"]))
    scored = [p for p in preds if p["outcome"] != "pending"]
    print(f"\n  scored {len(scored)}: fired {sum(p['outcome']=='fired' for p in scored)}, miss {sum(p['outcome']=='MISS' for p in scored)}, did not fire {sum(p['outcome']=='did not fire' for p in scored)}")
    # ---- the pre-committed reading ---------------------------------------------------
    if A3 and A2:
        if A3["removes_encoder_term"] and A3["relaid_helps"]:
            reading = ("SEQUENCE LENGTH DOES NOT EXPLAIN E23. Re-laying removes the encoder term at T=20 on a scratch LSTM "
                       "while it raises it on both pretrained backbones, so §4.2's scope sentence is TRAINING REGIME "
                       "(encoders overwritten to the current frame) and E23's reading stands.")
        elif not A3["removes_encoder_term"] and not A3["relaid_helps"]:
            reading = ("SEQUENCE LENGTH EXPLAINS IT. At T=20 the scratch LSTM behaves like the pretrained arms, so E23's "
                       "pretraining reading is WITHDRAWN as unidentified and §4.2's scope sentence is sequence length: "
                       "a twenty-task sequence over twenty layouts leaves no single current frame.")
        else:
            reading = ("INTERPOLATION: re-laying at T=20 " + ("removes the term but does not help the deployed head" if A3["removes_encoder_term"] else "helps the deployed head but leaves the term elevated")
                       + " — report the interpolation and claim neither scope sentence.")
        print(f"\nREADING (pre-committed, §5): {reading}")
        out["reading"] = reading
    out["predictions"] = preds
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
