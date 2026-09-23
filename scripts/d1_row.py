"""D1 sec 6 -- the registered predictions, scored from runs/d1/controls.json
(written by d1_diagnose.py). Convention as the appendix table: a prediction at
or above even odds that did not occur is a MISS; below even odds, "did not
fire". The two pretrained-arm predictions read "pending" until jobs_d1_pretrained
has run. Usage: python scripts/d1_row.py --out runs/d1_row.json
"""

import argparse
import json
import os

MLP = ["e18_pmd_mlp", "e18_rmd_mlp"]
LSTM = ["s72_off", "e18_pmd_lstm"]
ALL = LSTM + MLP


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--controls", default="runs/d1/controls.json"); ap.add_argument("--out", default="runs/d1_row.json")
    args = ap.parse_args()
    c = json.load(open(args.controls)); A = c["arms"]
    missing = [a for a in ALL if a not in A or "D1" not in A[a]]
    if missing:
        print(f"  incomplete -- arms without diagnostics: {missing}; nothing scored"); return
    d2p = c.get("D2_pooled", {})
    preds = [
        ("D1: transfer loss <= 5pp on adjacent tasks, all four arms", 60, all(A[a]["D1"]["adjacent"] <= 0.05 for a in ALL)),
        ("D1: transfer loss <= 5pp on all pairs, all four arms", 35, all(A[a]["D1"]["adjacent"] <= 0.05 and A[a]["D1"]["distant"] <= 0.05 for a in ALL)),
        ("D1: deploy(k) within resolution of own(k) on >= 3 of 4 arms", 40, sum(abs(A[a]["D1"]["deploy_minus_own"]) <= A[a]["D2"]["res95"] for a in ALL) >= 3),
        ("D2: spread < 0.5 on the MLP arms (T, re-laid)", 65, all(A[a]["D2"]["spread_T"] < 0.5 for a in MLP)),
        ("D2: spread < 0.5 on the LSTM arms (T, re-laid)", 45, all(A[a]["D2"]["spread_T"] < 0.5 for a in LSTM)),
        ("D2: F_enc correlates with log sigma_d across the 48 cells (r < -0.5)", 55, d2p.get("r_sigma_d", 0.0) < -0.5),
        ("D2: the residual explains F_enc better than sigma_d (|partial r| larger)", 45, abs(d2p.get("partial_r_resid", 0.0)) > abs(d2p.get("partial_r_sigma_d", 0.0))),
        ("D2: re-laid F_enc within resolution of 0 on >= 3 of 4 scratch arms", 55, sum(abs(A[a]["D2"]["F_enc_relaid"]) <= A[a]["D2"]["res95"] for a in ALL) >= 3),
        ("D3: ||D_in|| coefficient positive, interval excluding 0, pretrained arms", 60, None),
        ("D3: ||D_out|| coefficient's interval includes 0, pretrained arms", 55, None),
        ("D3: fit residual under 10% of ||Z_k|| (S, re-laid, test, all arms)", 70, all(A[a]["D2"]["residual_S_test"] < 0.10 for a in ALL)),
        ("C-SHUF passes 12/12 on every arm", 90, all(A[a]["C_SHUF"][0] == A[a]["C_SHUF"][2] and A[a]["C_SHUF"][1] == A[a]["C_SHUF"][2] for a in ALL)),
    ]
    print("D1 PREDICTIONS (sec 6)")
    rows = []
    for stmt, odds, hit in preds:
        if hit is None:
            verdict = "pending (jobs_d1_pretrained)"
        elif hit:
            verdict = "fired"
        else:
            verdict = "MISS" if odds >= 50 else "did not fire"
        rows.append({"prediction": stmt, "odds": odds, "outcome": verdict})
        print(f"  {stmt:<78} ~{odds:>2}%  {verdict}")
    scored = [r for r in rows if not r["outcome"].startswith("pending")]
    print(f"\n  scored {len(scored)}: fired {sum(r['outcome']=='fired' for r in scored)}, miss {sum(r['outcome']=='MISS' for r in scored)}, did not fire {sum(r['outcome']=='did not fire' for r in scored)}; pending {len(rows)-len(scored)}")
    # secondary print: D3 on the scratch arms (the lemma's head there is the STORED era head, not the deployed one)
    for a in ALL:
        d3 = A[a].get("D3")
        if d3:
            print(f"  [scratch D3, secondary] {a:<13} beta_in {d3['beta_in']:+.4f} {d3['ci_in']}  beta_out {d3['beta_out']:+.4f} {d3['ci_out']}  R2 {d3['r2']:.2f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"predictions": rows}, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
