"""E20-B -- the 66-88% band under uniform arms on har_subject.

Contract: docs/E20_prereg.md sec 8. Reads the (seeded, R2) linear
decompositions of the four arms and their floor pairs and prints, every verdict
from the printed values:

  * arm identity from artifacts (benchmark har_subject, era + shadow, adapters,
    the v1/v2 witnesses), uniform ACROSS arms;
  * per-arm reader share = pooled F_read / (F_enc + F_read), with per-seed
    shares beside it -- the band is reported as three-seed means AND the
    per-seed range, never the means alone;
  * the SHARE FLOOR per arm as the spread over ALL THREE seed-42 launches
    (main + floor a + floor b), not over the pair alone: a pair that agrees
    with itself is one observation of determinism (v1-ON's pair agreed with
    itself and disagreed with the main run in the same batch);
  * the E11-era band on the shared-window `har` construction beside it, cited
    as superseded by provenance and construction;
  * the sec-8 predictions scored.

Usage:
    python scripts/e20_band.py --root runs/e20/seeded --out runs/e20_band.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = [("OFF", "s72_off", "e10off_ec"), ("v1-ON", "s72_v1on", "e10v1on_ec"),
        ("v2-ON", "s72_v2on", "e10v2on_ec"), ("v2-control", "s72_v2ctl", "e10v2ctl_ec")]
E11_ERA_HAR = {"OFF": 66.2, "v1-ON": 75.6, "v2-ON": 77.2, "v2-control": 87.9}   # runs/e11_e6b/decomp.json, `har`
SEEDS = [42, 1337, 2024]
WITNESS = {"OFF": ("none", None), "v1-ON": ("none", None), "v2-ON": ("warm", "moves"), "v2-control": ("warm", "frozen")}


def share(rows):
    fe = float(np.mean([r["F_enc"] for r in rows])); fr = float(np.mean([r["F_read"] for r in rows]))
    return 100.0 * fr / (fe + fr), fe, fr


def witness(run):
    r = json.load(open(run)); a = r["arm"]
    warm = [t["epochs"][0]["warmup"] for t in r["task_history"][1:]]
    # the OFF arm has no adapter and no such key; its witness is "no warmup, no adapter"
    dist0 = [t["epochs"][0].get("adapter_dist_from_identity", 0.0) for t in r["task_history"][1:]]
    w = ("none" if not any(warm) else "warm", None if not any(warm) else ("frozen" if all(d == 0.0 for d in dist0) else "moves"))
    shadow = all(t["era_checkpoint"].get("fp32_shadow", {}).get("delta") == 0.0 for t in r["task_history"])
    return a, w, shadow, np.array(r["accuracy_matrix"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/e20/seeded")
    ap.add_argument("--out", default="runs/e20_band.json")
    args = ap.parse_args()
    out = {"arms": {}, "problems": []}

    print("=" * 100 + "\nE20-B  THE BAND ON har_subject, UNIFORM ARMS  (linear probe, seeded draw)\n" + "=" * 100)
    print(f"  {'arm':<12}{'share':>7}{'F_enc':>8}{'F_read':>8}{'R':>8}   {'per-seed shares':<26}{'3-launch s42 shares':<24}{'share floor':>12}{'E11-era har':>12}")
    shares = {}
    for label, arm, run_tag in ARMS:
        main_ = json.load(open(f"{args.root}/har_{arm}_linear.json"))
        fl = json.load(open(f"{args.root}/har_{arm}_floor_linear.json"))
        assert main_.get("probe_subset_seed") is not None and fl.get("probe_subset_seed") == main_.get("probe_subset_seed"), \
            f"{arm}: seeded artifacts required (R2)"
        # identity, uniformity, witnesses -- from the runs' own artifacts
        for s_ in SEEDS:
            a, w, shadow, _ = witness(f"runs/{run_tag}_seed{s_}/mafc_results.json")
            if a["benchmark"] != "har_subject" or not a["era_checkpoints"] or not shadow or a["torch_num_threads"] != 4 \
               or a["use_input_adapters"] != (label != "OFF") or w != WITNESS[label]:
                out["problems"].append(f"{run_tag}_seed{s_}: identity {a['benchmark']} era {a['era_checkpoints']} shadow {shadow} adapters {a['use_input_adapters']} witness {w}")
        sh, fe, fr = share(main_["rows"])
        per_seed = {s_: share([r for r in main_["rows"] if r["seed"] == s_])[0] for s_ in SEEDS}
        s42_main = per_seed[42]
        s42_a = share([r for r in fl["rows"] if r["seed"] == "a"])[0]
        s42_b = share([r for r in fl["rows"] if r["seed"] == "b"])[0]
        three = [s42_main, s42_a, s42_b]
        floor = max(three) - min(three)
        # matrix identity across the three launches
        _, _, _, m_main = witness(f"runs/{run_tag}_seed42/mafc_results.json")
        _, _, _, m_a = witness(f"runs/{run_tag}_floor4tec_a/mafc_results.json")
        _, _, _, m_b = witness(f"runs/{run_tag}_floor4tec_b/mafc_results.json")
        ident = (int((m_main == m_a).sum()), int((m_a == m_b).sum()))
        shares[label] = sh
        print(f"  {label:<12}{sh:7.2f}{fe:8.4f}{fr:8.4f}{main_['mean']['R']:+8.4f}   "
              f"{' / '.join(f'{per_seed[s_]:.1f}' for s_ in SEEDS):<26}{' / '.join(f'{v:.1f}' for v in three):<24}{floor:12.2f}{E11_ERA_HAR[label]:12.1f}"
              f"   matrix identical main/a {ident[0]}, a/b {ident[1]} /25")
        out["arms"][label] = {"share": sh, "F_enc": fe, "F_read": fr, "R": main_["mean"]["R"], "per_seed": per_seed,
                              "seed42_three_launches": three, "share_floor": floor, "matrix_identity": ident,
                              "identity_residual": main_["identity_residual"], "probe_subset_seed": main_["probe_subset_seed"],
                              "e11_era_har": E11_ERA_HAR[label]}
    for pr in out["problems"]:
        print("  PROBLEM:", pr)
    lo, hi = min(shares.values()), max(shares.values())
    print(f"\n  BAND (three-seed means): {lo:.1f}-{hi:.1f}%   per-seed range: "
          f"{min(min(v['per_seed'].values()) for v in out['arms'].values()):.1f}-{max(max(v['per_seed'].values()) for v in out['arms'].values()):.1f}%"
          f"   E11-era band on `har`: 66.2-87.9% (superseded by provenance and construction)")
    # ---- predictions, contract sec 8 ----
    order = [shares[l] for l, _, _ in ARMS]
    preds = {
        "all three ON arms inside 66-88": all(66 <= shares[l] <= 88 for l in ("v1-ON", "v2-ON", "v2-control")),
        "band narrower than 22pp": (hi - lo) < 22,
        "E11 ordering OFF<v1<v2<v2ctl preserved": order == sorted(order),
        "v2-control >= 85": shares["v2-control"] >= 85,
        "every ON floor pair bit-identical (pair test)": all(out["arms"][l]["matrix_identity"][1] == 25 for l in ("v1-ON", "v2-ON", "v2-control")),
        "every ON arm identical across all THREE s42 launches": all(out["arms"][l]["matrix_identity"] == (25, 25) for l in ("v1-ON", "v2-ON", "v2-control")),
    }
    print("\n  predictions (sec 8):")
    for k, v in preds.items():
        print(f"    {k:<58} {'fired' if v else 'MISS'}")
    out["band"] = {"lo": lo, "hi": hi}; out["predictions"] = preds; out["all_identity_ok"] = not out["problems"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
