"""
E11 — checkpoint audit table (catch 20, strengthened).

Catch 20 requires an arm x seed x EXISTS table before signing off any analysis
that claims to run on existing checkpoints. This program has now met the failure
mode one level down: a checkpoint that EXISTS, has a plausible size, and does not
load -- `modal volume get` can leave truncated files, and 35 of 847 local .pt
files are under 1KB.

So the audit criterion is LOADS, not exists. Every cell is opened through the
same PLCM.load_era() the analysis uses, so anything the audit passes is something
the chain can actually read.

Usage:
    python scripts/audit_checkpoints.py
    python scripts/audit_checkpoints.py --json runs/e11_ckpt_audit.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM

# Every arm the E11 recompute chain touches, by the experiment that needs it.
CHAIN = {
    "E6b/HAR": ["runs/ckpt_e7_heads_seed{s}/mafc_seed{s}",
                "runs/ckpt_e7_heads_adapt_seed{s}/mafc_seed{s}",
                "runs/ckpt_e5_seed{s}/mafc_seed{s}",
                "runs/ckpt_e5_off_seed{s}/mafc_seed{s}",
                "runs/ckpt_e5d_v2on_seed{s}/mafc_seed{s}",
                "runs/ckpt_e5d_v2ctl_seed{s}/mafc_seed{s}"],
    "E6b/MNIST": ["checkpoints/e4_off_seed{s}/mafc_seed{s}"],
    "E10":      ["runs/ckpt_e10_off_seed{s}/mafc_seed{s}",
                 "runs/ckpt_e10_on_seed{s}/mafc_seed{s}",
                 "runs/ckpt_e10_noshift_seed{s}/mafc_seed{s}"],
}
# MNIST ON is seed-keyed by directory, not template.
MNIST_ON = {42: "checkpoints/fullrank_ref/mafc_seed42",
            1337: "checkpoints/e4_on_seed1337/mafc_seed1337",
            2024: "checkpoints/e4_on_seed2024/mafc_seed2024"}
SEEDS = [42, 1337, 2024]
TASKS = [0, 1, 2, 3, 4]
EPOCH = 9


# E25 (2026-09-21, contract L0, gates L4-L8). A DIFFERENT chain, because the
# arms differ in both axes this file hardcoded for E11: task count (5 vs 20) and
# epoch (9 vs 4). Added BESIDE the E11 chain, never in place of it -- E11's table
# is cited and its default path is unchanged, so `--chain e11` remains what it was.
# The shadow suffix is part of the identity: the scratch analyses load the fp32
# shadow directories, the pretrained arms have none (E23 C-RELOAD), and auditing
# a directory the analysis does not open would prove nothing about the analysis.
E25_CHAIN = [
    # (steps that need it, label, dir template, n_tasks, epoch, seeds)
    ("A,B",  "lstm_har",      "{root}/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32",    5, 9, None),
    ("A,B",  "mlp_permuted",  "{root}/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
    ("B",    "lstm_permuted", "{root}/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", 5, 9, None),
    ("B",    "mlp_rotated",   "{root}/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
    ("B",    "e23b_t20 (A3)", "{root}/ckpt_e23b_t20_seed{s}/mafc_seed{s}_fp32",    20, 9, None),
    ("A",    "vit_b16",       "{root}/ckpt_e12_base_seed{s}/mafc_seed{s}",         20, 4, None),
    ("A",    "resnet50",      "{root}/ckpt_e14_base_seed{s}/mafc_seed{s}",         20, 4, None),
    ("ctrl6", "e17_mlp regr", "{root}/ckpt_e17_mlp_seed{s}/mafc_seed{s}",           5, 9, [42]),
]

# E26 (2026-09-21, contract L0). FS and DC load the B6 PERMUTED pretrained arms
# (`ckpt_e23_*`), which are different checkpoints from the B1 arms E25 audited
# (`ckpt_e12_base`, `ckpt_e14_base`). Same shape as the E25 chain otherwise.
E26_CHAIN = [
    ("FS,DC", "vit_b16 B6",    "{root}/ckpt_e23_vit_seed{s}/mafc_seed{s}",          20, 4, None),
    ("FS,DC", "resnet50 B6",   "{root}/ckpt_e23_rn_seed{s}/mafc_seed{s}",           20, 4, None),
    ("FS,DC", "e23b_t20 (A3)", "{root}/ckpt_e23b_t20_seed{s}/mafc_seed{s}_fp32",    20, 9, None),
    ("DC",    "lstm_har",      "{root}/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32",    5, 9, None),
    ("DC",    "mlp_permuted",  "{root}/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
    ("DC",    "lstm_permuted", "{root}/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", 5, 9, None),
    ("DC",    "mlp_rotated",   "{root}/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
]
# E27 (2026-09-22). theta_T only, but the criterion is LOADS, not exists. Five of
# these six were audited green under --chain e26; ARM B (`ckpt_e23_har_mlp`) has
# never been in an audit chain, which is the whole reason this chain exists.
E27_CHAIN = [
    ("E27", "lstm_har",      "{root}/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32",    5, 9, None),
    ("E27", "mlp_permuted",  "{root}/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
    ("E27", "lstm_permuted", "{root}/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", 5, 9, None),
    ("E27", "mlp_rotated",   "{root}/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
    ("E27", "e23b_t20 (A3)", "{root}/ckpt_e23b_t20_seed{s}/mafc_seed{s}_fp32",    20, 9, None),
    ("E27", "arm B mlp_har", "{root}/ckpt_e23_har_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
]
# E30 (2026-09-22). The contract opens "analysis only, on existing checkpoints",
# which is precisely the claim the standing rule requires a checkpoint AUDIT
# TABLE for -- arm x seed x LOADS -- verified before sign-off, because E6 made
# exactly that claim and its HAR OFF arm had no checkpoints at all. E30 is the
# widest chain yet: ten arms, and unlike E27 it needs EVERY era checkpoint, not
# just theta_T, because step 1 compares class geometry under theta_k against
# theta_T for every old task k.
E30_CHAIN = [
    ("B1",      "vit_b16 B1",    "{root}/ckpt_e12_base_seed{s}/mafc_seed{s}",         20, 4, None),
    ("B1",      "resnet50 B1",   "{root}/ckpt_e14_base_seed{s}/mafc_seed{s}",         20, 4, None),
    ("B6",      "vit_b16 B6",    "{root}/ckpt_e23_vit_seed{s}/mafc_seed{s}",          20, 4, None),
    ("B6",      "resnet50 B6",   "{root}/ckpt_e23_rn_seed{s}/mafc_seed{s}",           20, 4, None),
    ("scratch", "lstm_har S72",  "{root}/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32",    5, 9, None),
    ("scratch", "arm B mlp_har", "{root}/ckpt_e23_har_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
    ("scratch", "mlp_permuted",  "{root}/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
    ("scratch", "lstm_permuted", "{root}/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", 5, 9, None),
    ("scratch", "mlp_rotated",   "{root}/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32",  5, 9, None),
    ("scratch", "e23b_t20 (A3)", "{root}/ckpt_e23b_t20_seed{s}/mafc_seed{s}_fp32",    20, 9, None),
]
CHAINS = {"e25": E25_CHAIN, "e26": E26_CHAIN, "e27": E27_CHAIN, "e30": E30_CHAIN}


def audit_dir(d: str, tasks=None, epoch=None):
    """Try every load the chain performs. Returns (per-task status, files bad).

    `tasks`/`epoch` default to the E11 globals, so the E11 call is unchanged.
    """
    tasks = TASKS if tasks is None else tasks
    epoch = EPOCH if epoch is None else epoch
    status, bad = {}, []
    for k in tasks:
        p = Path(d) / f"task{k}_epoch{epoch}.pt"
        if not p.exists():
            status[k] = "MISSING"
            bad.append(str(p))
            continue
        if p.stat().st_size < 1024:
            status[k] = f"TRUNCATED({p.stat().st_size}B)"
            bad.append(str(p))
            continue
        try:
            PLCM.load_era(d, k, epoch=epoch)
            status[k] = "ok"
        except Exception as e:
            status[k] = type(e).__name__
            bad.append(str(p))
    return status, bad


def main_chain(chain_name: str, root: str, out: str | None) -> int:
    """E25/E26 L0. Criterion is LOADS, through the same PLCM.load_era the analyses use."""
    chain = CHAINS[chain_name]
    print("=" * 104)
    print(f"{chain_name.upper()} L0 — CHECKPOINT AUDIT, arm x seed x LOADS (catch 20)")
    print("=" * 104)
    print("  A listing is not a load. E11's audit found 14 of 33 directories unusable")
    print("  when opened, including one 4.9MB file that would not read.\n")
    print(f"  {'step':<6}{'arm':<16}{'seed':>6}{'T':>4}{'ep':>4}  {'loaded':>9}  dir")

    rows, all_bad = [], []
    for step, arm, tmpl, n_tasks, epoch, seeds in chain:
        for s in (seeds or SEEDS):
            d = tmpl.format(root=root, s=s)
            tasks = list(range(n_tasks))
            st, bad = audit_dir(d, tasks=tasks, epoch=epoch)
            all_bad += bad
            n_ok = sum(v == "ok" for v in st.values())
            ok = n_ok == n_tasks
            rows.append(dict(step=step, arm=arm, seed=s, dir=d, num_tasks=n_tasks,
                             epoch=epoch, n_loaded=n_ok, ok=ok,
                             status={str(k): v for k, v in st.items()},
                             failures={str(k): v for k, v in st.items() if v != "ok"}))
            mark = "" if ok else "   <- FAILS"
            print(f"  {step:<6}{arm:<16}{s:>6}{n_tasks:>4}{epoch:>4}  "
                  f"{n_ok:>4}/{n_tasks:<4}  {d}{mark}")

    n_ok = sum(r["ok"] for r in rows)
    print("\n" + "=" * 104)
    print(f"  {n_ok}/{len(rows)} directories fully loadable; {len(all_bad)} bad files")
    by_step = {}
    for r in rows:
        for stp in r["step"].split(","):
            by_step.setdefault(stp, []).append(r["ok"])
    for stp in sorted(by_step):
        v = by_step[stp]
        print(f"    step {stp:<6} {sum(v)}/{len(v)} directories green -> "
              f"{'CLEAR' if all(v) else 'BLOCKED'}")
    if all_bad:
        dirs = sorted({str(Path(p).parent.parent.name) for p in all_bad})
        print("\n  AUDIT FAILS. Re-sync or re-run these:")
        for d in dirs:
            print(f"    modal volume get plcm-runs /{d} ./runs --force")
        print("\n  L4-L8 are blocked on the arms above. L1-L3 are unaffected: they")
        print("  load no checkpoint.")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump({"chain": chain_name, "root": root, "criterion": "PLCM.load_era succeeds",
                   "n_dirs": len(rows), "n_green": n_ok, "n_bad_files": len(all_bad),
                   "by_step": {k: {"green": sum(v), "total": len(v), "clear": all(v)}
                               for k, v in by_step.items()},
                   "rows": rows}, open(out, "w"), indent=2)
        print(f"\n  wrote {out}")
    return 0 if not all_bad else 1


def main():
    ap = argparse.ArgumentParser(description="checkpoint audit; criterion is LOADS")
    ap.add_argument("--json", default=None)
    ap.add_argument("--chain", choices=["e11", "e25", "e26", "e27", "e30"], default="e11",
                    help="e11: the original recompute chain, unchanged. "
                         "e25: the arms E25's A and B load (contract L0).")
    ap.add_argument("--root", default="runs",
                    help="volume root for the e25 chain: 'runs' locally, '/runs' on Modal")
    args = ap.parse_args()

    if args.chain in CHAINS:
        return main_chain(args.chain, args.root, args.json)

    targets = []
    for exp, tmpls in CHAIN.items():
        for t in tmpls:
            for s in SEEDS:
                targets.append((exp, t.format(s=s)))
    targets += [("E6b/MNIST", MNIST_ON[s]) for s in SEEDS]

    print("=" * 104)
    print("E11 CHECKPOINT AUDIT — criterion is LOADS, not exists (catch 20)")
    print("=" * 104)
    print(f"  {'experiment':<12}{'checkpoint dir':<46}" +
          "".join(f"{'T'+str(k):>9}" for k in TASKS))

    rows, all_bad = [], []
    for exp, d in targets:
        st, bad = audit_dir(d)
        all_bad += bad
        rows.append(dict(experiment=exp, dir=d,
                         status={str(k): v for k, v in st.items()},
                         ok=all(v == "ok" for v in st.values())))
        print(f"  {exp:<12}{d:<46}" +
              "".join(f"{st[k]:>9}" for k in TASKS))

    n_ok = sum(r["ok"] for r in rows)
    print("\n" + "=" * 104)
    print(f"  {n_ok}/{len(rows)} checkpoint dirs fully loadable; "
          f"{len(all_bad)} bad files")
    if all_bad:
        dirs = sorted({str(Path(p).parent.parent.name) for p in all_bad})
        print(f"\n  AUDIT FAILS. Re-sync these run dirs from the Modal volume:")
        for d in dirs:
            print(f"    modal volume get plcm-runs /{d} ./runs --force")
        print("\n  No E11 recompute may be signed off until this table is green.")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)
        print(f"\n  wrote {args.json}")
    return 0 if not all_bad else 1


if __name__ == "__main__":
    sys.exit(main())
