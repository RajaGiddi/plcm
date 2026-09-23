"""
E16 / Item 1 — three-channel decomposition on LwF era checkpoints.

Contract: the Deck-Close Contract, Item 1.

THE QUESTION. Li & Hoiem state their mechanism plainly: preserving old-task
outputs "retain[s] the important shared structures learned for the previous
tasks." That is a claim about the TRUNK. Every measurement in this program says
the trunk was never what broke. So when LwF protects, where does the protection
land?

THIS IS A DRIVER, NOT A DECOMPOSITION. Every quantity is computed by
`scripts/channel_decomp.decompose()`, the audited LSTM-family path — the same
function that produced the 66-88% reader-share row. Nothing here re-implements
R / F_enc / F_read, and nothing here touches data loading: `load_task_data` is
imported, not rewritten. That is catch 32 applied at design time rather than
after a retraction:

    "two separate comprehensions over a shuffle=True train loader draw two
     DIFFERENT permutations ... the probe then trains on randomly relabelled
     data and collapses to chance"

A parallel extraction path is exactly how that bug reached a memo last time.

WHAT THIS FILE OWNS: arm -> checkpoint-directory addressing, the seed list, and
the provenance header. Nothing else.

PRECONDITION (contract amendment): the reference arm and the LwF arms are
decomposed by the SAME script version. The commit hash is recorded in every
output so a mismatch is visible in the artifact rather than assumed away —
three of this week's catches were flag-set drift between things that were
supposed to be identical.

Usage:
    python scripts/e16_decompose.py --arm lam0.25 --bench mnist --out out.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.channel_decomp as cd            # the audited path, imported whole
from src.data.har_shift import HARShiftBenchmark
from src.data.permuted_mnist import PermutedMNISTBenchmark

SEEDS = [42, 1337, 2024]

# arm -> (checkpoint-dir template, era-checkpoint status expected in the artifact)
# Every one of these was launched with `--era-checkpoints`; the status is
# re-asserted from the run's own result file, never assumed (the axis is worth
# up to 4.32pp and is configuration, not observability).
# NOTE THE NESTED `mafc_seed{s}` COMPONENT. `_save_era_checkpoint` writes to
# `{checkpoint-dir}/{model_type}_seed{seed}/taskK_epoch9.pt`, not to
# `{checkpoint-dir}` directly. The first version of this table stopped one level
# short and all 8 jobs died on `FileNotFoundError: task4_epoch9.pt`.
#
# THE DIRECTORIES EXISTING WAS NOT THE SAME AS THE CHECKPOINTS BEING ADDRESSABLE.
# "28 ckpt_e16* dirs are present, so the expensive part is done" was true and
# insufficient — the repo's own gotcha, quoted earlier in this same session:
# *"a checkpoint exists" is not "a checkpoint loads"*. Verified now by listing
# the files, not the directories.
ARMS = {
    "lam0.25": ("runs/ckpt_e16_{bench}_lam0.25_seed{s}/mafc_seed{s}", "e16_{bench}_lam0.25_seed{s}"),
    "lam4.0":  ("runs/ckpt_e16_{bench}_lam4.0_seed{s}/mafc_seed{s}",  "e16_{bench}_lam4.0_seed{s}"),
    "lam16.0": ("runs/ckpt_e16_{bench}_lam16.0_seed{s}/mafc_seed{s}", "e16_{bench}_lam16.0_seed{s}"),
    # The reference. NOT "already decomposed" — the only LSTM-family
    # decomposition on record is `runs/e11_e6b/decomp.json`, which is HAR-only,
    # E6b-lineage, and a different arm. That premise was in the contract draft
    # and was false; H-D3 needs this denominator, so the reference is measured
    # here alongside the arms it is compared against.
    # E17: the MLP scratch arm. Same script, same seeds, same era status as
    # every other cell — the point of the wave is a share that is comparable to
    # the ones already in the ledger, and comparability is a property of the
    # procedure, not of the intent.
    "mlp": ("runs/ckpt_e17_mlp_seed{s}/mafc_seed{s}", "e17_mlp_seed{s}"),
    # The FLOOR PAIR, decomposed as its own two "seeds". The pair ran with era
    # checkpoints, so it is decomposable — and that yields something better than
    # an inferred caveat: the SAME-SEED SHARE DELTA, measured. A 3.68pp AVG floor
    # does not translate into share points by itself; two decompositions of the
    # same seed do. |delta share| between these two IS the share-level floor.
    "mlp_floor": ("runs/ckpt_e17_mlp_floor_{s}/mafc_seed42", "e17_mlp_floor_{s}"),
    "mafc_off": {"mnist": ("runs/ckpt_e4off_ec_seed{s}/mafc_seed{s}", "e4off_ec_seed{s}"),
                 "har":   ("runs/ckpt_w1_har_off_ec_seed{s}/mafc_seed{s}", "w1_har_off_ec_seed{s}")},
    # E20 secondary / S72: the bridging benchmark's citable OFF arm (har_subject,
    # era + fp32 shadow). The screen reads the SHADOW directory and so does this
    # -- fp16 era reloads miss the matrix by one window. `--bench har_subject`
    # builds HARSubjectBenchmark, whose partition is x86-only (gated below).
    "s72_off": {"har_subject": ("runs/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32", "e10off_ec_seed{s}")},
    # E20-B: the band's three ON arms relaunched on har_subject (jobs_s72_band).
    # `use_adapter` for the decomposition is read from the artifact in main().
    "s72_v1on":  {"har_subject": ("runs/ckpt_e10v1on_ec_seed{s}/mafc_seed{s}_fp32",  "e10v1on_ec_seed{s}")},
    "s72_v2on":  {"har_subject": ("runs/ckpt_e10v2on_ec_seed{s}/mafc_seed{s}_fp32",  "e10v2on_ec_seed{s}")},
    "s72_v2ctl": {"har_subject": ("runs/ckpt_e10v2ctl_ec_seed{s}/mafc_seed{s}_fp32", "e10v2ctl_ec_seed{s}")},
    # E16 sec 7.2 closure: LwF at lambda* = 1.0 relaunched uniform with S72 (era + shadow,
    # har_subject; jobs_s72_lwf). Selected on AVG under the 5pp competence floor
    # (runs/s72_row.json sweep). Decomposed 2026-09-18 for the paper's sec 5.1.
    "s72_lwf":   {"har_subject": ("runs/ckpt_e16s_har_lam1.0_seed{s}/mafc_seed{s}_fp32", "e16s_har_lam1.0_seed{s}")},
    # E23 arm B (2026-09-19): the MLP backbone on the S72 construction (era + shadow, har_subject).
    "e23_har_mlp": {"har_subject": ("runs/ckpt_e23_har_mlp_seed{s}/mafc_seed{s}_fp32", "e23_har_mlp_seed{s}")},
    "e23_har_mlp_floor": {"har_subject": ("runs/ckpt_e23_har_mlp_floor4tec_{s}/mafc_seed42_fp32", "e23_har_mlp_floor4tec_{s}")},
    # floor pairs (replicates a/b of seed 42) -> the SHARE floor per arm
    "s72_off_floor":   {"har_subject": ("runs/ckpt_e10off_ec_floor4tec_{s}/mafc_seed42_fp32",   "e10off_ec_floor4tec_{s}")},
    "s72_v1on_floor":  {"har_subject": ("runs/ckpt_e10v1on_ec_floor4tec_{s}/mafc_seed42_fp32",  "e10v1on_ec_floor4tec_{s}")},
    "s72_v2on_floor":  {"har_subject": ("runs/ckpt_e10v2on_ec_floor4tec_{s}/mafc_seed42_fp32",  "e10v2on_ec_floor4tec_{s}")},
    "s72_v2ctl_floor": {"har_subject": ("runs/ckpt_e10v2ctl_ec_floor4tec_{s}/mafc_seed42_fp32", "e10v2ctl_ec_floor4tec_{s}")},
}


def script_version() -> dict:
    """Pin the script version across every arm of the comparison.

    The contract asked for a COMMIT HASH. This repo has zero commits
    (`git rev-parse HEAD` -> "does not have any commits yet"), so that clause
    had no mechanism -- the amendment written to close the flag-set-drift
    channel could not itself be satisfied. A content hash does the job the
    clause actually wants and does not depend on git hygiene: it pins THE CODE
    THAT RAN, which is the thing a mismatch would differ in.

    Both files are hashed: this driver AND `channel_decomp.py`, which owns every
    quantity reported. Hashing only the driver would pin the addressing layer
    and leave the arithmetic free to move between arms.
    """
    import hashlib
    h = {}
    for f in ("scripts/e16_decompose.py", "scripts/channel_decomp.py"):
        pth = Path(__file__).parent.parent / f
        h[f] = hashlib.sha256(pth.read_bytes()).hexdigest()[:16]
    try:
        g = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=pth.parent.parent)
        h["git"] = g.stdout.strip() if g.returncode == 0 else "no-commits"
    except Exception:
        h["git"] = "no-git"
    return h


def resolve(arm: str, bench: str):
    spec = ARMS[arm]
    if isinstance(spec, dict):
        spec = spec[bench]
    ckpt, run = spec
    return ckpt.replace("{bench}", bench), run.replace("{bench}", bench)


ROOT = "runs"


def era_status(run_tmpl: str) -> list:
    """Read `era_checkpoints` off each run's OWN artifact (contract precondition)."""
    out = []
    for s in SEEDS:
        p = Path(ROOT) / run_tmpl.format(s=s)
        hit = list(p.glob("*_results.json"))
        if not hit:
            out.append(None); continue
        arm = json.load(open(hit[0])).get("arm") or {}
        out.append(arm.get("era_checkpoints"))
    return out


def main():
    ap = argparse.ArgumentParser(description="E16 decomposition on LwF checkpoints")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--bench", required=True, choices=["mnist", "har", "har_subject"])
    ap.add_argument("--out", default=None)
    # E20: the probe axis. Default "linear" is byte-for-byte the pre-E20 path.
    ap.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--probe-seeds", default="0,1,2",
                    help="MLP probe init seeds; the spread is the probe's floor")
    ap.add_argument("--alpha", type=float, default=cd.MLP_ALPHA,
                    help="MLP probe L2; changed ONLY by the contract's alpha sweep")
    ap.add_argument("--probe-seed", type=int, default=None,
                    help="R2: seed the probe's training-subset draw (recorded). Default None = pre-fix path.")
    ap.add_argument("--device", default="cpu")
    # Checkpoints live in the Modal volume at /runs. Analysis runs WHERE THE
    # CHECKPOINTS WERE BUILT (the E11 lesson: E10 checkpoints re-evaluated
    # locally missed their own recorded matrices). Default matches the
    # container; pass --ckpt-root runs for a local run after a volume get.
    ap.add_argument("--ckpt-root", default="/runs")
    args = ap.parse_args()
    device = torch.device(args.device)

    global SEEDS
    if args.arm.endswith("_floor"):
        # Two runs of ONE seed. The loop variable is the replicate tag, and the
        # checkpoint dir keeps `mafc_seed42` because both replicates ARE seed 42.
        SEEDS = ["a", "b"]
    ckpt_tmpl, run_tmpl = resolve(args.arm, args.bench)
    ckpt_tmpl = ckpt_tmpl.replace("runs/", args.ckpt_root.rstrip("/") + "/", 1)
    commit = script_version()

    print("=" * 96)
    print(f"E16 DECOMPOSITION — arm={args.arm}  bench={args.bench}")
    print("=" * 96)
    print(f"  script version  {commit}")
    print(f"  checkpoint tmpl {ckpt_tmpl}")

    # ---- precondition: era-checkpoint status, from the runs' own artifacts ---
    global ROOT
    ROOT = args.ckpt_root.rstrip("/") or "runs"
    ec = era_status(run_tmpl)
    print(f"  era_checkpoints {ec}")
    if not all(v is True for v in ec):
        print("  PRECONDITION FAILED — era-checkpoint status not uniformly True; "
              "this arm cannot enter a matched comparison.")
        return 1

    # ---- precondition: the CHECKPOINT FILES, not their directories ---------
    # Directory existence proves a job started. This checks the file the loader
    # will actually open, for every seed, before any data is loaded.
    missing = [ckpt_tmpl.format(s=sd) + "/task4_epoch9.pt" for sd in SEEDS
               if not Path(ckpt_tmpl.format(s=sd), "task4_epoch9.pt").exists()]
    if missing:
        print(f"  PRECONDITION FAILED — {len(missing)} checkpoint file(s) absent:")
        for mm in missing[:3]:
            print(f"    {mm}")
        return 1
    print(f"  checkpoint files  present for seeds {SEEDS}")

    # ---- data: the AUDITED loader, seeded per run for MNIST -----------------
    if args.bench == "har":
        hdata = cd.load_task_data(HARShiftBenchmark(num_tasks=5, root=".",
                                                    batch_size=256), probe_subset_seed=args.probe_seed)
        data_for_seed = lambda _s: hdata
        n_classes = 6
    elif args.bench == "har_subject":
        from src.data.har_subject import HARSubjectBenchmark
        har = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=256)
        pfp = har.partition_fingerprint()
        print(f"  har_subject partition {pfp} -> "
              f"{'AS-EXECUTED' if pfp == '1104af185c87' else 'NOT the executed partition -- STOP'}")
        if pfp != "1104af185c87":
            return 1
        hdata = cd.load_task_data(har, probe_subset_seed=args.probe_seed)
        data_for_seed = lambda _s: hdata
        n_classes = 6
    else:
        cache = {}
        def data_for_seed(seed):
            # The floor pair's "seeds" are REPLICATE TAGS ("a"/"b"); both runs
            # are seed 42, so both must be scored against SEED 42's permutations.
            # Passing the tag through would have thrown (it did) -- and had it
            # not, `decompose`'s own docstring is the warning: seed-1337 weights
            # score CHANCE against seed-42 permutations, so a silent mis-pairing
            # produces garbage shares that look like a finding.
            if not isinstance(seed, int):
                seed = 42
            # Permuted-MNIST tasks are generated FROM the run's seed, so a
            # checkpoint must be scored against ITS OWN permutations —
            # seed-1337 weights read chance against seed-42 permutations.
            if seed not in cache:
                cache[seed] = cd.load_task_data(
                    PermutedMNISTBenchmark(num_tasks=5, batch_size=256, seed=seed),
                    probe_subset_seed=args.probe_seed)
            return cache[seed]
        n_classes = 10

    # `decompose` iterates the module-level SEEDS; set it explicitly rather than
    # relying on whatever the imported module happens to define.
    cd.SEEDS = SEEDS

    # use_adapter is READ FROM THE ARTIFACT (E20-B added adapter arms). Every
    # E16/E17 arm records False; the band's ON arms record True. A hardcoded
    # False here would have decomposed an ON arm with its adapter suppressed --
    # a different arm wearing this one's name (catch 30).
    adapters = []
    for s_ in SEEDS:
        hit = list((Path(ROOT) / run_tmpl.format(s=s_)).glob("*_results.json"))
        adapters.append((json.load(open(hit[0])).get("arm") or {}).get("use_input_adapters") if hit else None)
    assert len(set(adapters)) == 1 and adapters[0] is not None, f"use_input_adapters not uniform/recorded: {adapters}"
    use_adapter = bool(adapters[0])
    print(f"  use_input_adapters (from artifacts) {adapters} -> decomposing with use_adapter={use_adapter}")
    probe_seeds = tuple(int(x) for x in args.probe_seeds.split(","))
    rows = cd.decompose(ckpt_tmpl, use_adapter, data_for_seed, n_classes, device,
                        probe=args.probe, probe_seeds=probe_seeds, alpha=args.alpha)

    # ---- identity, printed before any channel number is read ---------------
    resid = max(abs(r["F_enc"] + r["F_read"] - r["R"] - r["F_total"]) for r in rows)
    print(f"\n  IDENTITY  max |F_enc + F_read - R - F_total| = {resid:.2e}"
          f"  -> {'OK' if resid < 1e-6 else 'COMPUTATION ERROR'}")

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"\n  cells {len(rows)}  (seeds {SEEDS} x 4 old tasks)")
    print(f"  R        {m('R'):+.4f}   (instrument bias)")
    print(f"  F_enc    {m('F_enc'):+.4f}")
    print(f"  F_read   {m('F_read'):+.4f}")
    print(f"  F_total  {m('F_total'):+.4f}")
    denom = m("F_enc") + m("F_read")
    share = m("F_read") / denom if abs(denom) > 1e-9 else float("nan")
    print(f"  reader share {100*share:+.2f}%")

    if args.probe == "mlp":
        spread = [max(r["mlp_ceiling"]["mlp_spread"], r["mlp_t4"]["mlp_spread"]) for r in rows]
        gaps = [r["mlp_t4"]["mlp_gap"] for r in rows]
        unconv = [i for i, r in enumerate(rows)
                  if not (r["mlp_ceiling"]["mlp_converged"] and r["mlp_t4"]["mlp_converged"])]
        print(f"  MLP probe: init spread max {max(spread):.4f} mean {np.mean(spread):.4f}"
              f" | t4 train-test gap mean {np.mean(gaps):+.4f} max {max(gaps):+.4f}"
              f" | unconverged cells {len(unconv)}/{len(rows)} {unconv}")

    print(f"  probe subset draw: {'seeded ' + str(args.probe_seed) if args.probe_seed is not None else 'UNSEEDED (pre-R2 path)'}")
    out = {"arm": args.arm, "bench": args.bench, "commit": commit, "use_adapter": use_adapter,
           "probe_subset_seed": args.probe_seed,
           "sklearn_version": cd._SKLEARN_VERSION,
           "probe_n_iter_max": (max(max(r["probe_n_iter"].values()) for r in rows)
                                if args.probe == "linear" else None),
           **({"probe": "mlp", "probe_seeds": list(probe_seeds), "alpha": args.alpha}
              if args.probe == "mlp" else {}),
           "era_checkpoints": ec, "seeds": SEEDS, "identity_residual": resid,
           "rows": rows,
           "mean": {k: m(k) for k in ("R", "F_enc", "F_read", "F_total")},
           "reader_share": share}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2, default=float)
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
