"""
Modal fan-out runner for the generality package.

Every experiment in this project is a call to scripts/train.py with different
flags, and the runs are fully independent — which is the shape Modal is good at.
Sequentially the remaining board is 30-48 runs x ~20 min = 10-16 hours; fanned
out it is bounded by the slowest single run.

Design notes:
  * The repo is mounted into the image, so the container runs the SAME
    scripts/train.py as local — no reimplementation, no drift between what we
    test locally and what produces the paper's numbers.
  * MNIST / FashionMNIST / CIFAR are baked into the image at build time, so
    containers do not each re-download (and so a run cannot fail on a network
    blip mid-fan-out).
  * Results are written to a Modal Volume and synced back with `modal volume get`,
    landing in runs/ exactly as local runs do.
  * Seeds and configs are unchanged, so a Modal run and a local run with the same
    flags are the same experiment. That matters: the paper's reproduction
    instructions must not fork by execution backend.

Usage (from the repo root):
    modal run modal_runner.py --experiment e1        # Fashion-MNIST, 12 runs
    modal run modal_runner.py --experiment e5        # UCI HAR
    modal run modal_runner.py --experiment e2        # CIFAR-10
    modal run modal_runner.py --experiment smoke     # 1 short run, verifies plumbing

Then:
    modal volume get plcm-runs /runs ./runs --force
"""

import modal

APP_NAME = "plcm-generality"
VOLUME_NAME = "plcm-runs"

# ---------------------------------------------------------------- image ----
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.13.0", "torchvision==0.28.0",
        # PINNED 2026-09-22 at the versions the running container reported
        # (scripts/env_versions.py), not at today's index. E30's contract asked
        # only for scipy to be pinned, because `quadratic_assignment` is
        # load-bearing there. But these five sat UNPINNED in one list, and
        # editing the list invalidates the image cache: a rebuild whose stated
        # purpose was to FIX scipy would have refetched numpy, scikit-learn,
        # pyyaml and tqdm at whatever versions were current that day. The
        # scikit-learn is the one that matters -- `channel_decomp.refit_probe`
        # runs LogisticRegression to the library's default tolerance, and every
        # decomposition in this program reads through it. Pinning one package in
        # an unpinned list is not a narrowing, it is a re-roll.
        "numpy==2.4.6", "pyyaml==6.0.3", "tqdm==4.70.0",
        "scikit-learn==1.9.0", "scipy==1.17.1",
        # E12/B2 — pinned; the weight hash is asserted at container start.
        "timm==1.0.20",
    )
    # Bake the datasets in so containers never download at run time.
    .run_commands(
        "python -c \"from torchvision import datasets; "
        "datasets.MNIST('/data', train=True, download=True); "
        "datasets.MNIST('/data', train=False, download=True); "
        "datasets.FashionMNIST('/data', train=True, download=True); "
        "datasets.FashionMNIST('/data', train=False, download=True); "
        "datasets.CIFAR10('/data', train=True, download=True); "
        "datasets.CIFAR10('/data', train=False, download=True); "
        # E12/B3 — CIFAR-100, same rule: baked, never fetched at run time.
        "datasets.CIFAR100('/data', train=True, download=True); "
        "datasets.CIFAR100('/data', train=False, download=True)\"",
        # E12/B2 — ViT-B/16 pretrained weights baked at BUILD time so no
        # container ever downloads them, and so the hash below is fixed by the
        # image rather than by whatever HuggingFace serves that day.
        "python -c \"import timm; timm.create_model("
        "'vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)\"",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TIMM_USE_OLD_CACHE": "0"})
    .add_local_dir(".", remote_path="/repo", ignore=[
        "runs/*", "checkpoints/*", "data/*", ".venv/*", ".git/*", "*.ipynb",
    ])
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ------------------------------------------------------------- the runner ----
@app.function(
    volumes={"/runs": volume},
    timeout=60 * 60 * 3,
    cpu=4.0,
    memory=8192,
    max_containers=16,
)
def train_one(flags: list[str], log_dir: str) -> dict:
    """Run scripts/train.py with `flags`, writing results into the Volume.

    Returns a small summary so the driver can print a table without syncing.
    """
    return _run(flags, log_dir, "cpu")


@app.function(
    volumes={"/runs": volume},
    timeout=60 * 60 * 4,
    gpu="L4",
    cpu=4.0,
    # 8GiB, not 16. Modal named this request as what was narrowing the eligible
    # worker pool ("Relaxing requirements (memory=16.8GiB) may lead to faster
    # scheduling") while E12's precondition re-run sat queued for ~3h and never
    # got an L4. This is CONTAINER RAM, not GPU RAM: ViT-B/16 at batch 64 with
    # num_workers=0 uses a small fraction of it, so this changes scheduling and
    # nothing about the experiment.
    memory=8192,
    max_containers=16,
)
def train_one_gpu(flags: list[str], log_dir: str) -> dict:
    """GPU variant, for CIFAR-scale runs.

    CIFAR is 50k samples against HAR's 7.3k, and LSTM cost scales with hidden^2.
    Extrapolating measured HAR timings, a single hidden=1024 CIFAR run is ~110min
    for ONE task on CPU — so the 5-task continual runs would be ~9h each and E2
    could not finish inside the timebox. This is an infrastructure fix, not a
    protocol change: identical code, identical flags, only the device differs.
    """
    return _run(flags, log_dir, "cuda")


@app.function(
    volumes={"/runs": volume},
    timeout=60 * 60 * 3,
    cpu=4.0,
    memory=8192,
    max_containers=16,
)
def train_one_1thread(flags: list[str], log_dir: str) -> dict:
    """CPU run pinned to ONE math thread. W2 diagnosis only.

    Multithreaded BLAS sums partial products in completion order, so a
    reduction over N threads is not associative in floating point and the same
    seed can land on different bits. Pinning to one thread removes that source
    entirely: if `plain_lstm` reproduces bit-identically here and not at default
    threading, the nondeterminism is THREADING, not the arm -- and every floor
    pair from now on records its thread config as part of the tuple it is a
    property of. If it still fails at one thread, threading is exonerated and
    the LSTM path's warmup phase is the next suspect.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    return _run(flags, log_dir, "cpu")


def _run(flags: list[str], log_dir: str, device: str) -> dict:
    import json, os, subprocess, sys, shutil

    os.chdir("/repo")
    # Datasets were baked into /data at image build; point the code at them.
    if not os.path.exists("/repo/data"):
        os.symlink("/data", "/repo/data")

    out = f"/runs/{log_dir}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "scripts/train.py", *flags, "--log-dir", out, "--device", device]
    print("RUN:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-4000:]); print(p.stderr[-4000:])
        volume.commit()
        return {"log_dir": log_dir, "ok": False, "error": p.stderr[-500:]}

    res = {}
    for f in os.listdir(out):
        if f.endswith("_results.json"):
            d = json.load(open(os.path.join(out, f)))
            res = {"average_accuracy": d["average_accuracy"], "forgetting": d["forgetting"]}
    volume.commit()
    return {"log_dir": log_dir, "ok": True, **res}


# ------------------------------------------------------------- job lists ----
CFG = ["--config", "configs/mafc_phase1.yaml", "--model", "mafc", "--mafc-arm", "lambda0"]
SEEDS = [42, 1337, 2024]


def jobs_e1() -> list[tuple[list[str], str]]:
    """E1 — Fashion-MNIST, both shifts, both arms, 3 seeds (12 runs)."""
    out = []
    for shift in ("permuted", "rotated"):
        for s in SEEDS:
            out.append((CFG + ["--dataset", "fashion", "--benchmark", shift,
                               "--seed", str(s)], f"e1_{shift}_adapt_seed{s}"))
            out.append((CFG + ["--dataset", "fashion", "--benchmark", shift,
                               "--no-adapters", "--seed", str(s)],
                        f"e1_{shift}_noadapt_seed{s}"))
    return out


def jobs_e5() -> list[tuple[list[str], str]]:
    """E5 — UCI HAR hardware-revision shifts, both arms, 3 seeds (6 runs).

    Adapter geometry is set by the benchmark, not here: HAR's shifts are 9x9
    channel maps applied identically at every timestep, so train.py switches the
    adapter to per_step (81 params/task). See src/data/har_shift.py.
    """
    out = []
    for s in SEEDS:
        out.append((CFG + ["--benchmark", "har", "--seed", str(s)],
                    f"e5_har_adapt_seed{s}"))
        out.append((CFG + ["--benchmark", "har", "--no-adapters", "--seed", str(s)],
                    f"e5_har_noadapt_seed{s}"))
    return out


def jobs_e5diag() -> list[tuple[list[str], str]]:
    """E5 diagnostic — checkpointed ON arm, 3 seeds.

    E5 failed both H-G1 bars with adapters slightly WORSE than none (3/3 seeds).
    Two readings are consistent with that headline and they imply different
    papers, so the distinguishing measurement is run before either is written:

      (i)  the adapters never learned to invert the shift (A_k stays near I),
           because the encoder can absorb a 9x9 channel map cheaply; or
      (ii) they did invert it and inversion simply did not help.

    Tasks 2 and 3 are EXACTLY invertible by the locked bias-free adapter, so the
    ideal target M_k^-1 is known in closed form and the comparison is exact.
    n=3 because a distinguishing claim cannot rest on n=1 (standing rule).
    """
    return [(CFG + ["--benchmark", "har", "--seed", str(s), "--save-checkpoints",
                    "--checkpoint-dir", f"/runs/ckpt_e5_seed{s}"],
             f"e5diag_har_adapt_seed{s}") for s in SEEDS]


def jobs_e5b() -> list[tuple[list[str], str]]:
    """E5b — no-shift control + OFF-arm reproducibility recheck.

    Control (6 runs): identical to E5 with every task's map set to the identity,
    so the five tasks are the same task. Forgetting measured here cannot be
    attributed to input shift and therefore bounds how much of E5's forgetting
    the mechanism was ever in a position to address.
    See docs/E5B_control_prereg.md for the branches, fixed before this ran.

    Recheck (3 runs): E5's ON/2024 was preempted-and-restarted and did NOT
    reproduce on relaunch (+5.50pp), while 42 and 1337 were bit-identical. The
    OFF arm has never been independently relaunched, so it is unverified by the
    same standard. These re-run it to check no other run is contaminated.
    """
    out = []
    for s in SEEDS:
        out.append((CFG + ["--benchmark", "har_noshift", "--seed", str(s)],
                    f"e5b_noshift_adapt_seed{s}"))
        out.append((CFG + ["--benchmark", "har_noshift", "--no-adapters",
                           "--seed", str(s)], f"e5b_noshift_noadapt_seed{s}"))
        out.append((CFG + ["--benchmark", "har", "--no-adapters", "--seed", str(s)],
                    f"e5recheck_har_noadapt_seed{s}"))
    return out


def jobs_e2cal() -> list[tuple[list[str], str]]:
    """E2 capacity calibration + reproduction-floor measurement (6 runs).

    Contract §2 E2 (v2 Finding 3) + v3 amendment 7.

    Single-task clean CIFAR-10 at hidden in {256, 512, 1024}. Take the best
    accuracy, record it, proceed regardless of whether 55% is reached; both arms
    later share the winner, never per-arm tuned.

    Each capacity runs TWICE under an identical config. The amendment requires
    relaunching only the winner, but running all three costs 2 extra short runs
    and answers a question the single relaunch cannot: whether the reproduction
    floor itself scales with capacity. That matters here — if d=3072 with an
    unsaturated encoder yields a floor near 8-10pp, H-G1b's +15pp bar sits
    barely above noise and E2 cannot cleanly answer the gate even if it passes.
    Learning that costs 6 short runs instead of 12 full ones.

    Calibration runs with --no-adapters. Capacity is a property of the ENCODER,
    which both arms share; measuring it with a 3072x3072 adapter attached would
    tune the choice to the ON arm, and §2 E2 requires capacity be "never per-arm
    tuned." Task 0 is the identity anyway, so the adapter has nothing to undo —
    it would only act as 9.4M parameters of extra capacity masking the very
    encoder differences the search is trying to resolve.
    """
    out = []
    for h in (256, 512, 1024):
        for rep in (1, 2):
            out.append((CFG + ["--benchmark", "cifar_permuted", "--no-adapters",
                               "--seed", "42", "--hidden-size", str(h),
                               "--num-tasks", "1"],
                        f"e2cal_h{h}_rep{rep}"))
    return out


def jobs_e2() -> list[tuple[list[str], str]]:
    """E2 — CIFAR-10 input shifts, both shifts x both arms x 3 seeds (12 runs)
    plus a 2-run reproduction-floor pair (14 total).

    Capacity fixed at hidden=1024 by the pre-registered rule: best clean
    single-task accuracy over {256, 512, 1024} measured with --no-adapters
    (0.5788 / 0.5952 / 0.6035), recorded in runs/e2/CALIBRATION.md. Both arms
    share it; it is never re-tuned per arm.

    FLOOR PAIR. e2cal measured 0.0000pp spread, but at num_tasks=1 with no
    adapters. E2 runs 5 tasks WITH adapters, where run-to-run divergence has
    four more task boundaries to accumulate across — and catch 19's lesson is
    that floors do not transfer between configurations. The pair re-runs
    cifar_permuted/seed42 in BOTH arms under identical configs, because an
    asymmetric relaunch would clean one side of the comparison and bias the
    delta in an unknown direction. Two extra runs is a cheap price for knowing
    whether H-G1b's +15pp bar sits above the noise on the config being read.
    """
    out = []
    for shift in ("cifar_permuted", "cifar_rotated"):
        for s in SEEDS:
            out.append((CFG + ["--benchmark", shift, "--hidden-size", "1024",
                               "--seed", str(s)], f"e2_{shift}_adapt_seed{s}"))
            out.append((CFG + ["--benchmark", shift, "--hidden-size", "1024",
                               "--no-adapters", "--seed", str(s)],
                        f"e2_{shift}_noadapt_seed{s}"))
    # Reproduction-floor pair — identical configs to the seed-42 permuted runs.
    out.append((CFG + ["--benchmark", "cifar_permuted", "--hidden-size", "1024",
                       "--seed", "42"], "e2floor_adapt_seed42_rep2"))
    out.append((CFG + ["--benchmark", "cifar_permuted", "--hidden-size", "1024",
                       "--no-adapters", "--seed", "42"], "e2floor_noadapt_seed42_rep2"))
    return out


def jobs_e5d() -> list[tuple[list[str], str]]:
    """E5d — method v2, adapter-first warmup (docs/E5D_prereg.md). 15 new runs.

    HAR (6 new): v2-ON and v2-control, 3 seeds each. v1-ON (`e5diag_*`) and OFF
    (`e5_har_noadapt_*`) are reused — both already verified.

    v2-control is the arm that makes H-D2/H-D3 attributable: it runs the
    IDENTICAL freeze schedule with the adapter frozen, so both arms rest the
    encoder for the same epochs and differ only in whether recruitment can
    happen. Without it, "warmup reduced forgetting" cannot be told apart from
    "the encoder trained less" — and Option A already measured that encoder
    freezing alone is worth +4.08pp.

    MNIST (9): v2-ON, v2-control, and v1-ON run CONCURRENTLY, 3 seeds each.
    v1-ON is re-run rather than compared against its historical 0.9331 +/- 0.0099
    band, because a +/-1pp tolerance against a +/-0.99pp spread is roughly one
    standard deviation — H-D4 would fail on noise. A same-session paired delta is
    the version whose failure would actually mean something.

    Checkpoints on for every arm: H-D1/H-D1b are unmeasurable without them.
    """
    out = []
    for s in SEEDS:
        for mode, tag in (("adapter", "v2on"), ("control", "v2ctl")):
            out.append((CFG + ["--benchmark", "har", "--seed", str(s),
                               "--warmup-epochs", "1", "--warmup-mode", mode,
                               "--save-checkpoints",
                               "--checkpoint-dir", f"/runs/ckpt_e5d_{tag}_seed{s}"],
                        f"e5d_har_{tag}_seed{s}"))
    for s in SEEDS:
        for flags, tag in (
            (["--warmup-epochs", "1", "--warmup-mode", "adapter"], "v2on"),
            (["--warmup-epochs", "1", "--warmup-mode", "control"], "v2ctl"),
            ([], "v1on"),
        ):
            out.append((CFG + ["--seed", str(s)] + flags,
                        f"e5d_mnist_{tag}_seed{s}"))
    return out


def jobs_e5dfloor() -> list[tuple[list[str], str]]:
    """E5d reproduction floor — BOTH v2 arms, seed 42, identical configs.

    Omitted from the original e5d launch; prereg §7 requires a measured floor
    before any delta is interpreted, and the v2-ON vs v2-control attribution in
    §3a is precisely a delta. Both arms, never one.
    """
    return [
        (CFG + ["--benchmark", "har", "--seed", "42", "--warmup-epochs", "1",
                "--warmup-mode", "adapter"], "e5dfloor_har_v2on_seed42_rep2"),
        (CFG + ["--benchmark", "har", "--seed", "42", "--warmup-epochs", "1",
                "--warmup-mode", "control"], "e5dfloor_har_v2ctl_seed42_rep2"),
    ]


def jobs_e6off() -> list[tuple[list[str], str]]:
    """E6 prerequisite — HAR OFF arm WITH checkpoints, 3 seeds.

    E6 was drafted as analysis-only. The audit found the OFF arm has no
    checkpoints at all: no job ever passed --no-adapters together with
    --save-checkpoints, so H-B3 and the ON/OFF drift_S ratio (H-B1) had no
    input. Second instance of the same error class as catch 15 (E4's contract
    also claimed existing checkpoints without auditing them).

    Interpretation lock (prereg v2 §5): E6's OFF anatomy is computed on THESE
    runs' own trajectories -- theta_0 and theta_4 come from one execution, so
    the anatomy is self-consistent regardless of whether the finals reproduce
    the original e5_har_noadapt_* numbers. The comparison against those finals
    is reported as a floor reading, never silently averaged in.
    """
    return [(CFG + ["--benchmark", "har", "--no-adapters", "--seed", str(s),
                    "--save-checkpoints",
                    "--checkpoint-dir", f"/runs/ckpt_e5_off_seed{s}"],
             f"e6off_har_noadapt_seed{s}") for s in SEEDS]


def jobs_e7() -> list[tuple[list[str], str]]:
    """E7 — per-task frozen heads on HAR, the reader-channel cure (8 runs).

    E6b: the shared reader carries 75-90% of HAR's forgetting even after charging
    the entire instrument bias against it, while the encoder channel is
    0.042-0.076. A frozen per-task head cannot walk.

      e7_heads       adapters OFF, heads ON  -- the cure ISOLATED. Adapters are
                     off deliberately: this is the version that can embarrass us,
                     since a collapse here shows adapters were UNNECESSARY on HAR
                     rather than merely insufficient.
      e7_heads_adapt adapters ON, heads ON   -- does the input path add anything
                     once the reader is fixed? (H-E3)

    Plus a 2-run reproduction-floor pair, BOTH arms (catch 19): E5d omitted its
    floor pair and had to launch it separately after the fact. The contract's
    section 6 requires it before any delta is interpreted.
    """
    out = []
    for s in SEEDS:
        out.append((CFG + ["--benchmark", "har", "--seed", str(s), "--task-heads",
                           "--no-adapters", "--save-checkpoints",
                           "--checkpoint-dir", f"/runs/ckpt_e7_heads_seed{s}"],
                    f"e7_heads_seed{s}"))
        out.append((CFG + ["--benchmark", "har", "--seed", str(s), "--task-heads",
                           "--save-checkpoints",
                           "--checkpoint-dir", f"/runs/ckpt_e7_heads_adapt_seed{s}"],
                    f"e7_heads_adapt_seed{s}"))
    out.append((CFG + ["--benchmark", "har", "--seed", "42", "--task-heads",
                       "--no-adapters"], "e7floor_heads_seed42_rep2"))
    out.append((CFG + ["--benchmark", "har", "--seed", "42", "--task-heads"],
                "e7floor_heads_adapt_seed42_rep2"))
    return out


def jobs_e10() -> list[tuple[list[str], str]]:
    """E10 — subject-disjoint HAR, the repaired benchmark (11 runs).

    Gate (G) cleared: the P3-v3 generative-honesty certificate validated on three
    controls (full-reconstruction rejected at 100% below tau; 10% injection
    metered at 9.98%; honest negative control 0.00%) and passed live at 0.00%.

    Arms: OFF and v1-ON (3 seeds each) measure the screen; the no-shift control
    (3) isolates how much forgetting comes from SUBJECT change rather than shift
    (P2 -- reported either way, never assumed away); the floor pair covers BOTH
    arms per the standing rule.

    --save-checkpoints on EVERY arm: catch 20 applied before it bites, since any
    arm here may later be a comparison baseline.
    """
    out = []
    for s in SEEDS:
        out.append((CFG + ["--benchmark", "har_subject", "--no-adapters", "--seed", str(s),
                           "--save-checkpoints", "--checkpoint-dir", f"/runs/ckpt_e10_off_seed{s}"],
                    f"e10_off_seed{s}"))
        out.append((CFG + ["--benchmark", "har_subject", "--seed", str(s),
                           "--save-checkpoints", "--checkpoint-dir", f"/runs/ckpt_e10_on_seed{s}"],
                    f"e10_on_seed{s}"))
        out.append((CFG + ["--benchmark", "har_subject_noshift", "--no-adapters",
                           "--seed", str(s), "--save-checkpoints",
                           "--checkpoint-dir", f"/runs/ckpt_e10_noshift_seed{s}"],
                    f"e10_noshift_seed{s}"))
    out.append((CFG + ["--benchmark", "har_subject", "--no-adapters", "--seed", "42"],
                "e10floor_off_seed42_rep2"))
    out.append((CFG + ["--benchmark", "har_subject", "--seed", "42"],
                "e10floor_on_seed42_rep2"))
    return out


def jobs_smoke() -> list[tuple[list[str], str]]:
    """One short run that exercises the whole path (image, repo, volume)."""
    return [(CFG + ["--dataset", "fashion", "--seed", "42", "--epochs", "1"],
             "modal_smoke")]


E12_CFG_PATH = "configs/e12_vit.yaml"
E12_CFG = ["--config", E12_CFG_PATH, "--model", "mafc", "--mafc-arm", "lambda0"]

# THE TWO FLAGS THAT MAKE THE ARM WHAT THE CONTRACT SAYS IT IS.
#
# `scripts/train.py` derives both from CLI flags and OVERWRITES the config:
# `--model mafc` sets adapters.enabled=True unless --no-adapters, and
# use_task_heads=bool(--task-heads). So configs/e12_vit.yaml reading
# `use_task_heads: true` / `adapters.enabled: false` describes a model that only
# exists if these flags say so. The first E12 jobs omitted both, and produced
# runs with adapters ON and ONE SHARED HEAD scoring all 20 tasks — neither the
# base arm nor the adapt arm, and not task-IL at all.
#
# Catch 21's premise rule, one level up: the config field cited as evidence was
# not the field that decided. The arm now travels in the results json
# (`arm` block, read off the built model) and the era-checkpoint P3b assert
# fails at the first task boundary if per-task heads are not there.
E12_TASK_IL = ["--task-heads"]                        # per-task heads == task-IL
E12_BASE = E12_TASK_IL + ["--no-adapters"]            # base: adapters OFF
E12_ADAPT = E12_TASK_IL                               # adapt: adapters ON


def _e12_seeds(need: int, path: str = E12_CFG_PATH) -> list[int]:
    """Seeds that have a REGISTERED class-order fingerprint, and only those.

    `train.py` gates the executed split against `expected_class_order[seed]` —
    but for an unregistered seed it prints a warning and trains anyway, so a
    5-seed headline over 3 registered seeds would run 2 of them ungated. The
    launcher refuses instead: the gate E10 bought is not optional for the two
    newest rows of the primary table.
    """
    import yaml

    cfg = yaml.safe_load(open(path))
    seeds = sorted(int(s) for s in
                   (cfg.get("benchmark", {}).get("expected_class_order") or {}))
    assert len(seeds) >= need, (
        f"{path} registers {len(seeds)} seed fingerprints {seeds}, contract needs "
        f"{need}. Compute the class-order (and shift) fingerprints for the extra "
        f"seeds IN THE EXECUTION ENVIRONMENT and add them to expected_class_order "
        f"before launching — a fingerprint certifies executed data only where the "
        f"data is built.")
    # PRIORITY ORDER, not numeric order. Registering seeds 7 and 1234 to meet the
    # five-seed lock silently changed which three `need=3` returned: sorted()
    # gives [7, 42, 1234], so the efficacy and class-IL controls would have run
    # on a different seed set from the frozen-probe arm and the preconditions
    # (both 42/1337/2024), breaking seed-level comparability between the controls
    # and the references they are read against.
    #
    # An amendment must re-verify what it did not touch. Extending the registry
    # touched the 5-seed rows by design and the 3-seed rows by accident.
    priority = [42, 1337, 2024, 7, 1234]
    ordered = [s for s in priority if s in seeds] + [s for s in seeds if s not in priority]
    return ordered[:need]


def jobs_e12() -> list[tuple[list[str], str]]:
    """E12 HEADLINE — base and adapt, 5 seeds each, fp16 era checkpoints.

    H-V1 (base, primary) and H-V2 (adapt, the falsifier). Both arms carry era
    checkpoints because the decomposition is computed FROM them: sec 3's
    measurement list needs task k's own end-of-task model, and catch 20's
    corollary says save for both arms whenever either might later be a
    comparison baseline.

    ~173MB x 20 tasks x 10 runs = ~35GB, which is the ruled storage plan. The
    boundary-time acc_ceiling/acc_refit_ceiling are authoritative; the reload is
    the reproduction path and the disagreement is recorded per task.
    """
    out = []
    for s in _e12_seeds(5):
        out.append((E12_CFG + E12_BASE + [
            "--seed", str(s), "--era-checkpoints",
            "--checkpoint-dir", f"/runs/ckpt_e12_base_seed{s}"], f"e12_base_seed{s}"))
        out.append((E12_CFG + E12_ADAPT + [
            "--seed", str(s), "--era-checkpoints",
            "--checkpoint-dir", f"/runs/ckpt_e12_adapt_seed{s}"], f"e12_adapt_seed{s}"))
    return out


def jobs_e12eff() -> list[tuple[list[str], str]]:
    """E12 efficacy control — the adapter shown capable of helping SOMEWHERE.

    Identical placement, optimizer, schedule and budget as `adapt`, on
    patch-consistent permuted Split-CIFAR-100 (B6). H-V2 is not readable until
    this arm passes; a flat control means "H-V2 untestable at this placement",
    reported, never silently read as confirmation.

    No era checkpoints: this arm is scored on accuracy, not decomposed. That
    accepts catch 20's exposure knowingly — if it ever becomes a comparison
    baseline it needs a re-run — in exchange for keeping the storage plan at the
    35GB the contract costed.
    """
    return [(E12_CFG + E12_ADAPT + ["--benchmark", "cifar100_permuted",
                                    "--seed", str(s)], f"e12eff_seed{s}")
            for s in _e12_seeds(3)]


def jobs_e12effb() -> list[tuple[list[str], str]]:
    """The efficacy control's MISSING COUNTERFACTUAL — base arm on permuted data.

    `jobs_e12eff` ran adapters-ON on patch-consistent permuted CIFAR-100 and
    nothing else. But "the adapter is capable of helping" is a COMPARATIVE claim,
    and the contract's spec — "identical placement, optimizer, schedule and
    budget as `adapt`" — matched the control to the treatment rather than to a
    counterfactual. With one arm there is no quantity "help" refers to: the
    control could be read as flat or not-flat against nothing.

    Sibling of catch 25 from a third direction. There: a gate that cannot fail.
    P3b: a gate whose green was vacuous. Here: a gate that cannot READ, because
    the comparison it scores was never run. All three pass inspection as written
    and none of them measures what its sentence claims.

    Adapters OFF, same benchmark, same seeds, same budget. Efficacy is then
    (adapt - base) on permuted, and H-V2 becomes readable — or honestly does not.
    """
    return [(E12_CFG + E12_BASE + ["--benchmark", "cifar100_permuted",
                                   "--seed", str(s)], f"e12effb_seed{s}")
            for s in _e12_seeds(3)]


def jobs_e12cil() -> list[tuple[list[str], str]]:
    """E12 class-IL (H-V3, secondary) — shared head over seen classes, 3 seeds.

    Era checkpoints ON: H-V3 is a decomposition claim ("reader share >= 0.50 in
    class-IL"), so the same era models the task-IL decomposition needs are
    required input here too. ~10GB on top of the 35GB.
    """
    return [(["--config", "configs/e12_vit_classil.yaml", "--model", "mafc",
              "--mafc-arm", "lambda0", "--no-adapters", "--seed", str(s),
              "--era-checkpoints",
              "--checkpoint-dir", f"/runs/ckpt_e12_cil_seed{s}"], f"e12cil_seed{s}")
            for s in _e12_seeds(3)]


def jobs_e12floor() -> list[tuple[list[str], str]]:
    """Catch 19's reproduction floor, on THIS GPU path, BOTH arms.

    Identical relaunch of base/42 and adapt/42. No delta in the ViT section — the
    arm delta, the fp16 reload delta, anything — may be interpreted before this
    pair reports, and it is a pair because an asymmetric relaunch cleans one side
    of a comparison and leaves the other dirty, which looks fixed and is worse.
    """
    return [(E12_CFG + E12_BASE + ["--seed", "42"], "e12floor_base_seed42_rep2"),
            (E12_CFG + E12_ADAPT + ["--seed", "42"], "e12floor_adapt_seed42_rep2")]


E14_CFG = ["--config", "configs/e14_resnet.yaml", "--model", "mafc",
           "--mafc-arm", "lambda0", "--task-heads", "--no-adapters"]


def jobs_e14() -> list[tuple[list[str], str]]:
    """E14 — ResNet-50 base arm, 3 seeds, 20 tasks, era checkpoints.

    --task-heads and --no-adapters are passed EXPLICITLY: train.py derives both
    from CLI flags and overwrites the config, which is how E12's "base arm" once
    ran adapters-ON with one shared head (catch 30). The config states the
    intent; only these flags make it true.
    """
    return [(E14_CFG + ["--seed", str(s), "--era-checkpoints",
                        "--checkpoint-dir", f"/runs/ckpt_e14_base_seed{s}"],
             f"e14_base_seed{s}") for s in _e12_seeds(3)]


def jobs_e14rpt() -> list[tuple[list[str], str]]:
    """E14 P2(b) — the repeat-task control, 5 tasks of IDENTICAL data.

    MISSING UNTIL AFTER THE HEADLINE RUNS LANDED. The contract (sec 2) states P2
    as two clauses; only (a) had a job, so P2 would have reached the memo marked
    PASS on half its definition. Catch 20's shape one level over: the audit
    covered artifacts and the build gate covered the build, and nothing checked
    that every contracted precondition HAD a job defined.

    Why it is load-bearing even with per-task frozen heads: (a) alone cannot
    distinguish drop-because-the-distribution-shifted from
    drop-because-the-trunk-kept-training. `--repeat-task` holds the data fixed
    and lets everything else proceed, so a near-zero reading attributes the
    headline drop to sequence content rather than to continued optimization.

    Same arm as jobs_e14 (E14_CFG carries --task-heads/--no-adapters, catch 30),
    5 tasks rather than 20 — matching jobs_e12rptb's shape exactly.
    """
    return [(E14_CFG + ["--seed", str(s), "--repeat-task", "--num-tasks", "5"],
             f"e14rpt_seed{s}") for s in _e12_seeds(3)]


def jobs_e14floor() -> list[tuple[list[str], str]]:
    """E14's OWN reproduction floor — identical config relaunched, seed 42.

    The floor does not transfer from E12: catch 19's wording is that it is a
    property of (seed x config x platform x BLAS), and E14 changes the config
    and the architecture. Without this pair, any base-vs-frozen delta would have
    to quote E12's floor as approximate, with the caveat stated. ~30 min buys
    the real number instead.
    """
    return [(E14_CFG + ["--seed", "42"], "e14floor_base_seed42_rep2")]


def jobs_e12pb() -> list[tuple[list[str], str]]:
    """E12 preconditions RE-RUN on the contracted base arm, 3 seeds.

    The P1/P2 numbers in runs/MEMO_e12.md came from `jobs_e12p`, which omitted
    --task-heads and --no-adapters (see E12_TASK_IL above), so they describe an
    adapters-on / shared-head configuration that is not the arm the headline
    runs. Preconditions gate the decomposition; a gate measured on a different
    arm does not gate this one.

    Cheap to redo (~100 min for 3 seeds, no checkpoints) and genuinely
    informative: with per-task frozen heads, P2's >= 0.15 task-0 drop is no
    longer close to guaranteed, which is exactly the branch-(G) outcome the
    contract priced at ~60% and flagged in advance.
    """
    return [(E12_CFG + E12_BASE + ["--seed", str(s)], f"e12pb_base_seed{s}")
            for s in _e12_seeds(3)]


def jobs_e12rptb() -> list[tuple[list[str], str]]:
    """P2's repeat-task control, on the contracted base arm — pairs with e12pb."""
    return [(E12_CFG + E12_BASE + ["--seed", str(s), "--repeat-task",
                                   "--num-tasks", "5"], f"e12rptb_seed{s}")
            for s in _e12_seeds(3)]


def jobs_e12p() -> list[tuple[list[str], str]]:
    """SUPERSEDED by jobs_e12pb — kept as the record of what produced
    runs/e12p_base_seed*.

    Labelled "base arm", but with neither --task-heads nor --no-adapters it ran
    per-task adapters ON and a single shared head for all 20 tasks. Evidence in
    the artifacts: every epoch dict carries `adapter_dist_from_identity` (~0.9,
    so the adapters trained) and none carries `head_weight_norm` (the trainer
    records it only when task k's head exists).

    E12 PRECONDITION probe — base arm, 3 seeds, no checkpoints.

    P1's second clause (trainable DIAG within 5pp of the frozen 0.9741) and P2
    (task-0 deployed drop >= 0.15) both read the accuracy matrix, so neither
    needs a checkpoint. Running the probe before the 5-seed headline keeps the
    branch-(G) outcome cheap: P2 is priced at ~60%, and its failure —
    "pretrained ViTs in task-IL do not forget enough to decompose at this
    scale" — is a reportable result, not a wasted 5-seed sweep.

    Three seeds, not one: a precondition decided at n=1 is a decision made on
    noise, which is this program's oldest standing rule.
    """
    return [(E12_CFG + ["--seed", str(s)], f"e12p_base_seed{s}") for s in SEEDS]


def jobs_e12rpt() -> list[tuple[list[str], str]]:
    """SUPERSEDED by jobs_e12rptb — same arm defect as jobs_e12p.

    E12 P2's repeat-task control — 5 tasks of IDENTICAL data, fresh heads.

    Establishes that whatever drop P2 measures is TASK-SEQUENCE-caused rather
    than optimization drift (E5b's lesson, ported). `--repeat-task` makes every
    task the same 5 classes; forgetting must come in <= 0.05.
    """
    return [(E12_CFG + ["--seed", str(s), "--repeat-task",
                        "--num-tasks", "5"], f"e12rpt_seed{s}") for s in SEEDS]


# ------------------------------------------------------------ E16 (LwF) ----
# Contract: docs/E16_lwf_prereg.md. Every entry below is named in that
# contract's sec 7 CLAUSE -> JOB -> ARTIFACT table (catch 33); if you add a
# clause there, add the entry here in the same edit.
#
# DEVICE: CPU, deliberately, and E16 must NEVER be added to GPU_EXPERIMENTS or
# GPU_JOB_PREFIXES. The contract header says "one day of GPU"; that is wrong and
# is corrected in the contract. Every arm E16 is compared against — the MNIST
# input-path table and the HAR OFF arm — is an existing CPU run, and the rule
# already encoded above is that every arm of a comparison runs on the same
# device as the arms it is compared against. Catch 19 makes this load-bearing:
# the reproduction floor is a property of (seed x config x platform x BLAS), so
# a device swap on one side is the infrastructure form of an asymmetric relaunch.

# AMENDMENT to contract sec 1: the registered grid was {0.5,1,2,5}; the binding
# grid is Li & Hoiem's geometric {0.25,1,4,16}. Wider and log-spaced, so the
# lambda response's SHAPE is visible rather than four points on a plateau.
E16_LAMBDAS = (0.25, 1.0, 4.0, 16.0)        # sec 1 sweep, 4 values, 3 seeds each

# The 5-seed headline set, declared HERE and not taken from _e12_seeds().
# _e12_seeds() returns seeds whose CIFAR-100 CLASS-ORDER fingerprint is
# registered in configs/e12_vit.yaml. E16 runs permuted MNIST and HAR, which
# have no class order at all -- so calling it here would invoke a gate that
# certifies nothing about this data while LOOKING like certification: a gate
# that cannot fail in the relevant direction (catch 25), resting on a premise
# that does not hold for the benchmark (catch 21).
#
# The fingerprints that DO apply here are per-benchmark and different in kind:
#   permuted MNIST -- the per-task permutation, derived from the run seed
#   HAR            -- the subject partition. The ledger records that the
#                     as-executed partition is 1104af185c87 and that
#                     REPRODUCTION REQUIRES x86 (an exact tie broken differently
#                     on ARM). Modal is x86; a local ARM relaunch is not the
#                     same experiment and must not be pooled with these.
E16_SEEDS_5 = [42, 1337, 2024, 7, 1234]
E16_ARM = ["--baseline", "lwf_lh17"]        # NOT --mafc-arm lwf (see below)

# MNIST = the OFF/vanilla configuration the input-path table compares against.
E16_MNIST = CFG + ["--benchmark", "permuted", "--no-adapters"]
# HAR pinned to the OFF arm (contract issue 9): bridging's 0.4812 -> 0.0886 is OFF.
E16_HAR = CFG + ["--benchmark", "har", "--no-adapters"]


def _require_lwf_support() -> None:
    """Refuse to build an E16 job list until train.py actually implements the arm.

    WHY A GUARD AND NOT A COMMENT. `--mafc-arm lwf` ALREADY EXISTS in this repo
    and means something else entirely — "both terms but RAW probe inputs (H3
    control)", a MAFC ablation. That collision is catch 30 pre-loaded: a launcher
    written from memory would produce a MAFC control arm wearing the field's
    baseline's name, and every downstream table would inherit it. The new arm is
    therefore `--baseline lwf_lh17`, with the citation in the name so the same
    collision cannot form again silently.

    Until that flag is implemented, argparse would reject these jobs one container
    at a time, after spawn, as N separate failures. This fails once, locally,
    before anything is submitted.
    """
    import pathlib as _p
    src = _p.Path(__file__).parent / "scripts" / "train.py"
    if "lwf_lh17" not in src.read_text():
        raise SystemExit(
            "E16 is not launchable yet: scripts/train.py does not implement "
            "`--baseline lwf_lh17`.\n"
            "  Do NOT substitute `--mafc-arm lwf` — that is the H3 raw-probe "
            "ablation, a different arm (contract sec 4b).\n"
            "  Implement the arm, fire the lambda=0 positive control "
            "(contract sec 4c), then launch.")


def _e16(base: list[str], lam: float, seed: int, tag: str,
         era: bool = True) -> tuple[list[str], str]:
    """One E16 run. Era checkpoints ON BY DEFAULT — contract sec 3, issue 2:
    the decomposition needs acc_ceiling from the ERA checkpoint, so a run
    launched without them cannot be decomposed later. The sweep arms get them
    too, because any configuration may become the compared one (catch 20)."""
    flags = base + E16_ARM + ["--lwf-lambda", str(lam), "--seed", str(seed)]
    if era:
        flags += ["--era-checkpoints", "--checkpoint-dir", f"/runs/ckpt_{tag}"]
    return (flags, tag)


def jobs_e16sweep_mnist() -> list[tuple[list[str], str]]:
    """sec 1 lambda sweep, MNIST — 4 lambda x 3 seeds. Selection is on AVG."""
    _require_lwf_support()
    return [_e16(E16_MNIST, lam, s, f"e16_mnist_lam{lam}_seed{s}")
            for lam in E16_LAMBDAS for s in SEEDS]


def jobs_e16sweep_har() -> list[tuple[list[str], str]]:
    """sec 1 lambda sweep, HAR/OFF — 4 lambda x 3 seeds."""
    _require_lwf_support()
    return [_e16(E16_HAR, lam, s, f"e16_har_lam{lam}_seed{s}")
            for lam in E16_LAMBDAS for s in SEEDS]


def jobs_e16head_mnist(lam: float = None) -> list[tuple[list[str], str]]:
    """sec 1 headline, MNIST — the sweep winner at 5 seeds.

    `lam` is REQUIRED at call time and has no default value that trains: the
    winner is selected from runs/e16_sweep.json on AVG, and hardcoding a lambda
    here would be selecting a hyperparameter in the launcher instead of from the
    measurement (the EWC lambda=200 lesson).
    """
    _require_lwf_support()
    if lam is None:
        raise SystemExit("jobs_e16head_mnist needs --lam: the AVG-selected "
                         "winner from runs/e16_sweep.json (contract sec 1)")
    return [_e16(E16_MNIST, lam, s, f"e16_mnist_head_seed{s}")
            for s in E16_SEEDS_5]


def jobs_e16head_har(lam: float = None) -> list[tuple[list[str], str]]:
    """sec 1 headline, HAR/OFF — the sweep winner at 5 seeds."""
    _require_lwf_support()
    if lam is None:
        raise SystemExit("jobs_e16head_har needs --lam: the AVG-selected "
                         "winner from runs/e16_sweep.json (contract sec 1)")
    return [_e16(E16_HAR, lam, s, f"e16_har_head_seed{s}")
            for s in E16_SEEDS_5]


def jobs_e16ctrl() -> list[tuple[list[str], str]]:
    """sec 4c POSITIVE CONTROL — lambda=0 must reproduce the vanilla arm EXACTLY.

    A certificate that has only ever passed is indistinguishable from one that
    cannot fail (catch 25), so the correctness gate ships with the case where it
    must fail. Non-tautological: it fails if the distillation term is wired into
    the wrong branch, or scaled where it should be added.

    THE RNG CAVEAT IS PART OF THE CONTROL. The teacher's forward pass can consume
    draws from the global generator and move the student's trajectory for an
    entirely benign reason — the control would then fire for the wrong cause and
    be waved through as "expected nondeterminism". The teacher pass must use a
    separate torch.Generator, and this run must be BIT-IDENTICAL to the vanilla
    arm's recorded matrix. Anything less is not this control.
    """
    _require_lwf_support()
    # warm-up OFF, deliberately. The control isolates the DISTILLATION wiring;
    # warm-up is a separate protocol element that legitimately changes training,
    # so leaving it on would make the run differ from the vanilla arm for a
    # reason that has nothing to do with the term under test, and the control
    # would fail honestly and uninformatively.
    #
    # THE CONTROL RUNS ON THE `mafc_off` ARM, WHICH HAS A PROVEN ZERO FLOOR
    # (four runs, two waves, 15/15 bit-identical at 4 threads). This matters
    # more than it looks: a bit-identity control is UNACHIEVABLE on an arm whose
    # own floor is nonzero -- `plain_lstm` reproduces 0/15 at default threading,
    # so the same control there would fail for reasons unrelated to LwF and be
    # uninterpretable. Match the control to an arm that can pass it.
    #
    # Target: `runs/e4off_v2_seed42` (AVG 0.4447), same flags minus the LwF ones.
    ctrl = ["--lwf-warmup-epochs", "0"]
    return [(_e16(E16_MNIST, 0.0, 42, "e16_mnist_lam0_seed42", era=False)[0] + ctrl,
             "e16_mnist_lam0_seed42"),
            (_e16(E16_HAR, 0.0, 42, "e16_har_lam0_seed42", era=False)[0] + ctrl,
             "e16_har_lam0_seed42")]


def jobs_e16floor_match(lam: float = None) -> list[tuple[list[str], str]]:
    """E16's VERDICT floors: both benchmarks at lambda, at the threading the
    reported runs actually used (4 threads, default).

    SUPERSEDES the pin-the-floor rule. Wave 1 concluded "floor pairs run pinned"
    on the theory that single-threading removes a BLAS reduction-order
    nondeterminism. HAR then produced the exact inverse: `mafc_off` is
    bit-identical at 4 threads (15/15, and it reproduces two-week-old runs
    exactly) and NONDETERMINISTIC at 1 thread (0/15, the two pinned runs
    disagreeing with each other and sitting up to 22pp from the 4-thread runs).

    A thread-count-monotone mechanism cannot produce both. The mechanism story
    is retracted; the rule keeps only what was measured. PINNING IS AN AXIS, NOT
    A STABILIZER -- so a floor is measured at the threading of the runs it
    bounds, or it bounds a configuration nobody reported.

    Note the deliberate name: `floor4t`, NOT `floor_`, so these do not match the
    `e16_mnist_floor_` prefix that routes to `train_one_1thread`.
    """
    if lam is None:
        raise SystemExit(
            "jobs_e16floor_match needs --lam: the AVG-selected winner. A floor "
            "at a guessed lambda bounds a configuration this experiment does "
            "not report.")
    # ROUTED THROUGH `_e16()`, NOT REBUILT. The first version of this function
    # composed its own flag list -- every flag it wrote was correct, and it
    # silently dropped `--era-checkpoints`, which `_e16()` appends by default
    # precisely so no run can lose it. The floor pair then measured a DIFFERENT
    # configuration from the sweep it was supposed to bound (HAR: floor runs
    # 0.5853 twice, sweep seed 42 0.5686 -- 1.67pp apart with the pair
    # bit-identical to itself).
    #
    # Catch 32's shape in a launcher: a parallel implementation escaping an
    # audited path, invisible at the call site because nothing WRITTEN is wrong,
    # only what is missing. Helpers that exist to enforce defaults are
    # load-bearing; bypassing one is a silent configuration change.
    #
    # Structural irony, for the memo: the dropped flag is `--era-checkpoints`,
    # default-on because of ISSUE 2 of this same contract's review. The fix for
    # one catch was bypassed by an instance of another, three days apart.
    out = []
    for tag, base in (("mnist", E16_MNIST), ("har", E16_HAR)):
        for t in ("a", "b"):
            # `floor4tec` = 4 threads + era checkpoints. NEW NAME, not a reuse
            # of `floor4t`: those runs are the EVIDENCE for the bypass diagnosis
            # (bit-identical to each other, 1.67pp from the sweep) and
            # overwriting them would destroy the artifact that proves the defect.
            out.append(_e16(base, lam, 42, f"e16_{tag}_floor4tec_lam{lam}_{t}"))
    return out


def jobs_e16floor_pinned(lam: float = None) -> list[tuple[list[str], str]]:
    """E16's MNIST floor pair at a NONZERO lambda, THREAD-PINNED (Wave 1 rule).

    Why a new pair when lambda=0 already reproduced bit-identically: nonzero
    lambda adds distillation gradients to the compute graph, which is a new
    (arm x config) pair under Wave 1's rule. Determinism measured at lambda=0
    does not transfer to lambda=4 any more than the mafc arm's zero floor
    transferred to plain_lstm. The lambda=0 control certified the WIRING; this
    certifies the CONFIGURATION the sweep will actually report.

    PINNED per the forward rule: floor pairs run at one thread (2.2x cost, and
    floors are cheap runs), comparison triples may run at default threading
    carrying whatever floor their threading earns.

    Lambda is REQUIRED -- no launcher default may select the reported config.
    Run it at the sweep's AVG-selected winner once runs/e16_sweep.json exists;
    running it at a guessed lambda measures the floor of a config we do not
    report.
    """
    if lam is None:
        raise SystemExit(
            "jobs_e16floor_pinned needs --lam: the AVG-selected winner from "
            "runs/e16_sweep.json. A floor measured at the wrong lambda is the "
            "floor of a configuration this experiment does not report.")
    return [(E16_MNIST + E16_ARM + ["--lwf-lambda", str(lam), "--seed", "42"],
             f"e16_mnist_floor_lam{lam}_{t}") for t in ("a", "b")]


def jobs_e16floor() -> list[tuple[list[str], str]]:
    """sec 1 reproduction floor for THIS arm — identical config relaunched.

    Requires the headline lambda, same reason as jobs_e16head_*; read it from
    runs/e16_sweep.json and pass --lam. Relaunch both sides of any comparison
    whose partner has no same-platform floor: an asymmetric relaunch cleans one
    side and biases the delta (catch 19).
    """
    raise SystemExit("jobs_e16floor: pass the AVG-selected lambda explicitly, "
                     "then relaunch the headline config unchanged. The floor is "
                     "measured on the config that is reported, not on the sweep.")


def jobs_e16combo() -> list[tuple[list[str], str]]:
    """sec 5b OPTIONAL fourth arm — LwF + bridging on HAR/OFF, 3 seeds.

    Composable, not alternatives: LwF changes training, bridging repairs the
    readout at read time on an unchanged trunk. FIRST THING CUT if the day runs
    long — it extends the result rather than securing it.
    """
    raise SystemExit("jobs_e16combo is the optional arm (contract sec 5b) and is "
                     "launched only after the headline rows land, with the "
                     "AVG-selected lambda passed explicitly.")


def jobs_e16ref() -> list[tuple[list[str], str]]:
    """sec 2a — re-run the cited reference numbers that the audit could not resolve.

    The audit found: ER-100 0.7916 CONFIRMED (runs/er_seed*), vanilla 0.4399 and
    EWC 0.4330 traceable only to runs/drift/MEMO.md, and adapters 0.9331 DOES NOT
    REPRODUCE LOCALLY — runs/e4_on_seed42 is absent and the two surviving seeds
    average 0.9261. "A number appears in a memo" is not "a number has an
    artifact": the same cheap-signal substitution as counting result directories.

    Only the adapters seed is mechanically recoverable from this launcher; the
    EWC/vanilla per-seed dirs need their original configuration identified first,
    which is a reading task, not a launch.
    """
    return [(CFG + ["--benchmark", "permuted", "--seed", "42"], "e4_on_seed42")]


def jobs_e16ref2() -> list[tuple[list[str], str]]:
    """sec 2a WHOLESALE SUPERSESSION — the adapters reference arm, all 3 seeds,
    under one configuration that every artifact RECORDS.

    WHY ALL THREE AND NOT THE TWO THAT ARE MISSING. The ruling asked for seeds
    1337 and 2024. Re-running 42 as well costs nothing in wall clock (three runs
    and two runs are both one wave) and is the only thing that separates the two
    live readings of the 2.91pp divergence:

      * `e4_on_seed42` (today, config RECORDED)   vs
      * `e4on_v2_seed42` (this job, same config)  = the REPRODUCTION FLOOR,
        measured on this exact path, seed and platform.

    Without that pair we would have three verified runs and still no way to say
    whether 0.9234-vs-0.9331 is noise or configuration -- which is the entire
    question. With it, the floor is measured and the residual is attributable.
    Catch 19's rule that the floor is a property of (seed x config x platform x
    BLAS) is why last night's E14 floor cannot be borrowed here.

    NEW DIRECTORY NAMES, old ones preserved. `e4_on_seed{1337,2024}` and
    `fullrank_ref` record no `arm` field and are the EVIDENCE for catch 34; they
    are not overwritten. Uniform naming across all three verified runs is the
    direct lesson of that catch -- `fullrank_ref` cost an hour of forensics and
    then turned out to be, on its own name's evidence, a different study's
    reference arm.
    """
    return [(CFG + ["--benchmark", "permuted", "--seed", str(s)],
             f"e4on_v2_seed{s}") for s in SEEDS]


def jobs_e16ref3() -> list[tuple[list[str], str]]:
    """sec 2a — the VANILLA/OFF reference arm, 3 seeds, config recorded.

    The other half of the comparison. jobs_e16ref2 superseded the adapters row
    (0.9331 -> 0.9183); the attribution (+50.3pp) is a DIFFERENCE, so it stays
    quarantined until both sides are measured at the same code state. Verifying
    one side and subtracting the other side's historical number would be an
    asymmetric relaunch in arithmetic form -- one clean operand, one dirty, and
    a delta biased in an unknown direction (catch 19's rule, applied to a
    subtraction rather than to a pair of runs).

    Same three seeds, same current HEAD. The old `e4_off_seed*` runs record no
    `arm` field -- arm-recording postdates them -- so, on a pipeline now MEASURED
    bit-deterministic (15/15 cells, max |delta| 0.000000, seed 42), they are not
    merely unverified: they are certainly a different code state.
    """
    return [(CFG + ["--benchmark", "permuted", "--no-adapters", "--seed", str(s)],
             f"e4off_v2_seed{s}") for s in SEEDS]


# ------------------------------------------------- Wave 1 (reference closure) --
# Contract: docs/W1_reference_closure.md. CPU, matching the arms these will sit
# beside (the E16 device rule). ERA CHECKPOINTS DELIBERATELY OFF -- see the
# deviation note in jobs_w1_vanilla.

W1_BASE = ["--config", "configs/mafc_phase1.yaml", "--benchmark", "permuted"]


def jobs_w1_vanilla() -> list[tuple[list[str], str]]:
    """1a -- the `plain_lstm` reference arm, 3 seeds, config recorded.

    NOT `mafc_off`. Those are two different arms one point apart (0.4399 vs
    0.4304) and the attribution uses the second; averaging them into a single
    "vanilla" is the hazard 1b exists to remove. Both get reported by name.

    DEVIATION FROM THE CONTRACT, RECORDED RATHER THAN ABSORBED: era checkpoints
    are OFF. The contract asked for them on both arms, but `_save_era_checkpoint`
    audits by loading through `PLCM.load_era`, which is PLCM-specific -- on
    `lstm`/`lstm_ewc` that clause cannot succeed, so it would have written
    checkpoints that fail their own audit. Making the audit model-aware is a
    shared-trainer change with blast radius across every experiment (~1-2h plus
    a regression pass) and is logged as a NAMED, PRICED, UNSCHEDULED follow-up.
    The honest consequence: if these arms ever need decomposition they re-run
    under a model-aware audit -- we do not pretend today's checkpoints would
    have been audit-clean.
    """
    return [(W1_BASE + ["--model", "lstm", "--seed", str(s)],
             f"w1_vanilla_seed{s}") for s in SEEDS]


def jobs_w1_ewc() -> list[tuple[list[str], str]]:
    """1a -- EWC at lambda=200, the historically cited value, 3 seeds.

    lambda=200 ONLY in this wave. The sweep behind the EWC null is 1c's
    RETRIEVAL job; if 1c finds the sweep unciteable, re-running it is a named
    follow-up (~12 CPU runs), not a silent extension of this wave.

    Arm-identity witness now exists in the artifacts (Wave-1 fixes 1 and 2):
    `ewc_penalty_mean` in every epoch dict, and `ewc_lambda` +
    `ewc_param_filter` in `arm_provenance`. READ-TIME ASSERT IS "NONZERO ON
    TASKS >= 1", not "nonzero everywhere": task 0 has no prior Fisher, so a zero
    there is correct by construction and is itself informative -- it confirms
    the penalty activates exactly when the mechanism says it should.
    """
    return [(W1_BASE + ["--model", "lstm_ewc", "--ewc-lambda", "200",
                        "--seed", str(s)], f"w1_ewc_l200_seed{s}") for s in SEEDS]


def jobs_w1_floor() -> list[tuple[list[str], str]]:
    """1a -- floor pair: `plain_lstm` seed 42 relaunched at TODAY's HEAD.

    Determinism was proven at the adapter wave's code state; HEAD has moved
    since (Wave-1 fixes 1 and 2, additive and regression-checked, but the point
    of a floor is not to trust that). If this reads 0.000000 the wave inherits
    the determinism claim and per-seed deltas against history are attributable
    to code state. IF IT READS NONZERO, STOP AND REPORT -- something changed in
    the path, and that finding outranks the wave.
    """
    return [(W1_BASE + ["--model", "lstm", "--seed", "42"],
             "w1_vanilla_seed42_rep2")]


def jobs_w3c_fullrank() -> list[tuple[list[str], str]]:
    """Wave 3c -- the storage table's full-rank cell, 3 seeds, recorded config.

    PRICED AT 3 RUNS, NOT 9. The sec 5.5 provenance pass found `lowrank_r32` and
    `lowrank_r64` DO have sibling seed dirs (`*_seed1337`, `*_seed2024`), so
    those two cells are a RECOMPUTE from existing artifacts plus an
    arm-provenance caveat -- no new compute. Only the full-rank cell is an
    orphan: `runs/fullrank_ref` has no siblings.

    That orphanhood resolves the run's identity as a side effect. A lone
    reference run with no seed set, sitting beside two arms that have them, is
    the shape of a STUDY REFERENCE -- consistent with the name. It was never an
    `e4_on` seed; it was the storage study's own full-rank arm, cited into the
    adapter table by an arithmetic coincidence (catch 34).

    Same run, two tables, one legitimate home -- and this job gives the
    legitimate one a verified replacement so the illegitimate citation can be
    retired without taking a real result down with it.
    """
    # `--adapter-rank 0` passed EXPLICITLY though it is already the default.
    # VERIFIED, not assumed: `configs/mafc_phase1.yaml` has no `rank` key, the
    # code documents 0 as "full-rank d x d", and a built adapter has 614,656
    # params = 784^2 exactly. It resolves correctly -- and it is spelled out
    # anyway so the arm's identity is in the command line rather than in a
    # default that a future config edit could move underneath it. Two-vanillas
    # rule, extended: the rank is part of the arm name.
    return [(W1_BASE + ["--model", "mafc", "--mafc-arm", "lambda0",
                        "--adapter-rank", "0", "--seed", str(s)],
             f"w3c_fullrank_r0_seed{s}") for s in SEEDS]


def jobs_w2d_mafc() -> list[tuple[list[str], str]]:
    """W2 diagnosis, cell A -- a SECOND `mafc` seed-42 pair at default threading.

    The existing mafc pair reproduced 15/15. One pass is not evidence of
    determinism, it is one observation of it -- and the whole reason this wave
    exists is that "the path is deterministic" was generalized from exactly that
    one pair. This tests whether the mafc pass replicates or was luck.
    """
    return [(W1_BASE + ["--model", "mafc", "--mafc-arm", "lambda0", "--seed", "42"],
             f"w2d_mafc_seed42_{t}") for t in ("a", "b")]


def jobs_w2d_lstm1t() -> list[tuple[list[str], str]]:
    """W2 diagnosis, cell B -- a `plain_lstm` seed-42 pair at ONE thread.

    The default-threaded plain_lstm pair already failed 0/15 (max |delta|
    0.1060), so it is not repeated -- that cell of the 2x2 is measured. This
    supplies the single-threaded cell. Routed to `train_one_1thread`.
    """
    return [(W1_BASE + ["--model", "lstm", "--seed", "42"],
             f"w2d_lstm1t_seed42_{t}") for t in ("a", "b")]


W1_HAR = ["--config", "configs/mafc_phase1.yaml", "--benchmark", "har",
          "--model", "mafc", "--mafc-arm", "lambda0", "--no-adapters"]


def jobs_w1_har_off() -> list[tuple[list[str], str]]:
    """Reference closure for the HAR OFF arm, 3 seeds, config recorded.

    THREE E16 CLAUSES BLOCK ON THIS ONE MISSING ARTIFACT: the lambda=0 control
    had no bit-identity target, the HAR sweep's competence floor had no
    reference DIAG, and E16's HAR comparison rows have no baseline. The existing
    `e5_har_noadapt_seed*` runs record no `arm` field and fall under the era
    presumption.

    PREDICTION ON RECORD (not a reference): the era-presumption runs read DIAG
    0.9133 / 0.9130 / 0.9037, mean ~0.9100. That glimpse is a PREDICTION until
    these land; the wave memo notes whether it is confirmed. Keeping the glimpse
    honest costs nothing and buys a calibration point.
    """
    return [(W1_HAR + ["--seed", str(x)], f"w1_har_off_seed{x}") for x in SEEDS]


def jobs_w1_har_off_floor() -> list[tuple[list[str], str]]:
    """The HAR OFF floor pair, THREAD-PINNED per Wave 1's forward rule.

    Pin where you are measuring the instrument; stamp-and-state where you are
    measuring the effect. This is the instrument.
    """
    return [(W1_HAR + ["--seed", "42"], f"w1_har_off_floor_{t}")
            for t in ("a", "b")]


def _with_era(flags: list[str], tag: str) -> tuple[list[str], str]:
    """Append era-checkpoint flags and ASSERT they are there.

    `_e16()` already does this for LwF arms; this is its counterpart for arms
    that carry no `--baseline`. The assert is the point: the last time a job
    builder composed its own flag list, every flag it WROTE was correct and it
    silently dropped `--era-checkpoints`, and the resulting floor pair bounded a
    configuration the sweep never ran. Structural guarantee beats care.
    """
    out = flags + ["--era-checkpoints", "--checkpoint-dir", f"/runs/ckpt_{tag}"]
    assert "--era-checkpoints" in out, f"{tag}: era flag missing"
    return (out, tag)


def jobs_w1_ref_ec() -> list[tuple[list[str], str]]:
    """The two reference arms re-run WITH era checkpoints — matched config.

    WHY: the E16 verdicts compare LwF arms (era ckpts ON) against reference arms
    (era ckpts OFF), and that axis is now MEASURED at up to 2.13pp with the sign
    varying by benchmark (MNIST +2.13, HAR -1.67). The mixed comparison is
    conservative -- matched-config seed 42 reads -2.89pp (MNIST) and -5.42pp
    (HAR) versus -2.45/-2.86 as reported -- so the direction is safe either way.

    FIX BEATS DISCLOSE. A reviewer meeting "mixed on a configuration axis worth
    2pp" discounts the table regardless of which way the confound runs, and the
    disclosure costs more than 40 minutes of compute. Second-order gain: the
    n=5 heads then inherit a clean comparison instead of carrying the caveat
    into the headline row.
    """
    out = []
    for x in SEEDS:
        out.append(_with_era(W1_BASE + ["--model", "mafc", "--mafc-arm", "lambda0",
                                        "--no-adapters", "--seed", str(x)],
                             f"e4off_ec_seed{x}"))
        out.append(_with_era(W1_HAR + ["--seed", str(x)], f"w1_har_off_ec_seed{x}"))
    return out


E16_DEC_ARMS = ("lam0.25", "lam4.0", "lam16.0", "mafc_off")


def jobs_e17dec() -> list[tuple[list[str], str]]:
    """E17 decomposition — the MLP arm, MNIST only (that is where it ran)."""
    return [(["scripts/e16_decompose.py", "--arm", a, "--bench", "mnist",
              "--device", "cpu", "--ckpt-root", "/runs",
              "--out", f"/runs/e16_decomp/mnist_{a}.json"], f"e17dec_mnist_{a}")
            for a in ("mlp", "mlp_floor")]


# ------------------------------------------------------- E17 (MLP share) ----
# Contract: E17, signed. A claim-integrity repair: "four architectures" is in
# the ledger one-liner, the E14 row, Figure 1 rev 1 and both shipped emails,
# and no MLP decomposition artifact exists.
#
# PRECONDITION 0 IS NOT NEEDED AND IS STRUCK FROM THE SIGNED CONTRACT. It
# assumed `--model mlp` and a model-aware era-checkpoint audit, because
# `PLCM.load_era` "reconstructs the wrong class for non-PLCM models". The MLP is
# not a non-PLCM model: it is a `--backbone` (train.py's choices are
# {lstm, mlp, vit, resnet}), the arm builds as `PLCM(backbone="mlp")` at
# plcm.py:222, and `runs/mlp_adapt_seed42` records `model_type: mafc`. So
# `load_era` reconstructs exactly the right class, the same path E12/E14/E16
# already use. Wave 1 struck era checkpoints for `--model lstm`/`lstm_ewc`,
# which ARE a different class; generalizing that to "non-PLCM" was a property
# of two model types read as a property of a category.
#
# CAPACITY VERIFIED, not assumed: hidden_size 256 from mafc_phase1.yaml gives an
# encoder of 266,752 params -- exactly the arm docs/BRITTLENESS_prereg.md names.

E17_MLP = W1_BASE + ["--model", "mafc", "--mafc-arm", "lambda0",
                     "--backbone", "mlp", "--no-adapters"]


def jobs_e17_mlp() -> list[tuple[list[str], str]]:
    """E17 -- the MLP scratch arm, 3 seeds, Permuted MNIST, era checkpoints ON.

    NOT pooled with the LSTM band. The scratch evidence stays per-dataset after
    this wave (LSTM arms on HAR, MLP on Permuted MNIST); no pooled point on any
    figure row, so the 77.0% failure is not repeated one level up.
    """
    return [_with_era(E17_MLP + ["--seed", str(x)], f"e17_mlp_seed{x}")
            for x in SEEDS]


def jobs_e17_floor() -> list[tuple[list[str], str]]:
    """E17 floor pair -- seed 42 twice, at the threading the runs use.

    The MLP arm has NO determinism record. It is not inherited from the other
    MNIST arms: `mafc_off` is bit-identical at 4 threads and `plain_lstm` is
    0/15 on the same benchmark, so a per-arm measurement is the only thing that
    licenses reading the share's delta. Default threading, matching the runs it
    bounds -- pinning is an axis, not a stabilizer.
    """
    return [_with_era(E17_MLP + ["--seed", "42"], f"e17_mlp_floor_{t}")
            for t in ("a", "b")]


# ------------------------------------ S72: E16 sec 7.2 closure, C0deg step 3 ----
# Ruling (runs/MEMO_c0deg.md sec 6, 2026-09-15): "the E16 sec 7.2 relaunch pair,
# both arms uniform era status, three-way row" -- LwF / C3 (bridging) / C0deg on
# the bridging benchmark.
#
# THE PAIR DIFFERED ON TWO AXES, NOT ONE. The ruling named era status. The
# artifacts' own `arm` field says the LwF HAR arm and its floor pair ran
# `benchmark: 'har'` -- the E5 shared-window construction -- while bridging's
# 0.4812 -> 0.0886 is `har_subject` (E10). The E16 contract pinned "UCI HAR
# (subject-disjoint, E10 construction)" (docs/E16_lwf_prereg.md sec 2) and
# E16_HAR above wrote `--benchmark har`. Catch 30's shape on the benchmark axis:
# the launcher's flag, not the contract's sentence, is what ran, and nothing
# inspected the artifact's `benchmark` field against the contract's words until
# this closure tried to use both arms in one row. So a uniform pair means BOTH
# arms on `har_subject`, and moving LwF to a benchmark it was never swept on
# means its lambda is selected outside the measurement unless the sweep moves
# with it (the EWC lambda=200 rule). The sweep moves with it.
#
# UNIFORM FLAGS, all jobs in this wave: `--era-checkpoints` (the arm-identity
# axis) AND `--fp32-shadow`. The shadow is not optional here: era checkpoints
# are fp16 and the cure screen's consistency gate demands acc_orig == the
# matrix's final row to 1e-6, which an fp16 reload misses by one window
# (e16_har_lam0.25_seed42 task 0: +0.00034). The shadow reloads and refits
# INSIDE the training loop exactly as the era feature does, so it is part of
# the run's configuration for the same reason, and it goes on every arm of the
# comparison or on none. Default threading (4), matching what the row bounds.
#
# `--save-checkpoints` is NOT passed: the trainer asserts it collides with era
# checkpoints on the same file names.

S72_OFF = CFG + ["--benchmark", "har_subject", "--no-adapters"]


def _s72(flags: list[str], tag: str) -> tuple[list[str], str]:
    """Era checkpoints + fp32 shadow, ASSERTED present. Routed through
    `_with_era` so the era flag cannot be dropped by a rebuilt flag list."""
    out, tag = _with_era(flags, tag)
    out = out + ["--fp32-shadow"]
    assert "--era-checkpoints" in out and "--fp32-shadow" in out, f"{tag}: wave flags missing"
    assert "--save-checkpoints" not in out, f"{tag}: collides with era checkpoints"
    return (out, tag)


def jobs_s72_off() -> list[tuple[list[str], str]]:
    """The bridging arm, relaunched: E10 OFF on har_subject, 3 seeds, era +
    shadow, plus its floor pair (seed 42 twice, default threading). These runs
    also bring the bridging benchmark's headline arm under arm recording --
    `runs/e10_off_seed*` predate it and carry no `arm` field."""
    out = [_s72(S72_OFF + ["--seed", str(s)], f"e10off_ec_seed{s}") for s in SEEDS]
    out += [_s72(S72_OFF + ["--seed", "42"], f"e10off_ec_floor4tec_{t}") for t in ("a", "b")]
    return out


def jobs_s72_lwf() -> list[tuple[list[str], str]]:
    """LwF (lwf_lh17) on har_subject OFF: the full E16 lambda sweep (4 x 3
    seeds), era + shadow, plus a floor pair at lambda 0.25 -- the winner on the
    OTHER construction, carried as a prior, not a selection. If AVG selects a
    different lambda here, a floor pair at that lambda is launched then and the
    0.25 pair is logged as a floor observation. Selection on AVG, competence
    floor against jobs_s72_off's DIAG, full sweep reported (E16 sec 1)."""
    _require_lwf_support()
    lwf = lambda lam, s: S72_OFF + E16_ARM + ["--lwf-lambda", str(lam), "--seed", str(s)]
    out = [_s72(lwf(lam, s), f"e16s_har_lam{lam}_seed{s}")
           for lam in E16_LAMBDAS for s in SEEDS]
    out += [_s72(lwf(0.25, 42), f"e16s_har_floor4tec_lam0.25_{t}") for t in ("a", "b")]
    return out


def jobs_s72_lwf_floor(lam: float = None) -> list[tuple[list[str], str]]:
    """Floor pair at the AVG-selected lambda on har_subject. `lam` is REQUIRED
    and has no default that trains: `scripts/s72_row.py` selects it from the
    sweep artifacts (AVG, competence floor), and a default here would select a
    hyperparameter in the launcher instead of from the measurement. The sweep
    selected 1.0 where the shared-window construction had selected 0.25; the
    0.25 pair already ran and is logged as a floor observation."""
    if lam is None:
        raise SystemExit("jobs_s72_lwf_floor needs --lam: the AVG-selected winner from "
                         "runs/s72_row.json. A floor at a guessed lambda bounds a "
                         "configuration this row does not report.")
    return [_s72(S72_OFF + E16_ARM + ["--lwf-lambda", str(lam), "--seed", "42"],
                 f"e16s_har_floor4tec_lam{lam}_{t}") for t in ("a", "b")]


# ---- S72-band: the 66-88% band's three ON arms on har_subject, uniform flags ----
# Ruling 2026-09-16 (E20 sign-off, runs/MEMO_c0deg.md sec 6g). The band's four
# arms are E5-family checkpoints on `--benchmark har` (shared-window), E11-era,
# no `arm` field, era OFF. Under Ruling A the OFF member's citable version is
# the S72 relaunch (har_subject, era + shadow); a band mixing provenances is not
# a caption problem: the SAME OFF arm reads 66.2% on E11-era provenance and
# 80.4% on recorded provenance (runs/e16_decomp/har_mafc_off.json, era ON, on
# `har`). So all four arms move together: OFF is done (jobs_s72_off); these are
# v1-ON, v2-ON and v2-control with the flags jobs_e5diag / jobs_e5d recorded,
# on har_subject, through _s72 (era + fp32 shadow asserted), 3 seeds + a floor
# pair each. Arm identity is read from the artifacts' `use_input_adapters`,
# `adapter_mode`, `warmup_epochs`, `warmup_mode` fields, never from this list.
S72_ON = CFG + ["--benchmark", "har_subject"]          # adapters ON by default = v1-ON
S72_BAND = {"v1on": [], "v2on": ["--warmup-epochs", "1", "--warmup-mode", "adapter"],
            "v2ctl": ["--warmup-epochs", "1", "--warmup-mode", "control"]}


def jobs_s72_band() -> list[tuple[list[str], str]]:
    out = []
    for tag, extra in S72_BAND.items():
        out += [_s72(S72_ON + extra + ["--seed", str(s)], f"e10{tag}_ec_seed{s}") for s in SEEDS]
        out += [_s72(S72_ON + extra + ["--seed", "42"], f"e10{tag}_ec_floor4tec_{t}") for t in ("a", "b")]
    return out


# ------------------------------------------------ E18: disjoint-content MNIST ----
# docs/E18_prereg.md sec 1, sec 8; sign-off condition 1 clears these to run.
# Every job through _s72 (era + fp32 shadow asserted), plus --disjoint-content.
# The construction is read back from each artifact's `arm` (disjoint_content,
# content_fingerprint, angles), never from this list.
E18_PMD_MLP = E17_MLP + ["--disjoint-content"]                       # W1_BASE + mafc lambda0 + mlp + no-adapters
E18_PMD_LSTM = W1_BASE + ["--model", "mafc", "--mafc-arm", "lambda0", "--no-adapters", "--disjoint-content"]
E18_RMD_MLP = ["--config", "configs/mafc_phase1.yaml", "--benchmark", "rotated", "--model", "mafc",
               "--mafc-arm", "lambda0", "--backbone", "mlp", "--no-adapters", "--disjoint-content"]


def _e18(base, tag):
    out = [_s72(base + ["--seed", str(s)], f"{tag}_seed{s}") for s in SEEDS]
    out += [_s72(base + ["--seed", "42"], f"{tag}_floor4tec_{t}") for t in ("a", "b")]
    assert all("--disjoint-content" in f for f, _ in out)
    return out


def jobs_e18_pmd_mlp():
    return _e18(E18_PMD_MLP, "e18_pmd_mlp")


def jobs_e18_pmd_lstm():
    return _e18(E18_PMD_LSTM, "e18_pmd_lstm")


def jobs_e18_rmd_mlp():
    return _e18(E18_RMD_MLP, "e18_rmd_mlp")


def jobs_e23b_seqlen() -> list[tuple[list[str], str]]:
    """E23-B (docs/E23B_seqlen_prereg.md, registered before launch 2026-09-20): does E23's
    crux failure come from pretraining or from sequence length? Scratch LSTM on
    disjoint-content Permuted MNIST at (T=5, 3k/task) and (T=20, 3k/task) -- `content_chunks`
    holds images-per-task fixed while tasks-per-sequence varies, and (T=5, C=20) uses the same
    chunks and permutations as the first five tasks of (T=20, C=20). A1 (T=5, 12k) already
    exists as `e18_pmd_lstm`. Era + fp32 shadow via `_s72`, 3 seeds + a seed-42 floor replicate
    per arm, CPU x86."""
    out = []
    for tag, T in (("e23b_t5c20", 5), ("e23b_t20", 20)):
        base = E18_PMD_LSTM + ["--num-tasks", str(T), "--content-chunks", "20"]
        out += [_s72(base + ["--seed", str(s)], f"{tag}_seed{s}") for s in SEEDS]
        out += [_s72(base + ["--seed", "42"], f"{tag}_floor4tec_a")]
    assert all("--disjoint-content" in f and "--content-chunks" in f for f, _ in out)
    return out


def jobs_e18_gate() -> list[tuple[list[str], str]]:
    """E18 P0 build gate: retrain the bit-deterministic MNIST anchor
    (`w2d_mafc_seed42_{a,b}`, 25/25 identical, benchmark permuted, adapters ON,
    era OFF, 4 threads) on the AMENDED code with the flag OFF. It must reproduce
    both existing replicates 25/25, or the flag-off path changed and nothing in
    E18 launches. Same flags as jobs_w2d_mafc, one replicate."""
    return [(W1_BASE + ["--model", "mafc", "--mafc-arm", "lambda0", "--seed", "42"],
             "e18_gate_w2d_mafc_seed42")]


def jobs_e16dec() -> list[tuple[list[str], str]]:
    """Deck-close Item 1 — decomposition on LwF era checkpoints, 4 arms x 2 benches.

    REBUILT. The first version took `--bench` and `--seed` and no lambda, so it
    could not address the checkpoint directories it was meant to read: the
    contract's table names lambda in {0.25, 4, 16} and the job had no lambda
    dimension at all. It also pointed at `scripts/e16_decompose.py`, which did
    not exist. Catch 33 twice in one entry, both mine.

    THE REFERENCE ARM IS IN THE TABLE. The contract draft said `mafc_off` was
    "already decomposed"; it is not. The only LSTM-family decomposition on
    record is `runs/e11_e6b/decomp.json` -- HAR-only, E6b-lineage, different
    arm. H-D3 ("protection concentrates in F_read") is meaningless without the
    unprotected arm's split as denominator, so the reference is measured here,
    same script, same seeds, same era-checkpoint status.

    Seeds are internal to the script (all three per call), so one job per
    arm x bench cell: 8 jobs.
    """
    return [(["scripts/e16_decompose.py", "--arm", a, "--bench", b,
              "--device", "cpu", "--ckpt-root", "/runs",
              "--out", f"/runs/e16_decomp/{b}_{a}.json"],
             f"e16dec_{b}_{a}") for b in ("mnist", "har") for a in E16_DEC_ARMS]


# ---- E23 (docs/E23_prereg.md, signed v2 2026-09-18; option (b): era checkpoints, no shadow) ------------------
E23_VIT = E12_CFG + E12_BASE + ["--benchmark", "cifar100_permuted", "--era-checkpoints"]   # = jobs_e12effb + era ckpts
E23_RN = E14_CFG + ["--benchmark", "cifar100_permuted", "--era-checkpoints"]              # E14_CFG carries --task-heads --no-adapters


def jobs_e23_smoke() -> list[tuple[list[str], str]]:
    """E23 sec 2 build gate: one epoch, two tasks, seed 42, both backbones on
    cifar100_permuted with era checkpoints. Checked afterwards for the recorded
    `benchmark`, shift fingerprint (asserted at train time against the config's
    expected_shift -- E12's values, now also in configs/e14_resnet.yaml),
    `backbone`, `use_task_heads`, `era_checkpoints`. Nothing headline launches
    before this lands (catch 30: arm identity from the run's own artifact)."""
    return [(E23_RN + ["--seed", "42", "--epochs", "1", "--num-tasks", "2", "--checkpoint-dir", "/runs/ckpt_e23_rn_smoke"], "e23_rn_smoke"),
            (E23_VIT + ["--seed", "42", "--epochs", "1", "--num-tasks", "2", "--checkpoint-dir", "/runs/ckpt_e23_vit_smoke"], "e23_vit_smoke")]


def jobs_e23_vit() -> list[tuple[list[str], str]]:
    """E23 arm A-ViT: the B6 base arm (jobs_e12effb) relaunched WITH era checkpoints, 3 seeds."""
    return [(E23_VIT + ["--seed", str(s), "--checkpoint-dir", f"/runs/ckpt_e23_vit_seed{s}"], f"e23_vit_seed{s}") for s in _e12_seeds(3)]


def jobs_e23_rn() -> list[tuple[list[str], str]]:
    """E23 arm A-RN: ResNet-50 on the same cifar100_permuted construction, era checkpoints, 3 seeds."""
    return [(E23_RN + ["--seed", str(s), "--checkpoint-dir", f"/runs/ckpt_e23_rn_seed{s}"], f"e23_rn_seed{s}") for s in _e12_seeds(3)]


def jobs_e23_floor() -> list[tuple[list[str], str]]:
    """E23 C-FLOOR: seed 42 of each backbone relaunched identically (era checkpoints ON, as the reported runs)."""
    return [(E23_VIT + ["--seed", "42", "--checkpoint-dir", "/runs/ckpt_e23_vit_floor42"], "e23_vit_floor_seed42"),
            (E23_RN + ["--seed", "42", "--checkpoint-dir", "/runs/ckpt_e23_rn_floor42"], "e23_rn_floor_seed42")]


def jobs_e23_har_mlp() -> list[tuple[list[str], str]]:
    """E23 arm B: S72_OFF (har_subject, adapters off, era + fp32 shadow, 4 threads) with
    `--backbone mlp` -- the MLP reads the flattened 128 x 9 window via the opt-in
    `mlp_input_dim` (src/models/plcm.py; regression 2026-09-19: MNIST-MLP and HAR-LSTM
    checkpoints forward bit-identically to the pre-change model). 3 seeds + the seed-42
    floor replicate. CPU, x86 (the har_subject partition)."""
    out = [_s72(S72_OFF + ["--backbone", "mlp", "--seed", str(s)], f"e23_har_mlp_seed{s}") for s in SEEDS]
    out += [_s72(S72_OFF + ["--backbone", "mlp", "--seed", "42"], "e23_har_mlp_floor4tec_a")]
    return out


def jobs_e23_rpt() -> list[tuple[list[str], str]]:
    """E23 P2(b): the repeat-task control on cifar100_permuted, both backbones, 3 seeds,
    5 tasks -- jobs_e12rptb / jobs_e14rpt shape. Under repeat_task + shift the dataset now
    holds ONE non-identity B6 permutation on every task (src/data/split_cifar100.py,
    opt-in, 2026-09-19; regression: registered fingerprints unchanged, unshifted control
    unchanged), so identical data means identical content AND layout. Bar <= 0.05 absolute
    (E12 read 0.0040, E14 0.0010 unshifted)."""
    return ([(E12_CFG + E12_BASE + ["--benchmark", "cifar100_permuted", "--seed", str(s), "--repeat-task", "--num-tasks", "5"], f"e23_rpt_vit_seed{s}") for s in _e12_seeds(3)]
            + [(E14_CFG + ["--benchmark", "cifar100_permuted", "--seed", str(s), "--repeat-task", "--num-tasks", "5"], f"e23_rpt_rn_seed{s}") for s in _e12_seeds(3)])


def jobs_e23_main() -> list[tuple[list[str], str]]:
    """E23 headline: A-ViT (3) + A-RN (3) + C-FLOOR (seed 42 of each), ONE app of 8 L4
    jobs -- one `modal run` stream, after the build gate (jobs_e23_smoke, 2026-09-19:
    both backbones recorded the contracted arm and wrote era checkpoints)."""
    return jobs_e23_vit() + jobs_e23_rn() + jobs_e23_floor()


E28_HO = ["--aug-heldout", "/runs/e28/heldout.json"]


def _e28(extra, tag):
    """Every E28 training arm goes through _s72, so it matches S72 OFF on era
    checkpoints and fp32 shadow -- both arm-identity fields with measured effects."""
    return _s72(S72_OFF + E28_HO + extra, tag)


def jobs_e28_smoke() -> list[tuple[list[str], str]]:
    """E28 sec 2: mu selection on ALL THREE seeds, not one. Selection is on
    competence only; a single-seed selection is what the standing rules forbid."""
    return [_e28(["--aug-perm-p", "0.5", "--aug-perm-seed", str(s),
                  "--aux-perm-weight", str(mu), "--seed", str(s)],
                 f"e28_b2_mu{mu}_seed{s}")
            for mu in ("0.1", "1.0") for s in (42, 1337, 2024)]


E28B_EP = ["--epochs", "30", "--epochs-first", "30"]


def jobs_e28b() -> list[tuple[list[str], str]]:
    """E28b: is the competence cost capacity, or steps?

    B1 and B2 both fail E28's competence bar by ~0.3-0.4 on the worst task, and
    they fail it by the SAME amount (worst -0.374 vs -0.394), so the cost is the
    augmentation and not the auxiliary head. Every arm had 10 epochs per task;
    classifying under random channel shuffles is a harder problem, so the budget
    is the obvious confound.

    Four arms at 30 epochs per task, competence measured against B0 AT THE SAME
    BUDGET. B1 is included although the ruling named only B0 and B2: B1 and B2
    are currently indistinguishable on competence, so without B1 at the new
    budget E28b cannot tell equivariance from invariance -- which is the
    distinction the whole pilot exists to draw.

    BOTH epoch fields are set. `--epochs` alone leaves `epochs_first_task` at the
    config's 10 (trainer.py:243), which would give task 0 a third of the budget
    of every other task -- and task 0 is exactly where B1 and B2 lose most."""
    out = []
    for s in (42, 1337, 2024):
        out.append(_s72(S72_OFF + E28B_EP + ["--seed", str(s)], f"e28b_b0_ep30_seed{s}"))
        out.append(_e28(E28B_EP + ["--aug-perm-p", "0.5", "--aug-perm-seed", str(s),
                                   "--seed", str(s)], f"e28b_b1_ep30_seed{s}"))
        out.append(_e28(E28B_EP + ["--aug-perm-p", "0.5", "--aug-perm-seed", str(s),
                                   "--aux-perm-weight", "1.0", "--seed", str(s)],
                        f"e28b_b2_ep30_seed{s}"))
        out.append(_e28(E28B_EP + ["--aug-perm-p", "0.25", "--aug-perm-seed", str(s),
                                   "--aux-perm-weight", "1.0", "--seed", str(s)],
                        f"e28b_b2_p025_ep30_seed{s}"))
    return out


def jobs_e28_b1() -> list[tuple[list[str], str]]:
    """B1: augmentation only, labels unchanged. Expected to become INVARIANT."""
    out = [_e28(["--aug-perm-p", "0.5", "--aug-perm-seed", str(s), "--seed", str(s)],
                f"e28_b1_seed{s}") for s in (42, 1337, 2024)]
    out += [_e28(["--aug-perm-p", "0.5", "--aug-perm-seed", "42", "--seed", "42"],
                 "e28_b1_floor4tec_a")]
    return out


JOBS = {"e28smoke": jobs_e28_smoke, "e28b1": jobs_e28_b1, "e28b": jobs_e28b, "e23main": jobs_e23_main, "e23rpt": jobs_e23_rpt, "e23bseq": jobs_e23b_seqlen, "e23harmlp": jobs_e23_har_mlp, "e23smoke": jobs_e23_smoke, "e23vit": jobs_e23_vit, "e23rn": jobs_e23_rn, "e23floor": jobs_e23_floor,
        "e1": jobs_e1, "e5": jobs_e5, "e5diag": jobs_e5diag, "e5b": jobs_e5b,
        "e2cal": jobs_e2cal, "e2": jobs_e2, "e5d": jobs_e5d, "e10": jobs_e10, "e7": jobs_e7, "e6off": jobs_e6off, "e5dfloor": jobs_e5dfloor,
        "smoke": jobs_smoke, "e12p": jobs_e12p, "e12rpt": jobs_e12rpt,
        "e12pb": jobs_e12pb, "e12rptb": jobs_e12rptb, "e12": jobs_e12,
        "e12eff": jobs_e12eff, "e12effb": jobs_e12effb, "e14": jobs_e14, "e14floor": jobs_e14floor, "e14rpt": jobs_e14rpt,
        "e16sweepm": jobs_e16sweep_mnist, "e16sweeph": jobs_e16sweep_har,
        "e16ctrl": jobs_e16ctrl, "e16ref": jobs_e16ref, "e16floorpin": jobs_e16floor_pinned, "e16floor4t": jobs_e16floor_match, "e16ref2": jobs_e16ref2, "e16ref3": jobs_e16ref3,
        "w1vanilla": jobs_w1_vanilla, "w1ewc": jobs_w1_ewc,
        "w1floor": jobs_w1_floor, "w3cfullrank": jobs_w3c_fullrank, "w2dmafc": jobs_w2d_mafc, "w2dlstm1t": jobs_w2d_lstm1t,
        "w1haroff": jobs_w1_har_off, "e17mlp": jobs_e17_mlp, "e17floor": jobs_e17_floor, "w1refec": jobs_w1_ref_ec, "w1haroffloor": jobs_w1_har_off_floor, "e12cil": jobs_e12cil, "e12floor": jobs_e12floor,
        "s72off": jobs_s72_off, "s72lwf": jobs_s72_lwf, "s72lwffloor": jobs_s72_lwf_floor,
        "s72band": jobs_s72_band,
        "e18gate": jobs_e18_gate, "e18pmdmlp": jobs_e18_pmd_mlp, "e18pmdlstm": jobs_e18_pmd_lstm,
        "e18rmdmlp": jobs_e18_rmd_mlp}

# Experiments routed to the GPU function (CIFAR-scale only).
GPU_EXPERIMENTS = {"e23main", "e23rpt", "e23smoke", "e23vit", "e23rn", "e23floor", "e2cal", "e2", "e14", "e14floor", "e14rpt", "e12p", "e12rpt", "e12pb", "e12rptb", "e12",
                   "e12eff", "e12cil", "e12floor"}   # E12 is ViT-B/16 at 224px

# Per-JOB device routing. Coarser experiment-level routing forces one device on a
# whole batch, which is wrong whenever a batch mixes benchmarks — E5d does: its
# HAR arms MUST stay on CPU because their comparison baselines (e5diag_*,
# e5_har_noadapt_*) are existing CPU runs, while its MNIST arms are all produced
# in the same batch and could safely have gone to GPU.
#
# The rule that matters is not "use the fast device" but: **every arm of a
# comparison runs on the same device as the arms it is compared against.**
# Changing the device for one side is the infrastructure form of an asymmetric
# relaunch — it biases the delta by an unknown amount in an unknown direction,
# and on HAR we measured same-config spread reaching 5.5pp, so a device swap is
# not a safe no-op against a 15pp bar.
GPU_JOB_PREFIXES = ("e2_", "e2cal_", "e2floor_", "e12")


def _wants_gpu(experiment: str, log_dir: str) -> bool:
    return experiment in GPU_EXPERIMENTS or log_dir.startswith(GPU_JOB_PREFIXES)


@app.function(
    volumes={"/runs": volume},
    timeout=60 * 60 * 3,
    cpu=4.0,
    memory=16384,
)
def analyze(argv: list[str]) -> str:
    """Run an ANALYSIS script in the same container the training ran in.

    E11/step 2 found that E10 checkpoints re-evaluated locally miss their own
    recorded accuracy matrix by ~0.6-2.8pp, while E4's MNIST checkpoints (trained
    on the laptop) reproduce theirs EXACTLY.

    The first diagnosis was WRONG and is kept here as the correction. It looked
    like float non-associativity -- different BLAS, last bits apart, flipping
    argmaxes near ties -- and it survived the batch-size control (128 and 256
    agree cell-for-cell). The partition control killed it: `HARSubjectBenchmark`
    assigns subjects 2 and 5 to DIFFERENT training groups on x86 and ARM (they
    are an exact tie in the greedy balancer), task 0's train subjects set the
    calibration constants for ALL five tasks, and so every task's test tensor
    hashes differently between the two environments. The data differs; the
    arithmetic was never the problem.

    A recompute that supersedes a Modal-computed number must therefore run on
    Modal -- not to match its arithmetic, but because it is the only environment
    that reproduces the data the checkpoints were trained and evaluated on.
    """
    import os, subprocess, sys

    os.chdir("/repo")
    if not os.path.exists("/repo/data"):
        os.symlink("/data", "/repo/data")
    # The analysis scripts address checkpoints as runs/ckpt_*; the volume is the
    # authoritative copy, so point runs/ at it rather than syncing a second one.
    if not os.path.exists("/repo/runs"):
        os.symlink("/runs", "/repo/runs")
    cmd = [sys.executable, *argv]
    print("RUN:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    volume.commit()
    return (p.stdout or "") + ("\n--- STDERR ---\n" + p.stderr[-4000:] if p.returncode else "")


@app.function(
    volumes={"/runs": volume},
    timeout=60 * 60 * 4,
    gpu="L4",
    cpu=8.0,
    memory=32768,
)
def analyze_gpu(argv: list[str]) -> str:
    """GPU variant of `analyze`, for E12: ViT-B/16 over 60k images at 224x224 is
    a GPU job, while every prior analysis in this program was CPU-sized."""
    import os, subprocess, sys

    os.chdir("/repo")
    if not os.path.exists("/repo/data"):
        os.symlink("/data", "/repo/data")
    if not os.path.exists("/repo/runs"):
        os.symlink("/runs", "/repo/runs")
    cmd = [sys.executable, *argv]
    print("RUN(gpu):", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    volume.commit()
    return (p.stdout or "") + ("\n--- STDERR ---\n" + p.stderr[-4000:] if p.returncode else "")


@app.local_entrypoint()
def analysis_gpu(argv: str = "scripts/e12_frozen_probe.py"):
    """modal run modal_runner.py::analysis_gpu --argv "scripts/e12_frozen_probe.py" """
    print(analyze_gpu.remote(argv.split()))


@app.local_entrypoint()
def analysis(argv: str = "scripts/audit_checkpoints.py"):
    """modal run modal_runner.py::analysis --argv "scripts/three_column.py --benchmark har_subject" """
    print(analyze.remote(argv.split()))


@app.function(
    volumes={"/runs": volume},
    timeout=60 * 60 * 3,
    cpu=4.0,
    memory=8192,
    max_containers=16,
)
def analyze_one(argv: list[str], label: str) -> dict:
    """CPU twin of `analyze_one_gpu`: one analysis script per worker, stdout
    persisted to /runs/<label>.log. E21's sweep is 36 independent jobs of
    12 cells each (~51 min continuous, ~25 min swap); serial through `analyze`
    it would be 26 hours, fanned out it is ~2.5 hours of wall clock."""
    import os, subprocess, sys

    os.chdir("/repo")
    if not os.path.exists("/repo/data"):
        os.symlink("/data", "/repo/data")
    if not os.path.exists("/repo/runs"):
        os.symlink("/runs", "/repo/runs")
    print(f"RUN({label}):", " ".join(argv), flush=True)
    p = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
    with open(f"/runs/{label}.log", "w") as f:
        f.write(p.stdout or "")
        if p.returncode:
            f.write("\n--- STDERR ---\n" + (p.stderr or ""))
    volume.commit()
    if p.returncode:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
    return {"label": label, "ok": p.returncode == 0}




@app.function(
    volumes={"/runs": volume},
    timeout=60 * 60 * 4,
    gpu="L4",
    cpu=8.0,
    memory=8192,
    max_containers=16,
)
def analyze_one_gpu(argv: list[str], label: str) -> dict:
    """Run ONE analysis script on a GPU worker. The fan-out unit for E12's
    decomposition, which is 10 independent (arm, seed) jobs of ~117k ViT
    forwards each — serial they are hours, fanned out they are one job's worth.

    stdout is written to /runs/<label>.log as well as returned. `_run` discards
    training stdout on success, which is how E12's per-epoch forgetting
    trajectory and its fp16 boundary/reload lines were lost; analysis output is
    the evidence for a claim, so it is persisted rather than streamed.
    """
    import os, subprocess, sys

    os.chdir("/repo")
    if not os.path.exists("/repo/data"):
        os.symlink("/data", "/repo/data")
    if not os.path.exists("/repo/runs"):
        os.symlink("/runs", "/repo/runs")
    print(f"RUN({label}):", " ".join(argv), flush=True)
    p = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
    with open(f"/runs/{label}.log", "w") as f:
        f.write(p.stdout or "")
        if p.returncode:
            f.write("\n--- STDERR ---\n" + (p.stderr or ""))
    volume.commit()
    if p.returncode:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
    return {"label": label, "ok": p.returncode == 0}


def jobs_e12dec() -> list[tuple[list[str], str]]:
    """E12 decomposition — one job per (arm, seed), 10 total."""
    out = []
    for s in _e12_seeds(5):
        for arm in ("base", "adapt"):
            out.append((["scripts/e12_decompose.py", "--arm", arm,
                         "--seed", str(s), "--device", "cuda",
                         "--out", f"/runs/e12_decomp_v2/{arm}_{s}.json"],
                        f"e12dec_{arm}_{s}"))
    return out


def jobs_e12cildec() -> list[tuple[list[str], str]]:
    """H-V3 — the decomposition on the class-IL era checkpoints, 3 seeds.

    Under branch (C) this is no longer merely secondary: with H-V2 unreadable,
    whether the encoder-side mechanism holds in class-IL as well is what tells us
    the flip is setting-independent rather than an artifact of task-IL routing.

    Two things differ from the task-IL arms and are handled in the script rather
    than assumed away: P3b is INAPPLICABLE (one shared readout, so "which head"
    has no referent), and the deployed head scores 100 classes while the refit
    probe only ever discriminates a task's 5 — so `acc_orig_masked` restricts the
    deployed logits to the task's own classes, and the gap between masked and
    unmasked is the task-identity component the recency-bias literature names.
    """
    return [(["scripts/e12_decompose.py", "--arm", "cil", "--seed", str(s),
              "--device", "cuda", "--out", f"/runs/e12_decomp_cil_v3/cil_{s}.json"],
             f"e12cildec_{s}") for s in _e12_seeds(3)]


def jobs_e12cildec_v4() -> list[tuple[list[str], str]]:
    """Class-IL probe CONVERGENCE AUDIT — re-fit the 57 cells with the witness.

    An audit of existing artifacts: no retraining, no checkpoint regeneration,
    same stored ViT era checkpoints, same max_iter / C / solver / standardization.
    What is new is that every fit now records `n_iter_` into the row and the
    100-way probe carries the same guard `refit_probe` always had.

    Writes to `e12_decomp_cil_v4/`, NOT v3: the v3 files are the recorded values
    the paper cites (97.86% [97.27, 98.45]; 100-way refit 0.6393) and this run
    is compared AGAINST them. Overwriting the reference would destroy the
    comparison this audit exists to make.

    Reading rule, fixed before the numbers exist: a nonzero delta against v3 is
    checked against this arm's reproduction floor at the same platform and
    thread count BEFORE it is called a discrepancy; if no such floor exists, that
    is stated. And if the guard fires on any cell, STOP AND REPORT — raising
    max_iter and re-running would be a silent fix to a finding about a reported
    number.
    """
    return [(["scripts/e12_decompose.py", "--arm", "cil", "--seed", str(s),
              "--device", "cuda", "--out", f"/runs/e12_decomp_cil_v4/cil_{s}.json"],
             f"e12cildec_v4_{s}") for s in _e12_seeds(3)]


def jobs_e13() -> list[tuple[list[str], str]]:
    """E13 HAR half — H-S1's re-derivation of the frozen 0.347.

    Runs on Modal because that is where the E5 checkpoints live AND where HAR's
    executed data reproduces (the platform gate refuses a cross-environment
    read). CPU is sufficient: analysis only, LSTM-scale.
    """
    return [(["scripts/e13_stack.py", "--datasets", "har", "--device", "cpu",
              "--out", "/runs/e13_stack_har.json"], "e13_har")]


def jobs_e15() -> list[tuple[list[str], str]]:
    """E15 — ViT tier-0, prototypes-only. One job per seed, base arm, 5 seeds."""
    return [(["scripts/e15_vit_tier0.py", "--seed", str(s), "--device", "cuda",
              "--out", f"/runs/e15_tier0/seed_{s}.json"], f"e15_{s}")
            for s in _e12_seeds(5)]


def jobs_e14dec() -> list[tuple[list[str], str]]:
    """E14 decomposition — one job per seed, base arm, 3 seeds. Carries H-R4."""
    return [(["scripts/e14_decompose.py", "--arm", "base", "--seed", str(s),
              "--device", "cuda", "--out", f"/runs/e14_decomp/base_{s}.json"],
             f"e14dec_{s}") for s in _e12_seeds(3)]


def jobs_e21() -> list[tuple[list[str], str]]:
    """E21 sweep (docs/E21_prereg.md sec 8): one job per (family, level,
    realization), 12 cells each. Level 0 has one realization by construction.
    Launched only after runs/e21/smoke.json exists (sec 9) -- asserted here
    from the volume-synced copy, not assumed."""
    import os
    assert os.path.exists("runs/e21/smoke.json"), "E21 sec 9: the smoke must exist before the sweep launches"
    levels = {"gain": 5, "offset": 5, "swap": 4}
    out = []
    for fam, n in levels.items():
        for li in range(n):
            for r in (range(3) if li > 0 else [0]):
                out.append((["scripts/e21_perturb.py", "--family", fam, "--level", str(li), "--realization", str(r),
                             "--out", f"/runs/e21/{fam}/level{li}_real{r}.json"], f"e21_{fam}_L{li}_r{r}"))
    return out


def jobs_e21_cont() -> list[tuple[list[str], str]]:
    """E21 continuous families re-run under the BOUNDED parameterization
    (docs/E21_prereg.md sec 3, amendment 2026-09-17): gain (13) + offset (13).
    The swap family is discrete, completed under the first sweep, and is NOT
    re-run -- its 10 artifacts stand. Refuses to launch unless the harness on
    disk records the bounded form, so the sweep cannot re-run the dead one."""
    import os
    src = open("scripts/e21_perturb.py").read()
    assert "LOG_GAIN_BOUND" in src and "OFFSET_BOUND_SIGMA" in src, "E21 amendment: harness is not the bounded one"
    assert not os.path.exists("runs/e21/gain/level0_real0.json"), "unbounded gain artifacts still in place -- move them to runs/e21/_unbounded first"
    return [j for j in jobs_e21() if j[0][2] in ("gain", "offset")]


def jobs_s72_lwf_dec() -> list[tuple[list[str], str]]:
    """Paper sec 5.1 (2026-09-18): decompose the E16 sec 7.2 LwF lambda*=1.0 arm
    relaunched uniform with S72 (era + fp32 shadow, har_subject), with the
    recorded probe draw -- the same instrument, seed and platform (x86: the
    har_subject partition is x86-only) as runs/e20/seeded/har_s72_*_linear.json.
    Until this ran, "F_total does not move" in the draft rested on the
    shared-window construction at lambda=0.25 (runs/e16_decomp/har_lam0.25.json)."""
    return [(["scripts/e16_decompose.py", "--arm", "s72_lwf", "--bench", "har_subject",
              "--device", "cpu", "--ckpt-root", "/runs", "--probe-seed", "20260916",
              "--out", "/runs/e20/seeded/har_s72_lwf_linear.json"], "s72_lwf_dec")]


def jobs_headdrift() -> list[tuple[list[str], str]]:
    """Paper sec 3.1 (2026-09-19): split F_read on the shared-head S72 HAR arm into
    its frozen-era-head part and its head-drift part (scripts/head_drift.py).
    HAR only: `har_subject`'s partition is x86-only, so this cannot run on the
    laptop; the three E18 MNIST arms run locally against the same script and the
    same seeded decomposition artifacts."""
    return [(["scripts/head_drift.py", "--arm", "s72_off", "--ckpt-root", "/runs",
              "--ref", "/runs/e20/seeded/har_s72_off_linear.json",
              "--out", "/runs/head_drift/s72_off.json"], "headdrift_s72_off")]


D1_REFS = {"s72_off": "/runs/e20/seeded/har_s72_off_linear.json", "e18_pmd_mlp": "/runs/e18_pmd_mlp/decomp.json",
           "e18_pmd_lstm": "/runs/e18_pmd_lstm/decomp.json", "e18_rmd_mlp": "/runs/e18_rmd_mlp/decomp.json"}


def jobs_d1_scratch() -> list[tuple[list[str], str]]:
    """D1 sec 2 (docs/D1_prereg.md, signed v2 2026-09-19): paired extraction and
    the linear drift fits on the four shared-head scratch arms, one CPU job per
    arm on x86 (har_subject's partition; uniform platform for the refit floor).
    Smoked locally on e18_pmd_mlp seed 42 (11 s, C-ID 0.0e+00) before launch.
    Refs are the seeded decompositions the cells are C-ID'd against."""
    return [(["scripts/d1_fit.py", "--arm", a, "--ckpt-root", "/runs", "--ref", D1_REFS[a],
              "--out-dir", f"/runs/d1/{a}"], f"d1_{a}") for a in sorted(D1_REFS)]


def jobs_e23_dec() -> list[tuple[list[str], str]]:
    """E23 sec 4: the raw + re-laid decomposition on the A-ViT / A-RN checkpoints
    (scripts/e23_decompose.py through the E12/E14 paths), one L4 job per (backbone, seed).
    Launch only after the corresponding e23_{vit,rn}_seed{s} runs are complete."""
    return [(["scripts/e23_decompose.py", "--backbone", b, "--seed", str(s), "--ckpt-root", "/runs", "--data-root", "./data",
              "--device", "cuda", "--out", f"/runs/e23/{'e23_vit' if b == 'vit' else 'e23_rn'}/decomp_seed{s}.json"], f"e23dec_{b}_{s}")
            for b in ("vit", "resnet") for s in (42, 1337, 2024)] + [
            (["scripts/e23_decompose.py", "--backbone", b, "--seed", "42", "--ckpt-root", "/runs", "--data-root", "./data", "--device", "cuda",
              "--ckpt-dir", f"/runs/ckpt_e23_{t}_floor42/mafc_seed42", "--run-dir", f"/runs/e23_{t}_floor_seed42",
              "--out", f"/runs/e23/e23_{t}/decomp_floor42.json"], f"e23dec_{b}_floor42")           # C-FLOOR on F_enc: the seed-42 relaunch, decomposed
            for b, t in (("vit", "vit"), ("resnet", "rn"))]


def jobs_e23_har_dec() -> list[tuple[list[str], str]]:
    """E23 arm B decomposition: the S72 instrument (scripts/e16_decompose.py, recorded probe draw) on the
    MLP-backbone HAR checkpoints, CPU x86 -- uniform with runs/e20/seeded/har_s72_off_linear.json."""
    return [(["scripts/e16_decompose.py", "--arm", "e23_har_mlp", "--bench", "har_subject", "--device", "cpu",
              "--ckpt-root", "/runs", "--probe-seed", "20260916", "--out", "/runs/e20/seeded/har_e23_mlp_linear.json"], "e23_har_dec")]


def jobs_e23b_dec() -> list[tuple[list[str], str]]:
    """E23-B sec 3: raw vs re-laid on the scratch LSTM at both sequence lengths, plus A1
    (e18_pmd_lstm) re-measured through the same script as C-ID. One CPU job per (arm, seed);
    runs where the checkpoints live so 60 era checkpoints stay on the volume."""
    out = [(["scripts/e23_seqlen.py", "--arm", "e18_pmd_lstm", "--seed", str(s), "--ckpt-root", "/runs",
             "--ref", "/runs/e18_pmd_lstm/decomp.json", "--out", f"/runs/e23b/e18_pmd_lstm/decomp_seed{s}.json"], f"e23b_dec_a1_{s}")
           for s in SEEDS]
    out += [(["scripts/e23_seqlen.py", "--arm", a, "--seed", str(s), "--ckpt-root", "/runs",
              "--out", f"/runs/e23b/{a}/decomp_seed{s}.json"], f"e23b_dec_{a}_{s}")
            for a in ("e23b_t5c20", "e23b_t20") for s in SEEDS]
    return out


def jobs_e23_mlp_screen() -> list[tuple[list[str], str]]:
    """E23 arm B C0deg screen (scripts/cure_screen.py --arms e23mlp, recorded probe draw), CPU x86."""
    return [(["scripts/cure_screen.py", "--arms", "e23mlp", "--probe-seed", "20260916", "--out", "/runs/e23_mlp_screen/"], "e23_mlp_screen")]


def jobs_e23_har_ref() -> list[tuple[list[str], str]]:
    """E23 arm B P1 reference: linear readout on flattened windows (scripts/e23_har_linear_ref.py), CPU, x86."""
    return [(["scripts/e23_har_linear_ref.py", "--out", "/runs/e23/frozen_mlp.json"], "e23_har_ref")]


def jobs_e23_frozen() -> list[tuple[list[str], str]]:
    """E23 P1 reference on cifar100_permuted: frozen trunk, per-task PERMUTED extraction
    (scripts/e23_frozen_probe.py; the E12/E14 frozen-probe scripts extract unpermuted
    features and are the B1 reference only). One L4 job per backbone."""
    return [(["scripts/e23_frozen_probe.py", "--backbone", b, "--root", "./data", "--device", "cuda",
              "--out", f"/runs/e23/frozen_{b}.json"], f"e23_frozen_{b}") for b in ("vit", "resnet")]


def jobs_d1_pretrained() -> list[tuple[list[str], str]]:
    """D1 sec 3 D3 on the pretrained arms (frozen per-task heads: F_frozen == F_total),
    scripts/d1_fit_pretrained.py through the E12/E14 decomposition paths; one L4 job
    per (backbone, seed), the three Table 1 seeds."""
    return [(["scripts/d1_fit_pretrained.py", "--arm", a, "--seed", str(s), "--ckpt-root", "/runs", "--data-root", "./data",
              "--device", "cuda", "--out-dir", f"/runs/d1/{a}"], f"d1pre_{a}_{s}") for a in ("e12_base", "e14_base") for s in (42, 1337, 2024)]


def jobs_e25_probe_floor() -> list[tuple[list[str], str]]:
    """The probe floor on every scratch arm (CPU). RULING 2026-09-21: `refit_probe`
    is NOT changed -- every refit number in the paper comes from it. This measures
    how far its default tolerance stops short, so sec 3.1's corrected sentence can
    carry a number instead of the word "convergence"."""
    return [(["scripts/e25_probe_floor.py", "--arm", a, "--ckpt-root", "/runs",
              "--out", f"/runs/e25/probe_floor/{a}.json"], f"e25pf_{a}")
            for a in ("s72_off", "e18_pmd_mlp", "e18_pmd_lstm", "e18_rmd_mlp")]


def jobs_e25_probe_floor_pre() -> list[tuple[list[str], str]]:
    """The same on the pretrained arms (GPU, uniform with E12/E14). These carry
    2048-d ResNet features, where the gap has no reason to match a 256-d arm's."""
    return [(["scripts/e25_probe_floor.py", "--arm", a, "--ckpt-root", "/runs",
              "--data-root", "./data", "--device", "cuda", "--seeds", str(s),
              "--out", f"/runs/e25/probe_floor/{a}_seed{s}.json"], f"e25pf_{a}_s{s}")
            for a in ("e12_base", "e14_base") for s in (42, 1337, 2024)]


def jobs_e25_c4_diag() -> list[tuple[list[str], str]]:
    """Diagnose control 4 on x86. The laptop builds HAR partition 5d047e4213d1;
    the executed runs are 1104af185c87, so the first local run of this diagnostic
    measured a partition no checkpoint was trained on. The script now asserts the
    fingerprint and this job set is the only way to run it."""
    return [(["scripts/e25a_c4_diag.py", "--arm", a, "--draws", "20",
              "--out", f"/runs/e25/a/c4_diagnosis_{a}.json"], f"e25c4diag_{a}")
            for a in ("s72_off", "e18_pmd_mlp")]


def jobs_e25_a_scratch() -> list[tuple[list[str], str]]:
    """E25 L5 + L7 (sec A): the supervision curve and A-ridge on the scratch arms.
    CPU, uniform with the scratch decompositions. A-ridge is not a separate job:
    it shares the curve's extraction and its draws, and a second implementation
    of the same draw is exactly what catch 32 is about."""
    return [(["scripts/e25a_curve.py", "--arm", a, "--ckpt-root", "/runs",
              "--out-dir", f"/runs/e25/a/{a}"], f"e25a_{a}")
            for a in ("s72_off", "e18_pmd_mlp")]


def jobs_e25_a_pre_smoke() -> list[tuple[list[str], str]]:
    """E25 L6 smoke: two tasks, one seed, per backbone, before the headline.
    The pretrained path cannot be exercised on the laptop -- no CIFAR, no GPU,
    checkpoints on the volume -- so the first live run is a smoke, as E23's was."""
    return [(["scripts/e25a_curve.py", "--arm", a, "--ckpt-root", "/runs", "--data-root", "./data",
              "--device", "cuda", "--seeds", "42", "--max-tasks", "2",
              "--out-dir", f"/runs/e25/a/smoke_{a}"], f"e25a_smoke_{a}")
            for a in ("e12_base", "e14_base")]


def jobs_e25_a_pre() -> list[tuple[list[str], str]]:
    """E25 L6 + L7 headline: ViT-B/16 and ResNet-50, three seeds, nineteen old
    tasks. GPU, uniform with E12/E14. Launch only after the smoke is green."""
    return [(["scripts/e25a_curve.py", "--arm", a, "--ckpt-root", "/runs", "--data-root", "./data",
              "--device", "cuda", "--seeds", str(s),
              "--out-dir", f"/runs/e25/a/{a}"], f"e25a_{a}_s{s}")
            for a in ("e12_base", "e14_base") for s in (42, 1337, 2024)]


def jobs_e25_b() -> list[tuple[list[str], str]]:
    """E25 L4 (docs/E25_prereg.md sec B): per-step drift on the four D1 scratch
    arms, five boundaries each. One job per arm, three seeds inside it -- the
    era checkpoints are loaded once per seed and reused across every k, so
    fanning out by seed would reload twenty models to save nothing."""
    return [(["scripts/e25b_steps.py", "--arm", a, "--ckpt-root", "/runs",
              "--out-dir", f"/runs/e25/b/{a}"], f"e25b_{a}")
            for a in ("s72_off", "e18_pmd_mlp", "e18_pmd_lstm", "e18_rmd_mlp")]


def jobs_e25_b_a3() -> list[tuple[list[str], str]]:
    """E25 L8: the same script on E23-B's A3 -- twenty tasks, 190 fits per seed.
    Separate job set from L4 because it is the long pole and its arm carries an
    extra construction field (`content_chunks`) that C-WIT asserts."""
    return [(["scripts/e25b_steps.py", "--arm", "e23b_t20", "--ckpt-root", "/runs",
              "--seeds", str(s), "--max-cross", "20",
              "--out-dir", "/runs/e25/b/e23b_t20"], f"e25b_a3_s{s}")
            for s in (42, 1337, 2024)]


# ---------------------------------------------------- E25 C: Mummadi at lambda 40 --
# A4 ALONE. A1, A3, A6, A0 and SNAP do not depend on lambda, so A3 is read from
# E21's artifacts and the cheap deterministic arms are recomputed as the C-ID
# witness that this harness is E21's. A2's bridging refit is the expensive arm and
# is excluded -- `run_cell` gates it on "A2" in arms, and the default tuple still
# contains it, so E21's own path is unchanged.
E25C_ARMS = "A0,A1,A4,A6,SNAP"


def jobs_e25_c() -> list[tuple[list[str], str]]:
    """E25 L3 (docs/E25_prereg.md sec C): A4 at lambda = 40, Mummadi's own
    weighting, on the swap family at every level and realization. Level 0 is RUN
    (the contract's scope is every level) but NOT READ for the comparison: one
    realization and a deterministic search make both floor components zero by
    construction, catch 35(a)."""
    out = []
    for li in range(4):
        for r in (range(3) if li > 0 else [0]):
            out.append((["scripts/e21_perturb.py", "--family", "swap", "--level", str(li),
                         "--realization", str(r), "--mummadi-lambda", "40", "--arms", E25C_ARMS,
                         "--out", f"/runs/e25/c/lam40/level{li}_real{r}.json"],
                        f"e25c_L{li}_r{r}"))
    return out


def jobs_e25_c_regress() -> list[tuple[list[str], str]]:
    """E25 L3 flag regression. The DEFAULT lambda = 1 with the DEFAULT arm tuple,
    on one level, must reproduce E21's existing artifact to 1e-6. Full arms, not
    the C subset: a regression that only re-ran what C re-runs would not test the
    gate it is there to protect (the A2/arms gating and the objective's rewrite).
    This is the flag's regression, not a claim about diversity."""
    return [(["scripts/e21_perturb.py", "--family", "swap", "--level", "1", "--realization", "0",
              "--out", "/runs/e25/c/lam1_regression_raw.json"], "e25c_regress_L1_r0")]


def jobs_e25_c_signflip() -> list[tuple[list[str], str]]:
    """E25 L3 must-fail: REWARD collapse instead of diversity (sign = +1) at
    lambda = 40, m = 1, all three realizations = 36 cells. Must do worse than A3
    on 36/36. A gate that has only ever passed is indistinguishable from one that
    cannot fail (catch 25), so this is the case it is known to have to fail."""
    return [(["scripts/e21_perturb.py", "--family", "swap", "--level", "1", "--realization", str(r),
              "--mummadi-lambda", "40", "--diversity-sign", "1", "--arms", E25C_ARMS,
              "--out", f"/runs/e25/c/signflip/level1_real{r}.json"], f"e25c_signflip_r{r}")
            for r in range(3)]


E26_FS_ARM = {"vit": "e23_vit", "resnet": "e23_rn"}


def _e26_fs(bb, s, extra, tag):
    return (["scripts/e26_fs.py", "--backbone", bb, "--seed", str(s), "--ckpt-root", "/runs",
             "--data-root", "./data", "--device", "cuda",
             "--ref", f"/runs/e23/{E26_FS_ARM[bb]}/decomp_seed{s}.json",
             "--out-dir", f"/runs/e26/fs/{E26_FS_ARM[bb]}"] + extra, tag)


def jobs_e26_fs_smoke() -> list[tuple[list[str], str]]:
    """E26 FS smoke: two old tasks, seed 42, both backbones, refit path included so
    the smoke exercises every code path the headline runs. Its output dir is the
    headline's, so a passing smoke's rows are simply overwritten by the full run."""
    return [_e26_fs(bb, 42, ["--max-tasks", "2"], f"e26fs_smoke_{bb}") for bb in ("resnet", "vit")]


def jobs_e26_fs_frozen() -> list[tuple[list[str], str]]:
    """E26 FS, frozen-trunk arm: the comparison that decides sec 4.3.

    FS's finding is that theta_T's refit reads best in the PRETRAINING layout.
    Two readings -- training taught the scrambled formats, or training cost skill
    on the original layout -- are separated only by measuring the same frames on
    the never-fine-tuned trunk. Same frames, same loaders, same probe, same draw
    stream as FS; the only difference is the encoder. C-ID against
    e23_frozen_probe's own-frame number, which is the one point where the two
    protocols overlap."""
    return [(["scripts/e26_fs_frozen.py", "--backbone", bb, "--seed", str(s),
              "--data-root", "./data", "--device", "cuda",
              "--ref-frozen", f"/runs/e23/frozen_{'vit' if bb == 'vit' else 'resnet'}.json",
              "--out-dir", f"/runs/e26/fs/frozen_{bb}"], f"e26fsfrozen_{bb}_s{s}")
            for bb in ("resnet", "vit") for s in (42, 1337, 2024)]


def jobs_e26_fs_vit1337() -> list[tuple[list[str], str]]:
    """Re-run of the one FS cell set that never landed: ViT seed 1337 ran ~3 h
    against its siblings' 52 and 114 min, wrote no artifact, and was cancelled
    2026-09-22. SAME seed, same code -- the 85% prediction registered against it
    (runs/MEMO_e26.md sec 1b, 2026-09-21 22:44) is unscored and nobody has seen
    a result, so it still counts."""
    return [_e26_fs("vit", 1337, [], "e26fs_vit_s1337_rerun")]


def jobs_e26_fs() -> list[tuple[list[str], str]]:
    """E26 FS headline: 19 old tasks x 20 frames, three seeds, both backbones. GPU."""
    return [_e26_fs(bb, s, [], f"e26fs_{bb}_s{s}") for bb in ("resnet", "vit") for s in (42, 1337, 2024)]


def _e26_dc_pre(bb, s, extra, tag):
    return (["scripts/e26_dc.py", "--arm", bb, "--seed", str(s), "--ckpt-root", "/runs",
             "--data-root", "./data", "--device", "cuda",
             "--out-dir", f"/runs/e26/dc/{'e23_vit' if bb == 'vit' else 'e23_rn'}"] + extra, tag)


def jobs_e26_dc_smoke() -> list[tuple[list[str], str]]:
    """E26 DC smoke: ResNet seed 42, three old tasks. Boundaries still run to T, so
    the smoke exercises every boundary's two extractions and both ports."""
    return [_e26_dc_pre("resnet", 42, ["--max-tasks", "3"], "e26dc_smoke_resnet")]


def jobs_e26_dc() -> list[tuple[list[str], str]]:
    """E26 DC headline, pretrained: both backbones, three seeds, L2-normalised; plus
    the unnormalised sensitivity arm on seed 42 only. GPU."""
    out = [_e26_dc_pre(bb, s, [], f"e26dc_{bb}_s{s}") for bb in ("resnet", "vit") for s in (42, 1337, 2024)]
    out += [_e26_dc_pre(bb, 42, ["--normalize", "none"], f"e26dc_{bb}_s42_unnorm") for bb in ("resnet", "vit")]
    return out


def jobs_e26_dc_scratch() -> list[tuple[list[str], str]]:
    """E26 DC on the scratch arms (CPU): the four D1 arms and E23-B's A3, three
    seeds. On these the methods' no-map assumption is violated by construction
    (the encoder is overwritten per frame, E23-B), so this is the expected-failure
    half of the comparison, not a place a method is expected to work."""
    return [(["scripts/e26_dc.py", "--arm", a, "--seed", str(s), "--ckpt-root", "/runs",
              "--out-dir", f"/runs/e26/dc/{a}"], f"e26dc_{a}_s{s}")
            for a in ("s72_off", "e18_pmd_mlp", "e18_pmd_lstm", "e18_rmd_mlp", "e23b_t20") for s in (42, 1337, 2024)]


def jobs_e26_fs_a3() -> list[tuple[list[str], str]]:
    """E26 FS on A3, the scratch comparison (CPU, uniform with E23-B's decomposition).
    One job per seed; C-ID against E23-B's decomp at 1e-6 (fp32 shadow)."""
    return [(["scripts/e26_fs_a3.py", "--seed", str(s), "--ckpt-root", "/runs",
              "--ref", f"/runs/e23b/e23b_t20/decomp_seed{s}.json",
              "--out-dir", "/runs/e26/fs/e23b_t20"], f"e26fs_a3_s{s}") for s in (42, 1337, 2024)]


E27_ARMS = ("s72_off", "e18_pmd_mlp", "e18_pmd_lstm", "e18_rmd_mlp", "e23b_t20", "e23_har_mlp")


def jobs_e27_defect() -> list[tuple[list[str], str]]:
    """E27 sec 1 + 1a: the equivariance defect and its per-arm random-init
    reference, six arms x three seeds. CPU, uniform with the scratch
    decompositions. Controls run at the top of every job (sec 3)."""
    return [(["scripts/e27_defect.py", "--arm", a, "--seed", str(s), "--ckpt-root", "/runs",
              "--out-dir", f"/runs/e27/{a}"], f"e27_{a}_s{s}")
            for a in E27_ARMS for s in (42, 1337, 2024)]


def jobs_e28_measure() -> list[tuple[list[str], str]]:
    """E28 sec 3: the measurement that tests the method, on E28b's 30-epoch arms.
    x86 (har_subject's partition), CPU, one job per (arm, seed)."""
    return [(["scripts/e28_measure.py", "--arm", a, "--seed", str(s), "--ckpt-root", "/runs",
              "--heldout", "/runs/e28/heldout.json", "--out-dir", "/runs/e28"],
             f"e28m_{a}_s{s}")
            for a in ("b0_ep30", "b1_ep30", "b2_ep30", "b2_p025_ep30") for s in (42, 1337, 2024)]


def jobs_e28_heldout() -> list[tuple[list[str], str]]:
    """E28 sec 1 + the sec 5 must-fail, on EXISTING S72 OFF checkpoints. Run
    BEFORE the trainer change: if a normally trained encoder loses nothing under
    an unseen channel permutation, there is nothing for an equivariance-trained
    encoder to fix and the pilot is void. x86 (har_subject's partition)."""
    return [(["scripts/e28_heldout.py", "--ckpt-root", "/runs",
              "--out", "/runs/e28/b0_mustfail.json",
              "--heldout-out", "/runs/e28/heldout.json"], "e28_heldout")]


def jobs_e27_audit() -> list[tuple[list[str], str]]:
    """E27's audit. Five arms were green under --chain e26; arm B
    (`ckpt_e23_har_mlp`) has never been audited and this is why the chain exists.
    Gates the run: loads, not exists."""
    return [(["scripts/audit_checkpoints.py", "--chain", "e27", "--root", "/runs",
              "--json", "/runs/e27/ckpt_audit.json"], "e27_audit")]


def jobs_e26_audit() -> list[tuple[list[str], str]]:
    """E26 L0: the B6 permuted pretrained arms and A3 that FS and DC load. Routed
    to GPU deliberately, as E25's audit was -- the only container proven to
    construct both pretrained backbones. Loadability is not device-dependent."""
    return [(["scripts/audit_checkpoints.py", "--chain", "e26", "--root", "/runs",
              "--json", "/runs/e26/ckpt_audit.json"], "e26_audit")]


def jobs_e25_audit() -> list[tuple[list[str], str]]:
    """E25 L0 (docs/E25_prereg.md): the arm x seed x LOADS table that gates L4-L8.

    ROUTING, stated rather than inherited. This is NOT in the contract's routing
    table because it is not a measurement: it opens every checkpoint A and B will
    load through `PLCM.load_era` and reports whether it opens. Device uniformity
    is a property a NUMBER needs; whether a file loads is not device-dependent.
    It is left OFF the CPU tuple in `spawn_analysis` deliberately, so it runs on
    `analyze_one_gpu` -- the only container proven to construct both pretrained
    backbones (it already runs e12dec, e23dec and d1pre against these same
    checkpoints). Falling to GPU by accident is the hazard the contract's routing
    rule names; falling to GPU on purpose, said out loud, is not that.

    One job: the chain is 22 directories and the cost is dominated by 120 ViT and
    ResNet constructions, which fan out to nothing useful.
    """
    return [(["scripts/audit_checkpoints.py", "--chain", "e25", "--root", "/runs",
              "--json", "/runs/e25/ckpt_audit.json"], "e25_audit")]


def jobs_e29_pools() -> list[tuple[list[str], str]]:
    """E29 Part A: pool composition x n, all 22 targets. CPU, x86 (har_subject's
    partition). One job -- the 9! table is precomputed once per reference and
    reused across the whole grid, so fanning out would repeat a 224MB x2
    precompute per worker for no wall-clock gain."""
    return [(["scripts/e29_pools.py", "--heldout", "/runs/e28/heldout.json",
              "--out", "/runs/e29/pools.json",
              "--controls-out", "/runs/e29/controls.json"], "e29_pools")]


def jobs_e29_rot_bias() -> list[tuple[list[str], str]]:
    """E29 Part B: the rotation estimator's transfer bias over subject pairings,
    n, and pool composition. `--n-start` is passed EXPLICITLY and has no default
    anywhere in the chain -- a default here would let the launcher, rather than
    the artifact, select the configuration that was measured."""
    return [(["scripts/e29_rot_bias.py", "--n-start", "8",
              "--out", "/runs/e29/rot_bias.json",
              "--controls-out", "/runs/e29/controls_rot.json"], "e29_rot_bias")]


def jobs_e29_reads() -> list[tuple[list[str], str]]:
    """E29 §0a re-measured on Modal's partition, with the contract's
    disagreement branches fired from the numbers."""
    return [(["scripts/e29_reads.py", "--n-start", "8",
              "--heldout", "/runs/e28/heldout.json",
              "--out", "/runs/e29/reads.json"], "e29_reads")]


def jobs_e29_tolerance() -> list[tuple[list[str], str]]:
    """E29 Part B.3: how much rotation error B0 absorbs, independent of any
    estimator. Three seeds, CPU."""
    return [(["scripts/e29_tolerance.py", "--seed", str(s), "--ckpt-root", "/runs",
              "--out-dir", "/runs/e29"], f"e29_tol_s{s}") for s in (42, 1337, 2024)]


def jobs_e29_downstream() -> list[tuple[list[str], str]]:
    """E29 downstream: estimated vs exact vs no repair, both families, on B0."""
    return [(["scripts/e29_downstream.py", "--seed", str(s), "--n-start", "8",
              "--ckpt-root", "/runs", "--heldout", "/runs/e28/heldout.json",
              "--out-dir", "/runs/e29"], f"e29_ds_s{s}") for s in (42, 1337, 2024)]


def jobs_e30_fidelity() -> list[tuple[list[str], str]]:
    """Does the repinned image reproduce numbers recorded BEFORE the rebuild?

    Version strings are the weak form of this check: two images can report the
    same packages and still differ in base image, baked dataset or BLAS linkage.
    This reproduces two exact CPU logits hashes from 2026-09-19 and the
    pretrained path's accuracy and path-identity gap from 2026-09-22, and
    records an exact GPU-path hash that never existed before.

    GATE: if a recorded number does not reproduce, it outranks E30."""
    return [(["scripts/e30_rebuild_fidelity.py", "--ckpt-root", "/runs",
              "--data-root", "/data", "--out", "/runs/e30/rebuild_fidelity.json"],
             "e30_fidelity")]


def jobs_e30_smoke() -> list[tuple[list[str], str]]:
    """E30 cost smoke: ONE pretrained arm, ONE seed, two tasks. The contract
    states no cost figure until this reports (the fourth standing question).

    It re-runs the version dump first, because the five-package pin invalidated
    the image cache: the rebuild must come back with the SAME versions the pin
    names, or the pin recorded a container that no longer exists."""
    return [(["scripts/env_versions.py"], "e30_env2"),
            (["scripts/e30_geometry.py", "--arm", "b1_vit", "--seed", "42",
              "--ckpt-root", "/runs", "--data-root", "/data", "--max-tasks", "2",
              "--out-dir", "/runs/e30/smoke"], "e30_smoke_b1vit")]


def jobs_e30_pre() -> list[tuple[list[str], str]]:
    """E30 pretrained cells: 4 arms x 3 seeds on GPU, through the E12/E14 path."""
    return [(["scripts/e30_geometry.py", "--arm", a, "--seed", str(s),
              "--ckpt-root", "/runs", "--data-root", "/data",
              "--out-dir", f"/runs/e30/{a}"], f"e30_pre_{a}_s{s}")
            for a in ("b1_vit", "b1_rn", "b6_vit", "b6_rn") for s in (42, 1337, 2024)]


def jobs_e30_scratch() -> list[tuple[list[str], str]]:
    """E30 scratch cells: 6 arms x 3 seeds on CPU, through the audited loader.
    Arms come from `e27_defect.ARMS`, not a second copy of the table."""
    return [(["scripts/e30_geometry.py", "--arm", a, "--seed", str(s),
              "--ckpt-root", "/runs", "--device", "cpu",
              "--out-dir", f"/runs/e30/{a}"], f"e30_scr_{a}_s{s}")
            for a in ("s72_off", "e23_har_mlp", "e18_pmd_mlp", "e18_pmd_lstm",
                      "e18_rmd_mlp", "e23b_t20") for s in (42, 1337, 2024)]


def jobs_e30_audit() -> list[tuple[list[str], str]]:
    """E30 L0: the arm x seed x LOADS table the contract's "analysis only, on
    existing checkpoints" claim requires (catch 20). Ten arms, EVERY era
    checkpoint, not just theta_T. Bundled with a version dump because the
    contract's other prerequisite -- pinning scipy -- edits an unpinned
    pip_install list and would re-roll numpy and scikit-learn along with it
    unless the pins reproduce what is installed now.

    Routed to GPU deliberately, as e25audit is: this constructs ViT-B/16 and
    ResNet-50, and the GPU container is the one proven to build both."""
    return [(["scripts/env_versions.py"], "e30_env"),
            (["scripts/audit_checkpoints.py", "--chain", "e30", "--root", "/runs",
              "--json", "/runs/e30/ckpt_audit.json"], "e30_audit")]


def jobs_classgeom_probe() -> list[tuple[list[str], str]]:
    """Pre-draft probe for the class-geometry check: the container's scipy
    version (unpinned in the image, so it must be asked, not read off
    requirements.txt) and whether the B1 pretrained checkpoints extract through
    B6's path. GPU by default -- this constructs ViT-B/16 and ResNet-50, and the
    GPU container is the one proven to build both."""
    return [(["scripts/classgeom_probe.py", "--seed", "42", "--ckpt-root", "/runs",
              "--out", "/runs/classgeom/probe.json"], "classgeom_probe")]


ANALYSIS_JOBS = {"e12dec": jobs_e12dec, "e12cildec": jobs_e12cildec, "e21": jobs_e21, "e21cont": jobs_e21_cont, "s72lwfdec": jobs_s72_lwf_dec, "headdrift": jobs_headdrift, "d1": jobs_d1_scratch, "d1pre": jobs_d1_pretrained, "e23frozen": jobs_e23_frozen, "e23dec": jobs_e23_dec, "e23harref": jobs_e23_har_ref, "e23hardec": jobs_e23_har_dec, "e23bdec": jobs_e23b_dec, "e23mlpscreen": jobs_e23_mlp_screen, "e25audit": jobs_e25_audit,
                 "e25c": jobs_e25_c, "e25cregress": jobs_e25_c_regress, "e25csignflip": jobs_e25_c_signflip,
                 "e25b": jobs_e25_b, "e25ba3": jobs_e25_b_a3, "e25ascratch": jobs_e25_a_scratch,
                 "e25apresmoke": jobs_e25_a_pre_smoke, "e25apre": jobs_e25_a_pre,
                 "e25pf": jobs_e25_probe_floor, "e25pfpre": jobs_e25_probe_floor_pre,
                 "e25c4diag": jobs_e25_c4_diag, "e26audit": jobs_e26_audit, "e27audit": jobs_e27_audit, "e28heldout": jobs_e28_heldout, "e28measure": jobs_e28_measure, "e27defect": jobs_e27_defect, "e26fssmoke": jobs_e26_fs_smoke, "e26fs": jobs_e26_fs, "e26fsvit1337": jobs_e26_fs_vit1337, "e26fsfrozen": jobs_e26_fs_frozen,
                 "e26dcsmoke": jobs_e26_dc_smoke, "e26dc": jobs_e26_dc, "e26dcscratch": jobs_e26_dc_scratch, "e26fsa3": jobs_e26_fs_a3,
                 "e13": jobs_e13, "e15": jobs_e15, "e14dec": jobs_e14dec,
                 "e16dec": jobs_e16dec, "e17dec": jobs_e17dec,
                 "e12cildec_v4": jobs_e12cildec_v4,
                 "e29pools": jobs_e29_pools, "e29rotbias": jobs_e29_rot_bias,
                 "e29reads": jobs_e29_reads, "e29tol": jobs_e29_tolerance,
                 "e29ds": jobs_e29_downstream,
                 "classgeomprobe": jobs_classgeom_probe,
                 "e30audit": jobs_e30_audit, "e30smoke": jobs_e30_smoke, "e30fidelity": jobs_e30_fidelity,
                 "e30pre": jobs_e30_pre, "e30scratch": jobs_e30_scratch}


@app.function(volumes={"/runs": volume}, timeout=120, cpu=1.0, memory=1024)
def _claim_jobset(experiment: str, ttl_s: int = 4 * 60 * 60, force: bool = False) -> dict:
    """Refuse to launch a job set that a live app already holds.

    WHY THIS IS CODE AND NOT A HABIT. E15 once ran as TWO apps of five
    containers each, both writing the same /runs/e15_tier0/seed_*.json paths.
    Nothing was contaminated — only because nothing had been written yet. That
    is timing, not design, and the sentence "nothing was contaminated, but only
    because..." is the one that precedes a catch in this program.

    The lock is per JOB SET, not per app, because every app here is named
    plcm-generality and the app list cannot tell two experiments apart. TTL is
    the function timeout: a crashed launch cannot wedge the door shut longer
    than its jobs could have run. `force=True` overrides, deliberately loudly.
    """
    import json as _j, os, time

    os.makedirs("/runs/.locks", exist_ok=True)
    path = f"/runs/.locks/{experiment}.json"
    now = time.time()
    if os.path.exists(path) and not force:
        try:
            held = _j.load(open(path))
        except Exception:
            held = {}
        age = now - float(held.get("t", 0))
        if age < ttl_s:
            volume.commit()
            return {"ok": False, "age_s": int(age), "held": held}
    _j.dump({"t": now, "experiment": experiment}, open(path, "w"))
    volume.commit()
    return {"ok": True}


@app.local_entrypoint()
def spawn_analysis(experiment: str = "e12dec", force: bool = False):
    """Fan out an ANALYSIS job list. Same no-live-client contract as
    spawn_experiment: `modal run --detach modal_runner.py::spawn_analysis`.
    """
    if experiment not in ANALYSIS_JOBS:
        raise SystemExit(f"unknown analysis {experiment!r}; "
                         f"choose from {sorted(ANALYSIS_JOBS)}")
    claim = _claim_jobset.remote(experiment, force=force)
    if not claim["ok"]:
        raise SystemExit(
            f"REFUSED: job set {experiment!r} was claimed {claim['age_s']}s ago and "
            f"may still be live. Two apps on one job set race the same output "
            f"paths. Stop it (modal app stop -y <id>) or pass --force.")
    js = ANALYSIS_JOBS[experiment]()
    handles = [(label, (analyze.spawn(argv) if experiment == "e13"
                        else analyze_one.spawn(argv, label) if experiment in ("e21", "e21cont", "s72lwfdec", "headdrift", "d1", "e23harref", "e23hardec", "e23mlpscreen", "e23bdec",
                                                                              # E25: C is uniform with E21 (CPU); B and the scratch A curve are
                                                                              # uniform with D1 and the scratch decompositions (CPU). The pretrained
                                                                              # A curve is NOT here -- it is uniform with E12/E14 on GPU.
                                                                              "e25c", "e25cregress", "e25csignflip", "e25b", "e25ba3", "e25ascratch", "e25pf", "e25c4diag", "e26dcscratch", "e26fsa3", "e27defect", "e28heldout", "e28measure",
                                                                              # E29 is all CPU: HAR statistics and a 3-parameter
                                                                              # SO(3) fit, no pretrained backbone anywhere.
                                                                              "e29pools", "e29rotbias", "e29reads", "e29tol", "e29ds", "e30scratch")
                        else analyze_one_gpu.spawn(argv, label)))
               for argv, label in js]
    print(f"{experiment}: spawned {len(handles)} analysis jobs\n")
    for label, h in handles:
        print(f"  {label:<24} call_id={h.object_id}")
    print("\nRuns server-side; poll  modal volume ls plcm-runs /e12_decomp")


@app.local_entrypoint()
def spawn_experiment(experiment: str = "e12pb", force: bool = False,
                     lam: float = None):
    """Submit an experiment's jobs and RETURN IMMEDIATELY — no live client.

    WHY THIS EXISTS. `main()` blocks on `.starmap()`, so the run only survives as
    long as the local process does. E12's precondition re-run was killed three
    times this way: twice when the launching shell died (an ephemeral app stops
    with its client), and once even under `--detach`, where a hard kill of the
    client left the app up for five minutes and then cancelled the in-flight
    calls ("Function call was cancelled by user or a failure"). Three launches,
    ~4 GPU-hours, zero results.

    `.spawn()` submits server-side and hands back a handle. Combined with
    `modal run --detach`, nothing local needs to stay alive: results land on the
    volume and are collected later by whatever asks for them.

    Usage:
        modal run --detach modal_runner.py::spawn_experiment --experiment e12pb
    """
    if experiment not in JOBS:
        raise SystemExit(f"unknown experiment {experiment!r}; choose from {sorted(JOBS)}")
    claim = _claim_jobset.remote(experiment, force=force)
    if not claim["ok"]:
        raise SystemExit(
            f"REFUSED: job set {experiment!r} was claimed {claim['age_s']}s ago and "
            f"may still be live. Stop it or pass --force.")
    js = (JOBS[experiment](lam) if lam is not None else JOBS[experiment]())
    handles = []
    for flags, log_dir in js:
        if log_dir.startswith(("w2d_lstm1t_", "e16_mnist_floor_",
                               "w1_har_off_floor_")):
            fn = train_one_1thread            # W2 diagnosis, thread-pinned
        else:
            fn = train_one_gpu if _wants_gpu(experiment, log_dir) else train_one
        handles.append((log_dir, fn.spawn(flags, log_dir)))
    print(f"{experiment}: spawned {len(handles)} jobs "
          f"({sum(_wants_gpu(experiment, d) for d, _ in handles)} on L4 GPU)\n")
    for log_dir, h in handles:
        print(f"  {log_dir:<28} call_id={h.object_id}")
    print("\nThe app now runs server-side and does NOT depend on this process.")
    print("Poll with:  modal volume ls plcm-runs /<log_dir>")


@app.local_entrypoint()
def main(experiment: str = "smoke"):
    if experiment not in JOBS:
        raise SystemExit(f"unknown experiment {experiment!r}; choose from {sorted(JOBS)}")
    js = (JOBS[experiment](lam) if lam is not None else JOBS[experiment]())
    gpu_js = [j for j in js if _wants_gpu(experiment, j[1])]
    cpu_js = [j for j in js if not _wants_gpu(experiment, j[1])]
    print(f"{experiment}: fanning out {len(js)} runs "
          f"({len(gpu_js)} on L4 GPU, {len(cpu_js)} on CPU)\n")
    # Both fan-outs are launched before either is awaited, so a mixed batch still
    # runs fully in parallel rather than GPU-then-CPU.
    gpu_h = train_one_gpu.starmap(gpu_js) if gpu_js else iter(())
    cpu_h = train_one.starmap(cpu_js) if cpu_js else iter(())
    results = list(gpu_h) + list(cpu_h)
    ok = [r for r in results if r.get("ok")]
    print(f"\n{len(ok)}/{len(results)} succeeded")
    for r in sorted(results, key=lambda r: r["log_dir"]):
        if r.get("ok"):
            print(f"  {r['log_dir']:<34} AVG {r['average_accuracy']:.4f}  forget {r['forgetting']:.4f}")
        else:
            print(f"  {r['log_dir']:<34} FAILED: {r.get('error','')[:120]}")
    print("\nSync results back with:")
    print(f"  modal volume get {VOLUME_NAME} /runs ./runs --force")
