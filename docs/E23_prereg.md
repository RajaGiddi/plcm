# E23 — Closing the Pretraining Confound (v2)

**Status:** **SIGNED 2026-09-18** (v2); **RUN 2026-09-19 and READ 2026-09-20** — `runs/MEMO_e23.md`, `runs/e23_row.json`. **P1 FAILS on all three arms** (the contracted reference cannot read the construction: frozen trunks 0.72 / 0.70 on B6, linear-on-windows 0.55 on HAR; the trainable arms beat their references by 6–24 pp) — **RULED 2026-09-20: (a) the A arms are read with P1 FAIL recorded and the competence gap beside (E1)/(E2); (c) arm B is read against the S72 LSTM (within 4.6 pp) and the imported 0.85 bar is the contract's error** — a defective gate (R3's category), not a failed arm. Measured: re-laying recovers nothing on either pretrained backbone ((E2) fails, re-laid worse by 2–6 pp deployed; wrong-permutation must-fail FAIL with cause); arm B share 100.2% vs the LSTM's 77.8. Sign-off: option (b) for shadow uniformity; † odds accepted with one adjustment (§7); the MLP `input_dim` change approved under the amendment rule with its regression. Redrafted after the v1 review:
v1's arm A (ResNet-50 on upsampled Permuted MNIST) is withdrawn; see §0.
**Not on the September path** — October work, November rebuttal material.
Two relaunches of an existing construction, one config-level arm, no new
benchmark to build or gate.

## 0. What v1 got wrong, and the grep that replaces its first sentence

v1 opened: *"Every pretrained result has no map."* The repo says otherwise.

```
$ grep -n "permuted\|shift_mode\|invert_perm" src/data/split_cifar100.py
1:  Split CIFAR-100 — E12's benchmark (B1), with the efficacy control's shift (B6).
30: Both modes are exactly invertible; B6's verification applies and inverts and
94: def apply_perm(x, perm)          100: def invert_perm(perm)
105: def build_patch_consistent_perm  125: def build_global_perm
```

**B6, `cifar100_permuted`** ([split_cifar100.py:1-35](../src/data/split_cifar100.py#L1-L35)):
pretrained ViT-B/16, twenty tasks, a per-task permutation applied at 224×224
**on the model's actual input** (identical within every 16×16×3 patch, 768
positions), exactly invertible, verified bitwise by the module, fingerprint
built from plain ints and identical on arm64 and x86. Six runs exist on it —
`runs/e12eff_seed{42,1337,2024}` (adapters ON) and `runs/e12effb_seed{42,1337,2024}`
(adapters OFF, the base arm) — with **`era_checkpoints: False`**, by a decision
the launcher records in its own docstring
([modal_runner.py:554-556](../modal_runner.py#L554-L556)): *"if it ever becomes
a comparison baseline it needs a re-run."* It has. That is catch 20's corollary
coming due and catch 21 on the premise: the fifth contract in a row whose
opening premise was written from a model of the program rather than from the
file. The rule that follows is in CLAUDE.md as of today: **a contract's first
paragraph is the grep of `src/data/` for the construction it claims does not
exist.**

## 1. The confound, restated against the artifacts

| cell | backbone | benchmark | map | decomposed? | artifact |
|---|---|---|---|---|---|
| scratch, known map | LSTM | `har_subject` (S72) | 9×9 affine per task | yes, seeded | `runs/e20/seeded/har_s72_off_linear.json` |
| scratch, known map | MLP / LSTM | Permuted MNIST, disjoint content | 784-perm | yes | `runs/e18_pmd_{mlp,lstm}/` |
| scratch, known map | MLP | Rotated MNIST | rotation | yes | `runs/e18_rmd_mlp/` |
| pretrained, **no map** | ViT-B/16 | `cifar100` (B1) | — | yes (corrected loader, unseeded draw) | `runs/e12_decomp_v2/base_*.json` |
| pretrained, **no map** | ResNet-50 | `cifar100` | — | yes (unseeded draw) | `runs/e14_decomp/base_*.json` |
| **pretrained, known map** | **ViT-B/16** | **`cifar100_permuted` (B6)** | **patch-consistent perm** | **no — no era checkpoints** | `runs/e12effb_seed*/` (matrices only) |
| pretrained, known map | ResNet-50 | `cifar100_permuted` | patch-consistent perm | never run | — |

The two-origins finding — with a known map both decomposition terms are
largely presentation drift; without one the readout is genuinely stranded —
is entangled with training regime only because the sixth row was never
decomposed. E23 fills the sixth row and adds the seventh, **on the same
benchmark as the fourth and fifth**: same backbone, same data, same twenty
tasks, same class order, map versus no map. Nothing else varies.

**Second, smaller gap:** the sensor benchmark has one architecture. Arm B adds
a second on the identical construction.

## 2. Constructions and arms — cited to the file that builds them

### Arm A-ViT — `cifar100_permuted`, ViT-B/16, base arm, relaunched with era checkpoints

Exactly `jobs_e12effb` ([modal_runner.py:566-586](../modal_runner.py#L566-L586)):
`E12_CFG + E12_BASE + ["--benchmark", "cifar100_permuted"]` — `configs/e12_vit.yaml`,
`--mafc-arm lambda0`, task-IL with per-task heads (`use_task_heads: True`, as
every E12/E14 decomposition), adapters OFF — plus `--era-checkpoints`. Seeds
42, 1337, 2024. The shift fingerprint per seed is asserted against the
registered value at [train.py:176-180](../scripts/train.py#L176-L180).

### Arm A-RN — `cifar100_permuted`, ResNet-50, base arm

`E14_CFG` ([modal_runner.py:615](../modal_runner.py#L615)) with
`--benchmark cifar100_permuted --era-checkpoints`, seeds 42, 1337, 2024. The
permutation is applied in the dataset's `__getitem__`
([split_cifar100.py:68-92](../src/data/split_cifar100.py#L68-L92)) and is
backbone-agnostic; [train.py:152](../scripts/train.py#L152) hard-wires
`shift_mode="patch"` for this benchmark name, so both backbones see the
**same** permutations and the same fingerprints (the shift depends on seed
and image size only). `shift_mode="global"` is not used: it would need a
`--shift-mode` flag and a regression, and it would make the two backbones'
constructions differ. **Build gate:** a one-epoch smoke of A-RN, checked for
the recorded `benchmark`, `shift_fingerprint`, `backbone`, `era_checkpoints`
in `arm`, and the E12 fingerprints added to `configs/e14_resnet.yaml` as
`expected_shift` so the assert at 176 fires for this backbone too.

### The no-map comparison column (B1) — existing runs, instrument re-run seeded

`runs/e12_base_seed{42,1337,2024}` (ViT; five seeds exist, the three shared
with A-ViT are used) and `runs/e14_base_seed{42,1337,2024}` (ResNet). Both
have era checkpoints. **Correction 2026-09-18 (catch 21 on this contract's
own caveat):** v2 as signed said these decompositions "carry the unrecorded
probe draw (R2)" and scheduled a seeded re-run. They do not. R2's draw is
`channel_decomp.load_task_data`'s first-4000 subsample of a shuffled loader;
`e12_decompose.py` / `e14_decompose.py` fit the probe on the **entire** task
train set through a `shuffle=False` loader
([e12_decompose.py:216-230](../scripts/e12_decompose.py#L216-L230)) — no
subset, no draw, `probe_subset_seed: None` means *nothing drawn*. The
existing `runs/e12_decomp_v2/base_*.json` and `runs/e14_decomp/base_*.json`
are the B1 column as they stand. The E23 decompositions of A-ViT / A-RN use
the same scripts, so the column is uniform by construction. (A reproduction
relaunch of one decomposition per backbone remains available as a catch-19
floor on the fp16 era reload across GPUs; it is not a provenance step.)

### Arm B — `har_subject` OFF, MLP backbone

`S72_OFF` ([modal_runner.py:1331](../modal_runner.py#L1331)) —
`configs/mafc_phase1.yaml`, `--benchmark har_subject --no-adapters` — with
`--backbone mlp --era-checkpoints --fp32-shadow`, seeds 42, 1337, 2024. Same
five tasks, same frozen channel-shift maps (fingerprint `3de66e205eb7`,
[har_subject.py:25-26](../src/data/har_subject.py#L25-L26)), same partition.
The LSTM comparator is the S72 OFF arm itself.

**One code change, opt-in.** [train.py:118-119](../scripts/train.py#L118-L119)
overwrites `adapters.dim` with `benchmark.input_size` (= 9) on HAR, and
[plcm.py:222-226](../src/models/plcm.py#L222-L226) builds
`MLPEncoder(input_dim=adapter_dim)`; `MLPEncoder` flattens `[batch, 128, 9]`
to 1152 internally ([mlp_base.py:45](../src/models/mlp_base.py#L45)). As-is,
arm B constructs a 9-input MLP for 1152-wide windows and fails at the first
forward. Fix: the MLP's `input_dim` is the flattened window size
(`seq_len × input_size`), taken from the benchmark, only when
`backbone == "mlp"`; every other path untouched. **Regression, asserted before
launch:** the flag-off path is bit-identical on `ckpt_e17_mlp_seed42` (MNIST,
where `784 = adapter_dim = input_dim`, so the change is a no-op there) and on
an S72 LSTM checkpoint (`backbone == "lstm"`, untouched). A fix that enables
the next experiment must not spend the proof the last one bought.

**Era/fp32-shadow uniformity — a sign-off choice.** The rule: fp32 shadow goes
on *every arm of a comparison or none*, because it reloads and refits inside
the loop. The B1 column's runs (E12, E14) have era checkpoints and **no
shadow**. Two consistent options:

| option | A-ViT / A-RN | B1 column | identity gate | cost |
|---|---|---|---|---|
| **(b) — RULED 2026-09-18** | `--era-checkpoints`, no shadow | as they exist | at the **measured fp16 reload floor per run** (E14: max 0.0040, `runs/MEMO_e14.md` §fp16 clause), not 1e-6 | 6 runs |
| (a) | `--era-checkpoints --fp32-shadow` | **relaunched** with the same | 1e-6 | 12 runs, ≈ 2× |

(b) is uniform on the axis that matters and half the price; (a) buys the
1e-6 gate. **Ruled (b):** a gate at the instrument's resolution (R3's principle) — 1e-6 would be a claim the fp16 reload cannot support; C-RELOAD prints the floor beside every check. Arm B is a within-benchmark comparison against S72, which has the
shadow, so arm B carries it regardless.

## 3. Preconditions — E14's form, cited, priced against existing artifacts

From [E14_resnet_prereg.md:87-93](E14_resnet_prereg.md#L87-L93):

- **P1 competence, scale-free.** Frozen-probe per-task DIAG ≥ **0.85**, and the
  trainable trunk within **5pp** of the frozen reference (catch 24: the
  reference is the trivial use of the pretrained trunk). For arm B the frozen
  reference is a linear probe on the flattened windows.
- **P2(a) forgetting exists.** Task-0 deployed accuracy ≥ **0.15** below its
  own ceiling. **Already read for A-ViT** from the existing permuted matrices:
  `runs/e12effb_seed{2024,1337,42}` task-0 drop **0.632 / 0.792 / 0.796**
  (unpermuted B1: 0.514–0.816; E14: 0.514–0.788). v1's "precondition at
  risk" was a property of the construction v1 chose.
- **P2(b) sequence-caused.** Repeat-task control (5 tasks of identical data,
  fresh heads) ≤ **0.05 absolute** — the bar E12 and E14 used, read 0.0040
  (`runs/MEMO_e12.md` §P2(b)) and 0.0010 (`runs/MEMO_e14.md` P2(b)). v1's
  "under 3% of the sequential drop" is withdrawn: at F_total = 0.15 it is
  0.0045, which E12's measured control clears by 0.0005 — a bar that fine
  detects noise. On B6 the control runs with **one fixed permutation** on all
  five tasks (identical data means identical presentation), `jobs_e12rptb` /
  `jobs_e14rpt` shape, 3 seeds.
- **P3 instrument exactness.** `head(captured_feature) == deployed_logits`,
  exact, on live inputs, asserted at the top of every consuming script, with
  the wrong-tensor positive control — re-derived per backbone: the ViT
  capture from E12, `avgpool→fc` from E14, and a **new** capture for the MLP
  on HAR with its own wrong-tensor control.
- **P4 configuration parity.** `arm` records `benchmark`, `shift_fingerprint`
  (or `no_shift`), `backbone`, `era_checkpoints`, `fp32_shadow`,
  `use_task_heads`, `torch_threads`. **Not** `input_map` / `label_space`: those
  fields do not exist in any artifact, and this contract does not add them —
  the construction is identified by `benchmark` + fingerprint, as every
  witness since S72 has been.

## 4. The measurement — displayed

Per §3.1, on the audited path (`channel_decomp.decompose`, `load_task_data`
with `probe_subset_seed`), per cell (seed × task k < T), for each arm:

```
F_enc   = acc_refit_ceiling − acc_refit_T          (era encoder refit − final encoder refit)
F_read  = acc_refit_T − acc_orig
R       = acc_refit_ceiling − acc_ceiling
share   = F_read / (F_enc + F_read)                  pooled-of-means, per seed, interval over seeds
```

On B6, `acc_refit_T` and `acc_orig` are computed **twice**, on two inputs:

```
raw      : x_k                       task-k test in its own frame P_k          (the paper's decomposition as run everywhere)
re-laid  : P_T · P_k⁻¹ · x_k         task-k test carried into the current frame   (C0deg's input path)
```

**Frame of the era terms, stated (2026-09-19, after D1 v1's pairing error):**
`acc_refit_ceiling` and `acc_ceiling` are computed on **x_k in its own frame
P_k, never re-laid** — the era encoder θ_k saw frames 0..k only and is at home
there. Only the θ_T-side terms are recomputed on the re-laid input, where θ_T
is at home. (E1) therefore compares the era refit in frame k with the final
refit in frame T on the *same samples*: both encoders in-distribution, same
content. A re-laid input fed to θ_k would put the era encoder on a frame it
never saw and measure nothing about drift.

```
```

using the module's own `apply_perm` / `invert_perm`
([split_cifar100.py:94-103](../src/data/split_cifar100.py#L94-L103)). No
parallel re-implementation (catch 32): `cifar_relayout(x, k, k) == x` exactly,
and `cifar_relayout(x_k, k, T)` equals the dataset's own rendering of the same
base image under `P_T`, bitwise, asserted before any decomposition is read.

**The crux — two displayed inequalities, both two-sided:**

```
(E1)  F_enc(B6, re-laid)  ≈  F_enc(B1)            |Δ| ≤ floor
(E2)  F_enc(B6, raw)      >  F_enc(B6, re-laid)    by more than floor
```

(E1) says the map removes exactly the presentation term and leaves the
content term the no-map cell already measured. (E2) says the map
*discriminates* on a pretrained backbone — the decomposition read in the wrong
frame attributes to the encoder what re-layout recovers. The floor for each is
`max(seed-interval half-width, R3's max(init spread, 1.96·SE_binomial))`,
computed per backbone before either is read.

**Arm B:** the full decomposition; comparator S72 OFF (seeded linear:
F_enc 0.087, F_read 0.306, **share 0.778** pooled-of-means,
`runs/e20/seeded/har_s72_off_linear.json`; ledger band 75–89%); plus C0deg
via `har_relayout` on the MLP checkpoints (the same screen as S72,
`scripts/cure_screen.py`, arm set `e10ec_seeded` shape).

Bridging is **not** an arm here. Its position is ruled (Ruling B, ledger row
"Bridging has no regime where the map is known"); nothing in E23 changes it.

## 5. Controls — each with the case where it must fail

| control | statement | must-fail / bar |
|---|---|---|
| **C-ID** map identity | at `P_k = I` (the B1 checkpoints) `cifar_relayout` is the identity and the re-laid decomposition equals the raw one | exact, per cell; a wrong permutation (`P_{k+1}` in place of `P_k`, permutation-only by construction) must read **below the right one in 19/19 tasks per seed**, stated here (CC′'s form) |
| **C-WIT** arm identity | `arm` fields per §3 P4, checked **across** the arms of each comparison, not within | any mismatch on `era_checkpoints`, `fp32_shadow`, `use_task_heads`, `backbone`, fingerprint → not compared |
| **C-PATH** | P3's assert, live, every arm | non-tautological: hooking the wrong tensor must fail it |
| **C-FLOOR** | relaunch one seed of A-ViT and one of A-RN identically at the reported threading; floors per arm × quantity | no delta below its floor is read; pinning is an axis |
| **C-DRAW** | the CIFAR decomposition path draws no probe subset (full task train set, sequential loader); every row records `probe_subset_seed: None` with that meaning, and arm B's HAR rows record 20260916 | a HAR row without the seed is not compared |
| **C-RELOAD** (option b) | fp16 era reload delta measured per run, printed beside every identity check | the identity gate's resolution is this number, not 1e-6 |

## 6. Readings, pre-committed — two-sided

| outcome | reading |
|---|---|
| (E1) and (E2) both hold, both backbones | The map is the discriminator; pretraining is not the explanation. The two-origins claim generalizes to pretrained encoders, on the benchmark where they were measured without it |
| (E2) holds, (E1) fails with F_enc(re-laid) **above** F_enc(B1) | Re-layout recovers the presentation term but training under layout shift damaged the pretrained features beyond what the no-map run shows — a second cost of the shift, on the encoder, that the map cannot undo. Reported as a third term |
| (E2) holds, (E1) fails with F_enc(re-laid) **below** F_enc(B1) | The permuted regime preserved features better than the unpermuted one. Unexpected; reported, not explained |
| (E2) fails — raw ≈ re-laid, both small | The pretrained encoder absorbs a within-patch permutation; there is nothing for the map to discriminate on this construction, and the finding is scoped to encoders that do not absorb the shift |
| (E2) fails — raw ≈ re-laid, both large | Re-layout does not recover on a pretrained backbone; the discriminator is scratch-scoped and §8 says so |
| holds on one backbone, not the other | reported per backbone; no pooling across them |
| Arm B share within floor of S72's 0.778 | the sensor result is not LSTM-specific |
| Arm B share outside it, either side | share is architecture-dependent on identical data; a Table 1 row, sign reported |

## 7. Predictions — registered before the runs

*Calibration series, read from `docs/appendix.tex` (`app:predictions`) after
E21's eight rows were entered and the scoring convention was applied uniformly,
2026-09-18:* **"Thirty-nine scored entries, six misses (2026-09-18)."** *(At
signing the line read "Thirty-one scored entries, seven misses" and this
contract said E21 would make it 36 / 10 — a count taken from the row script's
`MISS` label, which marks every non-event. The table's convention — a miss is
a prediction given above even odds that did not occur — re-read from the table
after the rows landed gave 39 / 8, and applying that convention to two 40%
entries the old count had carried as misses gives 39 / 6. Three counts in one
day, each replaced by reading the table under a stated rule; the rule now
stands in the appendix text.)*

Odds marked † were proposed at drafting and accepted at sign-off (one moved, marked); the others carry over from v1 where the clause survived.

| prediction | odds |
|---|---|
| P1 passes on both A arms (the E12/E14 trunks on the same data passed it) | ~90% † |
| (E2) holds on A-ViT: raw F_enc exceeds re-laid by more than floor | ~75% † — the patch embedding is linear, so a within-patch permutation is a linear re-mixing the fine-tuned embedding may partially absorb; E5d's `A_k ≈ I` says scratch LSTMs absorb little |
| (E2) holds on A-RN | ~80% † — a convolutional stem has no linear absorption path for a pixel permutation |
| (E1) holds on A-ViT, \|Δ\| ≤ floor | ~45% † — the honest number; the "above" branch is the live alternative |
| (E1) holds on A-RN | ~50% † |
| F_enc(B6, re-laid) − F_enc(B1) is **positive** on both backbones (the encoder-damage branch), whether or not it clears the floor | **~60%** (proposed 55%, moved at sign-off: twenty tasks under layout shift is a lot of drift for a fine-tuned trunk to absorb without cost) |
| Arm B passes all preconditions | ~80% |
| Arm B share within 10 points of S72's 0.778 | ~45% — E18's MLP/LSTM out-of-frame costs differ 7× (2.5 vs 17.6pp, `runs/MEMO_e18.md` §1) |
| Arm B F_enc below S72's 0.087 | ~65% |
| C-ID wrong-permutation must-fail passes 19/19 on every seed and backbone | ~90% † |

## 8. CLAUSE → JOB → ARTIFACT (catch 33) and the audit table (catch 20)

| clause | launcher / script | artifact |
|---|---|---|
| §2 A-ViT relaunch | `jobs_e23_vit` = `jobs_e12effb` + `--era-checkpoints` (+ shadow under (a)) | `runs/e23_vit_seed{s}/` + `ckpt_e23_vit_seed{s}/` |
| §2 A-RN | `jobs_e23_rn` = `E14_CFG + cifar100_permuted + --era-checkpoints` | `runs/e23_rn_seed{s}/` + checkpoints |
| §2 build gate | `jobs_e23_smoke` — **both** backbones, 1 epoch, 2 tasks, seed 42 (run 2026-09-19 19:03: `resnet`/`vit`, `cifar100_permuted`, per-task heads routed by hint, adapters off, `era_checkpoints: True`, era files `task{0,1}_epoch0.pt` written); `expected_shift` added to `configs/e14_resnet.yaml` (E12's values, verified locally to be the ResNet construction's too) | `runs/e23_{rn,vit}_smoke/mafc_results.json` |
| §2 Arm B code change | opt-in `mlp_input_dim` (`src/models/plcm.py`, `scripts/train.py` HAR branch); regression run inline 2026-09-19 | `runs/e23/mlp_regression.json` — logits bit-identical pre/post on `ckpt_e17_mlp_floor_a` (MNIST MLP, seed-42 replicate; `ckpt_e17_mlp_seed42` is not local) and `ckpt_e10off_ec_seed42` (S72 LSTM); HAR-MLP builds at 1152 and forwards; without the opt-in it fails at the first forward |
| §2 Arm B | `jobs_e23_har_mlp` = `_s72(S72_OFF + --backbone mlp)` (era + shadow asserted), 3 seeds + `e23_har_mlp_floor4tec_a`; launched 2026-09-19 20:05 | `runs/e23_har_mlp_seed{s}/`, `runs/e23_har_mlp_floor4tec_a/` |
| §3 P1 frozen references | `jobs_e23_frozen` (A arms) + `jobs_e23_har_ref` → `scripts/e23_har_linear_ref.py` (arm B: linear readout on flattened windows) → `scripts/e23_frozen_probe.py` (**new, 2026-09-19:** the E12/E14 frozen-probe scripts extract *unpermuted* features once and are the B1 reference only; on B6 each task's images carry that task's permutation, so the reference extracts per task through the benchmark's own loaders with the shift fingerprint gated). Arm B's MLP reference: a linear probe on flattened windows, with arm B | `runs/e23/frozen_{vit,resnet}.json` |
| §3 P2(b) repeat-task on B6 | `jobs_e23_rpt` (both backbones, 3 seeds, 5 tasks). **Catch 33 at build time (2026-09-19):** as signed, no working job existed — `--repeat-task` on `cifar100_permuted` repeated task 0's *classes* but drew a fresh permutation per task ([split_cifar100.py:174-180](../src/data/split_cifar100.py#L174-L180)), a shifted sequence rather than a control. Fixed by an opt-in in the dataset (repeat_task **and** shift_mode → every task carries task 1's B6 permutation); regression: the three registered shift fingerprints and the class order unchanged, the unshifted E12/E14 control unchanged (perms all `None`), the permuted control's five perms identical to the registered construction's task-1 perm | `runs/e23_rpt_{vit,rn}_seed{s}/` |
| §3 P3 capture asserts | top of `scripts/e23_decompose.py`, per backbone | printed + `runs/e23/path_identity.json` |
| §4 decomposition, raw + re-laid (B1 column: existing `e12_decomp_v2` / `e14_decomp` artifacts, no re-run — see §2 correction) | `jobs_e23_dec` → `scripts/e23_decompose.py` (imports the backbone's own `load_era` / `features_and_logits` / P3 gates; re-laid input = the dataset's own rendering under P_T, asserted bitwise equal to `apply_perm`/`invert_perm`; wrong-permutation must-fail per task) | `runs/e23/e23_{vit,rn}/decomp_seed{s}.json` |
| §4 arm B C0deg | `scripts/cure_screen.py` on the MLP checkpoints | `runs/e23/har_mlp_screen.json` |
| §5 C-ID incl. wrong-perm must-fail | first block of `e23_decompose.py` | `runs/e23/controls.json` |
| §5 C-FLOOR | `jobs_e23_floor` (inside `jobs_e23_main`: seed 42 of each backbone relaunched, era checkpoints on) + `jobs_e23_dec`'s `floor42` jobs (the relaunch decomposed, for the F_enc floor) | `runs/e23_{vit,rn}_floor_seed42/`, `runs/e23/e23_{vit,rn}/decomp_floor42.json` |
| §6 row | `scripts/e23_row.py` (written 2026-09-19 before any B6 artifact was read; the (E1)/(E2) floor is stated in its docstring) | `runs/e23_row.json` |
| audit | `scripts/audit_checkpoints.py` on every checkpoint above, by `PLCM.load_era` | `runs/e23/ckpt_audit.json` |

**Checkpoint audit table, as of writing:**

| arm × seed | exists | loads | era ckpts | shadow | usable for |
|---|---|---|---|---|---|
| `e12_base_{42,1337,2024}` | yes | to verify by loader | yes | no | B1 column (ViT), seeded re-run |
| `e14_base_{42,1337,2024}` | yes | to verify by loader | yes | no | B1 column (RN), seeded re-run |
| `e12effb_{42,1337,2024}` | matrices only | — | **no** | no | P2(a) read; **not** decomposable — hence the relaunch |
| A-ViT, A-RN, arm B, floors, rpt | to be built | — | — | — | — |

The audit is a precondition on *reading*; the two "to verify" rows are run
before the headline jobs launch, since 14 of 33 E11-era checkpoints once
failed the loader.

## 9. Cost — from the artifacts

| item | basis | GPU-h |
|---|---|---|
| A-ViT, 3 seeds | `training_time` 142–144 min per E12 run (`runs/e12eff_seed42` etc.) | 7.2 |
| A-RN, 3 seeds | 60–67 min per E14 run | 3.3 |
| floors, 1 seed each | same | 3.5 |
| P2(b) repeat-task on B6, 5 tasks × 2 backbones × 3 seeds | ¼ of a 20-task run | 2.6 |
| decompositions (raw, re-laid; B1 column exists), `analyze_one_gpu` | E12's decomposition budget | ~1 |
| **total, option (b)** | | **≈ 17.6 GPU-h** |
| option (a) adds B1 relaunches with shadow | 6 runs | +10.5 |
| Arm B (CPU): 3 runs + floor + frozen ref | S72 timings | ≈ 3 CPU-h |

## 10. Scope

**Grid, displayed** (backbone × benchmark; ✓ measured, **E23** this contract,
— not filled):

| | HAR (sensor) | Permuted MNIST | Rotated MNIST | CIFAR-100 |
|---|---|---|---|---|
| LSTM (scratch) | ✓ S72 | ✓ E18 | — | — |
| MLP (scratch) | **E23 B** | ✓ E18 | ✓ E18 | — |
| ViT-B/16 (pretrained) | — | — | — | ✓ B1 · **E23 B6** |
| ResNet-50 (pretrained) | — | — | — | ✓ B1 · **E23 B6** |

Eight cells stay empty. Pretrained image backbones on 9-channel windows and an
LSTM on 224×224 images are architecture–data mismatches that would fail P1 by
construction; the scratch MLP/LSTM on CIFAR-100 have no measured competence
number in this repo, so no claim about them is made here.

**Not tested here:** an *estimated* map (contract M, redesigned), the
wrong-locus cost on these arms (W's row script applies once the checkpoints
exist), the interference regime (E22).

## 11. Sign-off items — all resolved 2026-09-18

1. Option **(b)** in §2.
2. The † odds in §7 accepted; the encoder-damage-sign prediction moved 55 → 60%.
3. The MLP `input_dim` change approved as a build step under the amendment
   rule, with the two-checkpoint bit-identity regression (the shape E19's flag
   will need too).
