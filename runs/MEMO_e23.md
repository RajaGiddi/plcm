# MEMO — E23: Closing the Pretraining Confound

**Status (2026-09-20):** every contracted run and analysis landed and read
(`runs/e23_row.json`, `scripts/e23_row.py` — written before any B6 artifact
existed). **P1 ruled** (§1): the crux is read with P1 FAIL recorded and the
competence gap beside it; appendix rows and the ledger row entered. §8's scope
sentence is settled by E23-B (`runs/MEMO_e23b.md`): **training regime**.

Contract: `docs/E23_prereg.md` (signed v2, option (b)). Launch record: build
gate 19:03, headline 19:10 (8 L4), repeat controls / references / arm B
19:26–20:08, decompositions 23:44 (after one crash: the dataset's `apply_perm`
is per-image and the first relay code handed it a batch — fixed, tested
locally against the dataset's own P_T rendering to 0.0, respawned).

## 0. Runs and controls

18/18 runs pass the finish definition at their own shapes (20×5, 5×5, 5×10).
Arm identity from every artifact; the fingerprint gate asserted on every
20-task run (it fires before training; all eight wrote era checkpoints).

| control | ViT-B/16 (A-ViT vs B1 `e12_base`) | ResNet-50 (A-RN vs B1 `e14_base`) | arm B (MLP·HAR vs S72 LSTM) |
|---|---|---|---|
| C-WIT (heads, era, no shadow, fp16, backbone; across the arms of the comparison) | PASS | PASS | PASS (era + shadow + 4 threads, as S72) |
| C-FLOOR, seed-42 relaunch: matrix | 9/400 identical, max cell 0.284, **AVG +2.9 pp** | 12/400, max 0.124, AVG −0.1 pp | **identical** (AVG 0.4413 = 0.4413) |
| C-FLOOR on F_enc (main vs relaunch, pooled) | raw 0.016, re-laid **0.034** | raw 0.002, re-laid 0.010 | — |
| C-RELOAD fp16 reload floor (max) | 0.002 | 0.008 | 0.0 (shadow) |
| C-ID relay: k→k identity; algebraic vs the dataset's own P_T rendering | 0.0 / 0.0, every seed | 0.0 / 0.0 | n/a |
| **C-ID must-fail: wrong source permutation below the right one, 19/19** | **FAIL** 9–13 of 19 tasks per seed | **FAIL** 7–13 of 19 | n/a |
| P2(a) task-0 drop ≥ 0.15 | 0.700 PASS | 0.707 PASS | 0.167 PASS |
| P2(b) permuted repeat-task control ≤ 0.05 (opt-in fixed perm) | 0.005 PASS | 0.002 PASS | not contracted |
| **P1** reference ≥ 0.85 and trainable within 5 pp | frozen **0.722** (min 0.57); trainable 0.817, **above by 9.6** → FAIL | frozen **0.700** (min 0.54); trainable 0.760, above by 6.0 → FAIL | linear-on-windows **0.553**; MLP 0.791, above by 24 → FAIL |

**The wrong-permutation must-fail failed for CC's reason, not the
instrument's.** The relay is exact (0.0 against the dataset's own rendering,
every seed). The control's premise was that re-laying with the *right* map
recovers, so a wrong map must read below it; on these backbones re-laying
with the right map recovers nothing (§2), so wrong and right are
indistinguishable — pooled deployed accuracy 0.213 (wrong) vs 0.209 (right) on
ViT, 0.233 vs 0.234 on ResNet. Recorded FAIL with cause; the bar does not move.

## 1. P1 — RULED 2026-09-20: (a) for the A arms, (c) for arm B

*The arms are competent — they beat the reference by 6–24 pp. What failed is a
bar imported from a context where the reference could read the input: R3's
category, a defective gate, not a failed arm.* P1 recorded FAIL with cause on
the A arms and the competence gap printed beside (E1)/(E2); arm B read against
the S72 LSTM (0.837; within 4.6 pp), the imported 0.85 bar recorded as the
contract's error. **Reading added at the ruling:** the competence gap and the
crux failure may be one phenomenon — a trunk trained across twenty
permutations is spread over twenty layouts rather than specialized to the
last, which predicts both the lower DIAG (0.817 vs B1's 0.933) and "nothing
to re-lay into". **Confound named at the ruling:** the scratch arms have five
tasks, the pretrained arms twenty; "five permutations can be overwritten to
the last and twenty cannot" explains the crux without pretraining. Test:
the scratch LSTM on Permuted MNIST at twenty tasks (§5). §8's sentence waits
on it.

### 1a. As drafted before the ruling

The precondition fails on all three arms in one way: the contracted
**reference** cannot read the construction, and the trainable arm beats it.
The frozen ImageNet trunks read 0.97 / 0.96 on unpermuted CIFAR (B1) and
0.72 / 0.70 once each task's patches are permuted; arm B's linear-on-windows
reference reads 0.55. The 0.85 bar and the 5 pp margin were earned on
pretrained trunks reading natural images (E12/E14) and imported to B6 and to
a scratch MLP unchanged — catch 21's "validated is context-bound", on a bar.
As signed, *the pairing is void* on every arm.

The competence gap is itself a finding, whichever way the ruling goes: **the
permutation costs the pretrained trunks 12–19 pp of task competence** (ViT
DIAG 0.933 → 0.817, ResNet 0.945 → 0.760) that no amount of fine-tuning
recovered in 5 epochs, and the frozen trunks 25 pp.

Options, stated before the crux is read: **(a)** read (E1)/(E2) with P1
recorded FAIL and the competence gap printed beside them — B6 and B1 are arms
of different competence, so the F_enc comparison carries that qualifier;
**(b)** void the A-arm pairing as contracted and report §2 as descriptive
only; **(c)** for arm B, the scale-free form the rule intended — a scratch
MLP's reference on HAR is the S72 LSTM (DIAG 0.837), against which arm B's
0.791 is within 4.6 pp — with the imported 0.85 bar recorded as the
contract's error.

## 2. The crux — measured, read under the ruling

Pooled over 57 cells (3 seeds × 19 old tasks), pretrained arms; floor =
max(t95 half-width over seeds, binomial, F_enc relaunch floor):

| | ViT-B/16 | ResNet-50 |
|---|---|---|
| ceiling (era, frame k) | 0.816 | 0.756 |
| **raw** (θ_T on x_k in P_k): deployed → refit; F_enc; F_read | 0.271 → 0.768; **+0.080**; 0.498 | 0.258 → 0.745; **+0.012**; 0.487 |
| **re-laid** (θ_T on x_k under P_T): deployed → refit; F_enc; F_read | **0.208** → 0.750; **+0.099**; 0.541 | **0.234** → 0.732; **+0.025**; 0.497 |
| B1 (no map): deployed → refit; F_enc; F_read | 0.222 → 0.895; +0.056; 0.673 | 0.293 → 0.936; +0.008; 0.643 |
| share, raw / re-laid / B1 | 86.2 / 84.6 / 92 | 97.6 / 95.2 / 99 |
| (E1) \|F_enc(re-laid) − F_enc(B1)\| vs floor | 0.043 vs 0.044 → holds, at the floor; sign **+** on 3/3 seeds | 0.018 vs 0.020 → holds; sign + on 3/3 |
| (E2) F_enc(raw) − F_enc(re-laid) vs floor | **−0.019** vs 0.023 → fails | **−0.013** vs 0.013 → fails |
| cells where re-laying helps the deployed head / the refit | 13/57 / 15/57 | 15/57 / 18/57 |

**What the numbers say.** On both pretrained backbones **re-laying task-k
content into the current frame recovers nothing and costs**: deployed
accuracy falls 6.3 pp (ViT) and 2.4 pp (ResNet), the refit falls 1.9 / 1.3 pp,
and F_enc *rises* — the opposite of every scratch arm, where the same
operation took F_enc to zero within resolution on 4/4 arms (D1 §2). The final
pretrained trunk is not "in frame P_T": it reads task-k content **better in
P_k than in P_T**, i.e. it retained a per-task handling of twenty layouts
rather than being overwritten to the last one. (E2)'s registered "fails"
branch: *re-layout does not recover on a pretrained backbone; the map is not
the discriminator here.* (E1) holds only because the ViT floor is wide (seed
spread 0.044; its F_enc relaunch floor alone is 0.034): the point estimates
put B6's encoder term **above** B1's on every seed of both backbones (+2.2 to
+5.3 pp ViT, +0.8 to +2.3 pp ResNet), the "above" branch — training under
layout shift left a larger encoder term than the no-map run — as a
measurement, not a cleared claim.

**The two-origins claim, restated with its scope.** The map removes the
encoder term where the encoder was overwritten to the current frame
(scratch LSTM/MLP: F_enc → 0 on 4/4 arms). Where a pretrained trunk keeps
every frame, F_enc is small in every frame (0.01–0.10) and the map has
nothing to remove; the readout is stranded regardless (share 85–98%). The
discriminator "apply the map and re-measure" is therefore **scratch-scoped**
on the evidence, and §8 says so; the pretraining confound is closed in the
direction the contract did not expect.

## 3. Arm B — the sensor result is not LSTM-specific; it is *more* reader-dominated

`runs/e20/seeded/har_e23_mlp_linear.json` (same instrument, draw, partition;
identity 0.0; path identity per cell):

| | MLP · HAR (arm B) | LSTM · HAR (S72) |
|---|---|---|
| F_enc / F_read / R / F_total | **−0.001** / 0.373 / −0.020 / 0.392 | 0.087 / 0.306 / +0.007 / 0.386 |
| share, pooled [t95] | **100.2** [85.6, 114.2] (104.8 / 93.6 / 101.4) | 77.8 [49.7, 103.7] |

Same total forgetting; on the MLP all of it is reader. "F_enc below the
LSTM's" fired (65%); "share within 10 points" did not fire (45%) — it is 22
points off, in the reader direction; the C0deg screen on these checkpoints is
still to run (§5).

## 4. Predictions — 10 scored under the contract's convention

| prediction | odds | outcome |
|---|---|---|
| P1 passes on both A arms | 90% | **miss** |
| (E2) holds on A-ViT | 75% | **miss** (re-laid worse) |
| (E2) holds on A-RN | 80% | **miss** |
| (E1) holds on A-ViT | 45% | fired (at the floor) |
| (E1) holds on A-RN | 50% | fired |
| F_enc(B6 re-laid) − F_enc(B1) positive, both backbones | 60% | fired (3/3 seeds each) |
| arm B passes all preconditions | 80% | fired under the ruled reference (S72 LSTM, within 4.6 pp); the contracted 0.85 bar was the defective gate |
| arm B share within 10 points of 0.778 | 45% | did not fire |
| arm B F_enc below 0.087 | 65% | fired |
| wrong-permutation must-fail 19/19 | 90% | **miss** (cause: §0) |

Four misses in one contract (five under the contracted arm-B bar), all pointing the same way: the pretrained
regime does not behave like the scratch regime under a known map, and the
contract's priors — and its imported bars — assumed it would. Entered in the appendix 2026-09-20 under the ruling.

## 5. Open

- ~~§8's scope sentence: after the twenty-task scratch control~~ — **DONE 2026-09-20**: `runs/MEMO_e23b.md` rules out sequence length (T=20 scratch LSTM: F_enc → 0, +71.7 pp deployed, 57/57 cells). §4.2/§8 say **training regime**.
- **Twenty-task scratch control:** scratch LSTM on disjoint-content Permuted MNIST at T = 20 (the E18 construction with `--num-tasks 20`; 3k train / 500 test per task), era + shadow, 3 seeds; the E18 screen (C0deg) and the re-laid F_enc at T = 20. Pre-registered in `docs/E23_prereg.md` §10 before launch.
- Arm B's C0deg screen (`cure_screen.py` on `ckpt_e23_har_mlp_*`) — the
  re-layout column for a Table-1-style row; minutes, x86.
- Contract §8's "A-RN with `shift_mode=global`" was not run (declined at
  sign-off for uniformity); the within-patch permutation is the only shift
  tested, and the ResNet reads it as a 12 pp competence cost without a
  linear absorption path.

## 6. Artifacts

`runs/e23_{vit,rn}_seed{s}/`, `runs/e23_{vit,rn}_floor_seed42/`,
`runs/e23_rpt_{vit,rn}_seed{s}/`, `runs/e23_har_mlp_seed{s}/`,
`runs/e23_har_mlp_floor4tec_a/`, `runs/e23_{rn,vit}_smoke/`;
`runs/e23/e23_{vit,rn}/decomp_seed{s}.json` + `decomp_floor42.json`;
`runs/e23/frozen_{vit,resnet,mlp}.json`; `runs/e23/mlp_regression.json`;
`runs/e20/seeded/har_e23_mlp_linear.json`; `runs/e23_row.json`. Scripts:
`e23_decompose.py`, `e23_row.py`, `e23_frozen_probe.py`,
`e23_har_linear_ref.py`; launchers `jobs_e23_{smoke,main,rpt,frozen,dec,har_mlp,har_ref,har_dec}`.
