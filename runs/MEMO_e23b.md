# MEMO — E23-B: sequence length does not explain E23

**Status (2026-09-20, 01:15):** run and read. Contract
`docs/E23B_seqlen_prereg.md` (registered before launch, two false starts
recorded there). Artifacts `runs/e23b/`, row `runs/e23b_row.json`
(`scripts/e23_seqlen.py` → `e23_seqlen_row.py`). **Verdict: the pre-committed
reading selects TRAINING REGIME.** §4.2's scope sentence and §8's limitation
can now be written.

## 0. The question

E23 found that on pretrained backbones, re-laying task-k content into the
current frame **recovers nothing and costs**, where on every scratch arm it
takes F_enc to zero. The two regimes differed in a second variable — five
tasks versus twenty — and a third moved with it (12k versus 3k images per
task). This contract separates all three.

## 1. Result — three scratch points, one factor at a time

Pooled over cells, 3 seeds each; floor = max(t95 half-width over seeds, mean
per-cell 1.96·SE). Both new arms' floor replicates are **bit-identical** to
their seed-42 runs, so the reproduction floor is 0.

| arm | T | imgs/task | DIAG | raw F_enc | **re-laid F_enc** | floor | deployed raw → re-laid | cells improved |
|---|---|---|---|---|---|---|---|---|
| **A1** `e18_pmd_lstm` | 5 | 12,000 | 0.937 | +0.172 | **+0.011** | 0.011 | 0.195 → **0.935** (+74.0) | 12/12 |
| **A2** `e23b_t5c20` | 5 | 3,000 | 0.828 | +0.126 | **−0.005** | 0.033 | 0.159 → **0.844** (+68.5) | 12/12 |
| **A3** `e23b_t20` | 20 | 3,000 | 0.858 | +0.169 | **−0.006** | 0.030 | 0.161 → **0.878** (+71.7) | 57/57 |
| ViT-B/16 (E23) | 20 | — | 0.817 | +0.080 | **+0.099** | 0.044 | 0.271 → **0.208** (−6.3) | 13/57 |
| ResNet-50 (E23) | 20 | — | 0.760 | +0.012 | **+0.025** | 0.020 | 0.258 → **0.234** (−2.4) | 15/57 |

**At the same sequence length (twenty tasks, twenty layouts), the scratch LSTM
re-lays to +71.7 pp of deployed accuracy and F_enc → 0 within floor, while
both pretrained backbones lose accuracy and F_enc rises.** Twenty tasks do not
prevent overwriting. Images-per-task does not either (A2). The variable is the
training regime.

Per seed, A3's re-laid F_enc is −0.007 / −0.012 / +0.000 (raw +0.161 / +0.164
/ +0.183); every one of its 57 cells improves under re-layout, minimum gain
+37.8 pp. The pretrained arms improve in 13/57 and 15/57.

**A1 sits on the boundary and is reported as such.** Its re-laid F_enc is
+0.0108 against a floor of 0.0107 — over by one ten-thousandth, so the row
prints "removes the encoder term: NO" for A1 and YES for A2/A3. D1 read the
same arm as "within resolution" (+0.011 vs res 0.011); both statements
describe one number at its resolution, and neither is re-barred. Nothing in
the reading turns on it: A1's deployed accuracy rises 74 pp under the same
operation, and the two arms that carry the comparison are clearly inside.

## 2. Controls

| control | A1 | A2 | A3 |
|---|---|---|---|
| C-ID vs `runs/e18_pmd_lstm/decomp.json` | **PASS**, 0.0e+00, 12/12 | n/a | n/a |
| C-CONSTR (`num_tasks`, `content_chunks`, `construction_fingerprint`) | PASS | PASS | PASS |
| C-WIT (shared head, adapters off, era + shadow, 4 threads) | PASS | PASS | PASS |
| C-RELAY (k→k identity; k→T→k round trip) | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| identity residual | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| C-FLOOR (seed-42 replicate) | (E18) | **bit-identical** | **bit-identical** |
| **must-fail: wrong source permutation** | **PASS 4/4** | **PASS 4/4** | **PASS 19/19** ×3 seeds |

The wrong-permutation must-fail **passes here and failed on both pretrained
arms** (7–13 of 19 tasks). That is the same fact from the other side: the
control can only discriminate where the right map recovers something. On the
scratch arms the right map takes deployed accuracy to 0.84–0.94 and the wrong
one leaves it at 0.11–0.14; on the pretrained arms right and wrong are
indistinguishable because neither recovers.

## 3. Predictions — 6 scored, 5 fired, 1 miss

| prediction | odds | outcome |
|---|---|---|
| A3 (T = 20): re-laying still removes the encoder term | 55% | **fired** (−0.006 vs floor 0.030) |
| A2 (T = 5, 3k): re-laying removes it | 85% | **fired** (−0.005 vs 0.033) |
| A3: re-laid deployed accuracy above raw | 70% | **fired** (+71.7 pp, 57/57 cells) |
| total forgetting higher at T = 20 than T = 5, same images/task | 75% | **fired** (0.697 vs 0.665) |
| A2's F_enc within 2 pp of A1's | 50% | **miss** (0.126 vs 0.172 — 4.6 pp) |
| wrong-permutation must-fail passes on both new arms | 80% | **fired** |

The miss is informative and small: cutting images per task from 12k to 3k
**lowers** the raw encoder term by 4.6 pp (0.172 → 0.126) while leaving the
re-laid term at zero. Less data per task means less to encode and less to
lose, and it does not touch the map's ability to undo the presentation.

## 4. What this decides

**§4.2 / §8 scope sentence: training regime, not sequence length.** The
discriminator "apply the known map and re-measure" works on encoders
overwritten to the current frame — measured now at 5 and 20 tasks, at 12k and
3k images per task, on LSTM and MLP, on MNIST and HAR — and does not work on
pretrained trunks fine-tuned across the same twenty layouts, where re-laying
costs deployed accuracy and raises F_enc. E23's reading stands and its
alternative explanation is ruled out.

**The mechanism sentence the pair now supports:** a scratch encoder ends in
the last task's frame, so carrying old content into that frame restores it; a
pretrained trunk fine-tuned over twenty layouts ends in none of them in
particular — it reads task-k content better in P_k than in P_T — so there is
no current frame to carry anything into. The competence gap E23 measured
(pretrained DIAG 0.933 → 0.817 under permutation) and the crux failure are the
same phenomenon, and the scratch arms show it is not about how many layouts.

## 5. Process — the witness's two halves

Both false starts in this contract were witness failures, one per half:

1. **Computation.** `content_fingerprint` hashes the concatenated index
   tensors, so T = 5 and T = 20 collide (`c1a570a02d35` at seed 42) while
   sharing no per-task content — the guard was blind to the axis being
   varied. Fixed additively (`construction_fingerprint`), old hash kept
   byte-compatible so E18's artifacts stay readable.
2. **Recording.** `ContinualTrainer.arm_provenance` copies a whitelist of
   benchmark keys, so the new fields were set in the config, used to build
   the benchmark, and dropped before the artifact: eight correct runs whose
   own record could not say which construction they were. Re-ran rather than
   argue that `(num_tasks, content_fingerprint)` is injective over three arms.

*A witness has two halves — the computation and the recording — and fixing
one does not fix the other.* Both are now in CLAUDE.md. A3's artifacts carry
`content_fingerprint c1a570a02d35` beside `construction_fingerprint
9e33ed6ae4c5`, so the collision is visible in the files rather than only in
this memo.

(A third, unrelated: the first launch went out against a `train.py` broken by
an unanchored string-index insertion — CLAUDE.md's existing rule, third
instance. `train.py --help` now runs before any launcher edit ships.)

## 6. Artifacts

`runs/e23b_{t5c20,t20}_seed{42,1337,2024}/`, `..._floor4tec_a/`;
`runs/ckpt_e23b_*/mafc_seed*_fp32/`; `runs/e23b/{e18_pmd_lstm,e23b_t5c20,e23b_t20}/decomp_seed{s}.json`;
`runs/e23b/construction_regression.json`; `runs/e23b_row.json`. Scripts:
`scripts/e23_seqlen.py`, `scripts/e23_seqlen_row.py`; launchers
`jobs_e23b_seqlen`, `jobs_e23b_dec`. Construction: `content_chunks` in
`src/data/permuted_mnist.py` + `scripts/train.py`; record in
`src/training/trainer.py`.
