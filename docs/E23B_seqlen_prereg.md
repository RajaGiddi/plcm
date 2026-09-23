# E23-B — Is E23's crux failure about pretraining or about sequence length?

**Status:** pre-registration, written and registered **before launch**,
2026-09-20; **launched 00:28** after one false start. *Process record:* the
first launch (00:25) went out against a `scripts/train.py` broken by my own
unanchored string-index insertion — the new flag landed inside the previous
flag's help string. `ast.parse` caught it only *after* the spawn; the eight
jobs died on import and left empty directories and nothing else (the finish
definition's point: a directory proves a job launched). The rule already in
CLAUDE.md names this exact failure — *"a verification must EXERCISE the
failure mode … the unanchored `.replace()`"* — and the check that would have
caught it is the one now run before any launcher edit ships: `train.py
--help`, which builds the parser and resolves every string. Analysis-plus-training: six CPU runs on an existing construction,
one opt-in dataset parameter with its regression. Names the confound the E23
ruling named.

## 0a. Second false start — the field that was set but never recorded

The 00:28 launch trained all eight runs correctly (every recorded
`content_fingerprint` matches its rebuilt construction: A2 at 3k images/task,
A3 at T = 20), and **no artifact could say so**. `ContinualTrainer.arm_provenance`
copies a whitelist of benchmark keys into `arm` — `disjoint_content`,
`content_fingerprint`, `angles` — so the two fields this contract added
(`content_chunks`, `construction_fingerprint`) were set in the config, used by
the benchmark, and dropped on the way to the record: every artifact read
`content_chunks: None`. C-CONSTR as written could not pass on runs that were
in fact correct.

It is the same shape as the fingerprint defect one level up: *the fix for a
blind witness was itself not written into the witness.* Recorded rather than
argued around — the whitelist now carries both fields (`src/training/trainer.py`),
and the eight runs were **re-launched at 00:51** so the artifacts satisfy
C-CONSTR as written rather than by an argument about which pair of fields
happens to be injective over three arms. The first set's numbers are not read.

## 1. The confound, stated

E23 measured that on **pretrained** backbones (ViT-B/16, ResNet-50), re-laying
task-k content into the current frame with the known map **recovers nothing
and costs** — deployed accuracy −6.3 / −2.4 pp, F_enc rises (0.080 → 0.099,
0.012 → 0.025). On every **scratch** arm the same operation takes F_enc to
zero within resolution (D1 §2, 4/4 arms). The contract read that as
*pretraining*, and §4.2's scope sentence would say "encoders overwritten to
the current frame".

But the two regimes differ in a second variable: **the scratch arms have five
tasks, the pretrained arms twenty.** "Twenty permutations cannot be
overwritten to the last one; five can" explains the whole result without
pretraining. The B6-vs-B1 comparison inside E23 is clean (both twenty), but
the comparison that carries §4.2's claim is B6 against the scratch arms, and
that varies regime *and* sequence length.

**A third variable moves with the second.** `disjoint_content` splits 60k/10k
MNIST images into `num_tasks` chunks, so T = 5 gives 12k images per task and
T = 20 gives 3k. A T = 20 control alone would vary sequence length *and*
images-per-task. Fixed by an opt-in parameter (`content_chunks`, 2026-09-20):
split into C chunks and use the first `num_tasks`, so (T = 5, C = 20) has five
tasks of the same 3k chunks and the same permutations as the first five tasks
of (T = 20, C = 20). Verified: identical index sets and permutations,
default path byte-identical (`content_fingerprint` still `c1a570a02d35` at
seed 42, 12k/task).

**Defect found while building this control, recorded:** `content_fingerprint`
hashes the concatenation of the index tensors, so **chunking one permutation
into 5 or 20 pieces gives the same hash** — T = 5 and T = 20 both read
`c1a570a02d35` while sharing no per-task content. A consumer that rebuilt the
wrong T would have passed the construction assert. Kept byte-compatible (every
E18 artifact records it); `construction_fingerprint()` added beside it (hashes
the split's shape too: 3/3 distinct where the old one gives 2/3), and
`screen_e18` now asserts `num_tasks` and `content_chunks` as well.

## 2. Arms — three points, one factor at a time

Scratch LSTM, disjoint-content Permuted MNIST, adapters off, shared head, era
checkpoints + fp32 shadow, 4 threads, seeds 42/1337/2024 — i.e. `E18_PMD_LSTM`
with the flags below. Training compute is matched by construction: T × 3k =
T × images/task is 60k image-presentations per epoch sweep either way.

| arm | tasks | images/task | flags | status |
|---|---|---|---|---|
| **A1** `e18_pmd_lstm` | 5 | 12,000 | (as run) | **exists** (`runs/e18_pmd_lstm_seed*`, D1/E18) |
| **A2** `e23b_t5c20` | 5 | 3,000 | `--num-tasks 5 --content-chunks 20` | new, 3 seeds + floor replicate |
| **A3** `e23b_t20` | 20 | 3,000 | `--num-tasks 20 --content-chunks 20` | new, 3 seeds + floor replicate |

A2 vs A1 isolates images-per-task; A3 vs A2 isolates sequence length with
content and permutations held fixed on the shared tasks.

## 3. The measurement — the same quantities E23 read, displayed

Per arm, seed, old task k < T, on the task-k test set, through the audited
loader and `PLCM.load_era`, path identity asserted per cell:

```
raw       acc_orig      = acc( deployed θ_T, h_T  on x_k in FRAME k )
re-laid   acc_relaid    = acc( deployed θ_T, h_T  on mnist_relayout(x_k, k → T) )     [C0deg's input path]
ceiling   acc_ceiling   = acc( era θ_k, h_k on x_k )
F_enc          = refit(θ_k features) − refit(θ_T features, frame k)
F_enc_relaid   = refit(θ_k features) − refit(θ_T features, RE-LAID)
```

Era terms stay in frame k and are never re-laid (E23 §4's frame rule).
Resolution per cell: 1.96·SE_binomial on n_test (500 at 3k/task ⇒ 2.6–4.4 pp;
2,000 at 12k/task ⇒ 1.0 pp).

## 4. Controls

| control | statement | bar |
|---|---|---|
| **C-ID** | on A1, this script reproduces `runs/e18_pmd_lstm/decomp.json`'s `acc_orig`, `acc_ceiling` per cell | 1e-6, 12/12, else nothing read |
| **C-CONSTR** | every artifact's `num_tasks`, `content_chunks`, `construction_fingerprint` match the rebuilt benchmark | exact; the fingerprint defect above is why this is asserted, not the content hash alone |
| **C-WIT** | shared head, adapters off, era + shadow, 4 threads, from each run's own `arm` | uniform across the three arms or they are not compared |
| **C-FLOOR** | seed-42 replicate per new arm | no delta below it is read |
| **C-RELAY** | `mnist_relayout(x, k, k)` is the identity; `relayout(relayout(x, k→T), T→k) == x` | exact |
| **must-fail** | re-laying with the **wrong** source permutation (task k+1's) reads below the right one | 12/12 (A2) and 19/19 (A3) per seed — *if it fails where the right map also recovers nothing, that is E23's cause, recorded as such and not re-barred* |

## 5. Predictions — registered before the runs

*Calibration series, read from `docs/appendix.tex` (`app:predictions`) at the
moment of writing:* **"Sixty-one scored entries, thirteen misses
(2026-09-20)."*

| prediction | odds |
|---|---|
| **A3 (T = 20): re-laying still removes the encoder term** — \|F_enc_relaid\| ≤ resolution, pooled — i.e. sequence length does **not** explain E23 | **~55%** |
| A2 (T = 5, 3k): re-laying removes it — images-per-task does not explain E23 | **~85%** |
| A3: re-laid deployed accuracy **above** raw, pooled (the scratch direction, opposite to both pretrained backbones) | ~70% |
| Total forgetting higher at T = 20 than at T = 5 with the same images/task | ~75% |
| A2's F_enc within 2 pp of A1's (12k → 3k images/task does not move the split) | ~50% |
| The wrong-permutation must-fail passes on both new arms | ~80% |

**Two-sided reading, pre-committed.** If A3's re-laid F_enc is at zero while
the pretrained arms' rises, §4.2's scope sentence is **training regime**
("encoders overwritten to the current frame", pretrained trunks excepted) and
E23's reading stands. If A3's re-laid F_enc is also elevated and its deployed
accuracy also falls, the scope sentence is **sequence length**, E23's
pretraining reading is withdrawn as unidentified, and the paper says a
twenty-task sequence over twenty layouts leaves no single current frame — in
which case the ViT/ResNet result is a special case, not a regime difference.
If A3 lands between, report the interpolation and claim neither.

## 6. CLAUSE → JOB → ARTIFACT

| clause | launcher / script | artifact |
|---|---|---|
| §2 A2, A3 (+ floor replicates) | `jobs_e23b_seqlen` = `_s72(E18_PMD_LSTM + --num-tasks/--content-chunks)`, CPU x86 | `runs/e23b_{t5c20,t20}_seed{s}/`, `..._floor4tec_a/`, `runs/ckpt_e23b_*/mafc_seed{s}_fp32/` |
| §1 construction parameter | `content_chunks` (`src/data/permuted_mnist.py`, `scripts/train.py`); regression run 2026-09-20 | `runs/e23b/construction_regression.json` |
| §3 measurement, §4 controls | `scripts/e23_seqlen.py` (audited loader, `mnist_relayout`, `refit_probe`; C-ID against `runs/e18_pmd_lstm/decomp.json`) | `runs/e23b/{arm}/decomp_seed{s}.json` |
| §5 row | `scripts/e23_seqlen_row.py` | `runs/e23b_row.json` |

**Checkpoint audit:** A1's era checkpoints exist and were loaded today (D1,
48 cells). A2/A3 are built by this contract; the audit is a precondition on
reading, not on launching.

## 7. Cost

Six CPU runs; A1's T = 5 runs took 6–11 min each and the image-presentation
count is matched, plus era-checkpoint overhead at 20 boundaries instead of 5.
Under 2 CPU-hours total. Analysis: 12 + 57 cells of refits, minutes.
