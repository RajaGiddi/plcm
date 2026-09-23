# E31 §0 — filled from the code, before the contract is finalised

**Date:** 2026-09-23. Every number below is read from a file or measured on this
machine. No compute was launched: everything on Modal is stopped, and §0 needed
nothing there.

---

## 1. Task selection in the trainer — it does not exist

[`trainer.py:1244`](../src/training/trainer.py#L1244) is a bare
`for task_id in range(self.num_tasks)`. There is no `--only-task`, no
`--exclude-task`, no skip. **E31 is a training-loop change**, which is the
category E28's contract treated as the highest-risk kind, and the same
conditions apply: opt-in, default off, and a flag-off path proven
bit-identical rather than argued to be.

**Two recorded hashes are available for that regression**, both in
`runs/e23/mlp_regression.json` (2026-09-19), both already used by
`e28_regression.py`:

| checkpoint | batch | sha1 |
|---|---|---|
| S72 HAR OFF LSTM (fp32 shadow) | `[8, 128, 9]` | `50a2f2de57af0583` |
| e17 MLP on Permuted MNIST | `[8, 28, 28]` | `ca074a46f1805580` |

---

## 2. THE FINDING THAT DECIDES THE IMPLEMENTATION

**Reducing `num_tasks` silently changes every task's content.**
`permuted_mnist.py:147-152` chunks by `content_chunks or num_tasks`:

```python
chunks = content_chunks or num_tasks
self.train_index_sets = list(torch.randperm(N_TRAIN_IMAGES, generator=cgen).chunk(chunks))[:num_tasks]
```

Measured directly, seed 42:

| construction | tasks | task-0 size |
|---|---|---|
| `num_tasks=5, content_chunks=None` (what SEQ ran) | 5 | **12,000** |
| `num_tasks=4, content_chunks=None` | 4 | **15,000** |
| `num_tasks=4, content_chunks=5` | 4 | 12,000 |

So if `EXCL-k` is built by dropping `num_tasks` to 4, it trains on **different
images in different tasks** and the comparison against SEQ is void.

**RULE FOR THE FLAG:** `--exclude-task k` and `--only-task t` must keep
`num_tasks` at its full value and skip inside the loop. They must never reduce
it. The same applies to `ONLY-T`: `num_tasks=1` would put all 60,000 images in
one chunk, which is not task *T*'s content.

**And the fingerprint that exists will not catch the mistake.**
`content_fingerprint` reads `c1a570a02d35` for `e18_pmd_lstm` (T=5, C=5),
`e18_pmd_mlp` (T=5, C=5) *and* `e23b_t20` (T=20, C=20) — it hashes the
concatenated index tensors, and chunking one permutation differently gives the
same byte stream. That is catch 23-B exactly, firing in a new place.
`construction_fingerprint` does distinguish them, and it is **not among the 33
fields `arm_provenance` records**. E31 must record it.

**HAR is not exposed to this.** `har_subject.py` partitions with
`partition_subjects(subs, cnts)` using the module constant `N_GROUPS = 5`, not
`num_tasks`, so the grouping is fixed however many tasks are trained.

---

## 3. Content construction — the contract's question answered

**Task *k*'s test images are in no model's training set, by construction.**
Train chunks are drawn from MNIST's 60,000-image *train* split and test chunks
from the separate 10,000-image *test* split
([permuted_mnist.py:151-152](../src/data/permuted_mnist.py#L151)). The two never
intersect, for any task or any arm.

All three MNIST arms record `disjoint_content=True`, so tasks differ in content
and not only in permutation. `e18_pmd_*` use 5 chunks (12,000 images/task); A3
uses 20 (3,000/task).

**HAR:** five groups of six subjects, greedy-balanced by window count,
deterministic. Per group, five subjects train and **one is held out as that
task's test subject** — so each task's test subject is disjoint from its own
train subjects as well as from every other task.

---

## 4. Two design points the code changes

**(a) Review 1's premise holds structurally for permuted MNIST.** Re-laying task
*k*'s test content into format *T* yields a different random chunk of MNIST's
test split under `P_T`. MNIST chunks are exchangeable, so that is
distributionally *identical* to task *T*'s own test content. Δ_excl ≈ 0 is
therefore the structural expectation, not merely a likely outcome, and the two
75% predictions are well priced.

**(b) The HAR reading's premise is weaker than the contract states.** Reading 3
rests on "subjects are not exchangeable the way digits are". But
`partition_subjects` balances groups **by window count only** — no subject
property distinguishes group *k* from group *T*. And task *k*'s test subject is
held out from task *k*'s own training too. So training on task *k* supplies five
*other* people from a count-balanced group, with no evident reason to help on
group *k*'s held-out person more than group *T*'s five people do. Subjects are
non-identical; groups are not systematically different. **I would price the HAR
prediction nearer the MNIST ones than 60%,** and the reading should say
"individuals differ" rather than "subjects are not exchangeable".

---

## 5. Part B rules nothing out as specified

A linear probe on 784 raw pixels beating a 256-dimensional encoder's refit is
**what a bottleneck does**, with or without forgetting. As written, prediction 5
("raw-input probe exceeds the sequential refit by 10 points or more", ~80%)
fires on a fact about dimensionality.

The interpretable quantity is whether the raw-vs-refit gap is **larger on old
tasks than on task *T***, where there is no forgetting to explain it. **Part B
needs the task-*T* comparison as its reference**, or it is a measurement wearing
a control's clothes.

*(Note also that a linear probe is permutation-invariant up to a relabelling of
its input dimensions, so the raw probe is unaffected by which format it reads —
another reason its absolute level carries no information about drift.)*

---

## 6. Relay functions — read, with their exactness

[`cure_screen.py:145-177`](../scripts/cure_screen.py#L145):

* `mnist_relayout` — undo `perm_src`, apply `perm_dst`. **Exactly invertible.**
* `har_relayout` — composes `M_dst · M_src⁻¹` on the affine channel map. Its
  docstring records the v1 defect (mapping *through* `M_k`, max err 5.1985) and
  that this composition gives 0.0000.
* `rotated_relayout` — bilinear; exact only at multiples of 90°, lossy
  otherwise. **E31 does not use it** (permuted and HAR only).

---

## 7. Checkpoint audit — already green, today

All four SEQ arms were in the `e30` chain, run 2026-09-22, criterion **LOADS**:

| arm | E31 role | seeds | result |
|---|---|---|---|
| `e18_pmd_lstm` | LSTM · Permuted MNIST | 42/1337/2024 | 5/5 each |
| `e18_pmd_mlp` | MLP · Permuted MNIST | 42/1337/2024 | 5/5 each |
| `e10off_ec` | LSTM · HAR | 42/1337/2024 | 5/5 each |
| `e23b_t20` | A3 | 42/1337/2024 | 20/20 each |

Part of 30/30 directories fully loadable, 0 bad files. No new audit needed.

---

## 8. Cost — recorded, so no smoke is required

Every SEQ artifact records `training_time`. These are the same configurations
E31 re-runs, which is stronger evidence than a smoke of a proxy:

| arm | seed 42 | 1337 | 2024 | shape |
|---|---|---|---|---|
| `e18_pmd_lstm` | 10.8 min | 6.8 | 6.3 | 5 × 10 ep |
| `e18_pmd_mlp` | 3.3 min | 5.1 | 4.8 | 5 × 10 ep |
| `e10off_ec` | 7.1 min | 4.8 | 4.2 | 5 × 10 ep |
| `e23b_t20` | 13.1 min | 15.2 | 13.6 | 20 × 10 ep |

`EXCL-k` ≈ 0.8 × SEQ, `ONLY-T` ≈ 0.2 ×, `ONLY-T matched-steps` ≈ 1.0 ×. Per
construction per seed that is ≈ 4.4 × SEQ.

**Total: 61 jobs, ≈ 4.8 CPU-hours serial, ≈ 40 minutes wall fanned out at 16
containers** (the longest single job is A3's matched-steps arm at ~14 min). The
contract's "about 60 short CPU runs, roughly an hour" is accurate.

*The 10.8 vs 6.3 min spread on one arm across seeds is container variance, not
configuration — do not read it as a cost difference between seeds.*

---

## 9. Provenance fields to add before the first run

`arm_provenance` records 33 fields today, including `disjoint_content` and
`content_fingerprint`. E31 must add:

* `exclude_task` and `only_task` (None on every existing arm);
* `content_chunks` **unconditionally** — it is currently recorded only when set,
  so its absence is ambiguous between "not applicable" and "defaulted";
* `construction_fingerprint` — **the only field that can catch §2's error.**

---

## 10. Count line

Re-counted from `docs/appendix.tex` at the moment of writing:
**112 scored entries, 31 misses**, plus one not comparable and one unmeasurable.

---

## 11. One blocker that is not about E31

The five-package pin is still in `modal_runner.py`, so **the next Modal launch
of any kind triggers the full image rebuild** — the one that had run ~30 minutes
without finishing when it was stopped. It re-bakes four datasets and the ViT
weights, not just the pip layer.

E31 trains, so it needs Modal. Two options, and it is a decision rather than a
finding: keep the pin and pay the rebuild once (after which
`e30_rebuild_fidelity.py` is ready to prove the rebuild changed nothing), or
revert the five pins to bare names and get the cached image back immediately, at
the cost of the container's versions being unrecorded again.
