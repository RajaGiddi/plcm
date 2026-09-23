# Pre-Registration: E5b — No-Shift Control for the HAR Benchmark

**Status:** locked. Written **before** the control runs; thresholds fixed against
E5's already-measured numbers.

**Origin.** Not contract-derived. Raised in review as a challenge to E5's
construction: *UCI HAR is not a continual-learning benchmark.* It is a standard
supervised dataset with one train/test split, no task sequence, no established CL
protocol, and no published CL baselines. The five tasks were **constructed** by us
via simulated hardware-revision shifts. Permuted MNIST, by contrast, was purpose-built
as a CL benchmark with deliberately catastrophic permutations and a decade of
comparable numbers.

That asymmetry was not priced when E5 was designed. This control prices it.

---

## 1. The question

E5 measured forgetting of **0.3736** (OFF arm) and concluded the adapter mechanism
produced no benefit. That conclusion assumes the forgetting was caused by **input
shift** — the only thing the mechanism addresses.

If HAR forgets that much with **no shifts at all**, then E5's forgetting is
dataset-intrinsic, our constructed tasks never created the phenomenon the mechanism
targets, and E5 tested nothing about the mechanism's boundary.

> **"We found the edge" versus "our instrument didn't reach."**
> E5b decides which sentence we are allowed to write.

## 2. Design

Identical to E5 in every respect — same loader, encoder, seeds {42, 1337, 2024},
5 tasks × 10 epochs, both arms — with **one** change: every task uses the identity
map (`no_shift=True`). The five tasks become literally the same task.

**Prediction on record (before running):** a sane pipeline on five identical tasks
should forget ≈ 0, because there is nothing to forget. Forgetting near E5's 0.3736
would indicate the measured forgetting is not shift-induced at all.

## 3. Primary metric and branches (fixed before any number exists)

**Shift-attributable forgetting fraction:**

    f_attr = 1 − (F_noshift / F_shift),    F_shift = 0.3736  (E5 OFF, measured)

| branch | condition | reading |
|---|---|---|
| **(A)** | `f_attr ≥ 0.67` (F_noshift ≤ 0.1233) | Shifts caused most of the forgetting. The precondition holds, E5 is a **genuine test**, and its failure is a real — if narrow — boundary finding. E5 stays in the results. |
| **(B)** | `f_attr ≤ 0.33` (F_noshift ≥ 0.2503) | Forgetting is dataset-intrinsic. **The benchmark construction failed**: we built a CL problem that wasn't one. E5 moves to the appendix as a failed construction, and the mechanism's boundary on non-image modalities is reported as **UNTESTED**, not as a limit. |
| **(C)** | `0.33 < f_attr < 0.67` | Partial. Report the decomposition explicitly and scope any claim to the shift-attributable portion only. |

**Ties resolve DOWNWARD** — toward (B), the reading less favourable to our own
experiment having worked.

## 4. Secondary, reported not gated

- **ON arm under no shift.** The ideal adapter is the identity here. If adapters
  still cost accuracy, that is evidence they act as generic trainable capacity
  rather than as shift inverses — informative for E5's `e5diag` question.
- **Per-task severity audit** (already measured, from E5 row 0 — a task-0-only model
  evaluated on each later task with no adaptation):

  | benchmark | task-0 acc | later tasks | retained |
  |---|---|---|---|
  | Fashion-permuted | 0.891 | 0.103, 0.103, 0.105, 0.117 | **12.0%** |
  | Fashion-rotated | 0.891 | 0.314, 0.127, 0.077, 0.048 | **15.9%** |
  | **HAR** | 0.916 | **0.907, 0.856**, 0.286, 0.327 | **64.9%** |

  HAR tasks 1 (gain+offset) and 2 (rotation) are **nearly free** — a task-0 model
  handles them at 0.907 and 0.856 against its own 0.916, with no adaptation at all.
  On two of the four shifted tasks there was essentially **nothing for the mechanism
  to repair.** This is asymmetry #1, quantified: HAR retains ~4–5× more accuracy
  under its shifts than the MNIST-family benchmarks do.

## 5. Locks

Seeds {42, 1337, 2024}; n=3; no claim at n=1. No tuning. `F_shift = 0.3736` is
frozen at its measured value and does not move. Branch thresholds as above.

## 6. What E5b cannot settle

E5b tests whether **our HAR construction** created input-distributional forgetting.
It does not rehabilitate HAR as a CL benchmark, and under any branch the standing
caveat holds: these shifts are simulated. If (B) fires, the correct statement is that
**the mechanism remains untested outside image-like inputs** — not that it has a
demonstrated modality boundary.
