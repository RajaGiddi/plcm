# Pre-Registration: E12 — ViT on Split-CIFAR-100
## Decomposition generality + a falsifiable prediction from the engagement condition

**Status: SIGNED OFF (2026-08-03).** §§1–4 amended after contract review; B6 and
the weight-pin edits landed as the conditions of sign-off. §§5–8 of the original
draft are unchanged and carried forward.

**Execution order:** build gate (B1–B6, each verified) → P3 asserts with both
positive controls fired → preconditions (P0–P2) → training. No number is read
from a stage whose predecessor is not green.

---

## 0. Preamble — why this review caught what the draft missed

The three blocking catches in review were the program's three canonical error
classes, each **re-instantiating on first contact with a new architecture**:

| class | prior instance | E12 instance |
|---|---|---|
| unpriced trivial use of an assumed resource (catch 24) | E9's stored snapshot; E8's known shift map | ImageNet pretraining — frozen trunk has **zero forgetting by construction** |
| a gate that cannot fail (catch 25) | E4b's H-SUB3; E10's P3 threshold | H-V2 passes on an inert adapter *and* on a broken one |
| premise verified for **existence**, not **usage semantics** (catch 21/28) | E7's frozen heads (they existed, froze, and were never used per-task) | "task-IL" written as a setting the model does not have |

**The standing rules transfer the lessons; they do not transfer the checks.
Every new backbone re-earns them.** A rule learned on an LSTM is a hypothesis
about a ViT until re-derived there. This preamble exists so the next contract on
a fourth architecture starts by re-deriving rather than inheriting.

---

## 1. Setup

- **Backbone:** ViT-B/16, ImageNet-pretrained (timm), **trunk fully trainable**
  in the primary arms. Trainability is the program's differentiator and paper 2's
  seed — but it is now **priced against the frozen alternative** rather than
  asserted over it (arm C3-F below).
- **Benchmark:** Split-CIFAR-100, **20 tasks × 5 classes**, standard split, class
  order fixed by seed and frozen in config with a fingerprint **computed in the
  execution environment** (E10's ARM/x86 lesson: a fingerprint certifies executed
  data only where the data is built).
- **Settings:** **task-IL primary**; **class-IL secondary** (single shared head
  over seen classes), reported with the recency-bias caveat in the same sentence.

### Arms

| arm | trunk | adapters | seeds | purpose |
|---|---|---|---|---|
| **base** | trainable | off | **5** | decomposition measurement (primary, H-V1) |
| **adapt** | trainable | on | **5** | engagement-condition prediction (H-V2) |
| **frozen-probe** | **frozen** | off | 3 | the trivial-use baseline (catch 24) + P1 reference |
| **efficacy-control** | trainable | on | 3 | proves the adapter *can* help (catch 25) |

Five seeds on `base` and `adapt` — every row feeding an H-V1/H-V2 headline.
Three on the three control arms. (Resolves the original draft's contradiction
between §1's three seeds and §7's five-seed requirement.)

**frozen-probe (new, first-class).** ViT trunk frozen at pretrained weights;
per-task linear probes fit to convergence. Forgetting is **zero by construction**,
which is the point: it prices the assumed resource. The honest sentence this arm
buys is *"forgetting in the trainable-trunk regime, measured against the
zero-forgetting frozen alternative"* — and if the frozen probe sits near DIAG on
all 20 tasks, **the decomposition explicitly measures the cost of choosing
trainability, and says so.** Double duty: it is also P1's reference (§2).

**efficacy-control (new).** Identical adapter placement, optimizer, schedule and
budget as `adapt`, run on **permuted-pixel Split-CIFAR-100** — an
input-distributional shift where the framework predicts *engagement*. **H-V2 is
not readable until this arm has shown the adapter capable of helping somewhere.**
If it shows no help, the verdict is **"H-V2 untestable at this placement"** —
reported, never silently passed as confirmation.

**Adapter form.** Per-task linear map on **patch embeddings**, identity-init,
frozen after its task. Image-space square linear is 3072² — prohibitive. This is
**a different placement than paper 1's pixel-space map** and is stated as such;
a weaker adapter makes inertness easier to obtain, which is why the efficacy
control is a gate rather than a footnote.

---

## 1b. Build plan — six items, each with its verification

**No training run enters the record until all six are green.** None of this
infrastructure exists today; the original draft read as a configuration change.
Each item is a fresh premise under catch 21 and is verified, not assumed.

| # | item | today | verification step |
|---|---|---|---|
| B1 | CIFAR-100 data module, 20×5 split | **absent** — `src/data/` has `split_cifar.py` (CIFAR-**10**); zero repo matches for cifar100 | class-order fingerprint printed **and gated** in the execution environment; per-task counts and class-disjointness asserted |
| B2 | `timm` + ViT-B/16 weights | **absent** from `requirements.txt` and the Modal image | version pinned; weights baked at image-build time; **checkpoint hash recorded and asserted at container start — the hash, not file presence** (see below) |
| B3 | CIFAR-100 baked into the image | image bakes CIFAR-**10** only | dataset present at `/data` at container start, asserted before training |
| B4 | ViT backbone path | `PLCM.backbone` accepts `{"lstm", "mlp"}` only | forward-parity test: the ViT path produces logits of the right shape and trains a single task to DIAG |
| B5 | instrument binding for the ViT readout | E11's instrument is bound to `PLCM.forward` | P3's two asserts green on every arm × task cell, **both positive controls fired** (§2) |
| **B6** | **permuted-pixel Split-CIFAR-100** (the efficacy control's data) | `shifted_cifar.py` exists but for CIFAR-**10** shifts | shift spec fingerprinted in the execution environment, **plus** a direct assert that shifted and unshifted tensors differ by the registered permutation — apply, invert, compare, **exact** |

**Why B6 is its own item rather than part of B1.** B1's verification — class-order
fingerprint, per-task counts, disjointness — says nothing about whether a *shift*
was correctly applied. A silently broken permutation produces a **flat efficacy
control**, which under the new gate reads as *"H-V2 untestable at this
placement"*: a wrong verdict issued from an unverified premise. This is the one
arm whose failure mode is **indistinguishable from its null result**, so its data
gets its own gate.

**The weight pin (B2), and why it is a fingerprint rather than a dependency.**
"ImageNet-pretrained ViT-B/16" without a pinned checkpoint hash is the E10
partition gap wearing new clothes: *"the partition was frozen"* meant nothing
without knowing which environment built it, and *"pretrained"* means nothing
without knowing which weights. timm's default weights are superseded upstream, so
an image rebuilt months from now would train from a **different initialization
while every config file reads identical**.

**The full fingerprint set for this experiment — class order, shift spec, weight
hash — all computed where executed.**

---

## 2. Preconditions (gate everything; measured before any decomposition)

**P0 — build gate.** B1–B6 green. The full fingerprint set — **class order, shift
spec, ViT weight hash** — computed in the execution environment and recorded.

**P1 — competence, scale-free.** The frozen-probe arm is the reference:

- the **frozen-probe** per-task DIAG must clear **0.85** — if the *pretrained
  trunk itself* cannot solve 5-way CIFAR-100 subsets, the benchmark/backbone
  pairing is void by the rule that voided E2/CIFAR (DIAG 0.51,
  `docs/E5D_prereg.md:125`), and we report and stop;
- the **trainable-trunk** per-task DIAG must land **within 5pp of the frozen
  reference**.

An absolute 0.85 bar on a pretrained ViT is close to unfailable, which is catch
25's category. Anchoring to the frozen reference makes the gate scale-free and
gives the trivial baseline a second job.

**P2 — forgetting exists and is sequence-caused.** Deployed final-row accuracy on
task 0 falls **≥ 0.15** below its own ceiling; **plus** a repeat-task control
(5 tasks of identical data, fresh heads) showing **≤ 0.05** forgetting,
establishing the drop is task-sequence-caused rather than optimization drift
(E5b's lesson, ported).

**P3 — instrument exactness, re-derived for this architecture (catch 28).**
Two asserts, because the first does not imply the second. **This is the review's
sharpest finding: P3-as-originally-written verifies the capture point and is
silent on selection semantics — the exact E7 failure the draft cites as its own
cautionary tale.**

- **P3a — capture-point identity.** `head(captured_feature) == deployed_logits`,
  **max |Δ| = 0**, every arm × task cell, asserted permanently at the top of every
  consuming script. The ViT readout path (CLS token, post-norm, pre-head) is a
  fresh premise: verified, never assumed.
  *Positive control:* the assert must **FAIL** on a deliberately wrong capture
  point (pre-norm CLS), demonstrated before first live use.
- **P3b — selection semantics (new).** On task *k*, the readout module used **is
  head *k***. Verified by module identity, not by output comparison.
  *Positive control:* the assert must **FAIL** when the latest head is
  substituted for head *k*.

P3a passes with the wrong head selected. Only P3b catches that, and without it
E12 repeats E7 one experiment later.

### The API change P3b requires, and its blast radius

Verified in `src/models/plcm.py`: `current_key = str(self.current_task_id)`, and
`task_hint` routes **only the adapter**. On old task *k* the readout is either the
retrieval-weighted blend (non-empty bank — E7's finding) or `task_logits[current_key]`,
the **latest** head (empty bank). Neither is head *k*. "Task-IL primary" is a
capability the model does not have, written as a setting.

Per the C0 ruling, the fix is a **sanctioned model-API change under assert
protection**, not an instrument workaround: `task_hint` selects the head as well
as the adapter.

**BLAST RADIUS — the change must be opt-in, default off.** E11 established that
`PLCM.forward` reproduces every recorded accuracy matrix **exactly** (0.0000 in
8/8 E10 cells). Making `task_hint` route heads unconditionally would silently
alter the deployed path of **every prior arm** and destroy that reproduction —
the equipment's own proof of correctness. Therefore:

- the new behaviour is gated behind an explicit flag, **default preserving
  current semantics**;
- a regression assert shows that with the flag **off**, the path is **bit-identical**
  to the E11-verified behaviour on an E10 checkpoint;
- E12 sets the flag **on** and records that it does.

*(Amendments have blast radii; every amendment must re-check what it did not
touch. This one touches the one component every prior experiment shares.)*

---

## 3. Measurements

Per arm × seed × task *k*, at final θ_T:

- `acc_ceiling(k)` — task *k* accuracy at its own end-of-task checkpoint
- `acc_orig(k)` — deployed accuracy at θ_T
- `acc_refit(k)` — convex (lbfgs) probe refit on the frozen θ_T deployed feature,
  through the deployed pathway
- `acc_refit_ceiling(k)` — same recipe on the era checkpoint (recipe-gap control)

Decomposition per cell, printed with its identity column:
`R = acc_refit_ceiling − acc_ceiling` · `F_enc = acc_refit_ceiling − acc_refit(θ_T)` ·
`F_read = acc_refit(θ_T) − acc_orig` · `F_enc + F_read − R = F_total` (exact).

### Storage plan (new)

Era checkpoints are **required by the measurement**, not optional. ViT-B/16 ≈ 86M
params: fp32 × 20 tasks × 2 arms × 5 seeds ≈ **69 GB**; **fp16 ≈ 35 GB**, which is
the ruling.

- **fp16 era checkpoints on the volume**, audited **by loading at save time**
  (a checkpoint that exists is not a checkpoint that loads — 14/33 dirs were
  unusable when that was last checked).
- **Belt-and-braces:** `acc_ceiling` and `acc_refit_ceiling` are **also computed at
  the task boundary**, in-process, in full precision.
- **The boundary-time value is authoritative.** The fp16 reload is the
  reproduction path. **The disagreement between them is measured and reported,
  never assumed negligible** — a reloaded checkpoint is not the model that was
  trained until proven (catch 29), and fp16 rounding is exactly the kind of
  non-parameter difference that assumption hides. Required below the
  reproduction floor measured in the catch-19 step; if above, the storage
  precision is the finding and fp32 is re-costed.

Protocol locks inherited without amendment: per-cell `RHO_MIN_DENOM` guard applied
**per cell, never post-pooling** (catch 26); mean-of-ratios with
binomial-propagated CIs; per-seed R with a pooled **|R| ≤ 0.05** gate **per arm**;
components measured individually, never bundled; verdicts computed from the
printed arrays (catch 22); reproduction floor measured on this GPU path, **both
arms**, before any delta is interpreted.

---

## 4. Hypotheses

- **H-V1 (decomposition generality, PRIMARY).** Task-IL, base arm: reader share
  `F_read/(F_enc+F_read)` **≥ 0.50** pooled, CI lower bound above 0.50.
  *Derivation:* the program's measured range is **66–88%** (post-E11); 0.50 is the
  bar at which "the reader carries the majority" is the honest sentence.
  *Reported beside:* the frozen-probe arm's zero forgetting, so the claim reads as
  the cost of trainability rather than as forgetting in the abstract.
- **H-V2 (engagement prediction, the falsifier).** Adapters contribute **≤ 5pp**
  average accuracy over base, task-IL. **Readable only after the
  efficacy-control arm shows the adapter capable of helping.** If the control is
  flat, the verdict is *"H-V2 untestable at this placement."* A larger positive
  effect falsifies the framework's scope claim and is reported as such.
- **H-V3 (class-IL, SECONDARY).** Reader share ≥ 0.50 in class-IL, reported with
  the recency-bias caveat attached in the same sentence; never cited as primary
  evidence for H-V1.
- **H-V4 (depth/scale sanity, descriptive).** Report `F_enc` absolute magnitude
  against the **post-E11 LSTM/HAR range 0.052–0.112** (OFF 0.1115, v1-ON 0.0844,
  v2-ON 0.0853, v2-control 0.0523 — `runs/e11_e6b/decomp.json`). No gate.
  *The original draft cited 0.04–0.08, superseded by the E11 recompute.*

---

### Calibration note, recorded at sign-off — before any number exists

Joint (A) stays at **~50%**. Flagged on record: **H-V1 on a *pretrained* trunk is
the least-transferable prediction in the program's history.** Every reader-share
measurement we hold comes from encoders trained **from scratch**; a pretrained
trunk's stability under fine-tuning could move the F_enc/F_read balance in
**either** direction, and no prior result constrains which.

If the share comes back **below 0.50**, branch (C) is the honest reading: paper 1
reverts to the LSTM-family framing with this as its **measured boundary**. The
contract is built so that either answer is a result.

## §§5–8 — unchanged

Branches, predictions, locks and reporting carry forward from the original draft,
with two mechanical consequences of the above: §7's seed lock reads **5 seeds on
`base`/`adapt`, 3 on controls**, and §8's memo adds the frozen-probe pricing and
both P3 positive controls to the required contents.

**Masip et al. (arXiv:2601.22012) — verified by the author, fetch stood down.**
The related-work paragraph drafts against the confirmed abstract: geometric /
Crosscoder framework, ViT on sequential CIFAR-10, capacity-vs-readout split
described, **no additive per-task attribution**. The differentiation is real and
must be **argued against our own numbers**, not assumed.
