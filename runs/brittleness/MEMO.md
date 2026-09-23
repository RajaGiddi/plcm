# Frozen-Feature Brittleness — Result Memo

**Status:** Complete, n=3. **Pre-registered branch: DIES.** The recurrence-specific
brittleness claim is struck from the paper. The superseding joint branch fired
as pre-named and replaces it with a stronger, more general finding.

---

## 1. Results (frozen transfer = mean accuracy, tasks 1–4)

| arm | per-seed | mean ± std | task 0 (in-dist) |
|---|---|---|---|
| LSTM frozen | 0.7331 / 0.7387 / 0.7193 | **0.7304 ± 0.0081** | 0.9893 |
| MLP frozen | 0.7656 / 0.7653 / 0.7675 | **0.7661 ± 0.0010** | 0.9813 |
| raw-pixel reference | 0.9233 / 0.9233 / 0.9234 | **0.9234 ± 0.0001** | 0.9230 |

    GAP (MLP - LSTM) = +3.57pp   ->   DIES band (<= +5pp)

## 2. Verdict — claim struck

The pre-registered claim was:

> *frozen recurrent features fail across permutations because sequential
> processing bakes the input arrangement into temporal dynamics; feedforward
> features transfer better.*

At n=3 the architecture gap is **+3.6pp**, inside the "dies" band. Feedforward
features transfer *slightly* better, not materially so. **The claim is removed
from the paper**; the deletion is recorded in the process appendix per contract §6.

## 3. What replaced it — the joint branch (pre-named §3, fired)

**Both frozen encoders transfer far below a reference that absorbs permutation
by construction:** LSTM −19.3pp, MLP −15.7pp against 0.9234.

The failure is **representational commitment**, not recurrence. Any encoder
trained on one arrangement trades permutation-robustness for specialization —
feedforward nearly as much as recurrent.

**The trade, quantified.** Each encoder *beats* the reference on its own task and
loses badly off it:

| arm | task 0 vs reference | transfer vs reference |
|---|---|---|
| LSTM | **+6.6pp** (0.9893 vs 0.9230) | **−19.3pp** |
| MLP | **+5.8pp** (0.9813 vs 0.9230) | **−15.7pp** |

Roughly 6 points of in-distribution specialization bought at a cost of 16–19
points of transfer. That is the brittleness result, stated generally.

**The reference is invariant by construction, and the data shows it.** Across all
15 independent measurements (3 seeds × 5 tasks) the raw arm spans
**0.9228–0.9237**, σ = 0.0001 — flat on task 0 and every permuted task alike. A
fresh per-task linear map inverts any permutation exactly; nothing upstream has
committed to an arrangement.

## 4. Consequences for the paper

**(a) The frozen-regime section loses an explanation and gains a general one.**
The earlier frozen diagonals (25–59%) measured a *shared* head under drifting
conditions; this control measures *fresh* heads per task — the honest
comparison. Both numbers reported in one paragraph, with the strike noted.

**(b) The motivation section gains a derivation instead of an anecdote.** A
per-task linear map at the input is permutation-invariant by construction; deep
encoders are not, regardless of type. So *"install the per-task linear map in
front of the shared encoder"* is not a trick that happened to work — it is what
the control says any committed representation requires. The adapter is the
raw-pixel head's trick, installed in front of a deep network.

**Consistency check worth reporting:** the adapter result (0.9331 ± 0.0099) and
this invariance reference (0.9234 ± 0.0001) agree within ~1pp. The adapters
recover almost exactly what the invariance argument says is recoverable — and
slightly exceed it, since the deep encoder still contributes on top.

## 5. Consequence for the MLP-attribution control (next in queue)

This run was sequenced first to supply the discriminator for that contract's
branch (c) — *"adapters aren't needed on MLPs because the backbone barely
forgets."* **That reading just lost its footing:** frozen MLP features are nearly
as committed as LSTM ones (−15.7pp vs −19.3pp), so the MLP's free input layer
did **not** rescue its frozen features. Prior shifts toward branch (a)/(b):
adapters should buy a substantial delta on the MLP too. The discriminator points
*toward* architecture-generality, not away from it.

## 6. Deviations

None. Protocol, thresholds, seeds, and locks as pre-registered. The §3
amendment (raw line reframed from "floor" to "permutation-invariant reference",
plus the joint-branch clause) was made and signed off **before** launch, with the
1-epoch mechanics smoke explicitly excluded from evidence.
