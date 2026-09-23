# Pre-Registration: E9 — Estimating Current-Era Geometry Without Old Data

**Status:** **v2, locked.** Five review catches applied (§7). Analysis-only.
One-day timebox.

**Question:** E8 decomposed the recoverable reader channel into a storage-honest
stack (era head **0.236** → + era-prototype-span projection **0.347**) and an
oracle-bound remainder (**+0.130** to the θ₄-span treatment at 0.477; **+0.140**
via prototype alignment — two measurements of the same quantity: *what knowing
CURRENT feature geometry buys*). Can that geometry be estimated without old-task
data, from information legitimately available at read time?

**Premise (c), restated honestly:** C2-oracle shows a single global map per task
captures the alignment component — but that component is **~¼ of the recoverable
channel, not most of it**. This contract targets that quarter.

---

## 0. The resource audit (catch 24, applied before anything else)

| assumed resource | trivial use of it | ρ | verdict |
|---|---|---|---|
| era **head** (1542 floats) | apply it to θ₄ features | 0.236 | baseline, registered |
| era **prototypes** (1536 floats) | project onto their span | 0.347 | baseline, registered |
| era **encoder** θ_{k+1} (~264K params) | **run task k through it** | **1.237** | **DOMINATES EVERYTHING** |

**v1's T1 assumed the stored era encoder.** Storing it makes the problem
disappear: a snapshot has neither encoder damage nor reader mismatch, so it beats
the oracle ceiling. The estimator would have approximated what its own stored
object delivers exactly.

**Fix — write-time transport, rolling snapshot, permanent maps.** At each task
boundary the previous encoder is *still in memory*. Compute the per-step
Procrustes **then**, on that era's own current data — in-distribution **for that
pair**, which is the honest answer to RBST's shape at the single-step level —
store the **256×256 map** (65K floats; low-rank likely far less), and **discard
the encoder**. Read time chains stored maps with no snapshot present.

**Consequence for the risk profile, stated rather than discovered:** each link is
estimated in-distribution, so the risk **relocates from "transport OOD" to
"composition error growth."** T2's chaining is now the **primary** estimator and
H-T3's accumulation column is **load-bearing, not a side-finding.**

**The trivial baseline is itself a result.** ρ = 1.237 at ~1MB/task is the
degenerate upper-right anchor of the storage-accuracy frontier — the "just keep
every model" endpoint every storage-honest method is implicitly priced against.
It goes in the paper's frontier figure, which has been missing that corner.

## 1. The pipeline (one procedure, no ambiguity — catch 3)

At read time for old task k, with **no era encoder and no old-task data**:

1. encode x_k with θ₄ → features `f`
2. transport `f` **back** into the era frame via the chained stored inverses
   `Q_{4→k} = Q_{4→3} ∘ … ∘ Q_{k+1→k}`
3. project onto the **stored era-prototype span**
4. apply the **stored era head**

## 2. Estimators (each measured alone — composite rule)

- **T2 — chained transport (PRIMARY):** per-step maps composed through
  θ_{k+1}→…→θ₄. Each link estimated in-distribution at write time.
- **T1 — direct transport (secondary):** a single map θ₄→θ_{k+1} estimated on
  current data. Retained because its *gap* from T2 is the accumulation
  measurement: chained ≥ direct supports the accumulation account; the reverse
  bounds long-sequence viability.
- Procrustes is full-rank here (N≈2947 × 256), so no completion ambiguity —
  **catch 23 satisfied by construction**, verified by a 5-completion spot check
  anyway.

### 2a. The RBST guard — replaced (catch 2)

v1's split-half test estimated from two halves of *current* data. Both are the
same distribution, so **they agree by construction** and the test could pass with
the out-of-distribution problem fully intact. It tested estimator variance, not
transportability — not the failure it cited.

**Primary guard: estimated-vs-oracle map comparison.**

    relative deviation capture:  ||Q_est - Q_oracle||_F / ||Q_oracle - I||_F  <=  0.5

The estimated map must capture at least half the oracle's deviation-from-identity.
Principal angles reported descriptively. The oracle map is **tier-2 as a
diagnostic** — fine in the lab, flagged as such.

**This turns an underperformance into a diagnosis:**
- low ρ **with high** map agreement → the alignment component was smaller than
  C2-oracle suggested;
- low ρ **with low** map agreement → drift is not input-independent — **failure
  mode 6's second instance, sharpened.**

**Secondary (relabelled honestly): split-half stability** — retained because it
does measure estimator variance, which is worth knowing; it is not a
transportability test and is not called one.

### 2b. Failure mode 6, cited (standing practice)

RBST fit a **2-layer MLP, 256→256→256 ≈ 131,584 parameters**, by MSE, on raw cell
states, and **transport lost to doing nothing in 12 of 12 seed×horizon cells**
(−5.8pp to −78.5pp; 0.158 transported vs 0.346 untransported vs 0.785 reachable).
E9 differs in three declared ways: **(a)** closed-form orthogonal Procrustes, not
a learned MLP; **(b)** acting on readout-subspace features, where E6 locates the
consequential motion; **(c)** licensed by C2-oracle's evidence for a single global
map — at the honestly-restated ~¼ magnitude.

## 3. Hypotheses

- **H-T2 (gate):** relative deviation capture ≤ 0.5. **Evaluated FIRST**
  (catch 5); if it fails, H-T1 is not read.
- **H-T1:** T2 (or T1) pooled ρ ≥ **0.41** on HAR/OFF — the tier-0 stack (0.347)
  plus at least half the oracle gap (+0.065). Half-the-gap is the minimum that
  makes stored maps worth their 65K floats over the free stack.
- **H-T3 (accumulation, no gate):** T2 − T1 reported with its sign either way.

**Map spectra are reported regardless of branch** (ruled in at lock): if the
per-step maps are near-identity with a few strong directions, chaining should
survive; if they are dense rotations, both the accumulation account and the
estimator have a problem — **and the geometry shows it before ρ does.**

## 4. Branches — gates before grades (catch 5)

| order | branch | condition | reading |
|---|---|---|---|
| 1 | **(D)** | ¬H-T2 | Estimated map does not capture the oracle's deviation. **No repair claims.** Report the map geometry and stop. |
| 2 | **(A)** | H-T2 ∧ H-T1 | Current-era geometry **is** estimable from stored maps. Ledger: *"…partially repairable at read time from stored parameters and statistics, with the alignment component estimated — not oracle."* |
| 3 | **(B)** | H-T2 ∧ ¬H-T1 ∧ ρ > 0.347 | Transport adds something, under half the gap. Residual is oracle-bound; **the tier-0 stack (0.347) is the deployable cure.** |
| 4 | **(C)** | H-T2 ∧ ρ ≤ 0.347 | Transport adds nothing or hurts. **The RBST lesson generalises even to closed-form low-DOF maps** — drift is not input-independent enough. Failure mode 6 gains a second, sharper instance. Cure line caps at the tier-0 stack. Full prominence. |

## 5. References and arms

Fixed from E8's recorded output (HAR/OFF, bias-adjusted denominator 0.2311):
floor **0.5087**, tier-0 stack **0.347**, θ₄-span oracle **0.477**, full-Q oracle
**0.572**, snapshot anchor **1.237**. Arms: HAR/OFF primary, HAR/v1-ON secondary,
MNIST/OFF cross-regime. Seeds {42, 1337, 2024}.

## 6. Predictions and locks

H-T2 ~70% · **H-T1 ~45%** (below a coin flip; risk now relocated to composition
error growth, §0) · H-T3 sign chained ≥ direct ~55% · joint (A) ~35%.
*Calibration:* E5 (75% miss), E5b (75% miss), E7 (15%-branch fired), E8 premise
(c) (stated "most", measured ~¼).

Locks: analysis-only; E8's references reused, not recomputed; fingerprint
re-checked; verdict strings computed from printed arrays (catch 22); every
estimator measured alone (composite rule); per-seed values printed; provenance
from recorded output only; timebox one day.

## 7. Review log — five catches

1. **BLOCKING — T1's era encoder trivially dominates** (ρ 1.237 > every ceiling).
   Catch 3's error class recurring. → write-time maps, encoder discarded;
   promoted to standing rule **catch 24**; trivial baseline retained as the
   frontier's degenerate anchor.
2. **T3 did not guard against the failure it cited** — split-half on one
   distribution. → estimated-vs-oracle map comparison as primary guard;
   split-half demoted and relabelled.
3. **Application direction ambiguous** → the four-step pipeline in §1.
4. **"map disagreement large" unquantified** → relative deviation capture ≤ 0.5.
5. **(D) was a gate listed last** → branches ordered gates-before-grades.
