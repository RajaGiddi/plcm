# E2 — CIFAR-10 Generality: Result Memo

**Status:** Complete, 14/14 verified, n=3.
**BOTH hypotheses FAIL on BOTH shift types. The pre-registered venue rule fires.**

Pre-registration: `docs/GENERALITY_prereg.md` (v2 §2 E2, §3; v3 amendment 7).
Calibration: `runs/e2/CALIBRATION.md` (hidden=1024, selected --no-adapters).

---

## 1. Reproduction floor — read first, per protocol

Measured **in-condition**: 5 tasks, adapters in the graph, both arms.

| arm | original | rerun | spread | |
|---|---|---|---|---|
| ON | 0.3680 (`ee735d25`) | 0.3680 (`ee735d25`) | **0.0000pp** | bit-identical |
| OFF | 0.3012 (`92df37fe`) | 0.3012 (`92df37fe`) | **0.0000pp** | bit-identical |

**Determinism survives the adapter path.** This was not assumable from the
calibration floor: the ON arm executes a 3072×3072 matmul per timestep that the OFF
arm never touches, so the *determinism property itself* — not merely the spread
magnitude — could have differed between arms. It did not, and it held across four
additional task boundaries.

**Consequence: every delta below is real in the reproducibility sense.** No result
in this memo is within noise of zero. The +15pp bar is fully readable.

## 2. Verdict

CIFAR-10, LSTM (32 rows × 96 features), hidden=1024, 5 tasks × 10 epochs, n=3.

| shift | arm | AVG | DIAG | normalized retention |
|---|---|---|---|---|
| permuted | **ON** | 0.3682 ± 0.0009 | 0.5067 | **0.7266 ± 0.0055** |
| permuted | OFF | 0.3049 ± 0.0028 | 0.4912 | 0.6208 ± 0.0071 |
| rotated | **ON** | 0.3549 ± 0.0164 | 0.5069 | **0.7003 ± 0.0325** |
| rotated | OFF | 0.3972 ± 0.0051 | 0.5681 | 0.6991 ± 0.0071 |

    H-G1a  ON normalized retention >= 0.90
      permuted  0.7266   FAIL (short by 17.3pp)
      rotated   0.7003   FAIL (short by 20.0pp)

    H-G1b  ON - OFF >= +15pp AVG
      permuted   +6.32pp  FAIL
      rotated    -4.22pp  FAIL (WRONG SIGN)

## 3. What the numbers say, stated carefully

**The adapters did engage on permuted CIFAR — partially.** +6.32pp is a real effect
(floor = 0.0000pp), in the right direction, and the ON arm's normalized retention
exceeds the OFF arm's by 10.6pp. This is *not* the HAR null. But it is nowhere near
the +15pp bar, let alone Fashion's +35.2pp on the same shift class.

**On rotated CIFAR the adapters actively hurt** — −4.22pp, and the damage shows up
in the diagonal: ON DIAG 0.5069 vs OFF 0.5681. The adapter costs **6.1pp of
per-task learning** before any question of retention arises. A 9.4M-parameter
identity-initialized layer in front of an encoder that only reaches ~57% on the task
is not free.

**The engagement-condition prediction is partially disconfirmed.** The prediction on
record was that a 3072-dim pixel permutation is far too expensive for the encoder to
absorb internally, placing it firmly in adapter territory and arguing for a pass.
Dimensionality *did* predict the sign — adapters engaged at d=3072 where they were
inert at d=9 — but it badly over-predicted the magnitude. **Shift dimensionality
alone does not determine whether the mechanism delivers.**

## 4. The arbitration test — the OFF arm holds up, HAR-like

| benchmark | OFF normalized retention |
|---|---|
| Fashion-rotated | 0.4311 |
| Fashion-permuted | 0.5338 |
| **CIFAR-permuted** | **0.6208** |
| **HAR** | **0.6716** |
| **CIFAR-rotated** | **0.6991** |

The pre-named arbitration was: if the OFF arm collapses hard (Fashion-like), the
precondition is present and the mechanism should fire; if it holds up (HAR-like),
the condition is subtler than dimensionality alone.

**It holds up.** CIFAR's OFF arm sits with HAR, not with Fashion. The precondition
the mechanism addresses — severe input-shift-induced forgetting — is **weaker here
than on the MNIST family**, despite d=3072 and a genuine pixel permutation.

The reason is visible in DIAG: **0.49–0.57.** The encoder never learns CIFAR well
row-wise, so there is less competent representation to lose. The mechanism's measured
benefit tracks *how much the baseline had to forget*, not the dimensionality of the
shift. That is the second branch of the arbitration, and it is the honest reading.

## 5. Venue rule — fires as written

> *G1 fails on **both** E2 and E5 → ICLR off; scope the finding to demonstrated
> regimes in the CoLLAs/TMLR manuscript, plainly.*

- H-G2: **failed** at E4, took falsify-and-soften.
- H-G1 on E5 (HAR): **failed**.
- H-G1 on E2 (CIFAR): **failed**, both shift types.

**No non-MNIST dataset passes G1. The ICLR gate is not met.** This is the
pre-registered consequence of the measured results, invoked without amendment.

## 6. Story-changing threshold

ON-arm normalized retention across all six tested benchmark-shift pairs:

| pair | ON NR |
|---|---|
| MNIST permuted (ref) | 0.9540 |
| Fashion rotated | 0.9284 |
| Fashion permuted | 0.9238 |
| CIFAR permuted | 0.7266 |
| CIFAR rotated | 0.7003 |
| HAR | 0.6576 |

**Span 29.6pp against a 5pp trigger.** The invariance framing is dead, and so is
the pre-named fallback ("consistent large gains") — the gains are neither invariant
nor consistently large. **Conditionality is the organizing claim**, and it now has
three datasets defining the condition rather than an assertion about one.

## 7. What survives, stated at its real weight

The adapter mechanism produces **+35 to +50pp** on the MNIST family and is
**verified, reproducible, and large**. What E2 and E5 establish is the *boundary*:
the benefit tracks how much competent representation the baseline stands to lose,
which is high on saturating MNIST-family encoders (DIAG 0.88–0.98) and low on
unsaturated ones (CIFAR 0.51, HAR 0.92 with mild shifts).

That is a narrower claim than the package set out to support, and it is the one the
data licenses.

## 8. Deviations

None. Capacity fixed by the locked rule before any continual run and shared by both
arms; seeds, epochs, shift classes, and both thresholds as pre-registered. The
14-run design (12 gate + 2 floor) follows v3 amendment 7; the floor pair was
symmetric across arms per the rewritten catch 19.

**Not measured:** E2 ran without `--save-checkpoints`, so the adapter-travel
diagnostic (`‖A_k − I‖ / ‖M_k⁻¹ − I‖`) that explained HAR's null is unavailable
here. Given that permuted CIFAR shows partial engagement, that measurement would
now be genuinely informative — it is the direct test of whether "partial benefit"
corresponds to "partial travel." Flagged, not run.
