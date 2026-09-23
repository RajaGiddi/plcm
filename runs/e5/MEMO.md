# E5 — UCI HAR Generality: Result Memo

**Status:** Complete, 6/6 runs verified mechanically, n=3.
**BOTH contracted hypotheses FAIL. H-G1b fails with the WRONG SIGN — adapters are
slightly worse than no adapters, in 3/3 seeds.**

This is the experiment the contract designated as the ICLR gate. Reporting the
failure first and at full prominence, per standing rule.

Pre-registration: `docs/GENERALITY_prereg.md` (v2, §2 E5, §3).
Contract note: `docs/E5_CONTRACT_NOTE.md`. Frozen spec: `runs/e5/FROZEN_CONFIG.json`
(fingerprint `3de66e205eb7`).

---

## 1. Verdict — CORRECTED (see §1a; the first table was contaminated)

LSTM, 9 channels × 128 timesteps, 6 classes, 5 tasks × 10 epochs, seeds
{42, 1337, 2024}. ON = per-task per-timestep 9×9 input adapters (81 params/task).

| arm | AVG | DIAG | normalized retention |
|---|---|---|---|
| **ON (adapters)** | 0.6017 ± 0.0403 | 0.9150 | **0.6576 ± 0.0442** |
| OFF (vanilla) | 0.6111 ± 0.0272 | 0.9100 | 0.6716 ± 0.0306 |

    H-G1a  ON normalized retention >= 0.90 : 0.6576    FAIL  (short by 24.2pp)
    H-G1b  ON - OFF >= +15pp AVG           : -0.95pp   FAIL

Paired by seed, ON − OFF AVG: **−6.26pp, −0.49pp, +3.91pp — negative in 2 of 3.**

Both hypotheses still fail, and H-G1a fails by a margin no framing rescues — that
was never in doubt and no seed correction touches it.

The delta, however, is **−0.95pp with the per-seed range spanning zero (−6.26 to
+3.91)**. The defensible phrasing is therefore **"no measurable effect"** — not "no
benefit, possibly a small cost," and certainly not "adapters hurt."

The 3/3 → 2/3 correction matters for exactly this reason: *"worse in every seed"*
reads as a real negative effect, while *"worse in two of three, spanning zero"* reads
as **noise around nil**, which is what it is.

### 1a. Supersession — a preempted run that passed every completeness check

The first version of this memo reported ON = 0.5833 ± 0.0146, delta −2.78pp, and
**"negative 3/3 seeds."** That table was built on `e5_har_adapt_seed2024`, which was
**preempted and restarted** mid-flight.

A later experiment (`e5diag`) happened to re-run the identical ON configs:

| seed | E5 original | independent relaunch | |
|---|---|---|---|
| 42 | 0.5769 | 0.5769 | **bit-identical** |
| 1337 | 0.5696 | 0.5696 | **bit-identical** |
| **2024** | **0.6035** | **0.6585** | **differs +5.50pp** |

The pipeline is deterministic given a seed — proven by the two exact reproductions —
so the divergent run is the contaminated one, and it is exactly the run that was
preempted. The clean value supersedes it, and **the "3/3 negative" claim is
withdrawn.**

This is **catch 19**, promoted to the standing rules: a preempted-and-restarted run
can satisfy every clause of the finish definition and still be wrong. Only an
independent relaunch detects it. It was found by accident here.

**Not yet verified to the same standard:** the OFF arm has never been independently
relaunched. `e5recheck_*` is running to close that gap; until it lands, the OFF
numbers above carry weaker evidence than the ON numbers.

## 2. Against the other benchmarks — the span that fires the reframe

| benchmark | ON norm. retention | ON − OFF AVG |
|---|---|---|
| MNIST permuted (ref) | 0.954 | +50.3pp |
| Fashion permuted | 0.9238 | +35.2pp |
| Fashion rotated | 0.9284 | +43.6pp |
| **HAR** | **0.6376** | **−2.8pp** |

**Story-changing threshold (§3): span > 5pp → reframe.** Measured span across all
tested benchmark-shift pairs is **31.6pp**, six times the trigger.

The contract pre-named the fallback framing as *"consistent large gains."* **That
fallback is also refuted:** the gains are not consistent — there is a regime with no
gain at all. The reframe required is stronger than the one written down, and the
title's declarative register does not survive it unmodified.

## 3. This is not the inert-adapter bug

Checked before drawing any conclusion, given failure mode #5's precedent. The
adapters train — `adapter_dist_from_identity` at each task's final epoch (seed 42):

| task | shift | dist from identity |
|---|---|---|
| 0 | **identity (nothing to invert)** | **0.5853** |
| 1 | gain + offset | 0.2522 |
| 2 | rotation | 0.2154 |
| 3 | permutation | 0.1978 |
| 4 | combined | 0.1784 |

So the null is real, not a disabled code path.

**But note what this table says, and does not say.** Task 0's shift *is* the
identity, so a shift-inverting adapter should stay at `I` — and it moves the most,
while the genuinely shifted tasks move least. That is the opposite ordering
shift-inversion predicts, and it is suggestive that the adapters are acting as
generic extra capacity rather than as inverses.

**It is suggestive, not established.** Task 0 also trains from a randomly
initialized encoder, which alone would produce larger movement. Four mechanism
explanations have already been advanced and retracted in this project; this one is
therefore **not being asserted** until it is measured directly.

## 4. The distinguishing measurement (running)

Two readings survive the headline and imply different papers:

- **(i)** the adapters never learned to invert the shift, because a 9×9 channel map
  is cheap for the encoder to absorb in `W_ih` — whereas absorbing a 784-permutation
  would destroy MNIST features, which is why the adapter path won there;
- **(ii)** they did invert it, and inversion simply does not help on this data.

These are separable exactly: **tasks 2 and 3 are exactly invertible by the locked
bias-free adapter** (recovery residual 0.000, `FROZEN_CONFIG.json`), so the ideal
target `M_k⁻¹` is known in closed form.

**RESULT (n=3, from `e5diag` checkpoints) — reading (i) is confirmed. The adapters
never learned the inverse; they stayed at the identity.**

| task | shift | ‖A−M⁻¹‖ | ‖A−I‖ | ‖M⁻¹−I‖ (distance to travel) |
|---|---|---|---|---|
| 0 | identity | 0.154 | 0.154 | 0.000 |
| 1 | gain+offset | 0.120 | 0.080 | 0.097 |
| 2 | rotation *(exactly invertible)* | 0.413 | **0.068** | 0.423 |
| 3 | permutation *(exactly invertible)* | 1.425 | **0.067** | 1.414 |
| 4 | combined | 1.466 | 0.062 | 1.452 |

On task 3 the adapter needed to travel **1.414** to reach an ideal it could represent
*exactly*. It moved **0.067** — under **5% of the way** — and remained essentially at
the identity. Same picture on task 2.

**This is an optimization failure, not a capacity failure.** The 9×9 adapter can
represent the exact inverse for tasks 2 and 3; gradient descent simply never went
there, because the encoder could absorb a 9-dimensional channel map more cheaply.
On Permuted MNIST, absorbing a 784-permutation would wreck the encoder's features,
so the adapter path is the cheap one — which is why the same mechanism dominates
there and does nothing here.

**Stated carefully:** this explains *why the adapters were inert on this benchmark*.
It does **not** establish a modality boundary, because §5's severity audit shows two
of the four shifted tasks needed almost no repair in the first place. Whether E5 was
ever a valid test of the mechanism is E5b's question, not this one's.

## 4a. Is E5 a valid test at all? (raised in review; E5b pending)

**UCI HAR is not a continual-learning benchmark.** It is a standard supervised
dataset — one train/test split, no task sequence, no CL protocol, no published CL
baselines. We *constructed* the CL problem from simulated shifts. Permuted MNIST was
purpose-built for CL with deliberately catastrophic permutations and a decade of
comparable numbers. That asymmetry was not priced when E5 was designed.

**Severity audit** — a task-0-only model evaluated on each later task, no adaptation.
This is precisely the damage the mechanism exists to repair:

| benchmark | task-0 | later tasks | retained |
|---|---|---|---|
| Fashion-permuted | 0.891 | 0.103, 0.103, 0.105, 0.117 | **12.0%** |
| Fashion-rotated | 0.891 | 0.314, 0.127, 0.077, 0.048 | **15.9%** |
| **HAR** | 0.916 | **0.907, 0.856**, 0.286, 0.327 | **64.9%** |

HAR tasks 1 and 2 are **nearly free**: 0.907 and 0.856 against the model's own 0.916,
with no adaptation whatsoever. **On two of the four shifted tasks there was
essentially nothing to repair.** HAR retains 4–5× more accuracy under its shifts than
the MNIST-family benchmarks — the asymmetry, quantified.

Two further asymmetries, stated but not yet measured: HAR does not saturate
(DIAG 0.915 vs MNIST 0.978), and an unsaturated encoder keeps moving regardless of
task shift; and all five HAR tasks share their training data, differing only by a
linear channel map — possibly too little task separation to constitute a CL problem.

**This narrows what E5 can disconfirm.** "The mechanism failed on a hand-built CL
problem with mild shifts on a non-saturating dataset" is far weaker than "the
mechanism failed on sensor data." It does not rescue the result — H-G1a still misses
by 24.2pp — but it changes the candidate reading from *"the mechanism has a
boundary"* to possibly *"this experiment could not test the mechanism, because the
precondition (substantial input-shift-induced forgetting) was never established."*

**E5b decides between them** (`docs/E5B_control_prereg.md`, branches locked before
running): five **identical** tasks, no shifts. If forgetting stays near E5's 0.3736,
the forgetting is dataset-intrinsic, the construction failed, and E5 belongs in the
appendix as a failed benchmark construction with the mechanism's non-image boundary
reported as **untested** — not as a found edge.

## 5. Consequence for the venue rule — E2 is now load-bearing

The contract's gate: *ICLR = H-G2 pass (or falsify-with-soften) ∧ H-G1 pass on ≥1
non-MNIST dataset (E2 or E5).*

- H-G2 already **failed** at E4 and took the falsify-and-soften branch.
- H-G1 has now **failed on E5** — the non-MNIST dataset the contract rated at
  **~75% confidence**, higher than E2's ~60%, and preferred as *"answers the
  dataset-family objection at least as well as a second image dataset."*

**E2 (CIFAR-10) is therefore no longer a backup; it is the only remaining path to
the pre-registered ICLR gate.** The contract's own downside branch is explicit:
*G1 fails on both E2 and E5 → ICLR off; scope the finding to demonstrated regimes in
the CoLLAs/TMLR manuscript, plainly.*

Also worth stating plainly: E2 is a *second image benchmark*, so even an E2 pass
leaves the mechanism demonstrated only on image-like inputs consumed row-wise. E5
was the experiment that would have broadened the modality claim, and it did not.

## 6. What E5 is *not* evidence of

- Not evidence the adapter mechanism is wrong on MNIST/Fashion — those results
  stand, verified, at +35 to +50pp.
- Not evidence about semantic shift, which was never in scope.
- Not evidence about deployed sensor drift; the shifts are simulated, exactly as the
  contract's pre-named caveat says.

The defensible claim after E5 is narrower than before it: **the mechanism produces
large gains where the input shift is expensive for the encoder to absorb, and no
measurable gain on a sensor benchmark where the shift is a low-dimensional channel
map.** Whether that boundary is the right characterization is what §4 measures.

## 7. Deviations

None in protocol. Seeds, arms, epochs, adapter geometry, and both thresholds as
pre-registered; adapter shape (per-timestep 9→9) is one of the two forms §2 E5
explicitly permits, recorded pre-run.

Two pre-run corrections are logged in `docs/E5_CONTRACT_NOTE.md`: the offset
exclusion (reverted before any run) and the shift-application order (moved to the
calibrated stream after measurement showed raw-space application confounded the
offset manipulation with gravity-times-rotation).

Verification was mechanical (`scripts/verify_runs.py`): 6/6 complete matrices, full
task histories.

**Correction to this memo's own first version.** It stated that the preempted run's
"mtime postdates the restart, satisfying clause (c)." Two things were wrong with
that. First, the verifier was invoked **without `--after`**, so clause (c) was never
actually enforced — the mtime was eyeballed, not checked. Second, and more
importantly, clause (c) would not have helped: the run's mtime *was* late, and the
run was contaminated anyway (§1a). The completeness definition is necessary and
**not sufficient**; only independent relaunch settles it.
