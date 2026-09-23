# PLCM — Persistent Latent Cell Memory

**Where does catastrophic forgetting happen — in a network's representation,
or in the layer that reads it out?** This repository contains an empirical
study of that question across four architecture families, the code and
artifacts behind every number, and an account of which conclusions held up
under external review and which did not.

> **Status: concluded (September 2026).** Two of the study's original headline
> claims did not survive review. They are corrected below, before any results,
> so that readers meet the limitations first. The measurements, code and
> negative results remain useful, and several open problems are described in
> [§9](#9-open-problems).

---

## Contents

1. [Summary](#1-summary)
2. [Setup and terminology](#2-setup-and-terminology)
3. [Corrections to the original claims](#3-corrections-to-the-original-claims)
4. [Results](#4-results)
5. [Results that carry caveats](#5-results-that-carry-caveats)
6. [Approaches that did not work](#6-approaches-that-did-not-work)
7. [Recovering an unknown input change](#7-recovering-an-unknown-input-change)
8. [Why readout repair is hard: three causes](#8-why-readout-repair-is-hard-three-causes)
9. [Open problems](#9-open-problems)
10. [Methodology](#10-methodology)
11. [Using this repository](#11-using-this-repository)
12. [Provenance and limitations](#12-provenance-and-limitations)

---

## 1. Summary

**Measured.** After sequential training, most of the accuracy a model loses on
an old task can be recovered by fitting a new linear readout to its current
features. This holds across recurrent, feed-forward, convolutional and
transformer encoders, and replicates Davari et al. (2022) and Anthes et al.
(2023) on a broader set of architectures.

**Originally claimed, and not supported:**

- that recovering accuracy by transforming old inputs into the current input
  format shows the encoder *retained* them — on the benchmarks used, this
  outcome is guaranteed by how the tasks are constructed;
- that scratch-trained and pretrained models behave oppositely because of how
  they were trained — the comparison is confounded;
- that the decomposition's identity detects computational errors — it holds
  for any four numbers.

**Supported:**

- exemplar-free drift-compensation methods fail when tasks differ by
  discontinuous input changes;
- class-mean (prototype) readouts are capped far below a linear readout on
  pretrained features;
- fine-tuning a pretrained model erodes its reading of its native input layout;
- reproduction noise must be measured separately for each reported quantity;
- repairing a stale readout without labels faces a symmetry that no search
  can resolve.

**A recommendation for similar studies.** Before running an experiment, compute
what it would show for a model that learned nothing — one trained only on the
current task, a randomly initialised encoder, or a linear probe on raw inputs.
If that baseline already produces the result, the experiment cannot
distinguish the hypotheses.

---

## 2. Setup and terminology

### Protocol

A shared encoder $f_\theta$ is trained on tasks $0, \ldots, T$ in sequence.
Each task's performance is read out by a linear **readout** (classification
head). A snapshot of the model saved at the end of task $k$ is called its
**era checkpoint**, $\theta_k$.

- On the scratch-trained models, one readout is **shared** across tasks and all
  tasks use the same labels. This is **domain-incremental** learning.
- On the pretrained models, each task has its own readout over its own classes.
  This is **task-incremental** learning.

### The decomposition

For an old task $k$, four accuracies are measured:

| Symbol | Accuracy on task $k$ |
|---|---|
| $a = A_k(\hat h_{\theta_k}, \theta_k)$ | a freshly fitted linear probe on the era checkpoint |
| $b = A_k(\hat h_{\theta_T}, \theta_T)$ | a freshly fitted linear probe on the final model |
| $c = A_k(h_k, \theta_T)$ | the deployed readout on the final model |
| $d = A_k(h_k, \theta_k)$ | the model's own readout on the era checkpoint |

These define

$$
F_{\mathrm{enc}} = a - b, \qquad
F_{\mathrm{read}} = b - c, \qquad
R = a - d, \qquad
F_{\mathrm{total}} = d - c,
$$

where $F_{\mathrm{enc}}$ is loss the encoder no longer supports,
$F_{\mathrm{read}}$ is loss a new readout can recover, and $R$ is the probe's
advantage over the model's own readout. They satisfy

$$
F_{\mathrm{enc}} + F_{\mathrm{read}} - R = F_{\mathrm{total}}.
$$

This identity holds by construction: substituting gives $d - c = d - c$. It
cannot detect an error in any single accuracy.

The **readout share** is a ratio of pooled terms over tasks and seeds:

$$
\text{share} = \frac{\overline{F_{\mathrm{read}}}}{\overline{F_{\mathrm{enc}}} + \overline{F_{\mathrm{read}}}}.
$$

**The probe** is $\ell_2$-regularised multinomial logistic regression
($C = 1$), fitted with L-BFGS to scikit-learn's default tolerance. Refitting at
a tolerance of $10^{-10}$ changes the pooled encoder term by at most $0.0007$
where measured, and a single task by up to $0.014$, with no consistent sign.

### Other terms

| Term | Meaning |
|---|---|
| **Format change** | A transformation $M$ applied to one task's inputs relative to another's — a channel permutation, rotation, or gain and offset on sensor data; a pixel permutation or rotation on images |
| **Re-laying** | Applying the known transformation to an old task's inputs so they arrive in the format the final model was last trained on |
| **Bridging** | Regenerating old-format training data from current data using a stored checkpoint and the known map, then refitting the readout |
| **Frozen trunk** | A pretrained encoder that is never fine-tuned, used as a reference |

---

## 3. Corrections to the original claims

Two independent reviews of a draft write-up, each checked against the code,
established the following.

| Finding | Basis | Consequence |
|---|---|---|
| **Recovery by re-laying is guaranteed by the benchmark construction.** Tasks share labels and draw content from a common pool, so old inputs re-laid into the current format are statistically identical to fresh samples of the current task | Every task's training data is a chunk of one training split; every test set a chunk of one test split | Re-laying is a valid repair when the transformation is known, but not evidence that anything was retained |
| **The encoder discards old-format information.** A linear probe on raw permuted-MNIST pixels reaches roughly 0.92; a probe on the final encoder reaches 0.62–0.78 in old formats | Frame-sweep measurements | This is forgetting in the encoder |
| **The scratch-versus-pretrained comparison is confounded.** Scratch tasks share labels; pretrained Split-CIFAR tasks each have different classes. Architecture and dataset also differ between the two groups | Benchmark construction | No claim about training regime is supported |
| **A frozen pretrained trunk reads as well as a fine-tuned one** (0.955 against 0.945) and forgets nothing | Frozen-trunk reference | High probe accuracy on pretrained features is consistent with generic features rather than retained learning |
| **The identity is a tautology** | §2 | It is bookkeeping, not an error check |
| **The readout-share finding replicates prior work** | Davari et al. 2022; Anthes et al. 2023 | A replication across architectures, not a new finding |

**Unresolved.** The decomposition reported for Learning without Forgetting may
imply $R \approx 0.074$, above the study's own admissibility threshold of
$|R| \le 0.05$. It should not be cited until checked.

---

## 4. Results

### 4.1 A new readout recovers most lost accuracy

| Model and benchmark | Readout share [95% CI] |
|---|---|
| Scratch LSTM · subject-disjoint UCI HAR (four configurations) | 75–89% |
| Scratch MLP · disjoint-content Permuted MNIST | 89.5% [88.5, 90.6] |
| Pretrained ViT-B/16, fine-tuned, 20 tasks · Split-CIFAR-100 | 91.8% [89.6, 94.1] |
| Pretrained ResNet-50, fine-tuned, 20 tasks · Split-CIFAR-100 | 98.8% [98.3, 99.4] |

The pattern also holds under class-incremental evaluation. §3 explains why a
small encoder term does not, by itself, show retention.

### 4.2 Drift compensation fails on discontinuous input changes

Semantic Drift Compensation (Yu et al., 2020) and Learnable Drift Compensation
(Gomez-Villa et al., 2024), adapted to nearest-class-mean readouts:

| Benchmark | Uncompensated | LDC | Oracle prototypes |
|---|---|---|---|
| MLP · Rotated MNIST (smooth change) | 0.553 | **0.670** | 0.848 |
| MLP · Permuted MNIST (discontinuous change) | 0.797 | **0.135** | 0.906 |

Drift estimated on the current task's data does not describe an old task's data
when their formats are unrelated, and compensation can make accuracy much
worse.

### 4.3 Prototype readouts are capped on pretrained features

| Backbone | Oracle prototypes | Linear refit | LDC |
|---|---|---|---|
| ResNet-50 | 0.502 | 0.745 | 0.313 |
| ViT-B/16 | 0.548 | 0.769 | 0.245 |

Even exactly updated class means fall about 25 points short of a linear
readout, which caps every prototype-based compensation method. On these
features the exact solution of LDC's objective collapses all prototypes to a
single class; the published recipe avoids this only because its optimiser,
started from the identity, does not reach that solution.

### 4.4 Fine-tuning erodes the native input layout

Change in linear-probe accuracy on old tasks, fine-tuned trunk minus frozen
trunk, after twenty tasks with permuted image patches:

| Backbone | Every permuted layout | Original (pretraining) layout |
|---|---|---|
| ResNet-50 | +4.7 to +5.1 pp | **−17.2 pp** |
| ViT-B/16 | +3.8 to +4.9 pp | **−19.3 pp** |

The frozen trunk reads best in its original layout in every one of 108
measurements. Fine-tuning gains a little on every permuted layout and loses
about four times as much on the original.

### 4.5 Reproduction noise depends on the quantity

In one replicate pair, average accuracy differed by 3.68 points while readout
share differed by 0.01. A noise estimate for one reported quantity does not
transfer to another.

### 4.6 Labelled data needed for readout repair

Refitting the readout with ten labelled examples per class recovers 65–77% of
the gap between the deployed readout and a fully refitted one. A fully labelled
but orthogonally constrained alignment recovers 61–81%.

---

## 5. Results that carry caveats

| Result | Caveat |
|---|---|
| **Drift is not linear.** The best affine map leaves 41–45% of drift unexplained; per-step maps leave as much as a single end-to-end map; composing twenty per-step corrections scores 0.830 against 0.882 for one fit | Singular-value statistics were computed on unwhitened features. The residual stands; claims about orthogonality need whitening |
| **Drift concentrates in the readout subspace:** 32–42% of drift energy against a 3.9% chance baseline | Unwhitened; and the associated subspace condition fails empirically on every configuration ($R^2 \le 0.06$) |
| **Training makes encoders less equivariant than at initialisation:** ratio 1.51–2.19 for LSTMs, 1.18–1.34 for MLPs | A whitening control was not run. Trained features concentrate variance in fewer directions, which alone could produce this |
| **The equivariance defect tracks readout-repair loss:** Spearman $\rho = 0.72$ (raw defect), $0.596$ (normalised) | The normalised version missed its pre-registered threshold of 0.6, and measurements within a configuration are not independent |

---

## 6. Approaches that did not work

Recorded because each closes a direction that appears promising.

| Approach | Outcome | Cause (§8) |
|---|---|---|
| **External memory of cell states** (the original PLCM architecture) | Stored states became unreadable as the model reading them continued to change | Information bound |
| **Bridging** | Final forgetting 0.095, close to refitting with retained data (0.081) — but applying the known map directly gives −0.038, and deploying the stored checkpoint gives 0.000 | Dominated by simpler uses of the same resources |
| **Per-task input adapters** | Permuted MNIST accuracy 0.430 → 0.918, at roughly ten times the storage of the replay buffer they outperform. Similar methods exist (AdapterNet, CLR) | Not novel |
| **Label-free readout repair** — entropy minimisation, diversity objectives, a stored checkpoint as teacher, shared or orthogonal realignment (seven variants) | None met the pre-registered bar | Identifiability |
| **Per-step drift correction** | Drift across one task boundary is no more linear than across the whole sequence | Wrong drift model |
| **Training for equivariance with an auxiliary loss** | Preserved channel identity (0.836 against 0.549) but made readout repair costlier than plain augmentation. All augmented configurations failed their accuracy precondition, so none is formally compared | Targets the wrong property |

**Learning without Forgetting acts on the smaller component.** On UCI HAR it
reduced the encoder term from 0.087 to 0.0015, but total forgetting only from
0.386 to 0.226. See the open question in §3.

---

## 7. Recovering an unknown input change

If transforming the input repairs forgetting when the transformation is known,
can the transformation be recovered when it is not? Estimating sensor
orientation or channel order from signal statistics is established calibration
practice; these results are a baseline, not a new method.

### 7.1 Channel permutations

Every permutation $P$ of the nine UCI HAR channels is scored against stored
reference statistics — mean $\mu$, covariance $\Sigma$, and lag-1
cross-covariance $C_1$:

\[
\hat P = \arg\min_{P \in S_9}\;
\frac{\|\Sigma' - P\Sigma P^\top\|_F^2}{\|\Sigma\|_F^2}
+ \frac{\|C_1' - P C_1 P^\top\|_F^2}{\|C_1\|_F^2}
+ \frac{\|\mu' - P\mu\|_2^2}{\|\mu\|_2^2 + \epsilon}.
\]

Reference and query drawn from different subjects, 256 query windows,
22 target permutations:

| Activities in the query | Targets recovered |
|---|---|
| All six | 22/22 |
| **One static + one dynamic** | **22/22** |
| Two dynamic | 0/22 |
| Three static | 0/22 |

Recovery depends on the query containing **contrasting** activities, not on
how many it contains.

### 7.2 Sensor rotations

A rotation shared across the three sensor triples is identifiable from
covariance alone (0.00° error when reference and query come from the same
subjects). The estimator is exactly equivariant, so its error reflects only
the difference between subjects. Across 20 subject pairings the median error is
**12.2°**, the upper quartile 21.7°, and the worst 33.9°, against a model
tolerance of roughly 15°.

The stored reference is 171 numbers per task, and no labels are needed.

---

## 8. Why readout repair is hard: three causes

**A — Identifiability.** A label-free objective depends only on the distribution
of the current features, so it cannot distinguish a correct readout from one
with the classes relabelled:

$$
\mathcal{L}(\sigma \circ \hat h) = \mathcal{L}(\hat h) \quad \text{for all } \sigma \in S_c .
$$

With $c$ classes there are $c! - 1$ equally scored incorrect readouts for each
correct one. Resolving this requires information that is not a function of the
current features — for example, at least one label per class.

**B — An information bound.** If $M$ is the input change, any repair that acts
only on the readout is bounded by what the encoder retains of the changed input:

$$
\mathrm{acc}(\hat h \circ f \circ M) \;\le\; \max_h \mathrm{acc}(h \circ f \circ M).
$$

The bound is tight when the encoder is equivariant to the change:

$$
f(Mx) = A\,f(x),\; A \text{ invertible} \;\Longrightarrow\; \hat h = h A^{-1}.
$$

**C — Misspecified drift models.** Compensation methods assume drift is a
translation (SDC), a linear map (LDC), or a composition of small linear steps.
The measured drift fits none of these. What training preserves is the linear
separability of the classes, not the geometry of the features.

Every repair that worked drew on at least one of three resources: **stored
data or models**, **knowledge of the input change**, or **restricted
plasticity** (a frozen encoder). None worked without one of them.

---

## 9. Open problems

### 9.1 When does low encoder damage indicate retention?

Probe-based analyses report low encoder damage, but §3 shows this can occur
when nothing task-specific was retained. Settling this needs:

- **an exclusion control** — a model trained on every task except $k$,
  evaluated on task $k$'s re-laid inputs. On the benchmarks here, the analysis
  in `docs/E31_section0.md` predicts no difference;
- **raw-input and random-encoder probes**, compared against the same gap on
  the current task, where forgetting cannot explain it;
- sensitivity to the probe's regularisation strength, and label-budget curves.

### 9.2 Breaking the relabelling symmetry without labels

If the *ranking* of distances between class means survives drift, a stored
$c \times c$ distance matrix could identify which current cluster corresponds
to which class, enabling label-free readout repair on pretrained features. A
design for this test, including a check that each task's geometry is
distinctive enough to begin with, has been pre-registered (see §12) but not
run.

### 9.3 Equivariance by construction

For small finite families of input changes, such as channel permutations, a
model can learn to ignore the change entirely. Continuous families such as
sensor rotations carry task information — in activity recognition, the
direction of gravity distinguishes lying from standing — so ignoring them
costs accuracy. Whether an encoder that is equivariant by construction (for
example, vector-neuron layers) makes readout-only repair sufficient in that
setting is open. Related work: rotation-equivariant activity recognition
(TRI-HAR).

### 9.4 One label per class plus unlabelled data

Communications systems resolve the same ambiguity with a few known pilot
symbols. One label per class breaks the symmetry in §8; whether unlabelled data
can then refine the readout to the accuracy of ten labels per class has not
been tested.

### 9.5 Benchmarks whose outcome is not fixed in advance

Future studies of where forgetting occurs would benefit from:

- real, recorded input changes rather than synthetic ones;
- shared-label and class-disjoint versions of the same construction;
- scratch-trained and pretrained versions of the same architecture.

### 9.6 Composition in hyperbolic space

A second phase, composing representations in hyperbolic space with Möbius
operations, is scaffolded in `src/models/composition.py` but was not run.

---

## 10. Methodology

Each experiment followed the same sequence:

1. **Verify assumptions from the code.** Every mechanism, configuration value
   and cost a design relies on is checked in the implementing source before the
   design is written. Several experiments were redesigned at this step.
2. **Pre-register.** Arms, measurements, controls, aggregation, and predictions
   with probabilities are fixed before any compute.
3. **Audit checkpoints** by loading them, not by checking that files exist.
4. **Run, then score** each prediction from the output artifacts.
5. **Report**, including errors and failed controls.

`CLAUDE.md` records the working rules and the specific errors that motivated
each — including a control that could not fail, a gate passed on the wrong
configuration, a fingerprint blind to the property it was meant to guard, and
a statistic fixed by the design rather than measured.

**Calibration.** 112 predictions were scored: 31 were misses (predicted at
above even odds, did not occur). Misses clustered in three places: expectations
formed on scratch-trained models that failed on pretrained ones; drift repairs
that proved less linear and less composable than expected; and augmentation
that cost more accuracy than expected.

---

## 11. Using this repository

| Contents | Location |
|---|---|
| Model, training, benchmarks | `src/models/`, `src/training/`, `src/data/` |
| Pre-registrations | `docs/*_prereg.md` |
| Prediction ledger | `docs/appendix.tex` |
| Claim ledger | `docs/CLAIM_LEDGER.md` |
| Experiment index | `docs/INDEX.md` (generated by `scripts/make_index.py`) |
| Experiment reports | `runs/MEMO_*.md` |
| Output artifacts | `runs/` |
| Analysis scripts | `scripts/` |
| Cloud compute entry point | `modal_runner.py` |

Output artifacts are version-controlled so that every reported number can be
traced to the file that produced it. Model weights are excluded and can be
regenerated from the recorded configurations.

**Benchmarks.** Permuted and rotated MNIST (shared- and disjoint-content
variants), Split CIFAR-10 and CIFAR-100, CIFAR-100 with patch permutations, and
UCI HAR with simulated sensor changes in shared-window and subject-disjoint
variants. All input changes are synthetic.

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml --model plcm
pytest tests/ -v

# larger runs, executed remotely
modal run --detach modal_runner.py::spawn_analysis --experiment <name>
```

**Platform.** Exact reproduction requires x86. The UCI HAR subject partition
built on ARM differs from the one used in the reported runs.

---

## 12. Provenance and limitations

- **Missing pre-registrations.** The pre-registrations for experiments E25–E30
  were written outside this repository and are not yet included; they will be
  added unedited. The document `docs/E21_mummadi.md`, cited in the E21 report,
  was never written.
- **Experiments not completed.** The class-geometry test (§9.2), the exclusion
  control (§9.1), a whitening control for the equivariance results (§5), and an
  ablation of the external memory were designed but not run.
- **Scope.** Models of up to 86M parameters; sequences of up to 20 tasks;
  synthetic input changes only.
- **Unresolved number.** The Learning-without-Forgetting decomposition may
  exceed the study's own admissibility threshold (§3).

### References

- Anthes, D. et al. (2023). *Diagnosing catastrophic forgetting in continual learning.*
- Davari, M. et al. (2022). *Probing representation forgetting in supervised and unsupervised continual learning.* CVPR.
- Gomez-Villa, A. et al. (2024). *Exemplar-free continual representation learning via learnable drift compensation.* ECCV.
- Kirichenko, P., Izmailov, P., Wilson, A. G. (2023). *Last layer re-training is sufficient for robustness to spurious correlations.* ICLR.
- Yu, L. et al. (2020). *Semantic drift compensation for class-incremental learning.* CVPR.
