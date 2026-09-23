# PLCM — Persistent Latent Cell Memory

A continual-learning research program. It began as an architecture — an LSTM with
a persistent external memory of cell states — and became, mostly, an
investigation of **what catastrophic forgetting actually consists of**, run under
a pre-registration discipline strict enough that the negative results are the
ones worth reading.

The repository is organised so that **every number in a write-up traces to the
artifact that produced it, and every prediction to the contract that registered
it before the compute was spent.**

---

## The central finding

**Forgetting is mostly the readout, not the representation.**

A three-channel decomposition splits total forgetting on an old task into what
the encoder lost, what the readout lost, and instrument bias:

```
F_enc   = acc_refit_ceiling - acc_refit_T     the ENCODER's loss
F_read  = acc_refit_T       - acc_orig        the READOUT's loss
R       = acc_refit_ceiling - acc_ceiling     instrument bias
                                              F_enc + F_read - R = F_total
```

`acc_refit` is a linear probe fit to convergence on the features the deployed
model actually computes. The split matters because a readout term can be
repaired by refitting a head and an encoder term cannot.

The readout share, across four architecture families and one instrument:

| Arm | Reader share |
|---|---|
| Scratch LSTM · subject-disjoint UCI HAR | 75–89% (four arms) |
| Scratch MLP · disjoint-content Permuted MNIST | 89.5% [88.5, 90.6] |
| Pretrained ViT-B/16, full fine-tuning, 20 tasks | 91.8% [89.6, 94.1] |
| Pretrained ResNet-50, full fine-tuning, 20 tasks | 98.8% [98.3, 99.4] |

Old-task features survive fine-tuning nearly intact — `F_enc` is **0.06** on the
ViT and **0.0076** on the ResNet. The drift carrying the mismatch is *aimed* at
the readout subspace (32–42% of drift energy, against a 3.9% chance baseline),
not scattered. The same split holds under class-incremental evaluation.

`docs/CLAIM_LEDGER.md` is authoritative for the claim and its sourcing.

---

## What that bought, and what it did not

**Correcting the input path works — when the map is known.** On sensor data the
forgetting is presentation drift entirely, and re-laying old inputs into the
*current* frame with the known transform recovers more than the era checkpoint
had (forgetting **−0.038**, one encoder, no stored model, no refit). Into the
*base* layout it is harmful.

**And that is exactly what defeated the methods built on top of it.** Certified
generative refit ("bridging") reaches ρ ≈ **1.02 [0.82, 1.22]** and beats the
field's training-time baseline (LwF 0.226) — but it is dominated by the trivial
use of its own required resources: simply running old tasks through the stored
era snapshot scores ρ = **1.237**, above every ceiling in the program. *Bridging
has no regime where the map is known, because in that regime you apply the map.*
It is reported as a measured intermediate, not a contribution.

**The adapter result is a storage result, not a memory result.** Per-task input
adapters take permuted-MNIST accuracy from **0.4304** to **0.9183**, +50.3 pp —
and cost roughly **10× more storage than the replay buffer they beat**. The
`O(T·d²)` objection turns out to be regime-dependent rather than fundamental:
adapter cost tracks the shift's intrinsic dimensionality, not the input's (HAR's
9×9 channel map is 81 parameters per task; a flattened variant is 1.33 M with no
additional expressivity).

**The scratch result does not transfer to pretrained trunks.** On every scratch
arm, re-laying into the current frame removes the encoder term entirely. On
pretrained trunks it recovers nothing and costs. That transfer failure is the
largest single calibration event in the prediction ledger.

**Fine-tuning trades layouts rather than accumulating them.** Against a frozen
ImageNet ResNet-50 on the same frames, loaders and probe, the fine-tuned trunk
reads 4.7–5.1 pp *above* frozen on every permuted frame and **17.2 pp below** at
the base layout — the loss is 3.6× the gain.

---

## Negative results that closed directions

These are recorded because each one shut a door that looked open.

- **No rolling per-step repair.** Per-step drift is as non-linear as endpoint
  drift; no arm clears the 15% bar, and splitting the boundary buys nothing.
- **Encoders drift *away* from equivariance.** Every arm is less equivariant
  than its own random initialisation — LSTM arms 1.85, MLP arms 1.25.
- **Training for equivariance did not produce it.** Plain permutation
  augmentation achieves *invariance* on a finite family and cuts readout-repair
  cost 11×; adding an auxiliary equivariance loss preserved channel identity
  (0.836 vs 0.549 readable) and made repair **more** expensive, not less. Where
  the symmetry family is small enough to absorb, invariance suffices and the
  method is not needed.
- **Class means are too weak a reader to carry drift compensation.** Perfect
  prototypes reach 0.502 / 0.548 where a linear refit reads 0.745 / 0.769 —
  capping the whole family ~25 pp below the refit before any compensation runs.
- **No label-free readout repair has met the bar.** Label-free objectives cannot
  distinguish a correct readout from one with the classes relabelled, and every
  attempt has failed on that symmetry.

---

## The most recent result: recovering the map without labels

If correcting the input path is what works, the open question is whether the map
can be recovered when it is *unknown*. It can, with a stated requirement.

**Channel permutations** (exhaustive search over all 9! relabellings, scored
against stored input statistics, subject-disjoint):

| query pool, n = 256 | targets recovered |
|---|---|
| all six activities | 22/22 |
| **one static + one dynamic** | **22/22** |
| two dynamic activities | **0/22** |
| three static activities | **0/22** |

The requirement is **contrast, not count** — a calibration window must contain
at least one still activity and one moving one; more activities of one kind do
not help.

**Sensor rotations** are identifiable from covariance alone (0.00° on matched
subjects), with the estimator provably equivariant, so its error is a pure
subject-transfer bias: median **12.2°**, inside the model's ~15° tolerance,
with an upper quartile of 21.7° outside it. The method works for a typical
deployment and fails for a minority.

Storage: **171 floats per task, 684 bytes** — three orders of magnitude below
the per-task snapshot anchor, with no labels needed at repair time.

---

## Layout

| Layer | Where |
|---|---|
| Model, training, benchmarks | `src/models/`, `src/training/`, `src/data/` |
| Contracts (pre-registrations) | `docs/*_prereg.md` |
| Prediction ledger | `docs/appendix.tex` |
| Claim ledger (authoritative) | `docs/CLAIM_LEDGER.md` |
| Experiment index | `docs/INDEX.md` (generated — `scripts/make_index.py`) |
| Write-ups | `runs/MEMO_*.md` |
| Evidence those memos cite | `runs/` |
| Analysis and row scripts | `scripts/` |
| Compute | `modal_runner.py` |

`runs/` is version-controlled on purpose: a ledger row whose evidence lives
outside the repository is a citation nobody can check out. Model weights are the
exception — reproducible from the recorded configs, and excluded.

Benchmarks: permuted and rotated MNIST (shared- and disjoint-content), split
CIFAR-10/100, CIFAR-100 with a patch-consistent permutation, and UCI HAR with
simulated hardware-revision shifts in both shared-window and subject-disjoint
constructions.

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml --model plcm
pytest tests/ -v

# at scale, server-side, so a detached run survives the launching shell
modal run --detach modal_runner.py::spawn_analysis --experiment <name>
```

---

## How an experiment goes

1. **§0 is filled from the code first.** The mechanisms, config values and costs
   a contract is about to assume are pasted in from the files that implement
   them, *before* the contract is drafted. Several experiments were redesigned
   at this step because the premise turned out to be false — and one was
   answered outright, which is cheaper than running it.
2. **The contract is signed**: arms, measurements, controls, aggregation stated
   where the bar is, and predictions with odds.
3. **Checkpoints are audited** — criterion *loads*, not *exists*.
4. **It runs**, and a row script scores the registered predictions from the
   artifacts, never from prose.
5. **A memo reports it**, including what went wrong.

`CLAUDE.md` holds the standing rules and a numbered log of the errors that
produced each one. It is worth reading before trusting any number here: most
rules were written the day a plausible-looking result turned out to be an
artifact. Representative entries — a control that could not fail; a gate passed
emphatically on the wrong arm; a fingerprint blind to the axis it was guarding;
a statistic pinned by the design rather than measured.

### Prediction ledger

**112 scored entries, 31 misses**, plus one not comparable and one unmeasurable.
A miss is a prediction given above even odds that did not occur; below even odds
it is recorded as "did not fire" and not counted against the series.

The misses cluster, and the clusters are the useful part: priors formed on
scratch models transferred to pretrained ones and failed; drift repair proved
less linear and less composable than predicted on every arm measured; and
augmentation cost more within-task accuracy than predicted at every strength
tried.

---

## What this sets up

**Immediately open.**

- **E30 — does class geometry survive drift in order?** Signed; checkpoint audit
  green at 30/30 directories and 330 era checkpoints. A stored *c × c* matrix of
  distances between class means is side information that is not a function of
  the current feature distribution, so it could break the relabelling symmetry
  that has defeated every label-free repair. It is the only open item aimed at
  the pretrained regime — which is where a method contribution now has to come
  from, since input-path correction is dominated by trivially applying the map.
- **E31 — the exclusion control.** Does re-laid accuracy reflect *retention*, or
  is it current-task generalisation by construction? §0 (`docs/E31_section0.md`)
  already shows the premise holds structurally for permuted MNIST: re-laid
  old-task content is distributionally identical to current-task content. The
  control trains a model that never saw task *k* and compares.
- **Phase 2** — hyperbolic thought space and Möbius composition, scaffolded in
  `src/models/composition.py`, not yet run.

**Known gaps, recorded rather than left to be discovered** (`docs/INDEX.md`
lists them per experiment):

- Contracts for **E25–E28** are cited by their own memos and by scripts, and are
  not in `docs/`. They were signed in a working session and never committed, so
  ledger rows scoring their predictions currently cite a document this
  repository does not contain. They are **not** reconstructed here: a
  pre-registration written after the fact is the one document in this program
  that must never be back-filled.
- `docs/E21_mummadi.md` is cited by `runs/MEMO_e21.md` and is likewise absent.

These are provenance gaps, not disputed results.
