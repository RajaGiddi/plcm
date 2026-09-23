# PLCM — Persistent Latent Cell Memory

A continual-learning research program. It began as an architecture — an LSTM with
a persistent external memory of cell states — and became, mostly, an
investigation of **why continual-learning results are hard to believe**, run under
a pre-registration discipline strict enough that the negative results are the
ones worth reading.

The repository is organised so that **every number in a write-up can be traced to
the artifact that produced it, and every prediction to the contract that
registered it before the compute was spent.**

---

## What is actually here

| Layer | Where | What it holds |
|---|---|---|
| **Model** | `src/models/` | the GGC composition operator, memory bank, read/write controllers, consolidation, the full PLCM model, an LSTM baseline |
| **Training** | `src/training/` | the continual-learning loop, EWC, forgetting/transfer metrics |
| **Benchmarks** | `src/data/` | permuted and rotated MNIST, split CIFAR-10/100, UCI HAR with simulated hardware-revision shifts, and a subject-disjoint HAR variant |
| **Contracts** | `docs/*_prereg.md` | one pre-registration per experiment, signed before it ran |
| **Ledger** | `docs/appendix.tex` | every registered prediction and its outcome |
| **Write-ups** | `runs/MEMO_*.md` | one memo per experiment, citing artifacts by path |
| **Evidence** | `runs/` | the artifacts those memos cite, tracked deliberately |
| **Analysis** | `scripts/` | the decomposition instrument, screens, audits, and one row script per experiment |
| **Compute** | `modal_runner.py` | the Modal launcher and its job sets |

`runs/` is version-controlled on purpose. A ledger row whose evidence lives
outside the repository is a citation nobody can check out. Model weights are the
exception — they are reproducible from the recorded configs and stay out.

---

## The core instrument

Most findings here rest on a three-channel decomposition of forgetting. For an
old task *k*, comparing the checkpoint saved at the end of task *k* against the
final checkpoint:

```
F_enc   = acc_refit_ceiling - acc_refit_T     what the ENCODER lost
F_read  = acc_refit_T       - acc_orig        what the READOUT lost
R       = acc_refit_ceiling - acc_ceiling     instrument bias
                                              F_enc + F_read - R = F_total
```

`acc_refit` is a linear probe fit to convergence on the features the deployed
model actually computes. The split matters: a readout term can be repaired by
refitting a head, an encoder term cannot.

---

## Running it

```bash
pip install -r requirements.txt

# one arm, locally
python scripts/train.py --config configs/default.yaml --model plcm

# the benchmarks the program uses
#   permuted | rotated | har | har_subject | cifar100 | cifar100_permuted  (and no-shift controls)
python scripts/train.py --config configs/mafc_phase1.yaml --benchmark har_subject

pytest tests/ -v
```

Experiments at scale run on Modal, server-side, so a detached run survives the
launching shell:

```bash
modal run --detach modal_runner.py::spawn_experiment --experiment <name>
modal run --detach modal_runner.py::spawn_analysis   --experiment <name>
```

---

## How an experiment goes

1. **§0 is filled from the code first** — the mechanisms, config values and costs
   the contract is about to assume are pasted in from the files that implement
   them, before the contract is drafted. Several experiments were redesigned at
   this step because the premise turned out to be false.
2. **The contract is signed** (`docs/*_prereg.md`): arms, measurements, controls,
   aggregation stated where the bar is, and predictions with odds.
3. **Checkpoints are audited** — criterion *loads*, not *exists*.
4. **It runs**, and a row script scores the registered predictions from the
   artifacts, not from prose.
5. **A memo** (`runs/MEMO_*.md`) reports it, including what went wrong.

The rules this grew out of, and the numbered log of errors that produced each
one, are in `CLAUDE.md`. They are worth reading before trusting any number here:
most were written the day a plausible-looking result turned out to be an
artifact.

---

## Prediction ledger

`docs/appendix.tex` scores every registered prediction: **112 entries, 31
misses**, plus one recorded as not comparable and one as unmeasurable. A miss is
a prediction given above even odds that did not occur; below even odds it is
recorded as "did not fire" and not counted against the series.

The misses are the useful part. They cluster: priors formed on scratch models
transferred to pretrained ones and failed; drift repair proved less linear and
less composable than predicted on every arm measured; and a reader built on
class means turned out to be capped far below a linear probe before any method
was applied.

---

## Status

Phase 1 (Euclidean thought space, GGC composition) is the code that exists.
Phase 2 (hyperbolic space, Möbius composition) is scaffolded in
`src/models/composition.py` and not yet run.

**Known gaps**, recorded here rather than left to be discovered — `docs/INDEX.md`
lists them per experiment:

- Contracts for **E25–E28** are cited by their memos but are not in `docs/`.
  They were signed in a working session and never committed. The predictions
  they registered *are* scored in the ledger, so those rows currently cite a
  document the repository does not contain.
- `docs/E21_mummadi.md` is cited by `runs/MEMO_e25.md` and is likewise absent.

These are provenance gaps, not disputed results. They are listed so a reader
meets them in the index rather than in a dead link.
