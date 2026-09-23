# Deck-Close Contract — Review Amendments

**Status:** review complete, six issues, all resolved. Items 2–4 clear to sign.
Item 1 clear once launched with the rebuilt job list.

The contract's own preamble says the last four items are "the ones most likely
to be waved through as obvious, which is exactly the condition the catches keep
arriving under." The review found **six**, four of them in Item 1, **five of
them mine**.

---

## Issue 1 — `scripts/e16_decompose.py` did not exist — RESOLVED

Item 1's only job pointed at a file that was never written. Catch 33 in the
launcher entry I wrote three days ago.

**Built** as a **driver, not a decomposition**: every quantity comes from
`channel_decomp.decompose()`, the audited LSTM-family path that produced the
66–88% row, and data loading is `channel_decomp.load_task_data`, imported and
not rewritten. The file owns arm→directory addressing, the seed list, and the
provenance header. Nothing else. That is catch 32 applied at design time.

## Issue 2 — `jobs_e16dec` had no λ dimension — RESOLVED

The contract's table names λ ∈ {0.25, 4, 16}; the job took `--bench` and
`--seed` only and could not address the checkpoint directories it was meant to
read. **Rebuilt to 4 arms × 2 benchmarks = 8 jobs.** The 28 `ckpt_e16*`
directories exist in the volume, so the expensive part was already done — the
addressing layer was the whole gap.

## Issue 3 — "`mafc_off` — reference, already decomposed" was FALSE — RESOLVED

**The premise was false and is struck.** The only LSTM-family decomposition on
record is `runs/e11_e6b/decomp.json`, whose keys are `HAR/OFF`, `HAR/v1-ON`,
`HAR/v2-ON`, … — **HAR-only, E6b-lineage, and a different arm**. Triply not the
reference Item 1 assumes.

**Resolution: the reference joins the table**, per ruling — four decomposition
targets, both benchmarks, same script, same seeds, matched era-checkpoint
status. Rescoping to LwF-arms-only would have left **H-D3 unanswerable**:
*"protection concentrates in F_read"* is only meaningful against the
unprotected arm's split, and without the reference the fifth-practice claim has
no denominator.

*Origin, recorded because it is the recurring one: memory of what should exist
standing in for a check of what does.*

## Issue 4 — Item 2's full-rank arm was unverified — RESOLVED, verified

`configs/mafc_phase1.yaml` has **no `rank` key**, and `jobs_w3c_fullrank` passed
no `--adapter-rank`. "Probably the default" is how catch 30 happened.

**Verified live:** `adapter_rank` defaults to 0, the code documents 0 as
"full-rank d × d", and a built 784-dim adapter has **614,656 params = 784²
exactly**. It resolves correctly.

**Made explicit anyway:** the job now passes `--adapter-rank 0` and the artifact
is named `w3c_fullrank_r0_seed{s}`, so the arm's identity is in the command line
and in the name rather than in a default a future config edit could move
underneath it. *The two-vanillas rule, extended: the rank is part of the arm
name.*

## Issue 5 — checkpoints are in the volume, not local — RESOLVED

The script defaulted to `runs/` and the era checkpoints live at `/runs/` in the
Modal volume. **Added `--ckpt-root`, defaulting to `/runs`** so the analysis
runs **where the checkpoints were built** — the E11 lesson, where E10
checkpoints re-evaluated locally missed their own recorded matrices.

## Issue 6 — the new precondition had no mechanism — RESOLVED, substituted

The amendment added to close the flag-set-drift channel required **"one commit
hash recorded in each output."** This repository has **zero commits**:

```
fatal: your current branch 'main' does not have any commits yet
```

**The clause written to prevent unmechanized clauses was itself unmechanized.**

**Substituted:** a **content hash of both files** — this driver *and*
`channel_decomp.py`, which owns every reported quantity. Hashing only the driver
would pin the addressing layer and leave the arithmetic free to move between
arms. It pins *the code that ran*, which is what a mismatch would differ in, and
does not depend on git hygiene. Git status is recorded alongside as
`"no-commits"` rather than silently omitted.

**Separately flagged, not fixed here:** a repository with no commits is a
reproducibility gap for the paper independent of this contract. Priced at
minutes; not folded into a signed contract as a silent extension.

---

## Preconditions verified before sign-off

| clause | status |
|---|---|
| era-checkpoint status uniform across compared arms | **PASS** — asserted from each run's own artifact; `[True, True, True]` on the arm tested. The script **exits 1** if not uniformly True |
| catch-32 label alignment | **structural** — `load_task_data` imported, not reimplemented |
| identity `F_enc + F_read − R = F_total` | printed before any channel number, with an explicit `COMPUTATION ERROR` branch |
| script version pinned across arms | content hash of driver + `channel_decomp.py`, in every output |
| 28 `ckpt_e16*` directories present | **verified in the volume** |

## Signed as drafted

**H-D2 signs unchanged.** Writing *"a collapsed-DIAG arm's decomposition answers
where did what little it learned go, and is not comparable to a competent arm's"*
**before** the numbers are read is the same discipline whose absence produced
the mixed-config misread — installed this time ahead of the instrument rather
than after it.

**H-D3's two-sided framing signs unchanged**, including reporting an F_enc
result *at equal prominence*. If Li & Hoiem's stated mechanism is vindicated,
that is a genuine counterexample to §6 and it is worth as much as a
confirmation.
