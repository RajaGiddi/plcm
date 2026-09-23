# UNATTRIBUTED — not `e4_on` at current HEAD. Do not re-adopt.

**Status, 2026-08-16:** this run is **positively excluded** from the adapters
reference arm at the current code state. It must not be cited as `e4_on` seed 42,
and no audit should re-adopt it on the strength of its number matching.

## What it reads

`average_accuracy` **0.94696**, forgetting 0.0383. **No `arm` field** — the
artifact records nothing about its own configuration, because arm-recording
postdates it.

## Why it is excluded

The permuted-MNIST CPU path is **measured bit-deterministic**: `e4_on_seed42`
and `e4on_v2_seed42`, launched an hour apart under the recorded config, agree on
**15/15 cells, max |Δ| = 0.000000**. Both read **0.9179**.

A deterministic pipeline cannot produce this run's **−2.91pp** distance by
run-to-run variation. So `fullrank_ref` is a **different code state or a
different experiment — there is no third option.** Which one it is remains
unknown.

## The name is the standing hint

It sits beside `runs/lowrank_r32` and `runs/lowrank_r64`. The name says *the
full-rank reference arm of the low-rank adapter study*, not *`e4_on` seed 42*.
That lineage was visible before any compute was spent and was not read.

## How it entered the record (catch 34)

`mean(0.9470, 0.9279, 0.9243) = 0.9331` — matching the cited adapter row to four
decimals. The match was treated as identification. **A reconstruction that
matches a cited value is consistent with being its source, never evidence that
it is.**

## Supersession

The adapters reference row is now **0.9183** (n=3, `runs/e4on_v2_seed{42,1337,2024}`,
every artifact recording its own `arm`), superseding the cited **0.9331**.

## Still open

If the low-rank storage-frontier table survives into §5, **that table needs its
own provenance pass before it cites this run.** The number may be entirely
legitimate *in its own experiment* — same run, two tables, at most one
legitimate home.
