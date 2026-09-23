# E2 Capacity Calibration + Reproduction Floor

Contract: docs/GENERALITY_prereg.md §2 E2 (v2 Finding 3) + v3 amendment 7.
Single-task clean CIFAR-10, **--no-adapters** (capacity is an encoder property
both arms share; §2 E2 forbids per-arm tuning, and a 3072x3072 adapter would add
9.4M parameters of capacity that mask the encoder differences being resolved).

| hidden | rep 1 | rep 2 | reproduction spread |
|---|---|---|---|
| 256 | 0.5788 | 0.5788 | **0.0000pp** |
| 512 | 0.5952 | 0.5952 | **0.0000pp** |
| 1024 | 0.6035 | 0.6035 | **0.0000pp** |

**Capacity selected: hidden=1024** at 0.6035 clean single-task accuracy,
per the locked rule (take the best, record it, proceed regardless of the retired
55% reference — which it clears anyway). Both arms share it; never re-tuned.

## Reproduction floor

**0.0000pp at every capacity — bit-identical across independent launches.**

This is a sharp contrast with HAR's CPU runs, where the same-config spread reached
5.5pp (catch 19). The floor is a property of a configuration, not of the project.

**Scope limit, stated rather than assumed.** This floor was measured at
`num_tasks=1` with adapters OFF. E2's runs are 5 tasks with adapters ON, giving
run-to-run divergence four more task boundaries to accumulate across. Catch 19's
lesson is that floors do not transfer between configurations, so the E2 launch
includes a 2-run floor pair (`e2floor_*`) re-running cifar_permuted/seed42 in
**both** arms under identical configs. The +15pp H-G1b bar is only interpretable
against that measurement, not against this one.

## Timing

 hidden=1024: ~112s for one task -> ~9 min per 5-task run; 12 runs fan out in
parallel on L4 GPUs. CPU was infeasible (~9h/run), which is why E2 alone moved to GPU.
