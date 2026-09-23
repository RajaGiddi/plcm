# PLCM — Persistent Latent Cell Memory

**Tackling catastrophic forgetting through persistent cell state memory in a learned thought space.**

PLCM augments a standard LSTM with a persistent external memory bank of cell state vectors that survives across tasks. Instead of relying solely on weight updates (which destructively interfere across tasks), PLCM stores computational states in a "thought space" and retrieves them via attention when encountering related inputs.

The key contribution is the **Gated Geometric Composition (GGC)** operator, which composes current cell states with retrieved memories using multiplicative modulation rather than additive blending. Additive blending converges to the centroid of stored vectors, destroying task-specific information. GGC preserves information through three parallel pathways: modulation, injection, and gated composition.

## Architecture

```
Input xₜ → LSTM → cₜ, hₜ
  → Read Controller queries Memory Bank M
  → Retrieves top-k relevant past cell states c̃
  → GGC: cₜ' = γ⊙(cₜ⊙μ) + (1-γ)⊙ι
  → Write Controller scores cₜ' for storage
  → Consolidation compresses memory periodically
  → Output uses cₜ' for classification
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all models for comparison
python scripts/train.py --config configs/default.yaml --model all

# Run PLCM only
python scripts/train.py --config configs/default.yaml --model plcm

# Run tests
pytest tests/ -v
```

## Models compared

| Model | Forgetting strategy | What it tests |
|-------|-------------------|---------------|
| Vanilla LSTM | None | Full forgetting baseline |
| LSTM + EWC | Weight regularization | Standard anti-forgetting method |
| PLCM (additive) | External memory + additive blend | Shows blending washes out |
| PLCM (GGC) | External memory + geometric composition | Full system |

## Project structure

```
CLAUDE.md                   ← Project context for Claude Code
configs/default.yaml        ← Training configuration
src/models/
  composition.py            ← GGC, Additive, Möbius operators
  memory_bank.py            ← Key-value memory store
  controllers.py            ← Read (attention) + Write (importance)
  consolidation.py          ← Memory compression
  plcm.py                   ← Full model
  lstm_base.py              ← Vanilla LSTM baseline
src/training/
  trainer.py                ← Continual learning loop
  ewc.py                    ← EWC baseline
  metrics.py                ← Forgetting/transfer metrics
src/data/
  permuted_mnist.py         ← Standard CL benchmark
scripts/
  train.py                  ← Main entry point
tests/                      ← Unit + integration tests
```

## Research phases

- **Phase 1** (current): Euclidean thought space + GGC composition
- **Phase 2**: Hyperbolic thought space + Möbius composition (code scaffolded in `composition.py`)
- **Phase 3**: Scaling, theoretical analysis, paper writeup

## Key design decisions

See `CLAUDE.md` for the full architectural rationale, math, and conventions.
