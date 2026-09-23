"""
E11 addendum — how much of E7's deployed prediction is its OWN frozen head?

Ruling (2026-08-02): P-B failing on the E7 task-head arms means the deployed
system never read frozen heads AS frozen heads. `PLCM.forward` routes a convex
blend of ALL task heads by retrieval attention mass, then optionally blends the
bank's stored logits through the context-head gate. E7's branch-(D) reading
("frozen-early heads worsen forgetting") was interpreted as MISMATCH-to-theta_4;
it may be partly ROUTING. E7's premise check verified the heads existed and
froze -- it never verified the deployed path USED them per task. Catch 28's
sibling one level up: a premise verified EXISTENCE, not USAGE SEMANTICS.

WHAT IS MEASURED, per (arm, seed, old task k), on task k's full test set:

  alpha_own   attention mass routed to task k's OWN head, mean over samples
  alpha_max   mass on whichever head wins, and WHICH head that is
  lam         context-head gate: weight on the head mixture vs the bank's
              stored logits (1.0 = heads only, 0.0 = memory logits only)
  eff_own     lam * alpha_own -- the own head's actual share of the deployed logit
  agree       fraction of samples where own-head argmax == deployed argmax

READ: diffuse mass => E7's result is substantially a routing artifact and its
memo gets a supersession note. Concentrated mass => the original reading survives
with a footnote.

NOTHING IS REBUILT WITHOUT PROOF. alpha/lam are captured from the deployed call
by hooks, then recombined and asserted equal to the deployed logits. If the
recombination does not reproduce forward(), the measurement is wrong and the
script stops -- the same standard the feature lock is held to (catch 28).

Usage:
    python scripts/e7_routing_mass.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.channel_decomp import load, load_task_data, SEEDS
from src.data.har_shift import HARShiftBenchmark

ARMS = {"E7-heads":    ("runs/ckpt_e7_heads_seed{s}/mafc_seed{s}", False),
        "E7-heads+ad": ("runs/ckpt_e7_heads_adapt_seed{s}/mafc_seed{s}", True)}
TASKS = [0, 1, 2, 3]
RECON_TOL = 1e-4          # float32 recombination of a convex blend


@torch.no_grad()
def routing_of_batch(model, xb, task_k, use_adapter):
    """Capture the deployed routing, then PROVE the capture by recombining it."""
    cap = {}
    handles = [
        model.read_controller.register_forward_hook(
            lambda m, i, o: cap.__setitem__("info", o[2])),
        model.routing_gate.register_forward_hook(
            lambda m, i, o: cap.__setitem__("lam", torch.sigmoid(o).detach())),
    ]
    head_out = {}
    for key, head in model.task_classifiers.items():
        handles.append(head.register_forward_hook(
            lambda m, i, o, k=key: head_out.__setitem__(k, o.detach())))
    try:
        out = model(xb, store_memories=False, task_hint=task_k,
                    apply_adapter=use_adapter)
    finally:
        for h in handles:
            h.remove()

    info = cap["info"]
    bw, tids = info["bank_weights"], info["task_ids"]          # [B, top_k]
    alpha = {k: ((tids == int(k)).float() * bw).sum(-1, keepdim=True)   # [B, 1]
             for k in model.task_classifiers}

    # --- the proof: recombine exactly as forward() does, compare to forward() ---
    recon = sum(alpha[k] * head_out[k] for k in head_out)
    lam = cap.get("lam")
    if model.use_context_heads and info.get("logits") is not None and lam is not None:
        recon = lam * recon + (1.0 - lam) * info["logits"]
    d = float((recon - out["logits"]).abs().max())
    assert d < RECON_TOL, (
        f"routing reconstruction does not reproduce the deployed logits "
        f"(max |diff| = {d:.3e}) — the alpha/lam measurement is wrong, not the model")

    own = alpha[str(task_k)].squeeze(-1) if str(task_k) in alpha else torch.zeros(xb.shape[0])
    stacked = torch.cat([alpha[k] for k in sorted(alpha, key=int)], dim=-1)  # [B, n_heads]
    lam_v = lam.squeeze(-1) if lam is not None else torch.ones(xb.shape[0])
    own_head_pred = (head_out[str(task_k)].argmax(-1) if str(task_k) in head_out
                     else out["logits"].argmax(-1))
    return dict(
        own=own, alpha=stacked, lam=lam_v, recon_err=d,
        agree=(own_head_pred == out["logits"].argmax(-1)).float(),
        total_mass=stacked.sum(-1),
    )


def main():
    ap = argparse.ArgumentParser(description="E7 routing mass")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/e11_e7_routing.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    data = load_task_data(HARShiftBenchmark(num_tasks=5, root=".", batch_size=128),
                          n_tasks=5)
    rows = []
    for arm, (tmpl, use_ad) in ARMS.items():
        for seed in SEEDS:
            model = load(tmpl.format(s=seed), 4)
            for k in TASKS:
                xte = data[k][2]
                accs = [routing_of_batch(model, xte[i:i + 128].to(device), k, use_ad)
                        for i in range(0, xte.shape[0], 128)]
                cat = lambda key: torch.cat([a[key] for a in accs])
                alpha = torch.cat([a["alpha"] for a in accs])          # [N, heads]
                rows.append(dict(
                    arm=arm, seed=seed, task=k,
                    alpha_own=float(cat("own").mean()),
                    alpha_max=float(alpha.max(-1).values.mean()),
                    argmax_head=int(alpha.mean(0).argmax()),
                    total_mass=float(cat("total_mass").mean()),
                    lam=float(cat("lam").mean()),
                    eff_own=float((cat("own") * cat("lam")).mean()),
                    agree=float(cat("agree").mean()),
                    recon_err=max(a["recon_err"] for a in accs),
                ))

    print("=" * 100)
    print("E7 ROUTING MASS — is the deployed prediction the task's OWN frozen head?")
    print("=" * 100)
    print(f"  recombination proof: max error {max(r['recon_err'] for r in rows):.2e} "
          f"(< {RECON_TOL} required, else the script would have stopped)")
    print(f"\n  {'arm':<14}{'seed':>6}{'task':>5}{'alpha_own':>11}{'alpha_max':>11}"
          f"{'top head':>10}{'lam':>8}{'eff_own':>9}{'agree':>8}")
    for r in rows:
        print(f"  {r['arm']:<14}{r['seed']:>6}{r['task']:>5}{r['alpha_own']:>11.4f}"
              f"{r['alpha_max']:>11.4f}{r['argmax_head']:>10}{r['lam']:>8.4f}"
              f"{r['eff_own']:>9.4f}{r['agree']:>8.4f}")

    print("\n" + "=" * 100)
    print("READ (computed from the columns above)")
    print("=" * 100)
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        own = float(np.mean([r["alpha_own"] for r in rs]))
        eff = float(np.mean([r["eff_own"] for r in rs]))
        agree = float(np.mean([r["agree"] for r in rs]))
        n_own_top = sum(r["argmax_head"] == r["task"] for r in rs)
        mass = float(np.mean([r["total_mass"] for r in rs]))
        print(f"  {arm:<14} mean alpha_own {own:.4f} of total mass {mass:.4f}  "
              f"eff_own {eff:.4f}  own-head argmax agreement {agree:.4f}  "
              f"own head is top-weighted in {n_own_top}/{len(rs)} cells")
        share = own / mass if mass else float("nan")
        print(f"  {'':<14} -> own-task head carries {share*100:.1f}% of the routed "
              f"mass; deployed prediction matches its own head on "
              f"{agree*100:.1f}% of samples")

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
