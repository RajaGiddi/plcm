"""
RBST feasibility experiment — pre-registered protocol (amended).

Tests whether stored LSTM cell states can be transported through a sequence of
encoder updates (Task 0 -> Task 1 drift) such that they remain readable where
untransported states go to chance.

Amendments on record (approved before any results existed):
  A1. Readers are ERA-t probes R_t, trained per horizon on true era-t states of
      a disjoint probe-train input set, fixed recipe, never on transported
      states. M0 = R_t on raw era-0 vectors. Upper bound = R_t on true era-t
      states of S_old's inputs.
  A2 (PENDING sign-off, both readouts computed): M2's correct-step updates only
      sigma^2, so M2 means are IDENTICAL to M1 by construction and the original
      H2 (M2 - M1 >= 10pp) is unfalsifiable-in-favor. Proposed H2': probe
      accuracy on the lowest-sigma^2 half of memories exceeds accuracy on all
      memories by >= 10pp at the final horizon (variance-aware selection).

Locked hyperparameters (per approved amendment): see CONSTANTS below.
Transported object: raw LSTM cell state c_t = c_n[-1] (NOT c_prime).

Usage:
    python scripts/rbst_feasibility.py                     # all 3 seeds
    python scripts/rbst_feasibility.py --seeds 42          # smoke on one seed
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM
from src.data.permuted_mnist import PermutedMNIST, PermutedMNISTBenchmark

# ---------------- locked protocol constants (do not tune post-hoc) ------------
SEEDS = [42, 1337, 2024]
N_T1_EPOCHS = 10                 # task-1 epochs in the transport chain
HORIZONS = [3, 5, 8, 10]         # task-1 epochs (1-indexed); chain positions
N_S_OLD = 2000                   # task-0 TEST inputs, indices [0:2000]
N_PROBE = 8000                   # task-0 TRAIN inputs for probes, [0:8000]
N_PAIRS, N_HOLDOUT = 4096, 512   # task-1 TRAIN inputs for g_t, [0:4608]
T1_PROBE_SLICE = (8000, 16000)   # task-1 TRAIN inputs for P_task, disjoint from pairs
SIGMA0_SQ = 0.01                 # initial per-dim variance
KAPPA = 10.0                     # OOD variance inflation
TAU_PCT, TAU_OOD_PCT = 25, 90    # percentiles of live-target NN distances
EPS_JAC = 1e-3                   # finite-difference step for diagonal Jacobian
SHRINK = 0.5                     # correct-step shrink factor toward residual var
G_STEPS, G_LR, G_BATCH = 500, 1e-3, 256
PROBE_STEPS, PROBE_LR = 300, 1e-2
RECIPE_SEED = 0                  # fixed seed for every probe / drift-map fit
# ------------------------------------------------------------------------------


def get_inputs(seed: int) -> dict:
    """Deterministic input slices via direct dataset indexing (no shuffle)."""
    bench = PermutedMNISTBenchmark(num_tasks=2, batch_size=128, seed=seed)
    t0_train = PermutedMNIST(train=True, permutation=bench.permutations[0])
    t0_test = PermutedMNIST(train=False, permutation=bench.permutations[0])
    t1_train = PermutedMNIST(train=True, permutation=bench.permutations[1])

    def slab(ds, lo, hi):
        xs, ys = zip(*(ds[i] for i in range(lo, hi)))
        return torch.stack(xs), torch.tensor(ys)

    x_sold, y_sold = slab(t0_test, 0, N_S_OLD)
    x_probe, y_probe = slab(t0_train, 0, N_PROBE)
    x_pairs, _ = slab(t1_train, 0, N_PAIRS + N_HOLDOUT)
    x_t1probe, _ = slab(t1_train, *T1_PROBE_SLICE)
    return {
        "x_sold": x_sold, "y_sold": y_sold,
        "x_probe": x_probe, "y_probe": y_probe,
        "x_pairs": x_pairs, "x_t1probe": x_t1probe,
    }


@torch.no_grad()
def cell_states(model: PLCM, x: torch.Tensor, device, bs: int = 1024) -> torch.Tensor:
    """Raw final-layer LSTM cell states c_n[-1] (the transported object)."""
    out = []
    for i in range(0, x.shape[0], bs):
        _, (_, c) = model.lstm(x[i:i + bs].to(device))
        out.append(c[-1].cpu())
    return torch.cat(out)


def load_model(ckpt_path: str, device) -> PLCM:
    ck = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    m = PLCM.load_from_checkpoint(ck)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


class DriftMap(nn.Module):
    """Small MLP g_t: R^256 -> R^256 (2-layer, hidden 256)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)


def fit_drift_map(src: torch.Tensor, dst: torch.Tensor, device):
    """Fixed recipe. Returns (g, q) with q = per-dim residual variance on holdout."""
    torch.manual_seed(RECIPE_SEED)
    g = DriftMap(src.shape[1]).to(device)
    opt = torch.optim.Adam(g.parameters(), lr=G_LR)
    s_tr, d_tr = src[:N_PAIRS].to(device), dst[:N_PAIRS].to(device)
    s_ho, d_ho = src[N_PAIRS:].to(device), dst[N_PAIRS:].to(device)
    for step in range(G_STEPS):
        idx = torch.randint(0, s_tr.shape[0], (G_BATCH,), device=device)
        opt.zero_grad()
        nn.functional.mse_loss(g(s_tr[idx]), d_tr[idx]).backward()
        opt.step()
    g.eval()
    with torch.no_grad():
        q = (g(s_ho) - d_ho).var(dim=0).cpu()  # [dim]
    for p in g.parameters():
        p.requires_grad = False
    return g, q


def fit_probe(states: torch.Tensor, labels: torch.Tensor, n_classes: int, device):
    """Fixed-recipe linear probe. Never trained on transported states."""
    torch.manual_seed(RECIPE_SEED)
    probe = nn.Linear(states.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    s, y = states.to(device), labels.to(device)
    for _ in range(PROBE_STEPS):
        opt.zero_grad()
        nn.functional.cross_entropy(probe(s), y).backward()
        opt.step()
    probe.eval()
    for p in probe.parameters():
        p.requires_grad = False
    return probe


@torch.no_grad()
def probe_correct(probe, states: torch.Tensor, labels: torch.Tensor, device):
    pred = probe(states.to(device)).argmax(1).cpu()
    return (pred == labels).float()


@torch.no_grad()
def diag_jacobian(g: DriftMap, mu: torch.Tensor, device, dim_bs: int = 64):
    """Per-dimension diagonal finite-difference Jacobian of g at mu. [N, dim]"""
    mu_d = mu.to(device)
    base = g(mu_d)  # [N, dim]
    dim = mu.shape[1]
    J = torch.zeros_like(base)
    for lo in range(0, dim, dim_bs):
        hi = min(lo + dim_bs, dim)
        for i in range(lo, hi):
            pert = mu_d.clone()
            pert[:, i] += EPS_JAC
            J[:, i] = (g(pert)[:, i] - base[:, i]) / EPS_JAC
    return J.cpu()


@torch.no_grad()
def nn_dist(a: torch.Tensor, b: torch.Tensor, device, exclude_self=False):
    """For each row of a, distance to nearest row of b. [Na]"""
    d = torch.cdist(a.to(device), b.to(device))  # [Na, Nb]
    if exclude_self:
        d.fill_diagonal_(float("inf"))
    return d.min(dim=1).values.cpu()


@torch.no_grad()
def mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(nn.functional.cosine_similarity(a, b, dim=1).mean())


def build_chain(ckdir: str, n_tasks: int) -> tuple[list, list]:
    """Checkpoint chain and horizons.

    Short chain (n_tasks=2): Task 0 -> Task 1, horizons at epochs {3,5,8,10}.
    Long chain (n_tasks=5, escalation per Ruling 2): Task 0 -> Task 4, S_old
    still fixed at end-of-Task-0, horizons at each TASK BOUNDARY (steps
    10/20/30/40).
    """
    chain = [os.path.join(ckdir, "task0_epoch14.pt")]
    for t in range(1, n_tasks):
        chain += [os.path.join(ckdir, f"task{t}_epoch{e}.pt")
                  for e in range(N_T1_EPOCHS)]
    if n_tasks == 2:
        return chain, list(HORIZONS)
    return chain, [t * N_T1_EPOCHS for t in range(1, n_tasks)]


def run_seed(seed: int, ckpt_root: str, device, n_tasks: int = 2) -> dict:
    ckdir = os.path.join(ckpt_root, f"plcm_seed{seed}")
    chain, horizons = build_chain(ckdir, n_tasks)
    for p in chain:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing checkpoint: {p}")

    data = get_inputs(seed)
    x_sold, y_sold = data["x_sold"], data["y_sold"]
    dim = None

    # --- era-0 anchor: states + t=0 probe baseline (pre-registered record) ---
    m0_model = load_model(chain[0], device)
    sold_era0 = cell_states(m0_model, x_sold, device)          # the diary entries
    probe_era0_states = cell_states(m0_model, data["x_probe"], device)
    pairs_prev = cell_states(m0_model, data["x_pairs"], device)
    dim = sold_era0.shape[1]
    r0 = fit_probe(probe_era0_states, data["y_probe"], 10, device)
    baseline_t0 = float(probe_correct(r0, sold_era0, y_sold, device).mean())
    del m0_model

    # --- filter state (M2); M1 == M2 means by construction (H2 note) ---
    mu = sold_era0.clone()
    var = torch.full_like(mu, SIGMA0_SQ)
    m3_src_states = pairs_prev.clone()  # era-0 states of pair inputs (for M3)

    results = {"seed": seed, "baseline_t0_digit_probe": baseline_t0, "horizons": {}}

    for k in range(1, len(chain)):  # transport step k: chain[k-1] -> chain[k]
        model_k = load_model(chain[k], device)
        pairs_cur = cell_states(model_k, data["x_pairs"], device)
        g, q = fit_drift_map(pairs_prev, pairs_cur, device)

        # M2 predict
        with torch.no_grad():
            J = diag_jacobian(g, mu, device)
            mu = g(mu.to(device)).cpu()
            var = (J ** 2) * var + q.unsqueeze(0)
        # M2 correct / OOD inflate (percentile thresholds from live targets)
        live_tgt = pairs_cur[:N_PAIRS]
        self_nn = nn_dist(live_tgt, live_tgt, device, exclude_self=True)
        tau = float(np.percentile(self_nn.numpy(), TAU_PCT))
        tau_ood = float(np.percentile(self_nn.numpy(), TAU_OOD_PCT))
        d_mem = nn_dist(mu, live_tgt, device)
        near = d_mem < tau
        far = d_mem > tau_ood
        var[near] = SHRINK * var[near] + (1 - SHRINK) * q.unsqueeze(0)
        var[far] = var[far] * KAPPA
        var.clamp_(min=1e-8)

        if k in horizons:
            # --- era-k readers (amendment A1): fixed recipe, true era-k states ---
            probe_states_k = cell_states(model_k, data["x_probe"], device)
            t1_states_k = cell_states(model_k, data["x_t1probe"], device)
            r_digit = fit_probe(probe_states_k, data["y_probe"], 10, device)
            task_states = torch.cat([probe_states_k, t1_states_k])
            task_labels = torch.cat([
                torch.zeros(probe_states_k.shape[0], dtype=torch.long),
                torch.ones(t1_states_k.shape[0], dtype=torch.long),
            ])
            r_task = fit_probe(task_states, task_labels, 2, device)

            # --- M3: single direct map era-0 -> era-k (LDC-style) ---
            g3, _ = fit_drift_map(m3_src_states, pairs_cur, device)
            with torch.no_grad():
                m3_states = g3(sold_era0.to(device)).cpu()

            # --- ground truth era-k states of S_old inputs (upper bound + geometry) ---
            sold_true_k = cell_states(model_k, x_sold, device)

            conds = {
                "M0_untransported": sold_era0,
                "M1_naive": mu,          # identical to M2 means (H2 note)
                "M2_filtered": mu,
                "M3_snapshot": m3_states,
                "UPPER_true_era_t": sold_true_k,
            }
            h = {}
            digit_correct_m2 = None
            for name, s in conds.items():
                dc = probe_correct(r_digit, s, y_sold, device)
                tc = probe_correct(r_task, s, torch.zeros(s.shape[0], dtype=torch.long), device)
                h[name] = {"digit_acc": float(dc.mean()), "task_acc": float(tc.mean())}
                if name == "M2_filtered":
                    digit_correct_m2 = dc

            # --- geometry ---
            h["geometry_cosine"] = {
                "M0": mean_cosine(sold_era0, sold_true_k),
                "M1_M2": mean_cosine(mu, sold_true_k),
                "M3": mean_cosine(m3_states, sold_true_k),
            }

            # --- H3 calibration: sigma^2 vs probe error ---
            sbar = var.mean(dim=1).numpy()
            err = (1.0 - digit_correct_m2.numpy())
            rho, pval = spearmanr(sbar, err)
            deciles = []
            order = np.argsort(sbar)
            for dc_bin in np.array_split(order, 10):
                deciles.append(float(err[dc_bin].mean()))
            h["calibration"] = {"spearman_rho": float(rho), "p": float(pval),
                                "decile_err": deciles}

            # --- H2 readouts: original (identity check) + proposed H2' ---
            low_half = order[: len(order) // 2]
            h["H2_original_M2_minus_M1_pp"] = 0.0  # identical by construction
            h["H2prime_lowvar_half_acc"] = float(digit_correct_m2.numpy()[low_half].mean())
            h["H2prime_all_acc"] = float(digit_correct_m2.mean())

            results["horizons"][k] = h
            print(f"  seed {seed} horizon {k}: "
                  f"M0={h['M0_untransported']['digit_acc']:.3f} "
                  f"M1/M2={h['M1_naive']['digit_acc']:.3f} "
                  f"M3={h['M3_snapshot']['digit_acc']:.3f} "
                  f"UPPER={h['UPPER_true_era_t']['digit_acc']:.3f} "
                  f"| cos(M2,true)={h['geometry_cosine']['M1_M2']:.3f} "
                  f"| rho={h['calibration']['spearman_rho']:.3f}")

        pairs_prev = pairs_cur
        del model_k

    return results


def main():
    ap = argparse.ArgumentParser(description="RBST feasibility (pre-registered)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--ckpt-root", type=str, default="checkpoints/rbst/")
    ap.add_argument("--out", type=str, default="runs/rbst/")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--n-tasks", type=int, default=2,
        help="2 = short chain (Tasks 0->1, horizons {3,5,8,10}); "
             "5 = escalated chain (Tasks 0->4, horizons at task boundaries)",
    )
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(args.out, exist_ok=True)
    all_results = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        r = run_seed(seed, args.ckpt_root, device, n_tasks=args.n_tasks)
        all_results.append(r)
        with open(os.path.join(args.out, f"rbst_seed{seed}.json"), "w") as f:
            json.dump(r, f, indent=2)

    # ---- aggregate: mean +/- std across seeds, decision rules ----
    horizons = sorted(int(k) for k in all_results[0]["horizons"])
    final = horizons[-1]
    def agg(path_fn):
        vals = [path_fn(r) for r in all_results]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {"n_seeds": len(all_results), "chain_tasks": args.n_tasks,
               "horizons": horizons, "per_horizon": {}}
    for k in horizons:
        row = {}
        for cond in ["M0_untransported", "M1_naive", "M2_filtered", "M3_snapshot",
                     "UPPER_true_era_t"]:
            m, s = agg(lambda r, c=cond, k=k: r["horizons"][str(k)][c]["digit_acc"]
                       if str(k) in r["horizons"] else r["horizons"][k][c]["digit_acc"])
            row[cond] = {"digit_acc_mean": m, "digit_acc_std": s}
        summary["per_horizon"][k] = row

    m2_m, m2_s = agg(lambda r: r["horizons"][final]["M2_filtered"]["digit_acc"])
    m0_m, _ = agg(lambda r: r["horizons"][final]["M0_untransported"]["digit_acc"])
    m3_m, _ = agg(lambda r: r["horizons"][final]["M3_snapshot"]["digit_acc"])
    rho_m, rho_s = agg(lambda r: r["horizons"][final]["calibration"]["spearman_rho"])
    h2p_low, _ = agg(lambda r: r["horizons"][final]["H2prime_lowvar_half_acc"])
    h2p_all, _ = agg(lambda r: r["horizons"][final]["H2prime_all_acc"])

    summary["decision"] = {
        "H1_transport_ge_70pp_and_M0_lt_25pp": {
            "M2_final": m2_m, "M0_final": m0_m,
            "pass": bool(m2_m >= 0.70 and m0_m < 0.25),
        },
        "H2_original_note": "M2 means == M1 by construction; delta is 0pp always "
                            "(flagged pre-results; see H2prime)",
        "H2prime_selection_ge_10pp": {
            "lowvar_half": h2p_low, "all": h2p_all,
            "delta_pp": (h2p_low - h2p_all) * 100,
            "pass": bool((h2p_low - h2p_all) >= 0.10),
        },
        "H3_calibration_rho_ge_0.3": {
            "rho_mean": rho_m, "rho_std": rho_s, "pass": bool(rho_m >= 0.3),
        },
        "M3_vs_M2_streaming_delta_pp": (m2_m - m3_m) * 100,
    }
    with open(os.path.join(args.out, "rbst_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RBST FEASIBILITY SUMMARY (mean over seeds, {args.n_tasks}-task chain)")
    print("=" * 70)
    for k in horizons:
        row = summary["per_horizon"][k]
        print(f"  horizon {k:2d}: " + "  ".join(
            f"{c.split('_')[0]}={row[c]['digit_acc_mean']:.3f}±{row[c]['digit_acc_std']:.3f}"
            for c in row))
    for name, d in summary["decision"].items():
        print(f"  {name}: {d}")


if __name__ == "__main__":
    main()
