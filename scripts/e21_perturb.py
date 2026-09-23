"""E21 -- repair under an approximate map. Contract: docs/E21_prereg.md (SIGNED 2026-09-17).

One cell = (model seed s, old task k) on the HAR har_subject OFF arm, S72
checkpoints (fp32 shadow). The deployment's map is M_hat_k = M_k + eps; M_4 is
exact. Families, levels and realizations exactly as sec 1 displays them; arms
exactly as sec 2's where-eps-enters table, each cited to the line it copies:

  A0   deployed theta_T on X_k                                  (cure_screen.py:234)
  A1   relayout with M_hat: M_4 M_hat^-1 (x - c_hat) + c_4      (cure_screen.py:141-150, 249, 261)
  A2   bridging with M_hat: pseudo = M_hat M_4^-1 (x_4 - c_4) + c_hat on task-4 TRAIN inputs,
       labels argmax p_theta_k(pseudo), head refit on theta_T features of pseudo,
       tested on X_k in the TRUE frame                         (cure_screen.py:280, 285-288)
  A3   TENT-on-map: phi* = argmin mean_x H(p_theta_T(relayout_{M_hat(phi)}(x))); then A1 with M_hat(phi*)
  A4   Mummadi-objective on map params -- INTENT FORM (SLR confidence + diversity); the exact
       transcription is a precondition (docs/E21_mummadi.md) and this arm is NOT READ until it lands
  A5   era-teacher refinement: y_hat = argmax p_theta_k(x) on the TRUE frame (no map);
       phi* = argmin mean_x CE(p_theta_T(relayout_{M_hat(phi)}(x)), y_hat); then A1 with M_hat(phi*)
  A6   relayout with the exact M_k (anchor, constant across eps)
  A7   bridging on A5's refined map
  SNAP theta_k on X_k (anchor, 0 by construction)
  A5d  deranged teacher, y_hat -> (y_hat + 1) mod 6           (control C-DT)

The differentiable forward for A3-A5 is `model(x, store_memories=False,
task_hint=k, apply_adapter=False)["logits"]` -- the SAME call
`channel_decomp._run_deployed` makes (catch 28), with gradients enabled and
flowing only into phi. Optimizer pinned per sec 4: LBFGS, max_iter 50,
tolerance_grad 1e-6, strong Wolfe, full batch, 3 init jitters N(0, 0.01^2),
n_iter recorded; the swap family is an exhaustive search over 37 candidates
per greedy step. The probe subset draw is seeded (20260916) and recorded.

Usage:
    modal run modal_runner.py::analysis --argv "scripts/e21_perturb.py --smoke"
    modal run modal_runner.py::analysis --argv "scripts/e21_perturb.py --family gain --level 2 --realization 0"
"""

import argparse
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.har_shift import spec_fingerprint, N_CHANNELS
from src.data.har_subject import HARSubjectBenchmark
from scripts.channel_decomp import (load, features_and_logits, refit_probe, load_task_data,
                                    assert_path_identity, PROBE_SUBSET_SEED)
from scripts.cure_screen import har_maps, har_relayout, E10EC_HAR_ARMS, CUR, SEEDS, BATCH
from scripts.c0deg_controls import REGISTERED_SHIFT, AS_EXECUTED_PARTITION, banner

OLD = list(range(CUR))
N_CLASSES = 6
LEVELS = {"gain": [0.0, 0.05, 0.10, 0.20, 0.30], "offset": [0.0, 0.05, 0.10, 0.20, 0.30], "swap": [0, 1, 2, 3]}
N_REAL = 3
SEED_BASE = 20260917
LBFGS_MAX_ITER, LBFGS_TOL = 50, 1e-6
JITTER_SD, JITTERS = 0.01, (0, 1, 2)
TRANSPOSITIONS = list(itertools.combinations(range(N_CHANNELS), 2))    # 36
MUMMADI_LAMBDA = 1.0            # E25 C: set by --mummadi-lambda; 1.0 is the as-run E21 value
MUMMADI_SIGN = -1.0             # E25 C: -1 is the objective (slr - lam*H); +1 is the sign-flip must-fail
# TRANSCRIPTION (E25 sec 0, 2026-09-21; supersedes "pending"). Mummadi et al. 2021
# (arXiv:2106.14999) sec 3.2: L = L_div + delta*L_conf, delta = 0.025, with
# L_div = KL(pbar || uniform) = log C - H(pbar) and L_conf = L_slr of sec 3.2.2.
# Dividing by delta: argmin L = argmin [ L_slr - 40*H(pbar) ] up to a constant.
# The A4 objective below is exactly that at lambda = 40; E21 ran it at lambda = 1,
# i.e. Mummadi's form at 1/40th the diversity weight. docs/E21_mummadi.md was
# never written; this comment and E25's sec 0 are the transcription.


def realization_seed(level_idx: int, r: int) -> int:
    return SEED_BASE * 100 + 10 * level_idx + r


# ------------------------------------------------------------------ families --
def perturb(family: str, level_idx: int, r: int, M: np.ndarray, c: np.ndarray, sigma: np.ndarray):
    """Return (M_hat, c_hat, description) per docs/E21_prereg.md sec 1."""
    rng = np.random.default_rng(realization_seed(level_idx, r))
    lv = LEVELS[family][level_idx]
    if family == "gain":
        s = rng.choice([-1.0, 1.0], size=N_CHANNELS)
        return np.diag(1.0 + lv * s) @ M, c.copy(), {"g": lv, "signs": s.tolist()}
    if family == "offset":
        s = rng.choice([-1.0, 1.0], size=N_CHANNELS)
        return M.copy(), c + lv * sigma * s, {"o": lv, "signs": s.tolist()}
    if family == "swap":
        Q = np.eye(N_CHANNELS); chosen = []
        avail = list(range(N_CHANNELS))
        for _ in range(int(lv)):
            i, j = rng.choice(avail, size=2, replace=False); avail.remove(i); avail.remove(j)
            T = np.eye(N_CHANNELS); T[[i, j]] = T[[j, i]]; Q = T @ Q; chosen.append((int(i), int(j)))
        return Q @ M, Q @ c, {"m": int(lv), "transpositions": chosen}
    raise ValueError(family)


def relayout_maps(maps, k, M_hat, c_hat):
    """A maps dict in which task k carries the perturbed map and task 4 the exact one."""
    m = dict(maps); m[k] = (M_hat, c_hat); return m


# ------------------------------------------------------------ refined maps --
LOG_GAIN_BOUND = float(np.log(2.0))   # AMENDMENT 2026-09-17: gain corrections bounded to (1/2, 2)
OFFSET_BOUND_SIGMA = 1.0              # AMENDMENT 2026-09-17: offset corrections bounded to +-1 sigma per channel


def apply_phi(family, M_hat, c_hat, phi, sigma=None):
    """The refinement, per sec 3 AS AMENDED 2026-09-17.

    v1 used diag(e^phi) M_hat with phi unbounded. LBFGS's strong-Wolfe line
    search then took trial steps that sent some phi_i to huge magnitudes:
    exp(phi_i) underflowed to 0 and `relayout_t`'s inverse met a zero diagonal
    (11 gain-family jobs, `torch._C._LinAlgError`), or overflowed (1 job). The
    search space is now BOUNDED through a smooth squash, which the contract's
    own realistic range licenses: a gain correction outside (1/2, 2) or an
    offset correction beyond 1 sigma is outside every calibration error the
    sweep considers (max gain error 30% needs a correction of 0.77 or 1.43).
      gain   : M(phi) = diag( exp( ln2 * tanh(phi) ) ) M_hat        in (1/2, 2) per channel
      offset : c(phi) = c_hat + sigma * tanh(phi)                    within +-1 sigma per channel
      swap   : Q_phi M_hat  (unchanged; discrete)
    phi = 0 is still the identity refinement, so C-ID and the jitter inits are
    unchanged in meaning. The offset family ran to completion unbounded; it is
    re-run bounded so the optimizer is identical across the continuous
    families, and its unbounded artifacts are kept beside as superseded.
    """
    if family == "gain":
        return torch.diag(torch.exp(LOG_GAIN_BOUND * torch.tanh(phi))) @ M_hat, c_hat
    if family == "offset":
        assert sigma is not None, "offset refinement needs sigma_k"
        return M_hat, c_hat + OFFSET_BOUND_SIGMA * sigma * torch.tanh(phi)
    Q = phi                                       # a permutation matrix (tensor) for the swap family
    return Q @ M_hat, Q @ c_hat


def relayout_t(x, M_src, c_src, M_dst, c_dst):
    """Differentiable har_relayout: (x - c_src) M_src^-T M_dst^T + c_dst, on [N, T, 9].
    With the bounded parameterization M_src is never singular; the finiteness
    check is a loud guard rather than a silent NaN downstream."""
    Minv = torch.linalg.inv(M_src)
    if not torch.isfinite(Minv).all():
        raise RuntimeError("relayout_t: non-finite inverse -- the refined map left the bounded family")
    Z = (x - c_src) @ Minv.T
    return Z @ M_dst.T + c_dst


def logits_grad(model, x, k):
    """The deployed forward with gradients enabled -- the same call _run_deployed makes."""
    return model(x, store_memories=False, task_hint=k, apply_adapter=False)["logits"]


def objective(arm, model, x_true, k, M_src, c_src, M4, c4, teacher=None):
    """The three objectives of sec 2, as functions of the refined (M_src, c_src)."""
    logits = logits_grad(model, relayout_t(x_true, M_src, c_src, M4, c4), k)
    p = F.softmax(logits, dim=1)
    if arm == "A3":                                             # TENT: mean entropy
        return -(p * torch.log(p + 1e-12)).sum(1).mean()
    if arm == "A4":                                             # Mummadi: SLR + sign * lambda * H(mean p)
        slr = -(p * torch.log(p / (1 - p + 1e-12) + 1e-12)).sum(1).mean()
        pbar = p.mean(0)
        return slr + MUMMADI_SIGN * MUMMADI_LAMBDA * (-(pbar * torch.log(pbar + 1e-12)).sum())
    if arm == "A5":                                             # era teacher: CE to y_hat
        return F.cross_entropy(logits, teacher)
    raise ValueError(arm)


def refine_continuous(arm, family, model, x_true, k, M_hat, c_hat, M4, c4, teacher, device, sigma=None):
    """LBFGS on phi in R^9, three init jitters; returns per-jitter (phi, n_iter, converged, objective)."""
    outs = []
    sigma_t = None if sigma is None else torch.tensor(sigma, dtype=torch.float32, device=device)
    M_hat_t, c_hat_t = torch.tensor(M_hat, dtype=torch.float32, device=device), torch.tensor(c_hat, dtype=torch.float32, device=device)
    M4_t, c4_t = torch.tensor(M4, dtype=torch.float32, device=device), torch.tensor(c4, dtype=torch.float32, device=device)
    for j in JITTERS:
        g = torch.Generator().manual_seed(j)
        phi = (torch.randn(N_CHANNELS, generator=g) * JITTER_SD).to(device).requires_grad_(True)
        opt = torch.optim.LBFGS([phi], max_iter=LBFGS_MAX_ITER, tolerance_grad=LBFGS_TOL, line_search_fn="strong_wolfe")
        n_eval = [0]

        def closure():
            opt.zero_grad()
            Mr, cr = apply_phi(family, M_hat_t, c_hat_t, phi, sigma_t)
            loss = objective(arm, model, x_true, k, Mr, cr, M4_t, c4_t, teacher)
            loss.backward(); n_eval[0] += 1
            return loss
        t0 = time.time(); final = opt.step(closure); dt = time.time() - t0
        st = opt.state[opt._params[0]]
        n_iter = int(st.get("n_iter", 0))
        outs.append({"jitter": j, "phi": phi.detach().cpu().tolist(), "n_iter": n_iter, "n_eval": n_eval[0],
                     "converged": n_iter < LBFGS_MAX_ITER, "objective": float(final), "seconds": dt})
    return outs


def refine_swap(arm, model, x_true, k, M_hat, c_hat, M4, c4, teacher, device, depth):
    """Exhaustive over {identity} + 36 transpositions per greedy step, up to `depth` steps."""
    M_hat_t, c_hat_t = torch.tensor(M_hat, dtype=torch.float32, device=device), torch.tensor(c_hat, dtype=torch.float32, device=device)
    M4_t, c4_t = torch.tensor(M4, dtype=torch.float32, device=device), torch.tensor(c4, dtype=torch.float32, device=device)
    Q = torch.eye(N_CHANNELS, device=device); chosen = []; t0 = time.time(); n_eval = 0
    with torch.no_grad():
        for _ in range(max(depth, 1)):
            best, best_T = None, None
            for cand in [None] + TRANSPOSITIONS:
                T = torch.eye(N_CHANNELS, device=device)
                if cand is not None:
                    i, j = cand; T[[i, j]] = T[[j, i]]
                Mr, cr = apply_phi("swap", M_hat_t, c_hat_t, T @ Q)
                val = float(objective(arm, model, x_true, k, Mr, cr, M4_t, c4_t, teacher)); n_eval += 1
                if best is None or val < best:
                    best, best_T, best_c = val, T, cand
            if best_c is None:
                break
            Q = best_T @ Q; chosen.append(best_c)
    return [{"jitter": None, "phi": Q.cpu().tolist(), "chosen": chosen, "n_iter": len(chosen), "n_eval": n_eval,
             "converged": True, "objective": best, "seconds": time.time() - t0}]


# ---------------------------------------------------------------- one cell --
def run_cell(family, level_idx, r, seed, k, models, data, maps, sigma, device, arms=("A0", "A1", "A2", "A3", "A4", "A5", "A5d", "A6", "A7", "SNAP")):
    m4, m_era = models
    xtr, ytr, xte, yte = data[k]
    x4_tr = data[CUR][0]
    Mk, ck = maps[k]; M4, c4 = maps[CUR]
    M_hat, c_hat, desc = perturb(family, level_idx, r, Mk, ck, sigma[k]) if level_idx > 0 else (Mk.copy(), ck.copy(), {"eps": 0})
    maps_hat = relayout_maps(maps, k, M_hat, c_hat)
    acc = lambda l: float((l.argmax(1) == yte).float().mean())
    row = {"family": family, "level_idx": level_idx, "level": LEVELS[family][level_idx], "realization": r,
           "realization_seed": realization_seed(level_idx, r) if level_idx > 0 else None, "eps": desc,
           "seed": seed, "task": k, "probe_subset_seed": PROBE_SUBSET_SEED, "timing_s": {}}
    T = {}
    # references, deployed path (cure_screen.py:228-236)
    t0 = time.time()
    g4 = assert_path_identity(m4, xte, k, False, device, label=f"t4/T{k}", verbose=False)
    f_e_te, l_e_te = features_and_logits(m_era, xte, yte, k, False, device)
    f_4_te, l_4_te = features_and_logits(m4, xte, yte, k, False, device)
    row["p_b"] = bool(g4["p_b"]); row["SNAP"] = acc(l_e_te); row["A0"] = acc(l_4_te); T["A0+SNAP"] = time.time() - t0
    # A1 / A6
    t0 = time.time()
    _, l_a1 = features_and_logits(m4, har_relayout(xte, k, CUR, maps_hat), yte, k, False, device); row["A1"] = acc(l_a1)
    _, l_a6 = features_and_logits(m4, har_relayout(xte, k, CUR, maps), yte, k, False, device); row["A6"] = acc(l_a6)
    T["A1+A6"] = time.time() - t0

    def bridging(mh):                                             # cure_screen.py:280, 285-288 with a maps dict
        x_pseudo = har_relayout(x4_tr, CUR, k, mh)
        _, l_snap = features_and_logits(m_era, x_pseudo, None, k, False, device)
        y_pseudo = l_snap.argmax(1)
        f_ps, _ = features_and_logits(m4, x_pseudo, y_pseudo, k, False, device)
        return refit_probe(f_ps, y_pseudo, f_4_te, yte, N_CLASSES)
    # E25 C runs A4 alone, so the bridging refit -- by far the most expensive arm --
    # is gated. "A2" is in the default tuple, so the E21 path is unchanged.
    if "A2" in arms:
        t0 = time.time(); row["A2"] = bridging(maps_hat); T["A2"] = time.time() - t0

    # refined-map arms
    x_true = xte.to(device)
    teacher = l_e_te.argmax(1).to(device)                          # era model on the TRUE frame
    teacher_d = (teacher + 1) % N_CLASSES                          # C-DT derangement
    phi_star = {}
    for arm, tch in (("A3", None), ("A4", None), ("A5", teacher), ("A5d", teacher_d)):
        if arm not in arms:
            continue
        t0 = time.time()
        obj_arm = "A5" if arm == "A5d" else arm
        if family == "swap":
            fits = refine_swap(obj_arm, m4, x_true, k, M_hat, c_hat, M4, c4, tch, device, depth=max(int(LEVELS["swap"][level_idx]), 1))
        else:
            fits = refine_continuous(obj_arm, family, m4, x_true, k, M_hat, c_hat, M4, c4, tch, device, sigma=sigma[k])
        accs = []
        sig_t = torch.tensor(sigma[k], dtype=torch.float32)
        for ft in fits:
            phi = torch.tensor(ft["phi"], dtype=torch.float32)
            Mr, cr = apply_phi(family, torch.tensor(M_hat, dtype=torch.float32), torch.tensor(c_hat, dtype=torch.float32), phi, sig_t)
            mh = relayout_maps(maps, k, Mr.numpy().astype(np.float64), cr.numpy().astype(np.float64))
            _, l = features_and_logits(m4, har_relayout(xte, k, CUR, mh), yte, k, False, device)
            accs.append(acc(l)); ft["acc"] = accs[-1]
        row[arm] = float(np.mean(accs)); row[arm + "_fits"] = fits; row[arm + "_spread"] = float(max(accs) - min(accs))
        if arm == "A5":
            best = max(fits, key=lambda f: f["acc"]); phi_star["A5"] = best["phi"]
        T[arm] = time.time() - t0
    if "A7" in arms and "A5" in phi_star:
        t0 = time.time()
        phi = torch.tensor(phi_star["A5"], dtype=torch.float32)
        Mr, cr = apply_phi(family, torch.tensor(M_hat, dtype=torch.float32), torch.tensor(c_hat, dtype=torch.float32), phi, torch.tensor(sigma[k], dtype=torch.float32))
        row["A7"] = bridging(relayout_maps(maps, k, Mr.numpy().astype(np.float64), cr.numpy().astype(np.float64)))
        T["A7"] = time.time() - t0
    row["timing_s"] = {k_: round(v, 2) for k_, v in T.items()}
    return row


def main():
    ap = argparse.ArgumentParser(description="E21 harness")
    ap.add_argument("--smoke", action="store_true", help="one cell, family gain level 2 realization 0, all arms, timed")
    ap.add_argument("--family", choices=sorted(LEVELS)); ap.add_argument("--level", type=int); ap.add_argument("--realization", type=int, default=0)
    ap.add_argument("--out", default=None); ap.add_argument("--device", default="cpu")
    # ---- E25 C (2026-09-21). Defaults reproduce E21 exactly; the flag regression
    # (contract sec C, Controls) is what verifies that rather than this comment.
    ap.add_argument("--mummadi-lambda", type=float, default=MUMMADI_LAMBDA,
                    help="diversity weight in A4's objective. 1.0 = as E21 ran it. "
                         "40 = Mummadi's own weighting (see the transcription at the top).")
    ap.add_argument("--diversity-sign", type=float, default=MUMMADI_SIGN, choices=[-1.0, 1.0],
                    help="-1 is the objective (slr - lam*H). +1 is the SIGN-FLIP MUST-FAIL: "
                         "rewarding collapse instead of diversity must do worse than A3.")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset, e.g. 'A0,A1,A4,A6,SNAP'. Default: all ten.")
    args = ap.parse_args()
    globals()["MUMMADI_LAMBDA"] = float(args.mummadi_lambda)
    globals()["MUMMADI_SIGN"] = float(args.diversity_sign)
    arms = (tuple(a.strip() for a in args.arms.split(",")) if args.arms
            else ("A0", "A1", "A2", "A3", "A4", "A5", "A5d", "A6", "A7", "SNAP"))
    device = torch.device(args.device)
    banner("E21  repair under an approximate map")
    print(f"  platform {platform.machine()}  torch {torch.__version__}")
    assert spec_fingerprint() == REGISTERED_SHIFT
    har = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=BATCH)
    pfp = har.partition_fingerprint(); print(f"  partition {pfp} -> {'AS-EXECUTED' if pfp == AS_EXECUTED_PARTITION else 'STOP'}")
    if pfp != AS_EXECUTED_PARTITION:
        return 1
    maps = har_maps(har._sd)
    data = load_task_data(har, n_tasks=5, probe_subset_seed=PROBE_SUBSET_SEED)
    sigma = {k: data[k][0].reshape(-1, N_CHANNELS).std(0).numpy().astype(np.float64) for k in OLD}
    tmpl = E10EC_HAR_ARMS["OFF"][0]
    # C-WIT: arm identity from the run artifacts
    for s in SEEDS:
        a = json.load(open(f"runs/e10off_ec_seed{s}/mafc_results.json"))["arm"]
        assert a["benchmark"] == "har_subject" and a["era_checkpoints"] and not a["use_input_adapters"] and a["torch_num_threads"] == 4, a
    print("  C-WIT arm identity from artifacts: OK (har_subject, era ON, adapters off, 4 threads)")

    if args.smoke:
        family, level_idx, r, cells = "gain", 2, 0, [(42, 1)]
        out_path = args.out or "runs/e21/smoke.json"
    else:
        family, level_idx, r = args.family, args.level, args.realization
        cells = [(s, k) for s in SEEDS for k in OLD]
        out_path = args.out or f"runs/e21/{family}/level{level_idx}_real{r}.json"
    print(f"  family {family}  level {LEVELS[family][level_idx]}  realization {r}  cells {cells}")
    rows = []
    cache = {}
    for s, k in cells:
        if s not in cache:
            cache[s] = (load(tmpl.format(s=s), CUR), None)
        m4 = cache[s][0]; m_era = load(tmpl.format(s=s), k)
        t0 = time.time()
        row = run_cell(family, level_idx, r, s, k, (m4, m_era), data, maps, sigma, device, arms=arms)
        row["timing_s"]["cell_total"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"  cell s{s}/t{k}: " + "  ".join(f"{a} {row[a]:.4f}" for a in arms if a in row)
              + f"  | p_b {row['p_b']}  | seconds {row['timing_s']}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"family": family, "level_idx": level_idx, "level": LEVELS[family][level_idx], "realization": r,
               "optimizer": {"lbfgs_max_iter": LBFGS_MAX_ITER, "tol": LBFGS_TOL, "jitter_sd": JITTER_SD, "jitters": list(JITTERS),
                             "parameterization": "bounded (2026-09-17 amendment): gain exp(ln2*tanh), offset sigma*tanh"},
               "mummadi_form": {
                   "objective": "slr + sign * lambda * H(mean p)",
                   "lambda": MUMMADI_LAMBDA, "sign": MUMMADI_SIGN,
                   "transcription": ("Mummadi 2021 sec 3.2: L = L_div + 0.025*L_slr with "
                                     "L_div = log C - H(pbar); dividing by 0.025 gives "
                                     "L_slr - 40*H(pbar). lambda=40 sign=-1 IS their objective; "
                                     "lambda=1 sign=-1 is what E21 ran (1/40th the diversity weight); "
                                     "sign=+1 is the sign-flip must-fail."),
                   "supersedes": "INTENT -- transcription pending (docs/E21_mummadi.md); A4 not read"},
               "arms": list(arms),
               "rows": rows}, open(out_path, "w"), indent=2, default=float)
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
