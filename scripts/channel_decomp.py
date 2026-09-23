"""
E6b — Three-channel forgetting decomposition.

Pre-registration: docs/E6B_prereg.md (v2)

Separates, per arm, how much of the measured forgetting is carried by:
  C-enc   the encoder degrading old-task representations, and
  C-read  the reader walking away from representations that are still fine.

RECIPE-CONTROLLED. The naive version (ceiling-vs-refit) moves the encoder AND the
head recipe at once: a head fit to convergence beats an online-trained head, which
inflates the refit accuracy, understates F_enc and OVERSTATES F_read -- i.e. biases
toward the hypotheses. So a fourth quantity is measured, refitting at the ceiling
checkpoint too, which holds the recipe constant and turns the bias into a number:

    R(k)      = acc_refit_ceiling - acc_ceiling     # the instrument's own bias
    F_enc(k)  = acc_refit_ceiling - acc_refit_t4    # encoder effect, recipe fixed
    F_read(k) = acc_refit_t4      - acc_orig        # reader effect, encoder fixed
    F_total(k)= acc_ceiling       - acc_orig        # deployed forgetting
    identity:   F_enc + F_read - R = F_total        # exact; printed to catch bugs

R is also the instrument control (H-C3'): if |R| > 0.05 the refit advantage is too
large for F_read to mean "reader-walk", and no channel claim may be read.

REFIT RECIPE (amendment v3 -- identical in every cell, and knob-free):
    multinomial logistic regression, lbfgs, fit to CONVERGENCE (max_iter 5000,
    convergence checked), on features standardized with TRAIN statistics; fit on
    task-k TRAIN, evaluated on task-k TEST. The FEATURE is the deployed one,
    o_t*tanh(c_t), so the refit differs from the deployed head in fitting
    procedure only.

    v2's recipe (Adam, lr 1e-3, 5 epochs), inherited from brittleness.py, was
    REJECTED by H-C3' on first use: it underfit by ~50pp and diverged on raw c_t.
    Its validation certificate was for bounded 784-dim pixels and a transfer
    question -- it did not transfer to unbounded cell states (max |c| = 75).

DEPLOYED-PATHWAY LOCK: on ON arms the refit and its evaluation apply task-k's own
adapter, exactly as deployment does. Without this, F_read absorbs an input-pathway
swap instead of measuring reader-walk.

Usage:
    python scripts/channel_decomp.py
    python scripts/channel_decomp.py --skip-mnist
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM
from src.data.har_shift import HARShiftBenchmark
from src.data.permuted_mnist import PermutedMNISTBenchmark

SEEDS = [42, 1337, 2024]
LBFGS_MAX_ITER = 5000   # convex solver; convergence asserted, not assumed

# Recorded per row. The execution image installed scikit-learn unpinned, so the
# version behind the existing artifacts is unknown; from here on it travels
# with the numbers.
try:
    import sklearn as _sk
    _SKLEARN_VERSION = _sk.__version__
except Exception:            # never let the witness crash the measurement
    _SKLEARN_VERSION = None
N_TRAIN = 4000          # capped for runtime; identical in every cell
N_TEST = 2000
R_BAR = 0.05            # H-C3'
F_READ_BAR = 0.10       # H-C1 / H-C2 absolute floor

# E11 ruling — arms whose DEPLOYED readout is a routed blend of all five task
# heads (weighted by retrieval attention), not any one head. P-B fails on these:
# measured max |Δlogit| 5.44, and on E7-heads+ad task 2 **75% of predictions**
# differ from the own-head argmax. F_read is defined as deployed-reader vs
# matched-reader; a routed blend has no single reader to be mismatched, so
# F_read is not interpretable as reader-walk here. Their floors are still exact
# (the instrument returns deployed logits) — only the DECOMPOSITION is void.
# They are printed, labelled, and excluded from the pooled H-C3' gate.
PB_ROUTED_ARMS = {"E7-heads", "E7-heads+ad"}

HAR_ARMS = {
    "E7-heads":   ("runs/ckpt_e7_heads_seed{s}/mafc_seed{s}", False),
    "E7-heads+ad": ("runs/ckpt_e7_heads_adapt_seed{s}/mafc_seed{s}", True),
    "v1-ON":      ("runs/ckpt_e5_seed{s}/mafc_seed{s}", True),
    "OFF":        ("runs/ckpt_e5_off_seed{s}/mafc_seed{s}", False),
    "v2-ON":      ("runs/ckpt_e5d_v2on_seed{s}/mafc_seed{s}", True),
    "v2-control": ("runs/ckpt_e5d_v2ctl_seed{s}/mafc_seed{s}", True),
}
MNIST_ARMS = {
    "ON": ({42: "checkpoints/fullrank_ref/mafc_seed42",
            1337: "checkpoints/e4_on_seed1337/mafc_seed1337",
            2024: "checkpoints/e4_on_seed2024/mafc_seed2024"}, True),
    "OFF": ({s: f"checkpoints/e4_off_seed{s}/mafc_seed{s}" for s in SEEDS}, False),
}


def load(d: str, task: int, epoch: int = 9) -> PLCM:
    """The end-of-task-`task` model as DEPLOYED — via PLCM.load_era.

    E11/step 3. The previous body called load_from_checkpoint directly, which
    returns a model with task_stats == {} because that dict is not in state_dict.
    forward() gates coordinate alignment on task_stats being non-empty, so every
    analysis in this program since E4b silently evaluated a model missing that
    path: on E4/seed42 a bare reload missed its own accuracy matrix by up to
    3.4pp (systematically high, worst on the OFF arm), and load_era reproduced
    all eight cells exactly. load_era also enforces ERA correctness — a task-k
    model gets k entries, never task 4's.
    """
    m = PLCM.load_era(d, task, epoch=epoch)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def _run_deployed(model: PLCM, xb, task_k: int, use_adapter: bool):
    """One deployed forward, with every readout module hooked.

    Returns (output_dict, captures) where captures is a list of
    (module_name, readout_input, readout_output) recorded DURING the call --
    not reconstructed after it. Hooking is what makes the identity check able
    to fail: if the reported feature came from anywhere other than the tensor
    the heads consumed, the comparison below is unequal.
    """
    captures = []
    handles = []

    def mk(name):
        def hook(mod, inp, out):
            captures.append((name, inp[0].detach(), out.detach()))
        return hook

    handles.append(model.classifier.register_forward_hook(mk("classifier")))
    for key, head in model.task_classifiers.items():
        handles.append(head.register_forward_hook(mk(f"task_classifiers[{key}]")))
    try:
        out = model(xb, store_memories=False, task_hint=task_k,
                    apply_adapter=use_adapter)
    finally:
        for h in handles:
            h.remove()
    return out, captures


@torch.no_grad()
def features_and_logits(model: PLCM, x, y, task_k: int, use_adapter: bool, device,
                        bs: int = 256):
    """Return (DEPLOYED feature, DEPLOYED logits), read off the deployed path.

    AMENDMENT v4 (catch 28 -- path identity). v3 hand-rebuilt the readout from the
    RAW cell state c_t. The deployed classifier does not read c_t: PLCM.forward
    runs LSTM -> memory read -> GGC composition and reads the COMPOSED state's
    h' = o_t*tanh(c_t'). v3 got the form right and the argument wrong, and asserted
    path-identity in a docstring where nothing checked it -- so every number
    downstream of E6b inherited the error.

    This version calls PLCM.forward and takes `readout_feature` and `logits` from
    it. The feature is the deployed one because it IS the deployed one; the claim
    is verified per arm by assert_path_identity(), not stated here.

    `use_adapter=False` now SUPPRESSES the adapter through the deployed path
    (forward(apply_adapter=False)) rather than routing around it -- an instrument
    that bypasses the model to express a suppression reintroduces catch 28 in the
    place least able to afford it.

    v3's other two locks stand and are unchanged: the refit differs from the
    deployed head in FITTING PROCEDURE only, and the deployed head for a task-head
    arm is that arm's own head (here: whatever forward actually routes to).
    """
    F, L = [], []
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].to(device)
        out, _ = _run_deployed(model, xb, task_k, use_adapter)
        F.append(out["readout_feature"].cpu())
        L.append(out["logits"].cpu())
    return torch.cat(F), torch.cat(L)


def assert_path_identity(model: PLCM, x, task_k: int, use_adapter: bool, device,
                         label: str = "", n: int = 64, verbose: bool = True) -> dict:
    """PROVE the instrument reads the deployed path. Runs at the top of every
    consuming script, permanently -- not once (catch 28).

    Two distinct properties, checked separately because they can fail separately:

      P-A  path identity. The tensor features_and_logits() reports is bitwise the
           tensor every readout module consumed in that same call. Verified from
           forward hooks. NON-TAUTOLOGICAL: hooking a different module, or
           reporting a reconstruction, makes this unequal -- v3's o_t*tanh(c_t)
           fails it, which is how catch 28 would have been caught at birth.

      P-B  single-head readout. deployed logits == head_k(feature). True only when
           no routing blend sits between the head and the output. Where P-B fails,
           the deployed prediction is NOT any one head's argmax, so a refit-vs-
           deployed gap spans a routing difference on top of the fitting
           difference it is meant to measure.

    P-A failing is a bug in this instrument -> raises.
    P-B failing is a fact about the arm -> reported, and the caller stops per the
    E11 contract ("if any arm fails it: stop and report; do not special-case").
    """
    xb = x[:n].to(device)
    out, captures = _run_deployed(model, xb, task_k, use_adapter)
    feat = out["readout_feature"]

    # ---- P-A: every readout consumed exactly the reported feature -------------
    assert captures, f"{label}: no readout module was called -- nothing was hooked"
    for name, rin, _ in captures:
        assert rin.shape == feat.shape, (
            f"{label}: {name} consumed shape {tuple(rin.shape)}, reported "
            f"feature is {tuple(feat.shape)}")
        assert torch.equal(rin, feat), (
            f"P-A FAILED for {label}: {name} consumed a tensor that is not the "
            f"reported feature (max |diff| = {(rin - feat).abs().max():.3e})")

    # ---- P-B: does one head reproduce the deployed logits? -------------------
    key = str(task_k)
    head = (model.task_classifiers[key]
            if getattr(model, "use_task_heads", False) and key in model.task_classifiers
            else model.classifier)
    head_logits = head(feat)
    d_logit = float((head_logits - out["logits"]).abs().max())
    d_pred = float((head_logits.argmax(1) != out["logits"].argmax(1)).float().mean())
    p_b = d_logit < 1e-5

    # v3's feature, recomputed ONLY to size the defect this pass corrects.
    # LSTM-family only: the block below calls model.lstm expecting (out,(h,c)).
    # Non-recurrent backbones have their own gate (scripts/e12_p3.assert_p3);
    # say so rather than dying on a tuple unpack three frames deep.
    assert getattr(model, "backbone", "lstm") in ("lstm", "mlp"), (
        f"assert_path_identity is the LSTM-family gate; backbone="
        f"{getattr(model, 'backbone', '?')} must use scripts.e12_p3.assert_p3")
    lstm_out, (_, c_n) = model.lstm(
        model._apply_adapter(xb, key) if (use_adapter and key in model.task_adapters) else xb)
    c_t = c_n[-1]
    o_v3 = torch.sigmoid(model.output_gate(torch.cat([lstm_out[:, -1, :], c_t], -1)))
    d_v3 = float((o_v3 * torch.tanh(c_t) - feat).abs().max())

    if verbose:
        print(f"  {label:<28} P-A PASS (n={n}, {len(captures)} readout call(s))   "
              f"P-B {'PASS' if p_b else 'FAIL'}  max|Δlogit|={d_logit:.3e}  "
              f"pred-disagree={d_pred*100:.1f}%   v3-feature drift={d_v3:.3e}")
    return {"label": label, "p_a": True, "p_b": p_b, "d_logit": d_logit,
            "d_pred": d_pred, "d_v3_feature": d_v3,
            "readout_calls": [c[0] for c in captures]}


def refit_probe(ftr_tr, y_tr, ftr_te, y_te, n_classes: int, device=None,
                info: dict = None) -> float:
    """Multinomial logistic regression, L-BFGS at the library's DEFAULT tolerance.

    WORDING CORRECTED 2026-09-21 (E25; this docstring previously said "fit to
    OPTIMALITY"). It is not. sklearn's default `tol=1e-4` terminates 5-20x
    earlier than a tight fit and leaves the objective 0.5-1.9 units above its
    optimum. Measured floor against `tol=1e-10` on the same features, all six
    arms (`scripts/e25_probe_floor.py`, `runs/e25/probe_floor/`; the table in
    docs/appendix.tex app:probe is the same numbers and the two must agree):
    max |delta F_enc| is 0.002-0.003 on the 256-d MNIST arms, 0.0116 on HAR
    (test sets of 344-409 windows, so one window is already 0.24-0.29pp),
    0.0080 on 768-d ViT and 0.0140 on 2048-d ResNet. The induced error in F_enc
    is UNSIGNED -- across 143 cells it inflates in 57, deflates in 59 and leaves
    27 unchanged -- because both the era and the final refit stop short and
    which loses more decides the sign per cell.

    NOT CHANGED, deliberately: every refit number in the paper comes from this
    function, and tightening the tolerance would invalidate all of them. The
    ruling is to measure the gap and state it.

    `probe_n_iter` CANNOT WITNESS THIS. It is compared against `max_iter`, which
    detects exhausting the iteration budget and is blind to termination on
    tolerance -- 18 iterations never trips a cap of 5000. A second witness
    (objective at termination against a tight re-fit) is what would.

    Below is the v3 amendment that fixed a worse problem, retained:

    AMENDMENT v3 (solver lock). v2 used Adam at lr 1e-3 for 5 epochs -- ~80 steps
    -- which underfit by ~50pp and DIVERGED on raw c_t (accuracy fell from 0.2370
    to 0.1840 as epochs rose 5 -> 100). R measured the optimizer, not the model.

    A convex solver run to convergence removes lr, epochs, batch size and seed from
    the protocol entirely, so R measures the genuine deployed-vs-optimal gap, which
    is the quantity H-C3' was written to bound. Features are standardized on TRAIN
    statistics only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte = ftr_tr.numpy(), ftr_te.numpy()
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=LBFGS_MAX_ITER, solver="lbfgs", C=1.0)
    clf.fit(sc.transform(Xtr), y_tr.numpy())
    n_iter = int(getattr(clf, "n_iter_", [0])[0])
    if n_iter >= LBFGS_MAX_ITER:
        print(f"    WARNING: lbfgs hit max_iter={LBFGS_MAX_ITER} (not converged)")
    # THE WITNESS GOES IN THE ARTIFACT. The printed warning is a side effect on
    # stdout, and the orchestrator discards stderr on success -- so for 26 cells
    # "convergence" was evidenced only by retained log files. If the caller
    # passes `info`, the iteration count is written there and lands in the row
    # dict that gets serialized; the trajectory then travels with the number it
    # certifies rather than with the log that may not survive.
    acc_te = float((clf.predict(sc.transform(Xte)) == y_te.numpy()).mean())
    if info is not None:
        info["probe_n_iter"] = n_iter
        # E20: the train/test gap is the overfit witness the MLP probe is
        # compared against. Additive; the row key it feeds (`probe_gap`) is new
        # beside the existing ones and does not alter any existing key.
        info["probe_gap"] = float((clf.predict(sc.transform(Xtr)) == y_tr.numpy()).mean()) - acc_te
    return acc_te


MLP_HIDDEN = 512          # E20 contract sec 2
MLP_MAX_ITER = 2000
MLP_ALPHA = 1e-4          # sklearn default, STATED; swept only if the overfit flag fires
MLP_PROBE_SEEDS = (0, 1, 2)


def refit_probe_mlp(ftr_tr, y_tr, ftr_te, y_te, n_classes: int, device=None,
                    info: dict = None, seeds=MLP_PROBE_SEEDS, alpha: float = MLP_ALPHA) -> float:
    """E20: a two-layer MLP probe, same standardization and split as refit_probe.

    Everything knob-shaped is pinned or floored: lbfgs full-batch (no lr, batch,
    epochs), alpha stated, and the one thing left -- initialization -- is run at
    `seeds` and REPORTED as a spread. The returned number is the mean over
    seeds; `info` carries the per-seed accuracies, iteration counts, losses and
    the train/test gap (the overfit witness the contract's third branch reads).
    Unconverged fits (n_iter_ >= max_iter) are flagged, never silently used.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte = ftr_tr.numpy(), ftr_te.numpy()
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    accs, gaps, iters, losses, conv = [], [], [], [], []
    for sd in seeds:
        clf = MLPClassifier(hidden_layer_sizes=(MLP_HIDDEN,), activation="relu",
                            solver="lbfgs", alpha=alpha, max_iter=MLP_MAX_ITER,
                            random_state=int(sd))
        clf.fit(Ztr, y_tr.numpy())
        n_iter = int(getattr(clf, "n_iter_", 0))
        te = float((clf.predict(Zte) == y_te.numpy()).mean())
        tr = float((clf.predict(Ztr) == y_tr.numpy()).mean())
        accs.append(te); gaps.append(tr - te); iters.append(n_iter)
        losses.append(float(getattr(clf, "loss_", float("nan"))))
        conv.append(n_iter < MLP_MAX_ITER)
        if not conv[-1]:
            print(f"    WARNING: MLP probe seed {sd} hit max_iter={MLP_MAX_ITER} (not converged)")
    if info is not None:
        info["mlp_acc_per_seed"] = accs
        info["mlp_spread"] = float(max(accs) - min(accs))
        info["mlp_gap_per_seed"] = gaps
        info["mlp_gap"] = float(sum(gaps) / len(gaps))
        info["mlp_n_iter"] = iters
        info["mlp_loss"] = losses
        info["mlp_converged"] = all(conv)
        info["mlp_alpha"] = alpha
    return float(sum(accs) / len(accs))


def decompose(dirs, use_adapter: bool, data_for_seed, n_classes: int, device,
              probe: str = "linear", probe_seeds=MLP_PROBE_SEEDS, alpha: float = MLP_ALPHA):
    """Per-seed, per-task decomposition for one arm.

    `data_for_seed(seed)` is a CALLABLE, not a fixed dict. Permuted-MNIST tasks
    are generated from the run's seed, so a checkpoint must be evaluated against
    ITS OWN permutations: seed-1337 weights scored 0.1135 (chance) against
    seed-42 permutations versus 0.9470 against their own. HAR is unaffected --
    its shifts are fixed, not sampled -- and task 0's permutation is None, which
    is why the task-0-only analyses (E4b, E6) never met this.

    `probe` selects the refit recipe: "linear" (the default; every artifact in
    the ledger) or "mlp" (E20). The extraction, the gates and every other field
    are identical between the two, which is what makes the per-cell difference
    a paired measurement. The linear path is byte-for-byte the pre-E20 path.
    """
    assert probe in ("linear", "mlp"), probe
    out = []
    for seed in SEEDS:
        d = dirs[seed] if isinstance(dirs, dict) else dirs.format(s=seed)
        data = data_for_seed(seed)
        m_final = load(d, 4)
        for k in range(4):                       # old tasks only
            xtr, ytr, xte, yte = data[k]
            m_ceil = load(d, k)                  # theta_{k+1}: end of task k

            # The permanent gate (catch 28), on every cell this function reports
            # — both models, since a ceiling and a final checkpoint are different
            # objects and either could be hooked wrong.
            gate_c = assert_path_identity(m_ceil, xte, k, use_adapter, device,
                                          label="ceil", verbose=False)
            gate_f = assert_path_identity(m_final, xte, k, use_adapter, device,
                                          label="final", verbose=False)

            f_c_tr, _ = features_and_logits(m_ceil, xtr, ytr, k, use_adapter, device)
            f_c_te, l_c_te = features_and_logits(m_ceil, xte, yte, k, use_adapter, device)
            f_f_tr, _ = features_and_logits(m_final, xtr, ytr, k, use_adapter, device)
            f_f_te, l_f_te = features_and_logits(m_final, xte, yte, k, use_adapter, device)

            acc_ceiling = float((l_c_te.argmax(1) == yte).float().mean())
            acc_orig = float((l_f_te.argmax(1) == yte).float().mean())
            i_rc, i_r4 = {}, {}
            if probe == "linear":
                acc_rc = refit_probe(f_c_tr, ytr, f_c_te, yte, n_classes, device, info=i_rc)
                acc_r4 = refit_probe(f_f_tr, ytr, f_f_te, yte, n_classes, device, info=i_r4)
            else:
                acc_rc = refit_probe_mlp(f_c_tr, ytr, f_c_te, yte, n_classes, device,
                                         info=i_rc, seeds=probe_seeds, alpha=alpha)
                acc_r4 = refit_probe_mlp(f_f_tr, ytr, f_f_te, yte, n_classes, device,
                                         info=i_r4, seeds=probe_seeds, alpha=alpha)

            out.append({
                "seed": seed, "task": k,
                "acc_ceiling": acc_ceiling, "acc_refit_ceiling": acc_rc,
                "acc_refit_t4": acc_r4, "acc_orig": acc_orig,
                "R": acc_rc - acc_ceiling,
                "F_enc": acc_rc - acc_r4,
                "F_read": acc_r4 - acc_orig,
                "F_total": acc_ceiling - acc_orig,
                "p_b": bool(gate_c["p_b"] and gate_f["p_b"]),
                "d_logit": max(gate_c["d_logit"], gate_f["d_logit"]),
                # one key, sub-keyed per fit, so a single grep audits every path
                "probe_n_iter": {"ceiling": i_rc.get("probe_n_iter"),
                                 "t4": i_r4.get("probe_n_iter")},
                "sklearn_version": _SKLEARN_VERSION,
                # E20 (additive, both probes): the train/test gap of each fit.
                "probe_gap": {"ceiling": i_rc.get("probe_gap", i_rc.get("mlp_gap")),
                              "t4": i_r4.get("probe_gap", i_r4.get("mlp_gap"))},
                # Remaining E20 keys are ABSENT on the linear path; every key
                # that existed before E20 is unchanged in value.
                **({"probe": "mlp", "mlp_ceiling": i_rc, "mlp_t4": i_r4} if probe == "mlp" else {}),
            })
    return out


PROBE_SUBSET_SEED = 20260916   # E18/E20: the seed new analyses pass; recorded in their artifacts


def load_task_data(bench, n_tasks=4, n_train=None, n_test=None, probe_subset_seed=None):
    """Collect (x, y) in ONE pass per loader.

    E20 finding (2026-09-16): the N_TRAIN cap takes the FIRST batches of a
    shuffle=True loader, and nothing here seeded torch's global generator, so
    the probe's training subset depended on process history and library
    version rather than on anything recorded. Measured: reload and deployed
    extraction reproduce to 0.0e+00; refit quantities move by up to 1.6pp per
    cell (0.12pp pooled) between two invocations on the same checkpoints.
    `probe_subset_seed` fixes the draw: when given, the global generator is
    seeded immediately before each train loader is iterated, and the caller
    records the seed beside the numbers. Default None keeps the pre-E20 path
    byte-for-byte, so existing artifacts are not silently re-drawn; the
    ruling on re-running them seeded is the user's (runs/MEMO_e20.md).

    Two separate comprehensions over a shuffle=True train loader draw two
    DIFFERENT permutations, silently misaligning inputs and labels -- the probe
    then trains on randomly relabelled data and collapses to chance. That bug
    produced R ~ -0.63 and was misread as optimizer underfitting; test loaders
    use SequentialSampler, so the deployed accuracies stayed correct and the
    failure looked like an instrument problem rather than a data one.
    """
    # E13: the caps are overridable so a contract can demand the FULL split
    # (E11 measured the 2000-sample cap contributing 0.017 error on MNIST by
    # itself). The single-pass discipline below is unchanged — it is the whole
    # point of this function and the reason new scripts must route through it
    # rather than re-implement extraction (catch 32).
    cap_tr = N_TRAIN if n_train is None else n_train
    cap_te = N_TEST if n_test is None else n_test
    data = {}
    for k in range(n_tasks):
        tr, te = bench.get_task_loaders(k)
        if probe_subset_seed is not None:
            torch.manual_seed(int(probe_subset_seed) + k)   # the draw is now a recorded config
        xs, ys = [], []
        for xb, yb in tr:                      # single pass: pairs stay together
            xs.append(xb); ys.append(yb)
            if sum(t.shape[0] for t in xs) >= cap_tr:
                break
        xtr, ytr = torch.cat(xs)[:cap_tr], torch.cat(ys)[:cap_tr]
        xs, ys = [], []
        for xb, yb in te:
            xs.append(xb); ys.append(yb)
            if sum(t.shape[0] for t in xs) >= cap_te:
                break
        xte, yte = torch.cat(xs)[:cap_te], torch.cat(ys)[:cap_te]
        data[k] = (xtr, ytr, xte, yte)
    return data


def main():
    ap = argparse.ArgumentParser(description="E6b channel decomposition")
    ap.add_argument("--out", default="runs/e6b/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-mnist", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    results = {}
    # HAR shifts are FIXED (not seeded), so one dataset serves every seed.
    hdata = load_task_data(HARShiftBenchmark(num_tasks=5, root=".", batch_size=256))
    for arm, (tmpl, use_ad) in HAR_ARMS.items():
        results[("HAR", arm)] = decompose(tmpl, use_ad, lambda _s: hdata, 6, device)

    if not args.skip_mnist:
        # Permuted-MNIST tasks ARE seeded: each checkpoint gets its own dataset.
        _mcache = {}
        def mnist_data(seed):
            if seed not in _mcache:
                _mcache[seed] = load_task_data(
                    PermutedMNISTBenchmark(num_tasks=5, batch_size=256, seed=seed))
            return _mcache[seed]
        for arm, (dirs, use_ad) in MNIST_ARMS.items():
            results[("MNIST", arm)] = decompose(dirs, use_ad, mnist_data, 10, device)

    m = lambda rows, k: float(np.mean([r[k] for r in rows]))

    # ---- READ ORDER: identity -> H-C3' -> HAR hypotheses -------------------
    print("=" * 88)
    print("1. IDENTITY CHECK   F_enc + F_read - R == F_total   (exact by construction)")
    print("=" * 88)
    worst = 0.0
    for key, rows in results.items():
        e = max(abs(r["F_enc"] + r["F_read"] - r["R"] - r["F_total"]) for r in rows)
        worst = max(worst, e)
        print(f"  {key[0]+'/'+key[1]:<20} max |residual| = {e:.2e}")
    print(f"\n  worst residual {worst:.2e} -> {'OK' if worst < 1e-6 else 'COMPUTATION ERROR'}")

    print("\n" + "=" * 88)
    print("1b. P-B GATE (E11) — is the deployed readout a SINGLE head?")
    print("=" * 88)
    for key, rows in results.items():
        n_bad = sum(not r["p_b"] for r in rows)
        arm = key[1]
        if n_bad:
            print(f"  {key[0]+'/'+arm:<20} P-B FAIL on {n_bad}/{len(rows)} cells "
                  f"(max |dlogit| {max(r['d_logit'] for r in rows):.3f})"
                  f"  -> routed blend — F_read NOT interpretable as reader-walk")
        else:
            print(f"  {key[0]+'/'+arm:<20} P-B pass on all {len(rows)} cells")
    measured_routed = {k[1] for k, rr in results.items() if any(not r["p_b"] for r in rr)}
    if measured_routed != (PB_ROUTED_ARMS & {k[1] for k in results}):
        print(f"  NOTE: measured routed set {sorted(measured_routed)} differs from "
              f"the declared {sorted(PB_ROUTED_ARMS)} — the declaration follows the "
              f"measurement, not the reverse.")

    print("\n" + "=" * 88)
    print("2. H-C3' INSTRUMENT CONTROL   |R| <= 0.05")
    print("=" * 88)
    print("  Reported PER ARM and PER CELL, not only pooled. A pooled mean is a "
          "guard applied\n  after the average, where a failed cell is already "
          "averaged away (catch 26, 3rd instance).\n  The bar is unchanged at "
          f"{R_BAR}; only the granularity is.\n")
    ok = True
    for ds in (["HAR"] if args.skip_mnist else ["HAR", "MNIST"]):
        # Routed arms are excluded: R compares a refit single head against a
        # deployed routed blend there, so it does not measure fitting procedure.
        elig = [(a, rr) for (d, a), rr in results.items()
                if d == ds and a not in measured_routed]
        print(f"  --- {ds} ---")
        for a, rr in elig:
            Ra = float(np.mean([r["R"] for r in rr]))
            worst = max(rr, key=lambda r: abs(r["R"]))
            print(f"    {a:<16} mean R {Ra:+.4f}  worst cell R {worst['R']:+.4f} "
                  f"(seed {worst['seed']} T{worst['task']})  "
                  f"-> {'PASS' if abs(Ra) <= R_BAR else 'FAIL'}"
                  f"{'' if abs(worst['R']) <= R_BAR else '   [worst cell EXCEEDS bar]'}")
        pooled_rows = [r for _, rr in elig for r in rr]
        R = float(np.mean([r["R"] for r in pooled_rows])) if pooled_rows else float("nan")
        n_cell_fail = sum(abs(r["R"]) > R_BAR for r in pooled_rows)
        p = abs(R) <= R_BAR
        ok &= p
        print(f"    {'POOLED (P-B-clean arms)':<16} mean R {R:+.4f} -> "
              f"{'PASS' if p else 'FAIL'};  {n_cell_fail}/{len(pooled_rows)} "
              f"individual cells exceed the bar")
        for a, rr in [(a, rr) for (d, a), rr in results.items()
                      if d == ds and a in measured_routed]:
            print(f"    {a:<16} EXCLUDED — routed blend (P-B fail)")
    if not ok:
        print("\n  H-C3' FAILED -> STOP. No HAR reading is interpreted (prereg 6).")

    print("\n" + "=" * 88)
    print("3. DECOMPOSITION")
    print("=" * 88)
    print(f"  {'dataset/arm':<20}{'F_total':>10}{'F_enc':>10}{'F_read':>10}{'R':>9}"
          f"{'read share':>13}")
    for key, rows in results.items():
        ft, fe, fr, R = (m(rows, "F_total"), m(rows, "F_enc"),
                         m(rows, "F_read"), m(rows, "R"))
        share = fr / (fe + fr) if (fe + fr) else float("nan")
        routed = key[1] in measured_routed
        print(f"  {key[0]+'/'+key[1]:<20}{ft:>10.4f}{fe:>10.4f}{fr:>10.4f}{R:>+9.4f}"
              f"{share*100:>12.1f}%"
              f"{'   <- routed blend — F_read NOT interpretable as reader-walk' if routed else ''}")

    print(f"\n  per-task F_read / F_enc (task 3 = E6's damage-dominant transition)")
    for key, rows in results.items():
        line = "  ".join(
            f"T{k}: {np.mean([r['F_read'] for r in rows if r['task']==k]):+.3f}/"
            f"{np.mean([r['F_enc'] for r in rows if r['task']==k]):+.3f}"
            for k in range(4))
        print(f"  {key[0]+'/'+key[1]:<20}{line}")

    print("\n" + "=" * 88)
    print("4. HYPOTHESES")
    print("=" * 88)
    h1 = h2 = None
    if ("HAR", "OFF") in results:
        h1v = m(results[("HAR", "OFF")], "F_read")
        h1 = h1v >= F_READ_BAR
        print(f"  H-C1  HAR OFF F_read {h1v:.4f} >= {F_READ_BAR}  -> {'PASS' if h1 else 'FAIL'}")
    if ("HAR", "v2-control") in results:
        fr = m(results[("HAR", "v2-control")], "F_read")
        fe = m(results[("HAR", "v2-control")], "F_enc")
        h2 = (fr > fe) and (fr >= F_READ_BAR)
        print(f"  H-C2  v2-control F_read {fr:.4f} > F_enc {fe:.4f} AND >= {F_READ_BAR}"
              f"  -> {'PASS' if h2 else 'FAIL'}")

    if ok and h1 is not None and h2 is not None:
        br = ("(A) three-channel accounting confirmed -> draft E7" if (h1 and h2)
              else "(B) reader real but not dominant -> no single-mechanism E7" if h1
              else "(C) reader-walk is angle without consequence -> encoder-side after all")
    else:
        br = "(D) instrument confounded -> no channel claims"
    print(f"\n  BRANCH: {br}")

    json.dump({f"{d}/{a}": rows for (d, a), rows in results.items()},
              open(os.path.join(args.out, "decomp.json"), "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}decomp.json")


if __name__ == "__main__":
    main()
