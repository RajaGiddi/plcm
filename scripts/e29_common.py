"""E29 shared implementation — ONE audited path for every consumer.

Catch 32's rule, applied at the start rather than after the fact: every E29
script imports its statistics, its 9! search and its SO(3) fit from here. A
parallel implementation in any consumer is the defect that inverted E12's
conclusion, and the cheapest time to prevent it is before the second script
exists.

WHAT THE SCORE IS, and what §0 proved about it
----------------------------------------------
  S(P) = ||P S P^T - S'||_F^2 / ||S||_F^2  +  ||P C1 P^T - C1'||_F^2 / ||C1||_F^2

THE MEAN TERM IS NOT INCLUDED, and its absence is a finding, not an omission.
`har_subject._apply` (:126-129) standardizes with task-0 train statistics
BEFORE applying the channel map, so the reference mean is zero by construction
(||mu_ref|| = 1.14e-4, measured). Every candidate P then gives P.mu ~ 0, so the
mean term is the SAME for every candidate: it cannot discriminate, and divided
by a near-zero ||mu||^2 + eps it would swamp the two terms that can. A
denominator guard on a quantity that is zero by construction is catch 26's
shape; here the fix is to drop the term, not to guard it.

RANKING, NEVER AN ABSOLUTE BAR. Cross-subject drift raises the TRUE
permutation's score to 0.14-0.67 against a reference self-symmetry margin of
0.0297 -- from which the drafting of this experiment concluded, wrongly, that
recovery would fail. It does not: drift perturbs every candidate alike and
largely cancels in the ranking (100% exact at n >= 64, measured). Absolute
score was a cheap signal standing in for the expensive one. The ambiguity set
is therefore defined by RANK and nothing here may threshold a raw score.

THE SO(3) FIT'S RESTART COUNT IS LOAD-BEARING AND HAS NO DEFAULT.
`fit_rotation` requires `n_start` and refuses fewer than MIN_RESTARTS. At 1-2
restarts the optimizer settles in a ~172 degree local minimum and the
equivariance check fails with a 163 degree spread; at 8 it passes at 0.0000.
A hyperparameter with a default in the caller is a hyperparameter selected
outside the measurement.
"""

from __future__ import annotations

import hashlib
from itertools import permutations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

N_CHANNELS, N_STEPS, N_CLASSES, NUM_TASKS = 9, 128, 6, 5
HAR_PARTITION_X86 = "1104af185c87"          # Modal; the laptop builds 5d047e4213d1
DYNAMIC, STATIC = (0, 1, 2), (3, 4, 5)      # walking* vs sitting/standing/laying
MIN_RESTARTS = 8

# Part A aggregation, fixed here so no consumer can restate it (E28's lesson).
DRAWS_PER_TARGET = 10
TARGET_RECOVERS_AT = 9                      # exact in >= 9 of 10 draws
POOL_SUCCEEDS_AT = 20                       # >= 20 of 22 targets recover


# ---------------------------------------------------------------- statistics
def window_stats(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Covariance and lag-1 cross-covariance of a window stack [n, T, 9].

    C1 is computed WITHIN each window across its timesteps, so the 50% overlap
    between consecutive source windows does not corrupt it; overlap makes the
    DRAWS dependent, which is why the contiguous arms are split by stride.
    """
    Z = A.reshape(-1, N_CHANNELS)
    m = Z.mean(0)
    S = np.cov(Z.T)
    C = np.einsum('ntc,ntd->cd', A[:, :-1, :] - m, A[:, 1:, :] - m) / (A.shape[0] * (A.shape[1] - 1))
    return S, C


# ------------------------------------------------------- the exhaustive search
class PermSearch:
    """All 9! = 362,880 candidates, conjugated against a reference ONCE.

    The precompute is 0.4 s and 224 MB x2; each subsequent scored draw is
    0.11 s, so a consumer builds this once per reference and reuses it.
    """

    def __init__(self, S_ref: np.ndarray, C_ref: np.ndarray):
        self.P = np.array(list(permutations(range(N_CHANNELS))), dtype=np.int16)
        self.S_ref, self.C_ref = S_ref, C_ref
        self.nS, self.nC = (S_ref ** 2).sum(), (C_ref ** 2).sum()
        self.SP = S_ref[self.P[:, :, None], self.P[:, None, :]]
        self.CP = C_ref[self.P[:, :, None], self.P[:, None, :]]

    def scores(self, S_q: np.ndarray, C_q: np.ndarray) -> np.ndarray:
        return (((self.SP - S_q) ** 2).sum((1, 2)) / self.nS
                + ((self.CP - C_q) ** 2).sum((1, 2)) / self.nC)

    def search(self, A: np.ndarray, truth) -> dict:
        """Score one query pool and report against the true permutation BY RANK."""
        s = self.scores(*window_stats(A))
        order = np.argsort(s)
        best = self.P[order[0]]
        t = np.asarray(truth, dtype=np.int16)
        rank = int(np.where((self.P[order] == t).all(1))[0][0])
        return {"exact": bool(np.array_equal(best, t)),
                "channels_correct": int((best == t).sum()),
                "true_rank": rank,                      # 0 = the search found it
                "best": [int(v) for v in best],
                "score_best": float(s[order[0]]),
                "score_truth": float(s[order[rank]]),
                "score_runner_up": float(s[order[1]])}

    def ambiguity_set(self, A: np.ndarray, spread: float) -> int:
        """How many candidates sit within `spread` of the best, BY SCORE GAP.

        Reported for interpretation only. `spread` must come from observed
        draw-to-draw variation of the SAME pool, never from a fixed constant:
        a fixed bar is what §0's wrong turn thresholded.
        """
        s = self.scores(*window_stats(A))
        return int((s <= s.min() + spread).sum())


def apply_perm(x: np.ndarray, p) -> np.ndarray:
    """Output channel i holds input channel p[i] — `channel_affine`'s P
    convention and `e28_heldout.apply_perm`'s, verified by the search
    recovering PERM exactly."""
    return x[..., list(p)]


# --------------------------------------------------------------- SO(3) fitting
def blockdiag(R: np.ndarray) -> np.ndarray:
    """One 3x3 rotation on each of the three sensor triples — `channel_affine`
    applies the SAME `r` to all three (har_shift.py), so one R is estimated
    jointly from all of them."""
    B = np.zeros((N_CHANNELS, N_CHANNELS))
    for i in range(3):
        B[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    return B


def geodesic_deg(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(A.T @ B).as_rotvec())))


def fit_rotation(S_ref, C_ref, S_q, C_q, n_start: int, seed: int = 0):
    """Estimate the blockwise rotation carrying the reference to the query.

    `n_start` is REQUIRED and must be >= MIN_RESTARTS. See the module docstring:
    the objective has a strong spurious minimum near 180 degrees and a starved
    optimizer lands in it, which breaks the equivariance check rather than the
    estimate alone.
    """
    if n_start is None:
        raise ValueError("n_start is required; a default would select it outside the measurement")
    if n_start < MIN_RESTARTS:
        raise ValueError(f"n_start={n_start} < {MIN_RESTARTS}: the equivariance check fails "
                         f"at 1-2 restarts (163 deg spread) and passes at 8 (0.0000)")
    nS, nC = (S_ref ** 2).sum(), (C_ref ** 2).sum()

    def obj(v):
        B = blockdiag(Rot.from_rotvec(v).as_matrix())
        return ((B @ S_ref @ B.T - S_q) ** 2).sum() / nS + ((B @ C_ref @ B.T - C_q) ** 2).sum() / nC

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_start):
        r = minimize(obj, Rot.random(random_state=int(rng.integers(1e6))).as_rotvec(),
                     method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-12})
        if best is None or r.fun < best.fun:
            best = r
    return Rot.from_rotvec(best.x).as_matrix(), float(best.fun)


# --------------------------------------------------------------- pool builders
def pool_indices(y: np.ndarray, spec: str, n: int, rng) -> np.ndarray | None:
    """Indices for one composition spec. Returns None when the pool cannot be
    built at this n (reported as unbuildable, never silently shrunk)."""
    if spec == "all6":
        groups, w = list(range(6)), None
    elif spec == "two_dynamic":
        groups, w = [0, 1], None
    elif spec == "static_plus_dynamic":
        groups, w = [3, 0], None
    elif spec == "three_static":
        groups, w = [3, 4, 5], None
    elif spec == "all6_skew80":
        groups, w = list(range(6)), [0.8] + [0.04] * 5
    else:
        raise ValueError(spec)
    want = ([int(round(n * f)) for f in w] if w else
            [n // len(groups) + (1 if i < n % len(groups) else 0) for i in range(len(groups))])
    out = []
    for c, k in zip(groups, want):
        pool = np.where(y == c)[0]
        if len(pool) < k:
            return None
        out.append(rng.choice(pool, k, replace=False))
    return np.concatenate(out)


def contiguous_indices(subject_rows: np.ndarray, n: int, stride: int, start: int = 0):
    """`n` windows in recording order from one subject. stride 1 keeps the 50%
    overlap; stride 2 takes every second window so no two share samples."""
    idx = subject_rows[start::stride][:n]
    return idx if len(idx) == n else None


def assert_contiguous(idx: np.ndarray, stride: int) -> bool:
    d = np.diff(np.asarray(idx))
    return bool(len(d) and np.all(d == stride))


# -------------------------------------------------------------------- controls
def synthetic_pool(n: int, rng, anisotropic: bool = True, zero_mean: bool = True,
                   cov_seed: int = 0):
    """Gaussian windows with a non-degenerate covariance AND lag structure.

    THE POPULATION COVARIANCE IS FIXED BY `cov_seed`; THE SAMPLES BY `rng`.
    Separating them is load-bearing and was added after the isotropic must-fail
    read 0.0 degrees six times out of six. That control had drawn its reference
    and its query from the SAME array, so the query's covariance was the
    reference's conjugated exactly -- and since no finite sample is ever
    exactly isotropic (measured: diagonal 6.98-7.53, off-diagonal up to 0.27 at
    n = 2048), the estimator locked onto that realization's own sampling noise
    and recovered the rotation to 1.9e-07 with residual 8e-20.

    The control was therefore measuring self-consistency, not identifiability,
    and would have passed an estimator that could not generalise at all. A
    control rules out only the defects that would BREAK it: this one had to
    draw reference and query INDEPENDENTLY from one population before it could
    rule out anything.
    """
    if anisotropic:
        A = np.random.default_rng(cov_seed).normal(size=(N_CHANNELS, N_CHANNELS))
        L = np.linalg.cholesky(A @ A.T + N_CHANNELS * np.eye(N_CHANNELS))
    else:
        L = np.eye(N_CHANNELS)
    x = rng.normal(size=(n, N_STEPS, N_CHANNELS))
    x = np.cumsum(x, axis=1) * 0.3 + x            # a lag-1 structure C1 can see
    out = x @ L.T
    return out - out.mean((0, 1)) if zero_mean else out


def fingerprint(obj) -> str:
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
