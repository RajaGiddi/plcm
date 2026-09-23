# MEMO — E26: Frame Sweep and Drift-Compensation Ports

**Status (2026-09-22, 11:00):** contract `docs/E26_prereg.md` signed v2. L0 audit
**green 21/21**. **DC complete and read** on all seven arms (§2a, §2d): the
paper's claim is not narrowed. **FS read** on A3 (57/57 H-last, §1a) and on the
pretrained arms at ResNet 3/3 + ViT 2/3 seeds (§1b): no deployed-head hypothesis
reaches a majority; the refit reads best in the *pretraining* layout on 83% /
78% of cells. **E25 A complete on four arms with control 4 uniform** (§2e):
11/12 pass; Permuted MNIST fails below chance on 3/3 seeds and is the open
diagnosis.

**All E26 jobs complete (2026-09-22 14:40).** The held-out prediction on ViT
seed 1337 **MISSED** (§1b): 9/18, exactly half, registered at 85%, and the
two-seed pattern weakened from 78% to 69% pooled. The frozen-trunk comparison
(§1c) replicates on both backbones with no seed spread and is the finding §4.3
should rest on. The MNIST below-chance control is diagnosed as far as
measurement takes it (§2f): three explanations eliminated, mechanism open,
direction established as deflation not leakage.

## 0. L0 — checkpoint audit: GREEN, 21/21

`runs/e26/ckpt_audit.json`, criterion LOADS through `PLCM.load_era`, run on the
volume. FS 9/9 CLEAR, DC 21/21 CLEAR. These are the **B6 permuted** pretrained
arms (`ckpt_e23_vit`, `ckpt_e23_rn`, epoch 4) and E23-B's A3 — different
checkpoints from the B1 arms E25 audited. `audit_checkpoints.py` now takes
`--chain {e11,e25,e26}`; the E11 default path is unchanged.

## 1. FS smoke — controls pass, and an early signal recorded before the headline

`runs/e26/fs/{e23_rn,e23_vit}/deployed_seed42.json`, two old tasks each.

| control | ResNet | ViT |
|---|---|---|
| C-ID, $D_k(k)$ vs E23 `acc_orig` | **0.0000** | **0.0000** |
| C-ID, $D_k(T)$ vs E23 `acc_orig_relaid` | **0.0000** | **0.0000** |
| relay exactness, all 20 frames incl. $j=0$ | exact | exact |
| must-fail, unseen frame ≤ seen max | 2/2 | 2/2 |
| P3 positive controls | fired | fired |
| seconds per task | 101–127 | 166–190 |

Timing projects to ~36 min (ResNet) and ~56 min (ViT) per seed for 19 tasks,
inside the contract's 20/60 estimate.

**Early signal, recorded and not read.** On tasks 0 and 1 the deployed head
$h_k$ on $\theta_T$ features reads **at chance in every frame** — $D_k(j)$
spans 0.11–0.22 against a chance of 0.20 for five classes — so the location of
its peak is noise there. The refit $P_k(j)$ is informative everywhere,
0.65–0.78, and roughly flat across frames (ResNet task 0: own/base 0.734, last
0.678; ViT task 0: own 0.714, last 0.726). E23's pooled deployed accuracy was
0.26–0.27, so later tasks sit above chance and the headline's 19 tasks will show
whether the deployed peak is readable anywhere. The contract's H-flat branch
and its prediction 6 ("$P_k(j)$ flat within floor") already cover the case
where only the refit reads. **Note for the row:** at $k=0$ the own frame *is*
the base frame (`perms[0] is None`), so H-own and H-base coincide by
construction on that cell.

## 1a. FS on A3 — read: H-last on 57/57, with a recency ramp

`runs/e26/fs/e23b_t20/deployed_seed{42,1337,2024}.json`, 19 old tasks × 3
seeds. C-ID **exact (0.0e+00)** at both $j=k$ and $j=T$ against E23-B's
decomposition; relay exact; must-fail 19/19 on every seed.

| seed | peak at own | peak at last | peak at base | own ± 2 | flat |
|---|---|---|---|---|---|
| 42 | 0/19 | **19/19** | 0/19 | 2/19 | 0/19 |
| 1337 | 0/19 | **19/19** | 0/19 | 2/19 | 0/19 |
| 2024 | 0/19 | **19/19** | 0/19 | 2/19 | 0/19 |

**Prediction 5 fires (registered 90%): the scratch encoder converged to the
last frame.** The profile is the same on every cell: $D_k(j)$ sits at chance
(0.06–0.24 for ten classes) on frames 0–16 and ramps through the last three —
task 5, seed 42: frames 17/18/19 read 0.22 / 0.44 / **0.89**. The encoder
retains partial handling for the two frames it passed through most recently
and none for any earlier one. **The refit says the same**: $P_k(19)$ reads
0.88–0.89 against 0.62–0.78 on every other sampled frame, so this is not a
head effect — the encoder itself is specialised to frame $T$, and the recency
gradient is a property of the trunk. The "own ± 2" count of 2/19 is tasks 17
and 18, whose own frames lie inside the ramp.

This is E23-B's finding measured across all twenty frames instead of two, and
it is the baseline the pretrained arms are read against: where a scratch trunk
shows a single spike at $T$, the paper's claim is that a pretrained trunk does
not.

## 1b. FS on the pretrained arms — complete, 3/3 seeds each (2026-09-22)

`runs/e26/fs/{e23_rn,e23_vit}/deployed_seed{s}.json`, 19 old tasks × 3 seeds.
C-ID **exact (0.0000)** at both frames on every seed; relay exact; P3 controls
fired; must-fail 57/57 (ResNet) and **56/57** (ViT — the one failure is
itemised below).

**The deployed head: no hypothesis reaches a majority; own is the plurality.**

| arm | cells | peak at own | peak at last | peak at base ($k \ge 1$) | own ± 2 | peak at chance | mean D own / last / base |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 57 | 11 | 3 | 12 / 54 | 18 | 14 | 0.258 / 0.234 / 0.219 |
| ViT-B/16 | 57 | 21 | 6 | 5 / 54 | 23 | 14 | 0.271 / 0.209 / 0.223 |
| A3 scratch | 57 | 0 | **57** | 0 / 54 | 6 | 0 | 0.161 / **0.878** / 0.105 |

The own frame reads highest of the three named frames on both backbones, by
2.4 pp (ResNet) and 6.4 pp (ViT) over the last frame — E23's re-lay cost seen
again — but the peak *location* is noise: on average only 4–6 of 20 frames sit
within resolution of the peak, and a quarter of ResNet's cells peak at chance.
Registered predictions 1–4 and 6 do not fire; 5 fires at 57/57.

**The refit reads best in the pretraining layout, and it is not close.** For
each cell, which of the six sampled frames gives the highest refit:

| arm | cells ($k \ge 1$) | refit best at base | at own | at last | pooled P: own / last / base / other-3 |
|---|---|---|---|---|---|
| ResNet-50 | 54 | **45** (83%) | 3 | 2 | 0.745 / 0.732 / **0.783** / 0.729 |
| ViT-B/16 (3 seeds) | 54 | **37** (69%) | 13 | 2 | 0.769 / 0.750 / **0.781** / 0.742 |
| A3 scratch | 54 | 0 | 0 | **54** | 0.704 / **0.880** / 0.685 / 0.696 |

*(The ViT row was 28/36 = 78% on two seeds; the held-out third seed brought it
to 69%. See the prediction result below.)*

On the pretrained trunks the own frame, the last frame and three random
permuted frames all read the same (0.73–0.76), and the **unpermuted layout the
trunk was pretrained on reads ~4 pp above all of them, on 83% and 78% of
cells** — 83% on ResNet, 69% on ViT with a 50–94% per-seed range. The
seed-level t95 intervals overlap because the pooled level moves seed to seed;
the per-cell count is what carries it, and on ViT that count is not stable. On the scratch trunk the
same measurement says the opposite: last frame on 54/54.

**What this says, and what it changes.** E23's sentence was "the trunk keeps
handling for every frame rather than converging to the last." Measured across
all twenty frames: *no permuted frame is privileged over any other* — the trunk
did not converge to the last frame, and it did not keep per-task handling
either; every permuted frame is equally foreign. Its home is the frame it was
pretrained in. That is a stronger and more specific sentence than the
contract's H-own, and it is the contract's **H-base**, registered at 25% for
the deployed peak (where it does not fire: 12/54, 3/36) and firing
overwhelmingly on the refit, which the contract's prediction 6 expected to be
flat.

**Registered 2026-09-21 22:44, before ViT seed 1337 lands** (the only
pretrained cell set not read): *on ViT seed 1337, the refit's best of the six
sampled frames is the base frame on a majority of the 18 cells with
$k \ge 1$* — **~85%**. A post-hoc observation on 90 cells is a hypothesis; a
held-out seed is the test it needs.

**Result (2026-09-22, re-run 64 min): MISS.** Same seed, same code, nobody had
seen a result. ViT seed 1337 reads **base 9/18, own 7/18, last 1/18** — exactly
half, not a majority. Registered at ~85%.

**The two-seed observation was optimistic, and the held-out seed is why we know.**

| ViT seed | refit best at base ($k \ge 1$) |
|---|---|
| 2024 | 17/18 |
| 42 | 11/18 |
| **1337 (held out)** | **9/18** |
| all three | 37/54 = **69%** |

The pooled figure falls from 28/36 = 78% on the two seeds the observation was
made on to 69% on three, and the per-seed range is 50% to 94%. The finding
survives as a tendency and not as the near-universal pattern §1b's first draft
described. *Nothing enters a results table at n=1* has a sibling here: a
post-hoc pattern read off two seeds is a hypothesis, and this one came back
weaker when tested. The §1b text above is left as written, with this result
beneath it, because the sequence is the evidence.

**One must-fail cell failed**, ViT seed 1337 task 11: the unseen frame read
0.2640 against a seen-frame maximum of 0.2580, an excess of **0.16× that cell's
own resolution**. Recorded as a failure, not waived. Its context: across all
114 pretrained cells the seen-max clears the unseen frame by a median of 1.75
resolutions, but **31 of 114 sit within one resolution of failing**. On arms
whose deployed accuracy is near chance in every frame, "the best seen frame
beats an unseen one" is a comparison between two near-chance numbers, and the
control discriminates weakly there for the same reason §2a's must-fails do. It
is reported with that caveat and the bar is not moved.

## 1c. The frozen-trunk comparison — what §4.3 says (ResNet 3/3 seeds, 2026-09-22)

`runs/e26/fs/frozen_resnet/frozen_seed{s}.json`. The never-fine-tuned ImageNet
trunk on FS's own frames, loaders, probe, class count and draw stream — the only
difference from FS is the encoder. Controls: relay exact, timm parity **0.0**,
and C-ID against `e23_frozen_probe`'s own-frame number **0.0000 on 19/19 cells
per seed**, which is the single point where the two protocols overlap.

| arm | own frame | last frame | base frame | 3 random permuted |
|---|---|---|---|---|
| **Frozen** ResNet-50 | 0.697 ± .045 | 0.681 ± .049 | **0.954 ± .008** | 0.683 ± .028 |
| **Fine-tuned** $\theta_T$ | 0.745 ± .060 | 0.732 ± .058 | 0.783 ± .070 | 0.729 ± .064 |
| $\theta_T$ − frozen | **+0.048** | **+0.051** | **−0.172** | **+0.047** |

**Both things happened, and the loss is 3.6× the gain.** Training taught the
scrambled formats a little: on every permuted frame — the task's own, the last,
and three drawn at random — $\theta_T$ reads **4.7 to 5.1 pp above** the frozen
trunk, and the gain is the same size whichever permuted frame you pick. Training
cost skill on the original layout a lot: at the base frame $\theta_T$ reads
**17.2 pp below** frozen.

**The frozen trunk's profile is the sharp one.** It reads 0.954 at base against
0.68–0.70 on every permuted frame, a 26 pp gap, and base is its best frame on
**54/54 cells** (18/18 per seed). Fine-tuning compressed that 26 pp gap to 3.8 pp
by moving both ends toward each other — mostly by pulling the base end down.

**What §4.3 can now say.** FS alone showed $\theta_T$ reading best at base and
could not say why. With frozen beside it: *the pretrained trunk's preference for
its original layout is inherited, not learned — the frozen trunk prefers base by
26 pp and fine-tuning shrinks that to 4 pp. Twenty tasks of permuted training
buys about 5 pp on every permuted layout, uniformly, and spends 17 pp of the
layout the trunk came with.* The trunk does not specialise to any frame it was
trained on; it is dragged off its original one.

That also re-reads §1b's headline. "The refit reads best in the pretraining
layout" is true of $\theta_T$, but the frozen trunk shows the same preference
four times more strongly, so the right sentence is about what training *failed
to move*, not about a preference training created.

**ViT replicates it, at three seeds (2026-09-22).**

| arm | own | last | base | 3 random permuted |
|---|---|---|---|---|
| **Frozen** ViT-B/16 | 0.720 ± .030 | 0.707 ± .054 | **0.974 ± .001** | 0.704 ± .026 |
| **Fine-tuned** $\theta_T$ | 0.769 ± .048 | 0.750 ± .042 | 0.781 ± .058 | 0.742 ± .030 |
| $\theta_T$ − frozen | **+0.049** | **+0.043** | **−0.193** | **+0.038** |

Controls on all three ViT seeds: relay exact, timm parity 0.0, own-frame C-ID
against `e23_frozen_probe` **0.0000**. The numbers land on top of ResNet's:
gains of 3.8–5.1 pp on every permuted frame, a loss of 17–19 pp at base, and the
frozen trunk picking base on **18/18 cells in every seed of both backbones —
108/108**.

**This is the robust half of E26's FS result.** The frozen comparison replicates
across backbones with no seed spread at all, while the $\theta_T$-prefers-base
count it was built to explain ranges 50% to 94% across ViT seeds (§1b). The
sentence §4.3 should carry is the one supported by 108/108, not the one
supported by 69%.

## 2. DC — validated on a scratch arm before launch

`scripts/e26_dc.py`, local smoke on `e18_pmd_mlp` seed 42 (5 tasks, 4
boundaries). All controls behave:

| control | result |
|---|---|
| C-PLUMB (update arithmetic is the identity under zero drift) | exact |
| oracle identity (two NCM implementations) | exact |
| SDC must-fail (negated drift below naive) | 4/4 |
| LDC must-fail (shuffled pairs below naive) | 4/4 |
| LDC identity, from identity init | 0.00 rel-Fro |
| LDC identity, **from random init** | **1.19 rel-Fro vs tol 1e-4 — FAIL, recorded** |

**The first C-PLUMB draft compared an array to its own copy** — a control that
cannot fail, catch 25 — and was replaced before it ran with a test of the
update arithmetic: SDC with $Z_c = Z_p$ must add exactly zero, and LDC with the
identity projector must return the prototype exactly.

**The LDC identity control fails from random init and passes trivially from
identity init.** At the paper's hyperparameters (Adam $10^{-3}$, 20 epochs,
batch 128 — batch is ours) the optimizer does not reach the identity from a
random start. Whether that limits the *result* was settled by adding the exact
least-squares solution of Eq. 1 as an optimizer control beside the Adam port:

| task | boundaries | A0 | N-naive | N-LDC (Adam) | **N-LDC (LS)** | N-oracle |
|---|---|---|---|---|---|---|
| 0 | 4 | 0.603 | 0.587 | 0.112 | 0.102 | 0.865 |
| 1 | 3 | 0.553 | 0.584 | 0.095 | 0.116 | 0.858 |
| 2 | 2 | 0.876 | 0.916 | 0.100 | 0.102 | 0.941 |
| 3 | 1 | 0.938 | 0.934 | **0.004** | **0.001** | 0.942 |

Least squares and Adam agree on every cell, so the collapse is **the method
under a violated assumption, not the optimizer**. On this arm the encoder is
overwritten to the current frame (E23-B), so drift measured on frame-$t$ data
says nothing about frame-$k$ prototypes, and a linear map fit on frame 4 sends
frame-3 prototypes to the wrong classes consistently — below chance. SDC at
$\sigma=1$ survives on the last two tasks (0.742, 0.914) because a wide kernel
reduces to a global mean shift; at $\sigma \le 0.3$ it collapses. This is the
scratch half of the comparison, where the contract expects failure; the
pretrained arms, where the trunk keeps per-frame handling (E23), are the test.
The port uses identity init, stated, and the identity control is reported both
ways rather than re-barred.

## 2a. Scratch DC — read (`runs/e26/dc/{arm}/seed{s}.json`, 3 seeds each)

Pooled over cells, L2-normalised features, task-IL NCM among the task's own
classes:

| arm | cells | A0 | N-naive | N-SDC$_{0.3}$ | N-SDC$_{1.0}$ | N-LDC | N-LDC (LS) | N-oracle |
|---|---|---|---|---|---|---|---|---|
| HAR, LSTM | 12 | 0.467 | 0.477 | 0.460 | 0.468 | 0.404 | 0.423 | 0.698 |
| Permuted, MLP | 12 | 0.768 | 0.797 | 0.376 | 0.457 | 0.135 | 0.087 | 0.906 |
| Permuted, LSTM | 12 | 0.195 | 0.215 | 0.176 | 0.184 | 0.160 | 0.122 | 0.566 |
| **Rotated, MLP** | 12 | 0.512 | 0.553 | **0.613** | 0.567 | **0.670** | 0.671 | 0.848 |
| A3 (20 tasks) | 57 | 0.161 | 0.170 | 0.142 | 0.136 | 0.121 | 0.110 | 0.512 |

**N-naive tracks A0 within 0.03 on every arm**: NCM with era prototypes is a
faithful stand-in for the deployed head, which is what makes the comparison
apples-to-apples. **N-oracle sits 0.2–0.4 above naive everywhere**: the
prototype ceiling is high, so there is room for a correction to work.

**Both methods collapse on the permutation arms and work on the rotation
arm.** On rotated MNIST, LDC lifts naive 0.553 → **0.670** and SDC$_{0.3}$ to
0.613, recovering 40% of the naive→oracle gap. On the three permutation arms
both fall to chance or below. The split follows the methods' assumption
exactly: drift measured on task-$t$ content transfers to task-$k$ prototypes
when the shift between frames is *smooth* (22.5° between adjacent rotations)
and not when it is a discontinuous permutation. That is the scratch half of the
comparison saying what the contract expected it to say, with one arm showing
the methods are not broken — they work where their premise holds.

**Least squares agrees with Adam on every arm** (0.423/0.404, 0.087/0.135,
0.122/0.160, 0.671/0.670, 0.110/0.121), so none of this is the optimizer. The
LDC identity control from random init reads **0.97–1.36 rel-Fro on every arm
against the registered 1e-4**: recorded FAIL with cause — the bar was priced
for a solver that reaches the optimum, and Adam at the paper's settings does
not from a random start on 256-d — and the LS column is what shows the result
does not depend on it. The bar does not move.

**Must-fails.** LDC shuffled-pairs below naive: 12/12, 12/12, 11/12, 12/12,
47/57. SDC negated-drift below naive: 12/12, 12/12, **7/12**, 10/12, 43/57.
The two weakest SDC counts are on the arms where SDC's own signal is nil
(Permuted LSTM 0.176 vs naive 0.215): when a method moves prototypes in no
useful direction, negating the move has no direction to be wrong in, and the
control degenerates toward a coin flip. A control rules out only the defects
that would break it; on those cells it rules out little, and says so per cell.

## 2b. The first pretrained DC smoke failed in the right place

The P3b gate fired at the first boundary: `no head exists for task 1` on the
era-0 checkpoint. Correctly — heads are created lazily at task start
(`plcm.py set_task`), so $\theta_{t-1}$ has no head for task $t$, and DC's
feature pass under $\theta_{t-1}$ was asking P3b a question with no referent.
The fix is in the gate, not the assert: the $\theta_{t-1}$ pass asserts P3a
only (the feature is the deployed tensor), and P3b runs wherever head $k$
exists in $\theta_t$. The trunk feature is all DC reads on that pass; the
fallback head's logits are never used. Relaunched.

## 2c. Pretrained DC smoke — green on mechanics, and two things recorded before the headline

ResNet seed 42, three old tasks, all 19 boundaries: 915 s, oracle identity and
C-PLUMB exact, both must-fails weak on three near-chance cells (2/3, 1/3) for
the reason §2a names. Timing projects to ~25 min per ResNet seed. **Headline
launched** (8 jobs: both backbones × 3 seeds, plus the unnormalised arm on 42).

**(i) The NCM ceiling on these features is far below the refit.** N-oracle,
prototypes recomputed from task-$k$ data under $\theta_T$ — the best any
prototype correction can do — reads 0.38–0.47 on the three early tasks, where
E23's refit on the same features reads ~0.75. Nearest-class-mean is a much
weaker reader than a linear probe here, so the contract's "N-LDC within 5pp of
$A_\infty$" (registered 20%) is structurally out of reach on this arm: the
ceiling for the whole NCM family sits ~30pp under the refit. That is exactly
what N-oracle is in the design to show, and it is recorded before the headline
so it is not discovered as a surprise.

**(ii) Least squares and Adam DIVERGE on the pretrained arm, where they agreed
on every scratch arm.** LS reads **0.200 exactly** on all three cells — five
balanced classes of 100, so exactly one class predicted for everything: the
composition of 17–19 least-squares projectors (2048×2048, each fit on 2,500
samples) collapses the prototypes. Adam from identity init, 400 steps, stays
near the identity and degrades gracefully (0.17–0.25, spread up to ±0.11 over
projector seeds). So on 2048-d features **the Adam port is not at the optimum
of Eq. 1, and that is load-bearing**: its implicit regularisation — identity
init plus early stopping — is what keeps the composed projector from
collapsing. On 256-d scratch features the two agreed because 4,000 samples fit
a 256×256 map well; on 2048-d with 2,500 samples the exact solution
over-fits per boundary and the errors compound. The row reports both columns,
and the reading of N-LDC on the pretrained arms carries this qualifier: the
number is the paper's recipe, not the objective's optimum.

Per-boundary mean drift norm on unit-norm features is 0.9–1.25 on ResNet
(features rotate ~60° per boundary on average), against 1.3 on the scratch
MLP.

## 2d. Pretrained DC — read: the paper's claim is not narrowed

`runs/e26/dc/{e23_rn,e23_vit}/seed{s}.json`, 3 seeds × 19 old tasks, plus
the unnormalised sensitivity arm on seed 42. `runs/e26_dc_row.json`.

| arm | A0 | N-naive | N-SDC$_{0.3}$ | N-LDC | N-LDC (LS) | N-oracle | $A_\infty$ |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 0.258 ± .029 | 0.264 ± .033 | 0.269 | **0.313 ± .015** | 0.216 | 0.502 ± .124 | 0.745 |
| ViT-B/16 | 0.271 ± .017 | 0.282 ± .029 | 0.243 | **0.245 ± .075** | 0.232 | 0.548 ± .109 | 0.769 |

**Mixed across backbones, stated per backbone.** On ResNet, LDC beats naive
and A0 by 5 pp beyond both intervals, and recovers **14.5%** of the naive →
oracle gap. On ViT it reads 3.7 pp *below* naive, inside its own wide
projector-seed interval. Neither is anywhere near the refit: the best
prototype method sits **43 pp** under $A_\infty$ on ResNet and 52 pp on ViT.
The readings table's row: *drift compensation helps on one trunk, and the gap
to the refit is the price of not having labels at repair time.* **The abstract's
last sentence stands.** Class means stored at task end are not enough.

**Half of that shortfall is the reader, not the compensation** (§2c(i)). The
NCM oracle — perfect prototypes — reaches only 0.50 / 0.55, so the whole
prototype family is capped ~25 pp under the linear refit on these features
before any drift compensation is applied. LDC then recovers a seventh of what
remains. Both facts go in the reading; the second is the one a prototype
method's proponent will want to see.

**LDC degrades with boundaries crossed on every 20-task arm** ($r$ = −0.78
ResNet, −0.56 ViT, −0.43 A3; prediction 7 fires at its revised 75%). On
ResNet the whole gain is in the first seven boundaries (b = 4: LDC 0.41 vs
naive 0.22); by b ≥ 13 LDC is at or below naive. On ViT LDC is below naive from
b = 3 onward. This is E25 B's composed-repair finding in a published method's
clothes: twenty composed linear maps fit on current data do not hold.

**The Adam / least-squares divergence is confirmed at three seeds** (§2c(ii)):
LS reads 0.216 and 0.232, below naive on both, where the identity-initialised
Adam port reads 0.313 and 0.245. The paper's recipe works on ResNet *because*
it under-fits its objective. Unnormalised (seed 42): ResNet LDC 0.312, ViT
0.207 — normalisation is immaterial on ResNet and worth 4 pp on ViT.

**Controls.** Oracle identity and C-PLUMB exact on every seed. Must-fails on 19
cells per seed: SDC 11–16, LDC 10–16 — majorities, weaker than on the scratch
arms for the reason §2a gives (where a method's own effect is small, negating
it has little to be wrong about). LDC identity from random init 1.35–1.45
rel-Fro against 1e-4: FAIL with cause, as on every arm.

**One join refused.** The contract's $A_{10}$ is "E25 A, where read". E25 A's
pretrained arms record benchmark `cifar100` (B1, unpermuted); DC's record
`cifar100_permuted` (B6). Different checkpoints, different benchmark. The row
now refuses the join and prints why; the prediction reads "not comparable"
until an $A_{10}$ exists on B6. Caught by reading the `arm` field, not the
contract — catch 30's rule.

**Predictions (7):** 2 fired (SDC on rotated MNIST; degradation with
boundaries), 2 missed (LDC beats naive / A0 on *both* pretrained arms — true on
ResNet, false on ViT), 2 did not fire (within 5 pp of refit; ≥ 50% gap
recovery), 1 not comparable.

## 2e. E25 A, pretrained — read, with one control held

`runs/e25/a/{e14_base,e12_base}/curve_seed{s}.json`, B1 arms, 57 cells each.
Recovery fraction, mean of ratios over cells clearing the 0.05 guard:

| arm | A0 | $A_\infty$ | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|---|
| ResNet-50 (B1) | 0.293 | 0.936 | +0.327 | +0.656 | +0.765 | +0.848 | +0.920 |
| ViT-B/16 (B1) | 0.222 | 0.895 | +0.257 | +0.504 | +0.650 | +0.774 | +0.879 |
| HAR, LSTM | 0.467 | 0.773 | +0.330 | +0.698 | +0.738 | +0.793 | +0.937 |
| Permuted MNIST, MLP | 0.768 | 0.943 | −0.006 | +0.646 | +0.743 | +0.823 | +0.898 |

**No arm reaches 0.8 at ten labels per class** (E25's three headline
predictions at 55/45/40% do not fire); ResNet and MNIST clear it at twenty.
Ten labels per class recover 0.65–0.77 of the refit gap everywhere, which is
the same band R1 measured for C2 with *every* label under an orthogonal
constraint (0.61–0.81): **ten unconstrained labels ≈ all labels constrained.**

**Control 4 is now uniform across all four arms** (the pretrained relaunch
completed 2026-09-22 before it was cancelled; every artifact carries the
20-draw estimator). Eleven of twelve cell-sets pass, and the failure isolates
to one arm:

| arm | chance | shuffled mean per seed | bar | verdict |
|---|---|---|---|---|
| ResNet-50 (B1) | 0.200 | 0.2048, 0.1945, 0.1981 | 0.0351 | 3/3 PASS |
| ViT-B/16 (B1) | 0.200 | 0.2006, 0.1863, 0.2039 | 0.0351 | 3/3 PASS |
| HAR, LSTM | 0.1667 | 0.2001, 0.1669, 0.1609 | 0.0361 | 3/3 PASS |
| **Permuted MNIST, MLP** | 0.100 | **0.0908, 0.0843, 0.0833** | 0.0131 | **1/3 PASS** |

**The estimator is vindicated and the MNIST failure is real.** The 20-draw
correction was adopted because a single draw could not discriminate; under it,
three of four arms sit within 0.6 pp of chance and pass comfortably — including
HAR, whose single-draw 0.286 started the diagnosis. MNIST reads **below chance
on 3/3 seeds**, pooling to 0.0861 against 0.100, a consistent −1.4 pp that two
seeds clear the bar on individually.

Below chance is anti-correlation, not leakage: **a label leak inflates, so the
curve's recovery numbers are not inflated by whatever this is.** That is a
statement about direction, not an explanation. The bar does not move and the
failure stands recorded. §2f is the diagnosis.

## 2f. The MNIST below-chance control — three explanations eliminated, mechanism open

Analysis only (2026-09-22). Each candidate was stated, then measured.

**(i) Not the procedure.** The same balanced draw, balanced label shuffle and
solver on **pure-noise features** of each arm's shape reads chance exactly:

| shape | chance | noise features, balanced-permutation labels | iid random labels |
|---|---|---|---|
| C=10, d=256 (MNIST-MLP) | 0.1000 | **0.1001** | 0.0979 |
| C=6, d=256 (HAR) | 0.1667 | **0.1643** | 0.1705 |
| C=5, d=2048 (ResNet) | 0.2000 | **0.1996** | 0.2001 |

**(ii) Not clustering, and this killed my own proposed mechanism.** The
hypothesis was that a *balanced* shuffle is not independent of the truth when
features cluster by class: label $c$ is spread over all $C$ clusters, and the
balanced constraint means over-representing it in one cluster under-represents
it in the rest, so a test point should be slightly *less* than $1/C$ likely to
draw its own label. It predicts a deficit that grows with separability. Tested
on synthetic clusters at six separabilities:

| separability | true-label refit | shuffled | deficit |
|---|---|---|---|
| 0.0 | 0.100 | 0.0984 | −0.0016 |
| 0.25 | 0.865 | 0.0963 | −0.0037 |
| 0.5 | **1.000** | 0.0989 | −0.0011 |
| 1.0 | 1.000 | 0.1008 | +0.0008 |
| 2.5 | 1.000 | 0.1000 | +0.0000 |

Perfectly separable clusters read **exactly chance**. The deficit does not
scale with separability and does not appear at all. **The mechanism is wrong
and is recorded as wrong**; an earlier 20-draw synthetic that read 0.0977 had
sd 0.035 and was over-read as support.

**(iii) Not the marginals.** On the real features the predicted class frequency
is uncorrelated with the test frequency ($r$ = −0.02, −0.03, +0.04, −0.07 over
the four tasks), and the accuracy that those marginals *imply* under
independence is **0.0999–0.1001** — exactly chance. The observed 0.0833–0.0867
is therefore a genuine per-example anti-correlation, not a class-frequency
artifact. Prediction entropy is 2.287 against a maximum of 2.303 and all ten
classes are predicted on every draw, so it is not collapse either.

**Where that leaves it.** The effect is specific to the real MNIST-MLP features
at $\theta_T$, it is a per-example anti-correlation of about 1.6 pp, and no
mechanism I proposed survives measurement. What the diagnosis does establish is
the direction: **a label leak inflates a shuffled probe, and this deflates one**,
so the E25 A MNIST curve is not inflated by it. The control's failure stands,
the bar stays where it was priced, and the mechanism is open.

## 3. Launch record

| step | job set | jobs | state |
|---|---|---|---|
| L0 | `e26audit` | 1 | **green 21/21** |
| FS smoke | `e26fssmoke` | 2 | green |
| FS headline, pretrained | `e26fs` + `e26fsvit1337` | 6+1 | **complete**, read (§1b); held-out prediction MISSED |
| FS on A3 | `e26fsa3` | 3 | **landed, read (§1a)** |
| DC scratch | `e26dcscratch` | 15 | **landed, read (§2a)** |
| DC smoke | `e26dcsmoke` | 1 | green (§2c) |
| DC headline, pretrained | `e26dc` | 8 | **landed, read (§2d)** |
| E25 A pretrained, control-4 uniformity | `e25apre` | 6 | **completed**, read (§2e) |
| FS frozen-trunk arm (decides §4.3) | `e26fsfrozen` | 6 | **complete**, read (§1c) |
| MNIST below-chance diagnosis | analysis only | — | **done** (§2f) |

Rows: `scripts/e26_fs_row.py` (three arms, H-base counted on $k \ge 1$ where
own ≠ base) and `scripts/e26_dc_row.py` (joins $A_\infty$ from E23 and $A_{10}$
from E25 A; the boundaries prediction is not scored until A3 *and* both
pretrained arms are in, since scoring it on whichever subset landed is the n=1
habit in a new coat). Both run on the artifacts that exist and score what they
can; nothing pending is scored.

## 4. Artifacts

`runs/e26/ckpt_audit.json`; `runs/e26/fs/{e23_rn,e23_vit,e23b_t20}/{deployed,refit}_seed{s}.json`;
`runs/e26/dc/{arm}/seed{s}.json` (+ `_unnorm` on seed 42 pretrained);
`runs/e26_fs_row.json`, `runs/e26_dc_row.json`. Scripts: `e26_fs.py`,
`e26_fs_a3.py`, `e26_dc.py`, `e26_fs_row.py`, `e26_dc_row.py`;
`audit_checkpoints.py --chain e26`.
