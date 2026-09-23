"""
E12/P3 — instrument exactness for the ViT path, re-derived for this architecture.

Contract: docs/E12_prereg.md sec 2. Two asserts, because the first does not imply
the second, and E7 is the proof of that: its heads existed and froze exactly as
designed, and no task was ever scored by its own head (alpha_own 0.165, own head
top-weighted in 0/24 cells). A capture-point check would have passed there.

  P3a  capture-point identity — head(captured_feature) == deployed_logits,
       max |delta| = 0. The ViT readout path (CLS, post-norm, pre-head) is a
       FRESH premise on this architecture: verified, never assumed.
  P3b  selection semantics — on task k the readout module used IS head k,
       verified by MODULE IDENTITY, not by comparing outputs (two heads can
       agree numerically on a batch and still be the wrong module).

Each ships with a positive control demonstrated BEFORE first live use, because a
gate that has only ever passed is indistinguishable from one that cannot fail:

  P3a control  capture the PRE-norm CLS instead -> the assert must FAIL
  P3b control  route to the LATEST head instead -> the assert must FAIL

Import `assert_p3` into any consuming script and call it at the top, permanently.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _capture(model, x, task_hint, apply_adapter=True, pre_norm=False):
    """One deployed forward with every readout module hooked.

    Returns (out, feature, used_modules). `pre_norm=True` deliberately captures
    the wrong tensor — that is the P3a positive control, not an option.
    """
    used = []
    handles = []

    def hook(mod, inp, outp):
        used.append((mod, inp[0].detach()))

    handles.append(model.classifier.register_forward_hook(hook))
    for head in model.task_classifiers.values():
        handles.append(head.register_forward_hook(hook))
    try:
        out = model(x, store_memories=False, task_hint=task_hint,
                    apply_adapter=apply_adapter)
    finally:
        for h in handles:
            h.remove()

    if pre_norm:
        # The WRONG capture point is architecture-specific, so the ENCODER owns
        # it: a ViT's is the pre-norm CLS, a ResNet's is a different reduction of
        # the pre-pool map. A control that hardcodes one architecture's mistake
        # cannot guard another's — which is the E12 preamble in the gate itself.
        enc = model.lstm
        feature = enc.wrong_capture_point(x)
    else:
        feature = out["readout_feature"]
    return out, feature, used


def assert_p3a(model, x, task_hint, label="", pre_norm=False) -> float:
    """head(captured_feature) == deployed_logits, exactly. Returns max |delta|."""
    out, feature, used = _capture(model, x, task_hint, pre_norm=pre_norm)
    assert used, f"{label}: no readout module was called — nothing was hooked"
    mod = used[-1][0]
    d = float((mod(feature) - out["logits"]).abs().max())
    assert d == 0.0, (
        f"P3a FAILED {label}: head(captured_feature) != deployed_logits "
        f"(max |delta| = {d:.3e})")
    return d


def assert_p3b(model, x, task_k, label="") -> str:
    """The module used on task k IS head k — by identity, not by output."""
    _, _, used = _capture(model, x, task_k)
    assert used, f"{label}: no readout module was called"
    mod = used[-1][0]
    key = str(task_k)
    assert key in model.task_classifiers, (   # ModuleDict has no .get()
        f"P3b FAILED {label}: no head exists for task {task_k}")
    want = model.task_classifiers[key]
    assert mod is want, (
        f"P3b FAILED {label}: task {task_k} was scored by a different module "
        f"(id {id(mod)} vs head {task_k} id {id(want)}) — 'task-IL' is not "
        f"what the deployed path does")
    return f"head[{task_k}]"


def positive_controls(model, x, task_k, verbose=True) -> dict:
    """Both gates must FAIL where they are known to have to. Run before live use."""
    res = {}

    # P3a: capture the pre-norm CLS — a real tensor, one layer early.
    try:
        assert_p3a(model, x, task_k, label="CONTROL", pre_norm=True)
        res["p3a_control"] = False
    except AssertionError as e:
        res["p3a_control"] = True
        if verbose:
            print(f"    P3a control fired: {str(e)[:96]}...")

    # P3b: route to the latest head instead of head k.
    saved = model.head_routing_by_hint
    model.head_routing_by_hint = False          # falls back to current/latest head
    try:
        assert_p3b(model, x, task_k, label="CONTROL")
        res["p3b_control"] = False
    except AssertionError as e:
        res["p3b_control"] = True
        if verbose:
            print(f"    P3b control fired: {str(e)[:96]}...")
    finally:
        model.head_routing_by_hint = saved
    return res


def assert_p3(model, x, task_k, label="") -> dict:
    """Both asserts, for the top of every consuming script."""
    return {"p3a_delta": assert_p3a(model, x, task_k, label),
            "p3b_module": assert_p3b(model, x, task_k, label)}
