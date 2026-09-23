"""
ViT-B/16 backbone — E12's encoder (B4).

Pre-registration: docs/E12_prereg.md

NOT A DROP-IN, AND DELIBERATELY SO. `MLPEncoder` returns LSTM-shaped tensors so
the whole PLCM stack (memory read, GGC composition, learned output gate) runs on
top of it — correct for a brittleness control, where holding the machinery fixed
is the point. Doing that here would contradict the contract: P3a registers the
deployed feature as **CLS token, post-norm, pre-head**, and wrapping it in
`o_t * tanh(c')` would measure a ViT+PLCM hybrid whose readout no reader of the
ViT literature would recognize. E12 positions against Davari/Masip, who study
plain ViTs; the encoder must be one.

So the ViT path uses the NATIVE readout and bypasses memory/composition/output
gate entirely. Consequence, stated rather than buried: E12 measures a different
model family from the program's earlier arms, which is a comparability caveat on
H-V4 (descriptive, no gate) and is disclosed in the memo.

ADAPTER PLACEMENT. The registered form is a per-task linear map on **patch
embeddings**, so `forward_features` is reimplemented here to expose the injection
point between `patch_embed` and `_pos_embed`. A reimplementation is a premise:
`assert_matches_timm()` proves the no-adapter path is **bitwise** timm's own
`forward_features`, and it is called from the build gate. This is version-
sensitive by construction, which is why timm is pinned (B2).
"""

import torch
import torch.nn as nn

VIT_MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
VIT_EMBED_DIM = 768          # equals 16*16*3 — see split_cifar100's shift note


class ViTEncoder(nn.Module):
    """timm ViT-B/16 exposing (a) the patch-embedding injection point and
    (b) the post-norm CLS token that the deployed head reads.

    Args:
        model_name: timm identifier; pinned by B2 and hashed by the build gate
        pretrained: load ImageNet weights (baked into the image at build time)
    """

    def __init__(self, model_name: str = VIT_MODEL, pretrained: bool = True):
        super().__init__()
        import timm                      # lazy: the repo imports without timm
        self.model_name = model_name
        # num_classes=0 removes timm's own head; PLCM owns the readout.
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim

    # ------------------------------------------------------------- internals --
    def _features(self, x: torch.Tensor, adapter=None) -> torch.Tensor:
        """timm's forward_features, with the patch-embedding injection exposed.

        Sequence mirrors timm.models.vision_transformer.VisionTransformer:
            patch_embed -> [ADAPTER] -> _pos_embed -> patch_drop -> norm_pre
            -> blocks -> norm
        """
        v = self.vit
        x = v.patch_embed(x)                       # [B, N, C] patch embeddings
        if adapter is not None:
            x = adapter(x)                         # per-token linear, identity-init
        x = v._pos_embed(x)                        # prepends CLS, adds pos embed
        x = v.patch_drop(x)
        x = v.norm_pre(x)
        x = v.blocks(x)
        return v.norm(x)                           # [B, 1+N, C], post-norm

    def forward(self, x: torch.Tensor, adapter=None) -> torch.Tensor:
        """The DEPLOYED FEATURE: post-norm CLS, pre-head.

        `forward_head(..., pre_logits=True)` is timm's own pooling + fc_norm,
        stopping before the classifier — exactly the tensor P3a names.
        """
        tokens = self._features(x, adapter)
        return self.vit.forward_head(tokens, pre_logits=True)   # [B, embed_dim]

    @torch.no_grad()
    def wrong_capture_point(self, x: torch.Tensor) -> torch.Tensor:
        """P3a's POSITIVE CONTROL for this architecture: the PRE-norm CLS.

        Byte-for-byte the computation scripts/e12_p3._capture used inline; moved
        here so the control is the ENCODER's responsibility. A ResNet's wrong
        point is a different reduction of a different layer, and a gate that
        hardcodes one architecture's mistake cannot guard another's.
        """
        v = self.vit
        t = v.patch_embed(x)
        t = v._pos_embed(t)
        t = v.patch_drop(t)
        t = v.norm_pre(t)
        t = v.blocks(t)                       # <- deliberately no v.norm(t)
        return v.forward_head(t, pre_logits=True)

    # ---------------------------------------------------------- verification --
    @torch.no_grad()
    def assert_matches_timm(self, x: torch.Tensor) -> float:
        """B4's proof: the reimplemented path IS timm's, bitwise, with no adapter.

        Non-tautological — it compares against timm's own `forward_features`,
        so a wrong order, a skipped `norm_pre`, or a pos-embed applied on the
        wrong side makes it unequal. Returns max |diff| (0.0 on success).
        """
        mine = self._features(x, adapter=None)
        theirs = self.vit.forward_features(x)
        d = float((mine - theirs).abs().max())
        assert torch.equal(mine, theirs), (
            f"reimplemented forward_features is not timm's (max |diff| = {d:.3e}) "
            f"— the adapter injection point is not where it is claimed to be")
        return d
