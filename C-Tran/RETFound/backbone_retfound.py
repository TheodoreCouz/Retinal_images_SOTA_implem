"""RETFound (ViT-L/16) backbone for C-Tran.

Loads the `best_retfound_lse_select[_mured|_rfmid]` encoder — the RETFound MAE
checkpoint fine-tuned on the fundus signs — and exposes its post-norm token
sequence (CLS + 576 patch tokens at 384px, dim 1024) as the C-Tran image
features. Self-contained: only needs timm (no Fiber import).

The Fiber checkpoint is a state_dict of `ViTRetFoundRegressor`; its ViT weights
live under the `backbone.` prefix (the `_token_head`/`head` LSE-pooling keys are
discarded — C-Tran replaces that head with its transformer + label queries).
"""
import timm
import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed


class RETFoundBackbone(nn.Module):
    def __init__(self, weights_path=None, img_size=384, drop_path_rate=0.2,
                 grad_checkpointing=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch16_224", pretrained=False, num_classes=0,
            img_size=img_size, drop_path_rate=drop_path_rate)
        self.out_channels = self.backbone.num_features                 # 1024 for ViT-L
        self.num_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        if weights_path:
            self._load(weights_path)
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)                 # fit ViT-L@384 in ~20GB

    def _load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        if not bb:  # fall back: a raw ViT/MAE state_dict
            bb = {k: v for k, v in sd.items()
                  if not k.startswith(("decoder", "mask_token", "head.", "_token_head."))}
        if "pos_embed" in bb and bb["pos_embed"].shape != self.backbone.pos_embed.shape:
            bb["pos_embed"] = resample_abs_pos_embed(
                bb["pos_embed"], new_size=self.backbone.patch_embed.grid_size,
                num_prefix_tokens=self.num_prefix)
        missing, unexpected = self.backbone.load_state_dict(bb, strict=False)
        crit = [k for k in missing if not k.startswith("head")]
        print(f"[RETFound backbone] loaded {path}\n  tensors={len(bb)} "
              f"critical_missing={crit[:4]}({len(crit)}) unexpected={unexpected[:4]}({len(unexpected)})")

    def forward(self, x):
        # (B, num_prefix + N, C) post-norm tokens: CLS (global) + patch grid.
        return self.backbone.forward_features(x)

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
