"""C-Tran with a RETFound (ViT-L/16) backbone.

Faithful to Lanchantin et al. (CVPR 2021) — label queries + label-mask training
(LMT) state embeddings + a self-attention transformer + per-label diagonal
read-out — with the only change being the feature extractor: instead of a CNN
feature map flattened to tokens, we feed RETFound's post-norm token sequence
(CLS + 576 patch tokens, dim 1024) directly as the image features. Everything
downstream (label embeddings, state embeddings, transformer, read-out) is the
original C-Tran, imported from the parent repo so the methodology is identical.
"""
import os
import sys

import torch
import torch.nn as nn

_CTRAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CTRAN_ROOT not in sys.path:
    sys.path.insert(0, _CTRAN_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.transformer_layers import SelfAttnLayer          # noqa: E402  (parent C-Tran)
from models.utils import custom_replace, weights_init        # noqa: E402
from backbone_retfound import RETFoundBackbone                # noqa: E402


class CTranModelRETFound(nn.Module):
    def __init__(self, num_labels, use_lmt, layers=3, heads=4, dropout=0.1,
                 no_x_features=False, retfound_weights=None, img_size=384,
                 drop_path_rate=0.2, grad_checkpointing=True):
        super().__init__()
        self.use_lmt = use_lmt
        self.no_x_features = no_x_features

        # RETFound backbone -> token sequence (B, T, hidden), hidden = 1024
        self.backbone = RETFoundBackbone(retfound_weights, img_size,
                                         drop_path_rate, grad_checkpointing)
        hidden = self.backbone.out_channels

        # Label embeddings (one learned query per class)
        self.register_buffer("label_input", torch.arange(num_labels).view(1, -1).long())
        self.label_lt = nn.Embedding(num_labels, hidden)

        # State embeddings for LMT (unknown / known-negative / known-positive)
        self.known_label_lt = nn.Embedding(3, hidden, padding_idx=0)

        # Transformer + per-label classifier (diagonal read-out)
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout)
                                               for _ in range(layers)])
        self.output_linear = nn.Linear(hidden, num_labels)
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init everything except the pretrained backbone
        for m in (self.label_lt, self.known_label_lt, self.LayerNorm,
                  self.self_attn_layers, self.output_linear):
            m.apply(weights_init)

    def forward(self, images, mask):
        const_label_input = self.label_input.repeat(images.size(0), 1)
        init_label_embeddings = self.label_lt(const_label_input)

        features = self.backbone(images)                       # (B, T, hidden) already tokens

        if self.use_lmt:
            state_embeddings = self.known_label_lt(custom_replace(mask, 0, 1, 2).long())
            init_label_embeddings = init_label_embeddings + state_embeddings

        if self.no_x_features:
            embeddings = init_label_embeddings
        else:
            embeddings = torch.cat((features, init_label_embeddings), 1)

        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        output = self.output_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1), device=output.device).unsqueeze(0).repeat(
            output.size(0), 1, 1)
        output = (output * diag_mask).sum(-1)
        return output, None, attns

    # ── discriminative-LR / freeze helpers ───────────────────────────────────
    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        bb = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in bb]
