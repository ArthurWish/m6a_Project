"""ETD multi-task model entrypoint."""

from __future__ import annotations

import torch
from torch import nn

from .backbone import ETDBackbone
from .condition import ConditionEncoder
from .constants import PAD_TOKEN_ID, VOCAB_SIZE
from .heads import BindDirichletHead, MaskHead, ModHead, StructHead
from .utils import downsample_1d, downsample_mask, make_pair_features


class ETDMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_embed = nn.Embedding(VOCAB_SIZE, 32)
        self.pos_embed = nn.Embedding(12000, 32)#长度上限 12000，到 32 维
        self.struct_proj = nn.Linear(8, 32)#结构局部特征
        self.input_proj = nn.Linear(96, 256)#投影到 d_model=256
        self.input_norm = nn.LayerNorm(256)

        self.condition = ConditionEncoder(
            task_vocab=4,
            role_vocab=4,
            base_vocab=5,
            embed_dim=64,
            bottleneck_dim=512,
            pair_dim=128,
            decoder_dim=256,
        )

        self.backbone = ETDBackbone(
            d_model=256,
            channels=(320, 384, 448, 512, 512, 512, 512),
            n_layers=9,
            n_heads=8,
            mlp_ratio=4,
            pair_dim=128,
            dropout=0.1,
        )

        self.mask_head = MaskHead(d_model=256)
        self.mod_head = ModHead(d_model=256)
        self.bind_head = BindDirichletHead(d_model=256, hidden=256)
        self.struct_head = StructHead(bottleneck_dim=512, hidden=64)

    def forward(
        self,
        tokens: torch.Tensor,
        cond_task: torch.Tensor,
        cond_role: torch.Tensor,
        cond_base: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        struct_feats: torch.Tensor | None = None,
        site_positions: torch.Tensor | None = None,
        site_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if attn_mask is None:
            attn_mask = tokens != PAD_TOKEN_ID

        bsz, length = tokens.shape
        device = tokens.device

        pos_ids = torch.arange(length, device=device).clamp(max=11999)
        pos_emb = self.pos_embed(pos_ids).unsqueeze(0).expand(bsz, -1, -1)

        if struct_feats is None:
            struct_feats = torch.zeros(bsz, length, 8, device=device, dtype=torch.float32)

        x = torch.cat(
            [
                self.base_embed(tokens), #[bs, L, 32]
                pos_emb,                   #[bs, L, 32]
                self.struct_proj(struct_feats), #[bs, L, 32]
            ],
            dim=-1,
        )
        x = self.input_norm(self.input_proj(x)) #[bs, L, 256]

        down_mask = downsample_mask(attn_mask, factor=16)#序列主干 mask 下采样 [B, ceil(L/16)]
        pair_mask = downsample_mask(attn_mask, factor=32)#pair 分支 mask 下采样 [B, ceil(L/32)]

        struct_down_pair = downsample_1d(struct_feats, factor=32, mask=attn_mask) #[B, ceil(L/32),8]
        pair_feats = make_pair_features(struct_down_pair, pair_mask)

        cond = self.condition(cond_task, cond_role, cond_base)
        backbone_out = self.backbone(
            x=x,
            pair_feats=pair_feats,
            down_mask=down_mask,
            pair_mask=pair_mask,
            film_params=cond,
        )

        decoded = backbone_out["decoded"]
        bottleneck = backbone_out["bottleneck"]
        adjacency_logits = backbone_out["adjacency_logits"]

        mask_logits = self.mask_head(decoded)
        mod_logits_acu = self.mod_head(decoded)

        if site_positions is None:
            site_positions = tokens.new_full((bsz, 1), 0)
            site_mask = torch.zeros((bsz, 1), device=device, dtype=torch.bool)
        elif site_mask is None:
            site_mask = site_positions >= 0

        bind_out = self.bind_head(decoded, site_positions=site_positions, site_mask=site_mask)
        struct_logits = self.struct_head(bottleneck, adjacency_logits)

        return {
            "mask_logits": mask_logits,
            "mod_logits_acu": mod_logits_acu,
            "mod_logits": mod_logits_acu[..., 0],
            "struct_logits": struct_logits,
            **bind_out,
            "down_mask": down_mask,
        }
