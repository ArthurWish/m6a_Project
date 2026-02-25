import torch

from models.etd_multitask.backbone import ETDBackbone, MultiQueryAttention, PairUpdateBlock


def test_mqa_rope_clip_shape_stable():
    attn = MultiQueryAttention(embed_dim=512, n_heads=8)
    x = torch.randn(2, 17, 512)
    # Large bias to stress soft-clipping branch.
    attention_bias = torch.randn(2, 8, 17, 17) * 50.0
    out = attn(x, attention_bias=attention_bias, key_padding_mask=torch.ones(2, 17, dtype=torch.bool))

    assert out.shape == (2, 17, 512)
    assert torch.isfinite(out).all()


def test_pair_update_shape():
    block = PairUpdateBlock(seq_dim=512, pair_dim=128)
    seq = torch.randn(2, 19, 512)
    pair = torch.randn(2, 10, 10, 128)

    out = block(seq, pair)
    assert out.shape == (2, 10, 10, 128)
    assert torch.isfinite(out).all()


def test_backbone_forward_shapes():
    backbone = ETDBackbone(
        d_model=256,
        channels=(320, 384, 448, 512, 512, 512, 512),
        n_layers=9,
        n_heads=8,
        mlp_ratio=4,
        pair_dim=128,
        dropout=0.1,
    )

    bsz = 2
    length = 257
    seq_len = (length + 15) // 16
    pair_len = (length + 31) // 32

    x = torch.randn(bsz, length, 256)
    pair_feats = torch.randn(bsz, pair_len, pair_len, 4)
    down_mask = torch.ones(bsz, seq_len, dtype=torch.bool)
    pair_mask = torch.ones(bsz, pair_len, dtype=torch.bool)

    film = {
        "gamma_b": torch.zeros(bsz, 512),
        "beta_b": torch.zeros(bsz, 512),
        "gamma_p": torch.zeros(bsz, 128),
        "beta_p": torch.zeros(bsz, 128),
        "gamma_d": torch.zeros(bsz, 256),
        "beta_d": torch.zeros(bsz, 256),
    }

    out = backbone(x=x, pair_feats=pair_feats, down_mask=down_mask, pair_mask=pair_mask, film_params=film)
    assert out["decoded"].shape == (bsz, length, 256)
    assert out["bottleneck"].shape == (bsz, seq_len, 512)
    assert out["adjacency_logits"].shape == (bsz, seq_len, seq_len)
    assert out["pair_state"].shape == (bsz, pair_len, pair_len, 128)
