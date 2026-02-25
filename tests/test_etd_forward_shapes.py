import torch

from models.etd_multitask.model import ETDMultiTaskModel


def test_etd_forward_shapes():
    model = ETDMultiTaskModel()

    bsz = 2
    length = 128
    max_sites = 6

    tokens = torch.randint(0, 4, (bsz, length), dtype=torch.long)
    attn_mask = torch.ones((bsz, length), dtype=torch.bool)
    struct_feats = torch.zeros((bsz, length, 8), dtype=torch.float32)
    site_positions = torch.randint(0, length, (bsz, max_sites), dtype=torch.long)
    site_mask = torch.ones((bsz, max_sites), dtype=torch.bool)

    cond_task = torch.zeros((bsz,), dtype=torch.long)
    cond_role = torch.zeros((bsz,), dtype=torch.long)
    cond_base = torch.zeros((bsz,), dtype=torch.long)

    out = model(
        tokens=tokens,
        cond_task=cond_task,
        cond_role=cond_role,
        cond_base=cond_base,
        attn_mask=attn_mask,
        struct_feats=struct_feats,
        site_positions=site_positions,
        site_mask=site_mask,
    )

    assert out["mask_logits"].shape == (bsz, length, 4)
    assert out["mod_logits_acu"].shape == (bsz, length, 3)
    assert out["mod_logits"].shape == (bsz, length)
    assert out["bind_logits"].shape == (bsz, max_sites)
    assert out["bind_alpha"].shape == (bsz, max_sites, 2)
    assert out["alpha"].shape == (bsz, max_sites, 2)

    down_len = (length + 15) // 16
    assert out["struct_logits"].shape == (bsz, down_len, down_len)
