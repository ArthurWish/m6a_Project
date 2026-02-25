import random
from pathlib import Path

import numpy as np

from models.etd_multitask.data import BPPCache, TranscriptExample, collate_batch


def _make_bpp_cache(tmp_path: Path, transcript_id: str = "tx1", length: int = 7) -> BPPCache:
    out = tmp_path / f"{transcript_id}.npz"
    ij = np.asarray([[0, min(6, length - 1)], [1, min(5, length - 1)], [2, min(4, length - 1)]], dtype=np.int32)
    p_ref = np.asarray([0.2, 0.4, 0.6], dtype=np.float16)
    p_mod = np.asarray([0.3, 0.5, 0.7], dtype=np.float16)
    np.savez_compressed(out, ij=ij, p_ref=p_ref, p_modA=p_mod, L=np.asarray(length, dtype=np.int32))
    return BPPCache(tmp_path)


def test_collate_batch_mod_and_bind(tmp_path: Path):
    cache = _make_bpp_cache(tmp_path)

    example = TranscriptExample(
        transcript_id="tx1",
        sequence="AACGUAA",
        seq_len=7,
        m6a_positions=np.asarray([1, 5], dtype=np.int64),
        unlabeled_a_positions=np.asarray([0, 6], dtype=np.int64),
        role_labels={
            "writer": np.asarray([-1, 1], dtype=np.int64),
            "reader": np.asarray([1, -1], dtype=np.int64),
            "eraser": np.asarray([-1, -1], dtype=np.int64),
        },
        role_support={
            "writer": np.asarray([0, 8], dtype=np.int64),
            "reader": np.asarray([5, 0], dtype=np.int64),
            "eraser": np.asarray([0, 0], dtype=np.int64),
        },
    )

    batch_mod = collate_batch(
        examples=[example],
        task_name="mod",
        role_name="reader",
        cond_base="A",
        bpp_cache=cache,
        strong_binding_threshold=4.0,
        rng=random.Random(0),
        mod_unlabeled_ratio=1.0,
        mask_prob=0.15,
    )

    mod_mask = batch_mod["mod_pu_mask"].numpy()[0]
    mod_labels = batch_mod["mod_pu_labels"].numpy()[0]
    assert mod_labels[1] == 1
    assert mod_labels[5] == 1
    assert mod_mask[1]
    assert mod_mask[5]

    batch_bind = collate_batch(
        examples=[example],
        task_name="bind",
        role_name="reader",
        cond_base="A",
        bpp_cache=cache,
        strong_binding_threshold=4.0,
        rng=random.Random(0),
        mod_unlabeled_ratio=1.0,
        mask_prob=0.15,
    )

    site_pos = batch_bind["site_positions"].numpy()[0]
    site_labels = batch_bind["site_pu_labels"].numpy()[0]
    strong = batch_bind["strong_binding_mask"].numpy()[0]

    assert site_pos[0] == 1
    assert site_pos[1] == 5
    assert site_labels[0] == 1
    assert site_labels[1] == -1
    assert strong[0]
    assert not strong[1]


def test_struct_target_downsample_factor_16(tmp_path: Path):
    seq = ("AACGUAAACGUAAACGUAAACGUAAACGUAAAA")[:33]
    cache = _make_bpp_cache(tmp_path, transcript_id="tx_long", length=len(seq))

    example = TranscriptExample(
        transcript_id="tx_long",
        sequence=seq,
        seq_len=len(seq),
        m6a_positions=np.asarray([1, 9, 17, 25], dtype=np.int64),
        unlabeled_a_positions=np.asarray([i for i, ch in enumerate(seq) if ch == "A" and i not in {1, 9, 17, 25}], dtype=np.int64),
        role_labels={
            "writer": np.asarray([-1, 1, -1, 1], dtype=np.int64),
            "reader": np.asarray([1, -1, 1, -1], dtype=np.int64),
            "eraser": np.asarray([-1, -1, -1, -1], dtype=np.int64),
        },
        role_support={
            "writer": np.asarray([0, 8, 0, 9], dtype=np.int64),
            "reader": np.asarray([5, 0, 6, 0], dtype=np.int64),
            "eraser": np.asarray([0, 0, 0, 0], dtype=np.int64),
        },
    )

    batch = collate_batch(
        examples=[example],
        task_name="struct",
        role_name="reader",
        cond_base="A",
        bpp_cache=cache,
        strong_binding_threshold=4.0,
        rng=random.Random(0),
        mod_unlabeled_ratio=1.0,
        mask_prob=0.15,
    )

    expected_l_prime = (len(seq) + 15) // 16
    assert batch["struct_lengths"].numpy()[0] == expected_l_prime
    assert batch["struct_target"].shape[-1] == expected_l_prime
