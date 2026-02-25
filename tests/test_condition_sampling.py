import random

from scripts.training.train_etd_multitask import _apply_condition_mask


def test_condition_mask_ratio_close_to_target():
    rng = random.Random(123)
    n = 5000
    role_masked = 0
    base_masked = 0
    for _ in range(n):
        cond_role, cond_base = _apply_condition_mask(
            task_name="bind",
            sampled_role="reader",
            sampled_base="A",
            rng=rng,
            role_mask_prob=0.3,
            base_mask_prob=0.3,
        )
        if cond_role == "none":
            role_masked += 1
        if cond_base == "mask":
            base_masked += 1

    role_ratio = role_masked / n
    base_ratio = base_masked / n
    assert abs(role_ratio - 0.3) < 0.03
    assert abs(base_ratio - 0.3) < 0.03


def test_mask_task_keeps_fixed_condition():
    rng = random.Random(123)
    cond_role, cond_base = _apply_condition_mask(
        task_name="mask",
        sampled_role="none",
        sampled_base="mask",
        rng=rng,
        role_mask_prob=0.3,
        base_mask_prob=0.3,
    )
    assert cond_role == "none"
    assert cond_base == "mask"
