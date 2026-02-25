import torch

from models.etd_multitask.losses import NonNegativePULoss, evidential_positive_loss


def test_nnpu_loss_backward_stable():
    logits = torch.randn(32, requires_grad=True)
    labels = torch.tensor([1] * 8 + [-1] * 24)
    mask = torch.ones_like(labels, dtype=torch.bool)

    loss_fn = NonNegativePULoss(prior=0.2)
    loss = loss_fn(logits, labels, mask)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_nnpu_loss_handles_empty_masks():
    logits = torch.randn(10, requires_grad=True)
    labels = torch.full((10,), -1)
    mask = torch.zeros(10, dtype=torch.bool)

    loss_fn = NonNegativePULoss(prior=0.1)
    loss = loss_fn(logits, labels, mask)

    assert float(loss.detach().cpu()) == 0.0


def test_evidential_positive_loss_finite():
    alpha = torch.tensor(
        [
            [[2.0, 4.0], [1.2, 1.1], [3.0, 5.0]],
            [[1.0, 1.0], [2.5, 4.5], [1.1, 1.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ],
        dtype=torch.bool,
    )

    loss = evidential_positive_loss(alpha, positive_mask=mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert alpha.grad is not None
