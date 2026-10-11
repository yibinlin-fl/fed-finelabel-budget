import math
import pytest
import torch
from torch.nn import functional as F
from fed_finelabel_budget.training.losses import MixedSupervisionLoss


def test_mixed_is_one_loss_per_sample(hierarchy):
    z = torch.randn(3, 100, dtype=torch.double, requires_grad=True)
    coarse = torch.tensor([hierarchy.fine_to_coarse[7], 2, 3])
    fine = torch.tensor([7, -1, -1])
    actual = MixedSupervisionLoss(hierarchy)(z, coarse, fine, "none")
    expected = torch.stack([F.cross_entropy(z[:1], fine[:1]),
        -z[1].log_softmax(0)[list(hierarchy.fine_ids(2))].exp().sum().log(),
        -z[2].log_softmax(0)[list(hierarchy.fine_ids(3))].exp().sum().log()])
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(lambda x: MixedSupervisionLoss(hierarchy)(x, coarse, fine), (z,))


@pytest.mark.parametrize("known", [False, True])
def test_all_coarse_or_all_fine_uniform(hierarchy, known):
    z = torch.zeros(4, 100, requires_grad=True)
    coarse = torch.full((4,), hierarchy.fine_to_coarse[0])
    fine = torch.full((4,), 0 if known else -1)
    loss = MixedSupervisionLoss(hierarchy)(z, coarse, fine)
    assert loss.item() == pytest.approx(math.log(100 if known else 20))
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_extreme_logits_and_invalid_boundaries(hierarchy):
    loss_fn = MixedSupervisionLoss(hierarchy)
    z = torch.linspace(-10000, 10000, 200).reshape(2, 100).requires_grad_()
    loss = loss_fn(z, torch.tensor([0, 19]), torch.tensor([-1, -1]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    for coarse, fine in [([20, 0], [-1, -1]), ([0, 0], [-2, -1]),
                         ([0, 0], [100, -1]), ([0, 0], [7, -1])]:
        with pytest.raises(ValueError):
            loss_fn(z, torch.tensor(coarse), torch.tensor(fine))
    with pytest.raises(ValueError):
        loss_fn(torch.empty(0, 100), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
