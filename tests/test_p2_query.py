import math
import numpy as np
import pytest
import torch
from fed_finelabel_budget.query.strategies import BASELINES, choose_ids, conditional_entropy, allocate_quotas
from fed_finelabel_budget.data.label_oracle import FineLabelOracle
from fed_finelabel_budget.p2 import acquire_to_target


@pytest.mark.parametrize("strategy", BASELINES)
def test_all_strategy_interfaces_exact_unique_replay(hierarchy, strategy):
    ids = np.arange(50)
    coarse = ids % 20
    logits = np.random.default_rng(1).normal(size=(50, 100))
    features = np.random.default_rng(2).normal(size=(50, 8))
    result = choose_ids(strategy, ids, coarse, 17, 42, hierarchy, logits=logits, features=features)
    assert len(result) == len(set(result)) == 17
    assert set(result) <= set(ids)
    assert result == choose_ids(strategy, ids, coarse, 17, 42, hierarchy, logits=logits, features=features)
    assert choose_ids(strategy, ids, coarse, 0, 42, hierarchy) == []
    with pytest.raises(ValueError):
        choose_ids(strategy, ids, coarse, 51, 42, hierarchy)


def test_entropy_conditioned_only_on_public_group(hierarchy):
    z = torch.zeros(2, 100, dtype=torch.double)
    coarse = torch.tensor([0, 19])
    uniform = conditional_entropy(z, coarse, hierarchy)
    np.testing.assert_allclose(uniform, math.log(5))
    z[0, list(hierarchy.fine_ids(1))] = 10000
    np.testing.assert_allclose(conditional_entropy(z, coarse, hierarchy), uniform)
    z[1, hierarchy.fine_ids(19)[0]] = 10000
    scores = conditional_entropy(z, coarse, hierarchy)
    assert np.isfinite(scores).all() and scores[1] == pytest.approx(0)


def test_group_balancing_uses_purchased_coarse_only(hierarchy):
    ids = np.arange(20)
    coarse = np.repeat([0, 1], 10)
    chosen = choose_ids("B3", ids, coarse, 5, 1, hierarchy, purchased_coarse=[0] * 10)
    assert all(coarse[i] == 1 for i in chosen)


def test_exact_quota_and_stage_cap():
    assert allocate_quotas({0: 0, 1: 3, 2: 7}, 7) == {0: 0, 1: 2, 2: 5}
    oracle = FineLabelOracle({i: i for i in range(100)}, 8)
    acquire_to_target(oracle, [0, 1], 2)
    for ids in ([2, 2], [0, 2], [2, 3, 4]):
        with pytest.raises(ValueError):
            acquire_to_target(oracle, ids, 4)
        assert oracle.spent == 2
    acquire_to_target(oracle, [2, 3], 4)
    acquire_to_target(oracle, [4, 5, 6, 7], 8)
    oracle.query([0])
    assert oracle.spent == 8
