import numpy as np
import pytest
from fed_finelabel_budget.data.budget import (
    fraction_to_count, cumulative_targets, initial_random_ids, additional_random_ids
)
from fed_finelabel_budget.data.partition import iid_partition
from fed_finelabel_budget.data.label_oracle import FineLabelOracle

def test_budget_schedule_exact():
    assert cumulative_targets(50000, .02, [.02, .04]) == [1000, 2000, 4000]
    assert cumulative_targets(101, .02, [.02, .04]) == [2, 4, 8]
    assert fraction_to_count(101, .01) == 1
    with pytest.raises(ValueError):
        cumulative_targets(100, .9, [.2])

def test_initial_spending_exact_global_count():
    parts = iid_partition(101, 10, 12)
    ids = initial_random_ids(parts, .08, 13)
    assert len(ids) == len(set(ids)) == 8
    assert set(ids).issubset(set(range(101)))

def test_atomic_rollback_on_over_budget_and_invalid_id():
    oracle = FineLabelOracle({i: i for i in range(10)}, 3)
    oracle.query([0, 1])
    with pytest.raises(ValueError):
        oracle.query([2, 3])
    assert oracle.spent == 2
    with pytest.raises(KeyError):
        oracle.query([2, 500])
    assert oracle.spent == 2
    with pytest.raises(TypeError):
        oracle.query([np.int64(2)])
    assert oracle.spent == 2
    assert oracle.query([0, 0])[0] == 0
    assert oracle.spent == 2
    oracle.query([2])
    assert oracle.spent == 3 and oracle.remaining == 0

def test_extra_random_never_returns_purchased():
    parts = iid_partition(100, 10, 2026)
    ids = additional_random_ids(parts, {0, 1, 2}, 20, 111)
    assert len(ids) == len(set(ids)) == 20
    assert not set(ids) & {0, 1, 2}
