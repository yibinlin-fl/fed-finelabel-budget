import numpy as np
import pytest
from fed_finelabel_budget.data.partition import (
    iid_partition, coarse_dirichlet_partition, validate_partition
)
from fed_finelabel_budget.data.budget import initial_random_ids

def test_iid_replay():
    a = iid_partition(100, 10, 111)
    b = iid_partition(100, 10, 111)
    assert all(np.array_equal(a[i], b[i]) for i in a)
    assert not all(np.array_equal(a[i], iid_partition(100, 10, 112)[i]) for i in a)

def test_dirichlet_replay_nonempty_and_complete():
    coarse = np.repeat(np.arange(20), 40)
    a = coarse_dirichlet_partition(coarse, 10, .3, 20261011, min_client_samples=5)
    b = coarse_dirichlet_partition(coarse, 10, .3, 20261011, min_client_samples=5)
    assert all(np.array_equal(a[i], b[i]) for i in a)
    assert min(map(len, a.values())) >= 5
    validate_partition(a, len(coarse))
    assert initial_random_ids(a, .02, 99) == initial_random_ids(b, .02, 99)

def test_impossible_minimum_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        coarse_dirichlet_partition(np.tile(np.arange(20), 2), 10, .3, 1, min_client_samples=5)
