import numpy as np
import pytest
from fed_finelabel_budget.data.public_pool import PublicPool
from fed_finelabel_budget.data.label_oracle import FineLabelOracle
from fed_finelabel_budget.data.partition import coarse_dirichlet_partition

def test_public_samples_never_contain_fine_label():
    image = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    pool = PublicPool(image, np.arange(8, dtype=np.int64))
    sample = pool.sample(0)
    assert not hasattr(sample, "fine_label")
    assert set(sample.__dataclass_fields__) == {"sample_id", "coarse_label", "image"}
    assert "fine" not in repr(sample).lower()
    sample.image[0, 0, 0] = 250
    assert pool.sample(0).image[0, 0, 0] == 0
    given_coarse = pool.coarse_labels()
    given_coarse[0] = 19
    assert pool.coarse_labels()[0] == 0

def test_unpurchased_fine_labels_unavailable():
    oracle = FineLabelOracle({0: 51, 1: 36, 2: 7}, max_queries=2)
    assert dict(oracle.purchased_labels()) == {}
    assert not hasattr(oracle, "fine_labels")
    assert dict(oracle.query([1])) == {1: 36}
    assert dict(oracle.purchased_labels()) == {1: 36}
    with pytest.raises(TypeError):
        oracle.purchased_labels()[0] = 51

def test_coarse_partition_not_affected_by_hidden_fine():
    # Partition API accepts ONLY public coarse labels, never an oracle/fine array.
    coarse = np.tile(np.arange(20, dtype=np.int64), 20)
    before = coarse_dirichlet_partition(coarse, 10, 0.3, 42, min_client_samples=2)
    _unused_hidden_truth_A = np.arange(400) % 100
    _unused_hidden_truth_B = 99 - (_unused_hidden_truth_A % 100)
    after = coarse_dirichlet_partition(coarse, 10, 0.3, 42, min_client_samples=2)
    assert all(np.array_equal(before[c], after[c]) for c in before)
