from __future__ import annotations
import numpy as np
import pytest
from fed_finelabel_budget.data.hierarchy import Cifar100Hierarchy

@pytest.fixture
def synthetic_labels():
    # The real CIFAR IDs are NOT contiguous by superclass; use permuted ids.
    fine = np.tile(np.arange(100), 2)
    fine_to_coarse = np.array([(f * 7) % 20 for f in range(100)], dtype=np.int64)
    coarse = fine_to_coarse[fine]
    return fine, coarse, fine_to_coarse

@pytest.fixture
def hierarchy(synthetic_labels):
    fine, coarse, _mapping = synthetic_labels
    return Cifar100Hierarchy.from_label_pairs(fine, coarse)
