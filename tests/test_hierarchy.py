import numpy as np
import pytest
from fed_finelabel_budget.data.hierarchy import Cifar100Hierarchy

def test_actual_numeric_mapping_not_assumed(synthetic_labels):
    fine, coarse, mapping = synthetic_labels
    h = Cifar100Hierarchy.from_label_pairs(fine, coarse)
    assert h.fine_to_coarse == tuple(mapping)
    assert len(h.coarse_to_fine) == 20
    assert all(len(group) == 5 for group in h.coarse_to_fine)
    assert h.fine_ids(int(mapping[7])) != tuple(range((int(mapping[7])) * 5, int(mapping[7]) * 5 + 5))

def test_conflict_rejected(synthetic_labels):
    fine, coarse, _ = synthetic_labels
    broken = coarse.copy()
    broken[100] = (broken[100] + 1) % 20
    with pytest.raises(ValueError, match="conflicting"):
        Cifar100Hierarchy.from_label_pairs(fine, broken)

def test_missing_fine_label_rejected(synthetic_labels):
    fine, coarse, _ = synthetic_labels
    with pytest.raises(ValueError, match="100 distinct"):
        Cifar100Hierarchy.from_label_pairs(fine[fine != 99], coarse[fine != 99])

def test_test_labels_checked(hierarchy, synthetic_labels):
    fine, coarse, _ = synthetic_labels
    hierarchy.verify_pairs(fine, coarse)
    bad = coarse.copy()
    bad[1] = (bad[1] + 1) % 20
    with pytest.raises(ValueError, match="does not match"):
        hierarchy.verify_pairs(fine, bad)
