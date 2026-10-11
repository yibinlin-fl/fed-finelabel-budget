import pickle
import numpy as np
import pytest
from fed_finelabel_budget.data.cifar_io import bootstrap_train, inspect_dataset

@pytest.fixture
def fake_cifar_files(tmp_path, synthetic_labels):
    fine, coarse, _ = synthetic_labels
    folder = tmp_path / "cifar-100-python"
    folder.mkdir()
    for split in ("train", "test"):
        count = 200 if split == "train" else 100
        payload = {
            b"data": np.zeros((count, 3072), dtype=np.uint8),
            b"fine_labels": fine[:count].tolist(),
            b"coarse_labels": coarse[:count].tolist(),
        }
        with (folder / split).open("wb") as handle:
            pickle.dump(payload, handle)
    return tmp_path

def test_trusted_bootstrap_separates_oracle(fake_cifar_files):
    pool, oracle, hierarchy = bootstrap_train(fake_cifar_files, 20, strict=False)
    assert len(pool) == 200
    assert len(pool.coarse_labels()) == 200
    assert not hasattr(pool.sample(0), "fine_label")
    assert oracle.spent == 0
    assert dict(oracle.query([0])) == {0: 0}
    assert hierarchy.fine_to_coarse[7] != 7 // 5

def test_data_check_reports_sizes(fake_cifar_files):
    summary = inspect_dataset(fake_cifar_files, strict=False)
    assert summary["train_samples"] == 200
    assert summary["test_samples"] == 100
    assert summary["fine_classes"] == 100

def test_strict_mode_rejects_tiny_fake_data(fake_cifar_files):
    with pytest.raises(ValueError, match="expected 50000"):
        bootstrap_train(fake_cifar_files, 20, strict=True)


def test_p1_real_data_controller_subset_ids(fake_cifar_files, tmp_path, monkeypatch):
    """Exercise the official-format adapter on a fixture, NOT real CIFAR accuracy."""
    import json
    from pathlib import Path
    import yaml
    from fed_finelabel_budget import p1

    monkeypatch.setattr(p1, "bootstrap_train", lambda root, budget: bootstrap_train(root, budget, strict=False))
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/p1_smoke.yaml").read_text(encoding="utf-8"))
    cfg["mode"] = "cifar100"
    cfg["dataset"]["root"] = str(fake_cifar_files)
    cfg["dataset"]["real_smoke_samples"] = 100
    cfg["output"] = str(tmp_path / "fixture-smoke")
    summary = p1.run_smoke(cfg)
    assert summary["fine_queries_spent"] == summary["fine_labels_available"] == 2
    assert summary["budget_targets"] == [2, 4, 8]
    assert sum(c["fine_examples_seen"] for c in summary["history"][0]["clients"]) == 2
    assert sum(c["examples_seen"] for c in summary["history"][0]["clients"]) == 100
    assert summary["changed_parameter_tensors"] > 0 and summary["checkpoint_verified"]
    assert summary["test_or_validation_read"] is False
    assert json.loads((Path(cfg["output"]) / "run.json").read_text())["status"] == "PASS"
