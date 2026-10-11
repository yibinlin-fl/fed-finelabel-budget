from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np
import pytest
import torch
from torch import nn
import yaml
from fed_finelabel_budget.data.cifar_io import SimulationData, prepare_experiment
from fed_finelabel_budget.data.partition import iid_partition
from fed_finelabel_budget.eval.offline import OfflineEvaluator
from fed_finelabel_budget.p2 import run_baseline, validate_config
from fed_finelabel_budget.p2 import query_increment
from fed_finelabel_budget.data.budget import initial_random_ids


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 100)

    def forward(self, images):
        return self.fc(self.avgpool(self.conv(images)).flatten(1))


@pytest.fixture
def experiment(hierarchy):
    torch.set_num_threads(1)
    fine = np.arange(1000) % 100
    coarse = np.array(hierarchy.fine_to_coarse)[fine]
    images = np.random.default_rng(10).integers(0, 256, (1000, 32, 32, 3), dtype=np.uint8)
    val = OfflineEvaluator(images[:10], fine[:10], coarse[:10])
    simulation = SimulationData(images, coarse, fine, hierarchy, val, None,
                                {"train_original_ids": list(range(1000)), "validation_original_ids": list(range(1000, 1010))})
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/p2_cpu_pilot.yaml").read_text())
    return cfg, simulation, iid_partition(1000, 10, 101)


def test_five_baselines_share_initial_ids_budgets_and_training_cost(experiment, tmp_path):
    cfg, simulation, parts = experiment
    results = [run_baseline(cfg, simulation, parts, b, tmp_path / b, model_factory=TinyNet) for b in cfg["baselines"]]
    for result in results:
        assert result["initial_ids"] == results[0]["initial_ids"]
        assert [s["budget"] for s in result["stages"]] == [20, 40, 80]
        assert [r["gradient_steps"] for r in result["rounds"]] == [10, 10, 10]
        assert [r["actual_sampled_examples"] for r in result["rounds"]] == [40, 40, 40]
        acquired = [i for entry in result["ledger"] for i in entry["new_ids"]]
        assert len(set(acquired)) == len(acquired) == 80
        assert result["checkpoint_verified"] and result["test_final_only"] is None
    # Random queries are paired even though B0 and B1 train different models.
    assert [x["new_ids"] for x in results[0]["ledger"]] == [x["new_ids"] for x in results[1]["ledger"]]
    assert all(c["coarse_examples_seen"] == 0 for r in results[0]["rounds"] for c in r["clients"])
    assert any(c["coarse_examples_seen"] > 0 for r in results[1]["rounds"] for c in r["clients"])


def test_round_checkpoint_resume_matches_uninterrupted(experiment, tmp_path):
    cfg, simulation, parts = experiment
    uninterrupted = run_baseline(cfg, simulation, parts, "B2", tmp_path / "full", model_factory=TinyNet)
    stopped = run_baseline(cfg, simulation, parts, "B2", tmp_path / "resume", stop_after_rounds=1, model_factory=TinyNet)
    assert stopped["status"] == "PAUSED_AT_ROUND_BOUNDARY"
    resumed = run_baseline(cfg, simulation, parts, "B2", tmp_path / "resume", resume=stopped["checkpoint"], model_factory=TinyNet)
    assert [x["new_ids"] for x in uninterrupted["ledger"]] == [x["new_ids"] for x in resumed["ledger"]]
    assert uninterrupted["stages"] == resumed["stages"]
    a = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/checkpoint.pt", weights_only=True)
    assert a["purchased"] == b["purchased"]
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    altered = deepcopy(cfg)
    altered["seeds"]["training"] += 1
    with pytest.raises(ValueError, match="protocol"):
        run_baseline(altered, simulation, parts, "B2", tmp_path / "bad", resume=stopped["checkpoint"], model_factory=TinyNet)


def test_real_format_split_excludes_validation_and_test(tmp_path, synthetic_labels):
    fine, coarse, _ = synthetic_labels
    folder = tmp_path / "cifar-100-python"
    folder.mkdir()
    for name in ("train", "test"):
        with (folder / name).open("wb") as handle:
            pickle.dump({b"data": np.zeros((200, 3072), dtype=np.uint8),
                         b"fine_labels": fine.tolist(), b"coarse_labels": coarse.tolist()}, handle)
    data = prepare_experiment(tmp_path, split_seed=42, validation_size=50, train_samples=100,
                              validation_samples=10, test_samples=10, strict=False)
    manifest = data.manifest
    assert set(manifest["train_original_ids"]).isdisjoint(manifest["validation_original_ids"])
    assert len(data.pool) == 100 and len(manifest["test_original_ids"]) == 10
    oracle = data.new_oracle(8)
    with pytest.raises(KeyError):
        oracle.query([100])  # no validation or test IDs in the training namespace
    assert oracle.spent == 0
    assert not hasattr(data.validation, "fine_labels")
    assert not hasattr(data.test, "fine_labels")


def test_cpu_guard_rejects_full_resnet_experiment(experiment):
    cfg, *_ = experiment
    validate_config(cfg)
    cfg["dataset"]["train_samples"] = None
    with pytest.raises(ValueError, match="CPU"):
        validate_config(cfg)


def test_hidden_truth_changes_do_not_affect_query_inputs(experiment):
    cfg, data, parts = experiment
    initial = initial_random_ids(parts, .02, cfg["seeds"]["query"])
    oracle_a = data.new_oracle(80)
    oracle_a.query(initial)
    # Change every unpurchased fine within the SAME public coarse group.
    truth_b = {i: (i % 100 if i in initial else (i + 20) % 100) for i in range(1000)}
    from fed_finelabel_budget.data.label_oracle import FineLabelOracle
    oracle_b = FineLabelOracle(truth_b, 80)
    oracle_b.query(initial)
    assert dict(oracle_a.purchased_labels()) == dict(oracle_b.purchased_labels())
    model = TinyNet()
    for strategy in ("B2", "B3", "B4"):
        kwargs = dict(count=20, seed=123, model=model, hierarchy=data.hierarchy, device="cpu", batch_size=32)
        a = query_increment(strategy, data.pool, parts, oracle_a.purchased_labels(), **kwargs)
        b = query_increment(strategy, data.pool, parts, oracle_b.purchased_labels(), **kwargs)
        assert a == b and not set(a) & set(initial)
