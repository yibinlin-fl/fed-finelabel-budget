import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from fed_finelabel_budget.eval.metrics import classification_metrics
from fed_finelabel_budget.data.public_pool import PublicPool
from fed_finelabel_budget.data.public_pool import ClientPublicView
from fed_finelabel_budget.training.baselines import build_clients, FineOnlyLoss
from fed_finelabel_budget.training.losses import MixedSupervisionLoss
from fed_finelabel_budget.training.fedavg import train_fedavg
from fed_finelabel_budget.training.seeds import seed_everything


def test_metrics_all_100_class_macro_and_missing_groups():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1])
    assert result["accuracy"] == .75
    assert result["macro_f1"] == pytest.approx((2/3 + 4/5) / 100)
    assert result["coarse_accuracy"]["0"] == .5
    assert result["coarse_accuracy"]["2"] is None


def test_b0_b1_data_and_loss_distinction(hierarchy):
    pool = PublicPool(np.zeros((4, 32, 32, 3), dtype=np.uint8), np.zeros(4, dtype=int))
    b0, ce = build_clients("B0", pool, {0: [0, 1, 2, 3]}, {0: 0}, hierarchy)
    b1, mixed = build_clients("B1", pool, {0: [0, 1, 2, 3]}, {0: 0}, hierarchy)
    assert len(b0[0]) == 1 and len(b1[0]) == 4
    assert isinstance(ce, FineOnlyLoss) and isinstance(mixed, MixedSupervisionLoss)
    with pytest.raises(ValueError):
        ce(torch.zeros(4, 100), torch.zeros(4, dtype=torch.long), torch.tensor([0, -1, -1, -1]))
    z = torch.randn(1, 100)
    torch.testing.assert_close(ce(z, torch.tensor([0]), torch.tensor([0])), mixed(z, torch.tensor([0]), torch.tensor([0])))


def test_fixed_steps_and_round_resume_exact(hierarchy):
    torch.set_num_threads(1)
    pool = PublicPool(np.ones((6, 32, 32, 3), dtype=np.uint8), np.zeros(6, dtype=int))
    clients, criterion = build_clients("B0", pool, {0: [0, 1], 1: [2, 3, 4, 5]}, {0: 0, 2: 0}, hierarchy)
    kwargs = dict(clients_per_round=2, local_epochs=1, batch_size=4, learning_rate=.001,
                  max_steps=2, sampling_seed=13, training_seed=14, fixed_steps=True,
                  aggregation_counts={0: 2, 1: 4})
    seed_everything(12)
    full = nn.Sequential(nn.Flatten(), nn.Linear(3072, 100))
    history = train_fedavg(full, clients, criterion, rounds=2, **kwargs)
    seed_everything(12)
    resumed = nn.Sequential(nn.Flatten(), nn.Linear(3072, 100))
    first = train_fedavg(resumed, clients, criterion, rounds=1, **kwargs)
    state = {k: v.clone() for k, v in resumed.state_dict().items()}
    resumed.load_state_dict(state)
    second = train_fedavg(resumed, clients, criterion, rounds=1, start_round=1, **kwargs)
    assert history == first + second
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())
    assert all(c["steps"] == 2 and c["examples_seen"] == 8 for r in history for c in r["clients"])


def test_training_query_no_truth_or_evaluation_imports():
    base = Path(__file__).parents[1] / "src/fed_finelabel_budget"
    forbidden = ("cifar_io", "label_oracle", "eval", "p1", "p2", "pickle")
    for folder in ("training", "query"):
        for path in (base / folder).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    assert not set((node.module or "").split(".")) & set(forbidden)
                if isinstance(node, ast.Import):
                    assert all(not set(a.name.split(".")) & set(forbidden) for a in node.names)


def test_client_public_view_cannot_read_other_clients():
    pool = PublicPool(np.zeros((4, 32, 32, 3), dtype=np.uint8), np.zeros(4, dtype=int))
    view = ClientPublicView(pool, [1, 3])
    assert set(view.coarse_labels()) == {1, 3}
    with pytest.raises(KeyError):
        view.sample(0)
    image = view.sample(1).image
    image[0, 0, 0] = 255
    assert view.sample(1).image[0, 0, 0] == 0
