import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from fed_finelabel_budget.data.label_oracle import FineLabelOracle
from fed_finelabel_budget.data.public_pool import PublicPool
from fed_finelabel_budget.models.resnet_cifar import resnet18_cifar
from fed_finelabel_budget.training.dataset import ClientDataset
from fed_finelabel_budget.training.fedavg import aggregate_states, train_fedavg
from fed_finelabel_budget.training.losses import MixedSupervisionLoss
from fed_finelabel_budget.training.seeds import seed_everything


def test_weighted_aggregation_and_bn_counter():
    states = [{"weight": torch.tensor([1., 3.]), "num_batches_tracked": torch.tensor(2)},
              {"weight": torch.tensor([5., 7.]), "num_batches_tracked": torch.tensor(5)}]
    result = aggregate_states(states, [1, 3])
    torch.testing.assert_close(result["weight"], torch.tensor([4., 6.]))
    assert result["num_batches_tracked"].item() == 5
    assert result["num_batches_tracked"].dtype == torch.long
    with pytest.raises(ValueError):
        aggregate_states(states, [0, 1])


def test_client_snapshot_isolated_from_truth_and_other_clients(hierarchy):
    pool = PublicPool(np.zeros((4, 32, 32, 3), dtype=np.uint8), np.zeros(4, dtype=np.int64))
    oracle = FineLabelOracle({0: 0, 1: 20, 2: 40, 3: 60}, 2)
    oracle.query([0, 3])
    ds = ClientDataset(pool, [0, 1], oracle.purchased_labels(), hierarchy)
    assert ds.fine == (0, -1)
    assert ds.ids == (0, 1) and len(ds.samples) == 2
    assert ds[1][2] == -1
    with pytest.raises(ValueError):
        ClientDataset(pool, [1], {1: 7}, hierarchy)
    with pytest.raises(ValueError):
        ClientDataset(pool, [1, 1], {}, hierarchy)


def test_training_modules_have_no_trusted_imports():
    root = Path(__file__).parents[1] / "src/fed_finelabel_budget/training"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(x in (node.module or "") for x in ("cifar_io", "label_oracle", "p1"))
            if isinstance(node, ast.Import):
                assert all(not any(x in alias.name for x in ("cifar_io", "label_oracle", "p1")) for alias in node.names)


def test_serial_training_replay_and_no_fine_client(hierarchy):
    torch.set_num_threads(1)
    pool = PublicPool(np.full((6, 32, 32, 3), 120, dtype=np.uint8), np.zeros(6, dtype=np.int64))
    clients = {0: ClientDataset(pool, [0, 1], {0: 0}, hierarchy),
               1: ClientDataset(pool, [2, 3, 4, 5], {}, hierarchy)}
    runs = []
    for _ in range(2):
        seed_everything(12)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3072, 100))
        initial = model[1].weight.detach().clone()
        history = train_fedavg(model, clients, MixedSupervisionLoss(hierarchy), rounds=2,
            clients_per_round=2, local_epochs=1, batch_size=2, learning_rate=.01, max_steps=1,
            sampling_seed=13, training_seed=14)
        assert not torch.equal(initial, model[1].weight)
        assert history[0]["aggregation_weights"] == [1/3, 2/3]
        assert history[0]["clients"][1]["fine_examples_seen"] == 0
        runs.append((model.state_dict(), history))
    assert runs[0][1] == runs[1][1]
    assert all(torch.equal(v, runs[1][0][k]) for k, v in runs[0][0].items())


def test_resnet_cifar_forward_backward():
    torch.set_num_threads(1)
    model = resnet18_cifar().train()
    assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)
    z = model(torch.randn(2, 3, 32, 32))
    assert z.shape == (2, 100)
    z.square().mean().backward()
    assert model.conv1.weight.grad.abs().sum() > 0
