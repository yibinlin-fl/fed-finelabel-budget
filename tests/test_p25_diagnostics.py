import numpy as np
import pytest
import torch
from torch import nn
from fed_finelabel_budget.eval.metrics import classification_metrics
from fed_finelabel_budget.eval.offline import OfflineEvaluator
from fed_finelabel_budget.training.losses import MixedSupervisionLoss
from fed_finelabel_budget.training.dataset import ClientDataset
from fed_finelabel_budget.training.diagnostic import supervised_epoch
from fed_finelabel_budget.data.public_pool import PublicPool
from fed_finelabel_budget.p25 import gradient_audit, validate_diagnostic_config
from fed_finelabel_budget.training.fedavg import train_fedavg
from fed_finelabel_budget.training.baselines import FineOnlyLoss
from copy import deepcopy


@pytest.mark.parametrize("correct", [True, False])
def test_constructed_logits_metric_control(hierarchy, correct):
    labels = np.arange(100)
    logits = torch.full((100, 100), -20.)
    logits[torch.arange(100), (torch.arange(100) + (0 if correct else 1)) % 100] = 20.
    metrics = classification_metrics(labels, logits.argmax(1), hierarchy.fine_to_coarse)
    assert metrics["accuracy"] == metrics["macro_f1"] == (1. if correct else 0.)


def test_fixed_100_class_macro_f1_subset():
    metrics = classification_metrics([2, 7], [2, 7], [0, 1])
    assert metrics["accuracy"] == 1 and metrics["macro_f1"] == .02


def test_gradient_dilution_is_quantified(hierarchy):
    result = gradient_audit(hierarchy)
    assert result["sample_mean"]["finite"] and result["group_mean"]["finite"]
    assert result["fine_norm_ratio"] == pytest.approx(50.)
    assert result["coarse_norm_ratio"] == pytest.approx(100/98)


@pytest.mark.parametrize("known", [True, False])
def test_loss_group_mean_absent_group_and_weights(hierarchy, known):
    z = torch.randn(8, 100, requires_grad=True)
    f = torch.arange(8) if known else torch.full((8,), -1, dtype=torch.long)
    c = torch.tensor(hierarchy.fine_to_coarse[:8])
    raw = MixedSupervisionLoss(hierarchy)(z, c, f, reduction="none")
    loss = MixedSupervisionLoss(hierarchy, aggregation="group_mean", fine_weight=2., coarse_weight=.5)(z, c, f)
    assert torch.allclose(loss, raw.mean() * (2. if known else .5))
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_default_loss_remains_per_sample_mean(hierarchy):
    z = torch.randn(20, 100)
    f = torch.full((20,), -1, dtype=torch.long)
    f[:2] = torch.tensor([0, 1])
    c = torch.tensor(hierarchy.fine_to_coarse[:20])
    original = MixedSupervisionLoss(hierarchy)
    assert torch.equal(original(z,c,f), original(z,c,f,reduction="none").mean())


def test_offline_diagnostics_do_not_expose_labels_or_mutate_bn(hierarchy):
    images = np.zeros((4,32,32,3), dtype=np.uint8)
    evaluator = OfflineEvaluator(images, [0]*4, [hierarchy.fine_to_coarse[0]]*4)
    model = nn.Sequential(nn.Conv2d(3,4,1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4,100))
    state = {k:v.clone() for k,v in model.state_dict().items()}
    result = evaluator.evaluate(model, diagnostics=True)
    assert model.training and all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert sum(result["prediction_histogram"]) == 4
    assert 0 <= result["prediction_entropy"] <= np.log(100) + 1e-5
    assert np.isfinite(result["fine_ce"])
    assert not hasattr(evaluator, "fine_labels")
    histogram = evaluator.label_distribution()
    histogram["fine_histogram"][0] = 999
    assert evaluator.label_distribution()["fine_histogram"][0] == 4


def test_diagnostic_training_rejects_unpurchased_fine(hierarchy):
    pool = PublicPool(np.zeros((2,32,32,3), dtype=np.uint8), [0,0])
    dataset = ClientDataset(pool, [0,1], {}, hierarchy)
    model = nn.Linear(1,100)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    with pytest.raises(ValueError, match="authorized"):
        supervised_epoch(model, dataset, optimizer, seed=1, batch_size=2, device="cpu")


@pytest.mark.parametrize("options", [{"aggregation":"invalid"}, {"fine_weight":-1}, {"coarse_weight":float('nan')}, {"fine_weight":0,"coarse_weight":0}])
def test_loss_invalid_configuration(hierarchy, options):
    with pytest.raises(ValueError):
        MixedSupervisionLoss(hierarchy, **options)


def test_fedavg_matches_manual_local_sgd_updates(hierarchy):
    torch.set_num_threads(1)
    fine_labels = hierarchy.coarse_to_fine[0][:2]
    pool = PublicPool(np.stack([np.full((32,32,3),60,dtype=np.uint8)]*2 +
                              [np.full((32,32,3),180,dtype=np.uint8)]*6), np.zeros(8,dtype=np.int64))
    purchased = {i:fine_labels[0 if i<2 else 1] for i in range(8)}
    clients = {0:ClientDataset(pool,range(2),purchased,hierarchy),
               1:ClientDataset(pool,range(2,8),purchased,hierarchy)}
    model = nn.Sequential(nn.Flatten(),nn.Linear(3072,100))
    states=[]
    for cid in (0,1):
        local=deepcopy(model)
        optimizer=torch.optim.SGD(local.parameters(),lr=.001,momentum=.9)
        x,c,f,_=clients[cid][0]
        for _ in range(2):
            optimizer.zero_grad()
            nn.functional.cross_entropy(local(x[None]),torch.tensor([f])).backward()
            optimizer.step()
        states.append(local.state_dict())
    history=train_fedavg(model,clients,FineOnlyLoss(),rounds=1,clients_per_round=2,local_epochs=1,
        batch_size=1,learning_rate=.001,momentum=.9,max_steps=2,sampling_seed=1,training_seed=2,fixed_steps=True)
    assert history[0]["aggregation_weights"] == [.25,.75]
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,states[0][k]*.25+states[1][k]*.75)


def test_diagnostic_cpu_guard_prevents_full_experiment():
    from pathlib import Path
    import yaml
    cfg=yaml.safe_load((Path(__file__).parents[1]/"configs/p25_cpu_diagnostic.yaml").read_text())
    validate_diagnostic_config(cfg)
    cfg["central"]["samples"]=1024
    with pytest.raises(ValueError,match="CPU diagnostics"):
        validate_diagnostic_config(cfg)
