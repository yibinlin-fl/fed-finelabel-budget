from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
import yaml
from fed_finelabel_budget.data.cifar_io import SimulationData
from fed_finelabel_budget.data.partition import iid_partition
from fed_finelabel_budget.data.public_pool import PublicPool
from fed_finelabel_budget.training.dataset import ClientDataset
from fed_finelabel_budget.training.paired import supervision_terms,train_round
from fed_finelabel_budget.training.losses import MixedSupervisionLoss
from fed_finelabel_budget.training.fedavg import train_fedavg
from fed_finelabel_budget.training.baselines import build_clients
from fed_finelabel_budget.p26 import run_staged,health_checks,verify_pair,validate_config


class TinyBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,2,1),nn.BatchNorm2d(2),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(2,100))
    def forward(self,x):
        return self.net(x)


@pytest.mark.parametrize("aggregation",["sample_mean","group_mean"])
def test_paired_fine_term_gradient_exact_match(hierarchy,aggregation):
    z=torch.randn(6,100,requires_grad=True)
    fine=torch.tensor([0,-1,-1,3,-1,-1])
    coarse=torch.tensor(hierarchy.fine_to_coarse[:6])
    control,a=supervision_terms(z,coarse,fine,hierarchy,"PS_FINE",aggregation=aggregation)
    treated,b=supervision_terms(z,coarse,fine,hierarchy,"PS_COARSE",aggregation=aggregation)
    assert a["fine_count"]==b["fine_count"]==2 and a["fine_component"]==b["fine_component"]
    grad,=torch.autograd.grad(control,z)
    assert torch.count_nonzero(grad[fine<0])==0
    # Treatment's fine part has precisely the same normalization as control.
    raw=MixedSupervisionLoss(hierarchy)(z,coarse,fine,reduction="none")
    fine_only=raw[fine>=0].sum()/(len(fine) if aggregation=="sample_mean" else 2)
    expected,=torch.autograd.grad(fine_only,z)
    torch.testing.assert_close(grad,expected)
    assert b["coarse_count"]==4 and treated>control


def test_paired_forward_stream_and_fine_exposure_exact(hierarchy):
    torch.set_num_threads(1)
    fine=np.arange(40)%100
    pool=PublicPool(np.random.default_rng(2).integers(0,256,(40,32,32,3),dtype=np.uint8),np.array(hierarchy.fine_to_coarse)[fine])
    clients={0:ClientDataset(pool,range(40),{0:0,3:3},hierarchy)}
    initial=TinyBN()
    args=dict(round_index=0,clients_per_round=1,batch_size=4,max_steps=12,learning_rate=.01,momentum=.9,sampling_seed=1,training_seed=2)
    a,b=[train_round(deepcopy(initial),clients,hierarchy,m,**args) for m in ("PS_FINE","PS_COARSE")]
    assert a["clients"][0]["stream_hash"]==b["clients"][0]["stream_hash"]
    assert a["clients"][0]["fine_exposure_hash"]==b["clients"][0]["fine_exposure_hash"]
    assert a["fine_examples_seen"]==b["fine_examples_seen"]
    assert a["bn"]["counter_max"]==b["bn"]["counter_max"]==12
    assert a["coarse_examples_seen"]==0 and b["coarse_examples_seen"]>0
    assert a["zero_gradient_batches"]>0 and a["optimizer_steps"]<b["optimizer_steps"]


def test_all_unknown_control_updates_bn_but_not_parameters(hierarchy):
    pool=PublicPool(np.ones((4,32,32,3),dtype=np.uint8),np.zeros(4,dtype=int))
    clients={0:ClientDataset(pool,range(4),{},hierarchy)}
    model=TinyBN()
    before=[p.detach().clone() for p in model.parameters()]
    r=train_round(model,clients,hierarchy,"PS_FINE",round_index=0,clients_per_round=1,batch_size=2,max_steps=3,
                  learning_rate=.1,momentum=.9,sampling_seed=1,training_seed=2)
    assert all(torch.equal(a,p) for a,p in zip(before,model.parameters()))
    assert r["optimizer_steps"]==r["nonzero_gradient_steps"]==0 and r["bn"]["counter_max"]==3


def test_supervised_zero_gradient_retains_original_sgd_step(hierarchy):
    class ZeroGradient(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=nn.Parameter(torch.ones(100))
        def forward(self,x):
            return (self.weight*0).expand(len(x),100)
    fine=hierarchy.coarse_to_fine[0][0]
    pool=PublicPool(np.ones((4,32,32,3),dtype=np.uint8),np.zeros(4,dtype=int))
    clients={0:ClientDataset(pool,range(4),{i:fine for i in range(4)},hierarchy)}
    r=train_round(ZeroGradient(),clients,hierarchy,"B0",round_index=0,clients_per_round=1,batch_size=2,max_steps=3,
                  learning_rate=.1,momentum=.9,sampling_seed=1,training_seed=2)
    assert r["optimizer_steps"]==r["backward_calls"]==3
    assert r["nonzero_gradient_steps"]==r["parameter_update_steps"]==0


@pytest.mark.parametrize("method",["B0","B1"])
def test_original_supervision_replica_matches_legacy_update(hierarchy,method):
    pool=PublicPool(np.random.default_rng(3).integers(0,256,(8,32,32,3),dtype=np.uint8),np.zeros(8,dtype=int))
    clients,criterion=build_clients(method,pool,{0:list(range(8))},{0:hierarchy.coarse_to_fine[0][0]},hierarchy)
    a=TinyBN(); b=deepcopy(a)
    common=dict(clients_per_round=1,batch_size=4,max_steps=2,learning_rate=.01,momentum=.9,sampling_seed=1,training_seed=2)
    train_fedavg(a,clients,criterion,rounds=1,local_epochs=1,fixed_steps=True,**common)
    train_round(b,clients,hierarchy,method,round_index=0,**common)
    for key,v in a.state_dict().items():
        torch.testing.assert_close(v,b.state_dict()[key],rtol=1e-5,atol=1e-6)


class FakeValidation:
    """Explicit test double for 5k health logic; never an experiment metric."""
    def evaluate(self,*args,**kwargs):
        return {"samples":5000,"accuracy":.01,"macro_f1":.001,"fine_ce":4.6,"prediction_entropy":4.6,
                "prediction_histogram":[50]*100,"max_prediction_fraction":.01}


@pytest.fixture
def simulation(hierarchy):
    torch.set_num_threads(1)
    f=np.arange(1000)%100
    c=np.array(hierarchy.fine_to_coarse)[f]
    images=np.random.default_rng(10).integers(0,256,(1000,32,32,3),dtype=np.uint8)
    sim=SimulationData(images,c,f,hierarchy,FakeValidation(),None,{"train_original_ids":list(range(1000)),"validation_original_ids":list(range(1000,6000))})
    cfg=yaml.safe_load((Path(__file__).parents[1]/"configs/p26_v100.yaml").read_text())
    cfg["device"]="cpu"
    cfg["training"].update(clients_per_round=2,batch_size=4,max_steps=1)
    parts=iid_partition(1000,2,101)
    return cfg,sim,parts


def meta():
    return {"source_hash":"fixture","torch":str(torch.__version__),"config_hash":"fixture","git_commit":"TEST_DOUBLE","hardware":"unit-test CPU"}


def test_stage_resume_budget_chain_and_pair_verification(simulation,tmp_path):
    cfg,sim,parts=simulation
    results={}
    for method in ("PS_FINE","PS_COARSE"):
        output=tmp_path/method
        r=run_staged(cfg,sim,parts,method,output,until_round=5,model_factory=TinyBN,metadata=meta())
        assert r["rounds"][-1]["purchased"]==20
        for until,budget in ((30,20),(35,40),(60,40),(65,80),(90,80)):
            r=run_staged(cfg,sim,parts,method,output,until_round=until,resume=r["checkpoint"],model_factory=TinyBN,metadata=meta())
            assert r["rounds"][-1]["purchased"]==budget and len(r["rounds"])==until
        assert len({i for e in r["ledger"] for i in e["new_ids"]})==80
        assert r["checkpoint_verified"]
        results[method]=r
    assert verify_pair(results["PS_FINE"]["rounds"],results["PS_COARSE"]["rounds"])["rounds"]==90
    altered=deepcopy(results["PS_COARSE"]["rounds"])
    altered[0]["clients"][0]["stream_hash"]="wrong"
    with pytest.raises(ValueError,match="stream"):
        verify_pair(results["PS_FINE"]["rounds"],altered)


def test_fresh_restart_and_invalid_resume_are_rejected(simulation,tmp_path):
    cfg,sim,parts=simulation
    with pytest.raises(ValueError,match="round5"):
        run_staged(cfg,sim,parts,"B1",tmp_path/"bad",until_round=90,model_factory=TinyBN,metadata=meta())
    r=run_staged(cfg,sim,parts,"B1",tmp_path/"good",until_round=5,model_factory=TinyBN,metadata=meta())
    with pytest.raises(ValueError,match="empty"):
        run_staged(cfg,sim,parts,"B1",tmp_path/"good",until_round=5,model_factory=TinyBN,metadata=meta())
    with pytest.raises(ValueError,match="continuation"):
        run_staged(cfg,sim,parts,"B1",tmp_path/"good",until_round=90,resume=r["checkpoint"],model_factory=TinyBN,metadata=meta())
    altered=deepcopy(cfg); altered["loss_protocol"]="group_mean"
    with pytest.raises(ValueError,match="protocol"):
        run_staged(altered,sim,parts,"B1",tmp_path/"good",until_round=30,resume=r["checkpoint"],model_factory=TinyBN,metadata=meta())
    state=torch.load(r["checkpoint"],weights_only=True)
    state["ledger"][0]["new_ids"]=list(state["ledger"][0]["new_ids"])
    state["ledger"][0]["new_ids"][0]=state["ledger"][0]["new_ids"][1]
    path=tmp_path/"corrupt.pt"; torch.save(state,path)
    with pytest.raises(ValueError,match="ledger"):
        run_staged(cfg,sim,parts,"B1",tmp_path/"other",until_round=30,resume=path,model_factory=TinyBN,metadata=meta())


@pytest.mark.parametrize("fault",["bn","ce","budget"])
def test_health_rejects_bn_ce_budget_pathology(simulation,tmp_path,fault):
    cfg,sim,parts=simulation
    rows=run_staged(cfg,sim,parts,"B1",tmp_path/"health",until_round=5,model_factory=TinyBN,metadata=meta())["rounds"]
    assert health_checks(rows)["engineering_pass"]
    if fault=="bn":
        rows[-1]["bn"]["negative_variances"]=1
    elif fault=="ce":
        rows[-1]["validation"]["fine_ce"]=501.
    else:
        rows[-1]["purchased"]+=1
    assert not health_checks(rows)["engineering_pass"]


def test_resume_matches_manual_uninterrupted_training(simulation,tmp_path):
    from fed_finelabel_budget.p26 import build_datasets
    from fed_finelabel_budget.training.seeds import seed_everything
    from fed_finelabel_budget.data.budget import initial_random_ids
    cfg,sim,parts=simulation
    r=run_staged(cfg,sim,parts,"PS_COARSE",tmp_path/"resumed",until_round=5,model_factory=TinyBN,metadata=meta())
    r=run_staged(cfg,sim,parts,"PS_COARSE",tmp_path/"resumed",until_round=30,resume=r["checkpoint"],model_factory=TinyBN,metadata=meta())
    oracle=sim.new_oracle(80)
    oracle.query(initial_random_ids(parts,.02,cfg["seeds"]["query"]))
    clients=build_datasets("PS_COARSE",sim.pool,parts,oracle.purchased_labels(),sim.hierarchy)
    seed_everything(cfg["seeds"]["init"])
    model=TinyBN()
    for ri in range(30):
        record=train_round(model,clients,sim.hierarchy,"PS_COARSE",round_index=ri,clients_per_round=2,batch_size=4,
            max_steps=1,learning_rate=cfg["training"]["learning_rate"],momentum=.9,sampling_seed=103,training_seed=105,
            aggregation_counts={k:len(ids) for k,ids in parts.items()})
        assert [c["stream_hash"] for c in record["clients"]]==[c["stream_hash"] for c in r["rounds"][ri]["clients"]]
    saved=torch.load(r["checkpoint"],weights_only=True)
    assert all(torch.equal(v,saved["model"][k]) for k,v in model.state_dict().items())


def test_cuda_entry_does_not_fallback_to_cpu():
    cfg=yaml.safe_load((Path(__file__).parents[1]/"configs/p26_v100.yaml").read_text())
    cfg["device"]="cpu"
    with pytest.raises(RuntimeError,match="no CPU fallback"):
        validate_config(cfg)


@pytest.mark.parametrize("fault",["weights","validation","diagnostic_name"])
def test_v100_configuration_protocol_guards(monkeypatch,fault):
    # Hardware is mocked ONLY to unit-test schema checks, not GPU execution.
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda _idx:"MOCK V100")
    cfg=yaml.safe_load((Path(__file__).parents[1]/"configs/p26_v100.yaml").read_text())
    validate_config(cfg)
    if fault=="weights":
        cfg["loss_weights"]["fine"]=2
    elif fault=="validation":
        cfg["dataset"]["validation_samples"]=100
    else:
        cfg["loss_protocol"]="group_mean"
        cfg["protocol_role"]="group_mean_diagnostic"
    with pytest.raises(ValueError):
        validate_config(cfg)
