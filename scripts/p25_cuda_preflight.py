"""V100 launch gate: real-data gradients; diagnostic weights discarded."""
import io
import argparse
import json
from pathlib import Path
import torch
from fed_finelabel_budget.p25 import authorize_diagnostic
from fed_finelabel_budget.training.seeds import seed_everything
from fed_finelabel_budget.training.losses import MixedSupervisionLoss
from fed_finelabel_budget.models.resnet_cifar import resnet18_cifar


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/p25_cuda_preflight.json")
    args=parser.parse_args()
    assert torch.cuda.is_available(), "CUDA preflight requires an available GPU"
    seed_everything(102)
    cfg = {"dataset_root":"data/cifar100", "split_seed":20261011,"validation_samples":32}
    sim,_purchased,ds,_evaluator=authorize_diagnostic(cfg,16)
    model=resnet18_cifar().cuda().train()
    x,c,f,_ids=next(iter(torch.utils.data.DataLoader(ds,batch_size=16)))
    x,c,f=x.cuda(),c.cuda(),f.cuda()
    criterion=MixedSupervisionLoss(sim.hierarchy).cuda()
    optimizer=torch.optim.SGD(model.parameters(),lr=.02)
    records={}
    for name,labels in (("fine",f),("mixed",torch.cat((f[:1],torch.full_like(f[1:],-1))))):
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)
        assert logits.shape==(16,100)
        loss=criterion(logits,c,labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        records[name]={"loss":float(loss.detach()),"gradients_finite":True}
    buffer=io.BytesIO()
    torch.save(model.state_dict(),buffer)
    buffer.seek(0)
    restored=torch.load(buffer,map_location="cuda",weights_only=True)
    assert all(torch.equal(v,restored[k]) for k,v in model.state_dict().items())
    result={"purpose":"diagnostic_full_fine","authorized_training_samples":16,"official_test_read":False,
            "gpu":torch.cuda.get_device_name(0),"torch":str(torch.__version__),"losses":records,"checkpoint_exact":True}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
