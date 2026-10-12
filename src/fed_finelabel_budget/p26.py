"""Trusted staged P2.6 controller; never launched as an automatic 90-round job."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime, timezone
from uuid import uuid4
import torch
import yaml
from .data.cifar_io import prepare_experiment
from .data.partition import iid_partition, coarse_dirichlet_partition, validate_partition
from .data.budget import cumulative_targets, initial_random_ids
from .training.dataset import ClientDataset
from .training.paired import METHODS, train_round
from .training.seeds import seed_everything
from .models.resnet_cifar import resnet18_cifar
from .eval.metrics import client_budget_statistics
from .p2 import _protocol_hash, _save_checkpoint, acquire_to_target, query_increment


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def runtime_metadata(cfg):
    source_hash=hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        source_hash.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        source_hash.update(path.read_bytes())
    cuda=cfg["device"]=="cuda"
    train=Path(cfg["dataset"]["root"])/"cifar-100-python"/"train"
    training_hash=hashlib.sha256()
    with train.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            training_hash.update(block)
    return {"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()),
        "config_hash":digest(cfg),"seeds":cfg["seeds"],"source_hash":source_hash.hexdigest(),
        "python":platform.python_version(),"torch":str(torch.__version__),"platform":platform.platform(),"training_file_hash":training_hash.hexdigest(),
        "hardware":torch.cuda.get_device_name(0) if cuda else platform.processor(),
        "cuda_runtime":torch.version.cuda,"gpu_total_memory":torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "nvidia_smi":subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip() if cuda else None,
        "utc_time":datetime.now(timezone.utc).isoformat()}


def validate_config(cfg):
    if cfg["device"]!="cuda" or not torch.cuda.is_available() or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("P2.6 staged pilot requires accessible V100; no CPU fallback")
    d,t=cfg["dataset"],cfg["training"]
    if d["train_samples"] is not None or d["validation_size"]!=5000 or d["validation_samples"] is not None or d["test_samples"]!=0:
        raise ValueError("P2.6 requires 45k train / full 5k validation / no official test")
    if cfg["n_clients"]!=10 or t["rounds_per_stage"]!=30 or cfg["stage_model_policy"]!="continue":
        raise ValueError("invalid stage protocol")
    if cfg["budget"]!={"initial_fraction":.02,"additional_fractions":[.02,.04]}:
        raise ValueError("invalid budget")
    if cfg["loss_protocol"] not in ("sample_mean","group_mean"):
        raise ValueError("invalid loss protocol")
    if cfg["loss_weights"]!={"fine":1.0,"coarse":1.0}:
        raise ValueError("P2.6 fixes identical loss weights at one for all methods")
    if (cfg["loss_protocol"]=="group_mean") != (cfg["protocol_role"]=="group_mean_diagnostic"):
        raise ValueError("group_mean must be explicitly named diagnostic")
    if not 1<=t["clients_per_round"]<=10 or min(t["batch_size"],t["max_steps"])<1 or not 1<=cfg["threads"]<=8:
        raise ValueError("invalid training limits")
    if not math.isfinite(t["learning_rate"]) or t["learning_rate"]<=0 or not math.isfinite(t["momentum"]) or not 0<=t["momentum"]<1:
        raise ValueError("invalid optimizer")
    if not 1<=cfg["evaluation_batch_size"]<=256 or cfg["query_batch_size"]<1:
        raise ValueError("invalid batch size")
    if cfg["partition"]["type"] not in ("iid","dirichlet_coarse"):
        raise ValueError("invalid partition")
    output=Path(cfg["output"])
    # Prevent pointing a new phase at any earlier results directory.
    if output.parent!=Path("results") or not output.name.startswith("p26_"):
        raise ValueError("output must be results/p26_*")
    if cfg["loss_protocol"]=="group_mean" and "group_mean_diagnostic" not in output.name:
        raise ValueError("group_mean output must be explicitly named diagnostic")


def health_checks(rows):
    errors,warnings=[],[]
    for i,r in enumerate(rows):
        if r["round"]!=i+1:
            errors.append("round sequence mismatch")
        stage=(r["round"]-1)//30
        if stage>2 or r["stage"]!=stage or r["purchased"]!=cumulative_targets(r["budget_denominator"],.02,[.02,.04])[stage]:
            errors.append("round budget/stage mismatch")
        if any(not math.isfinite(r[k]) for k in ("train_loss","fine_loss_component","coarse_loss_component")):
            errors.append("nonfinite loss")
        v=r["validation"]
        if not all(math.isfinite(v[k]) for k in ("accuracy","macro_f1","fine_ce","prediction_entropy")):
            errors.append("nonfinite validation")
        if not 0<=v["accuracy"]<=1 or not 0<=v["macro_f1"]<=1 or v["fine_ce"]<0 or not 0<=v["prediction_entropy"]<=math.log(100)+1e-5:
            errors.append("invalid metric range")
        if sum(v["prediction_histogram"])!=v["samples"] or v["samples"]!=5000:
            errors.append("validation size/histogram mismatch")
        bns=[r["bn"]]+[c["bn"] for c in r["clients"]]
        if any(not b["finite"] or b["negative_variances"] for b in bns):
            errors.append("invalid BN state")
        if any(b["modules"] and (b["variance_min"]<1e-6 or b["variance_max"]>1e4 or b["mean_max_abs"]>100) for b in bns):
            warnings.append(f"round {r['round']}: extreme BN statistics")
        if v["fine_ce"]>100:
            errors.append("validation CE >100: investigate numerical/BN pathology")
        if v["max_prediction_fraction"]>.95 or v["accuracy"]==0:
            warnings.append(f"round {r['round']}: concentrated/zero accuracy; not an engineering failure alone")
        if r["fine_examples_seen"]==0:
            warnings.append(f"round {r['round']}: no fine exposure")
        if r["examples_seen"]!=r["fine_examples_seen"]+r["coarse_examples_seen"]+r["unsupervised_examples_seen"]:
            errors.append("exposure accounting mismatch")
        if not (r["parameter_update_steps"]<=r["optimizer_steps"]<=r["batches"] and r["nonzero_gradient_steps"]<=r["backward_calls"]<=r["batches"]):
            errors.append("gradient/update accounting mismatch")
    return {"engineering_pass":not errors,"errors":sorted(set(errors)),"warnings":warnings,
            "policy":"fixed safety thresholds; not early stopping/model selection by accuracy"}


def build_datasets(method,pool,parts,purchased,hierarchy):
    return {cid:ClientDataset(pool,[int(i) for i in ids if int(i) in purchased] if method=="B0" else ids,
                               purchased,hierarchy) for cid,ids in parts.items()}


def run_staged(cfg,sim,parts,method,output,*,until_round,resume=None,model_factory=resnet18_cifar,metadata=None):
    """Controller accepts a test fixture; CLI imposes V100/full validation gates."""
    if method not in METHODS:
        raise ValueError("unsupported method")
    metadata=runtime_metadata(cfg) if metadata is None else metadata
    validate_partition(parts,len(sim.pool))
    if sim.test is not None:
        raise ValueError("official test forbidden in P2.6")
    protocol_hash=_protocol_hash(cfg,sim.manifest,parts)
    initial=initial_random_ids(parts,.02,cfg["seeds"]["query"])
    targets=cumulative_targets(len(sim.pool),.02,[.02,.04])
    oracle=sim.new_oracle(targets[-1])
    seed_everything(cfg["seeds"]["init"])
    model=model_factory().to(cfg["device"])
    init_hash=hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    rows,ledger=[],[]
    next_round=0
    output=Path(output)
    checkpoint=output/"checkpoint.pt"
    if resume is None:
        if until_round!=5 or output.exists() and any(output.iterdir()):
            raise ValueError("fresh run must stop at round5 in an empty new directory")
    else:
        saved=torch.load(resume,map_location="cpu",weights_only=True)
        if saved.get("purpose")!="p26_budget_protocol" or saved["method"]!=method or saved["protocol_hash"]!=protocol_hash:
            raise ValueError("checkpoint purpose/method/protocol mismatch")
        if saved["source_hash"]!=metadata["source_hash"] or saved["torch"]!=metadata["torch"]:
            raise ValueError("checkpoint source/runtime mismatch")
        if saved["initial_ids"]!=initial or saved["init_hash"]!=init_hash:
            raise ValueError("checkpoint initialization mismatch")
        next_round=saved["next_round"]
        transitions={5:30,30:35,35:60,60:65,65:90}
        if transitions.get(next_round)!=until_round:
            raise ValueError("invalid stage continuation; use 5->30->35->60->65->90")
        rows,ledger=saved["rounds"],saved["ledger"]
        health=health_checks(rows)
        if not health["engineering_pass"]:
            raise ValueError("engineering health gate failed: "+str(health["errors"]))
        # Strict round-boundary purchase/ledger reconciliation before replay.
        expected=targets[(next_round-1)//30]
        acquired=[i for entry in ledger for i in entry["new_ids"]]
        valid_ledger=len(ledger)==(next_round-1)//30+1 and len(rows)==next_round
        for i,e in enumerate(ledger):
            valid_ledger=valid_ledger and e["stage"]==i and e["target"]==targets[i] and len(e["new_ids"])==targets[i]-(targets[i-1] if i else 0)
        if not valid_ledger or ledger[0]["new_ids"]!=initial or len(acquired)!=len(set(acquired)) or set(acquired)!=set(saved["purchased"]) or len(acquired)!=expected:
            raise ValueError("checkpoint ledger/budget mismatch")
        oracle.query(acquired)
        if dict(oracle.purchased_labels())!=saved["purchased"]:
            raise ValueError("checkpoint purchased-label mismatch")
        model.load_state_dict(saved["model"])
        if output.exists() and any(output.iterdir()) and checkpoint.resolve()!=Path(resume).resolve():
            raise ValueError("refuse overwriting a different existing run")
    output.mkdir(parents=True,exist_ok=True)
    public_manifest={"manifest":sim.manifest,"partitions":{str(k):v.tolist() for k,v in parts.items()},
                     "hierarchy":sim.hierarchy.coarse_to_fine,"protocol_hash":protocol_hash}
    manifest_path=output/"protocol_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["protocol_hash"]!=protocol_hash:
        raise ValueError("output manifest mismatch")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(public_manifest,indent=2),encoding="utf-8")
    invocation={**metadata,"method":method,"from_round":next_round,"until_round":until_round,"resume":str(resume) if resume else None,
                "protocol_hash":protocol_hash,"loss_protocol":cfg["loss_protocol"],"config":cfg}
    (output/f"invocation-{uuid4().hex}.json").write_text(json.dumps(invocation,indent=2),encoding="utf-8")
    t,s=cfg["training"],cfg["seeds"]
    for ri in range(next_round,until_round):
        stage=ri//30
        if oracle.spent<targets[stage]:
            ids=initial if stage==0 else query_increment("B0",sim.pool,parts,oracle.purchased_labels(),targets[stage]-oracle.spent,
                    s["query"]+stage*100003,model,sim.hierarchy,cfg["device"],cfg["query_batch_size"])
            acquire_to_target(oracle,ids,targets[stage])
            ledger.append({"stage":stage,"new_ids":ids,"target":targets[stage],"purchased":oracle.spent,
                "client_budget":client_budget_statistics(parts,oracle.purchased_labels(),sim.pool.coarse_labels())})
        if oracle.spent!=targets[stage]:
            raise ValueError("budget stage mismatch")
        clients=build_datasets(method,sim.pool,parts,oracle.purchased_labels(),sim.hierarchy)
        row=train_round(model,clients,sim.hierarchy,method,round_index=ri,clients_per_round=t["clients_per_round"],
            batch_size=t["batch_size"],max_steps=t["max_steps"],learning_rate=t["learning_rate"],momentum=t["momentum"],
            sampling_seed=s["client_sampling"],training_seed=s["training"],device=cfg["device"],
            aggregation=cfg["loss_protocol"],aggregation_counts={cid:len(ids) for cid,ids in parts.items()})
        tick=time.perf_counter()
        row["validation"]=sim.validation.evaluate(model,cfg["device"],cfg["evaluation_batch_size"],diagnostics=True)
        row.update({"evaluation_seconds":time.perf_counter()-tick,"purchased":oracle.spent,"stage":stage,
            "budget_denominator":len(sim.pool),"budget_fraction":[.02,.04,.08][stage],"protocol_hash":protocol_hash,
            "config_hash":metadata["config_hash"],"git_commit":metadata["git_commit"],"seeds":s,
            "source_hash":metadata["source_hash"],"torch":metadata["torch"],
            "hardware":metadata["hardware"],"init_hash":init_hash,"purchased_hash":digest(dict(oracle.purchased_labels()))})
        if cfg["device"]=="cuda":
            row["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()
            row["peak_cuda_reserved_bytes"]=torch.cuda.max_memory_reserved()
        rows.append(row)
        payload={"purpose":"p26_budget_protocol","model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
            "method":method,"next_round":ri+1,"rounds":rows,"ledger":ledger,"purchased":dict(oracle.purchased_labels()),
            "initial_ids":initial,"init_hash":init_hash,"protocol_hash":protocol_hash,"config":cfg,
            "source_hash":metadata["source_hash"],"torch":metadata["torch"]}
        _save_checkpoint(checkpoint,payload)
        (output/"rounds.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
        print(f"{method} round={ri+1} purchased={oracle.spent} fine={row['fine_examples_seen']} coarse={row['coarse_examples_seen']} loss={row['train_loss']:.5f} val_acc={row['validation']['accuracy']:.5f}",flush=True)
    health=health_checks(rows)
    (output/"health.json").write_text(json.dumps(health,indent=2),encoding="utf-8")
    restored=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if not all(torch.equal(v.detach().cpu(),restored["model"][k]) for k,v in model.state_dict().items()):
        raise RuntimeError("checkpoint roundtrip mismatch")
    _save_checkpoint(output/f"checkpoint-round{until_round:03}.pt",restored)
    return {"rounds":rows,"ledger":ledger,"health":health,"checkpoint_verified":True,"checkpoint":str(checkpoint)}


def verify_pair(a,b):
    if len(a)!=len(b) or not a or {a[0]["method"],b[0]["method"]}!={"PS_FINE","PS_COARSE"}:
        raise ValueError("paired arms and same nonempty round prefix required")
    for x,y in zip(a,b):
        for k in ("round","stage","init_hash","purchased_hash","examples_seen","fine_examples_seen","loss_protocol","protocol_hash",
                  "source_hash","torch","config_hash","hardware","learning_rate","momentum"):
            if x[k]!=y[k]:
                raise ValueError("paired mismatch: "+k)
        left=[(c["client"],c["stream_hash"],c["fine_exposure_hash"]) for c in x["clients"]]
        right=[(c["client"],c["stream_hash"],c["fine_exposure_hash"]) for c in y["clients"]]
        if left!=right or x["bn_policy"]!=y["bn_policy"] or x["optimizer_policy"]!=y["optimizer_policy"]:
            raise ValueError("paired stream/BN policy mismatch")
    return {"paired_stream_verified":True,"rounds":len(a),"fine_exposures":sum(x["fine_examples_seen"] for x in a)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/p26_v100.yaml")
    p.add_argument("--method",choices=METHODS,required=True)
    p.add_argument("--until-round",type=int,choices=(5,30,35,60,65,90),required=True)
    p.add_argument("--resume")
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_config(cfg)
    certificate=Path(cfg["output"])/"preflight.json"
    if not certificate.exists():
        raise RuntimeError("Run P2.6 CUDA/data/tests preflight before training")
    certified=json.loads(certificate.read_text())
    metadata=runtime_metadata(cfg)
    if not certified.get("preflight_pass") or any(certified[k]!=metadata[k] for k in ("config_hash","source_hash","torch","hardware","git_commit","training_file_hash")):
        raise RuntimeError("stale/mismatched P2.6 preflight; rerun preflight")
    torch.set_num_threads(cfg["threads"])
    d,s,pc=cfg["dataset"],cfg["seeds"],cfg["partition"]
    sim=prepare_experiment(d["root"],split_seed=s["split"],validation_size=5000,test_samples=0)
    parts=(iid_partition(len(sim.pool),cfg["n_clients"],s["partition"]) if pc["type"]=="iid" else
           coarse_dirichlet_partition(sim.pool.coarse_labels(),cfg["n_clients"],pc["alpha"],s["partition"],pc["min_client_samples"],pc["max_attempts"]))
    result=run_staged(cfg,sim,parts,args.method,Path(cfg["output"])/args.method,until_round=args.until_round,resume=args.resume,metadata=metadata)
    if not result["health"]["engineering_pass"]:
        raise RuntimeError("engineering health gate failed; checkpoint retained for investigation")


if __name__=="__main__":
    main()
