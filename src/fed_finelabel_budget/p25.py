"""Trusted P2.5 diagnostics controller; full fine access is isolated from baselines.

No official test is read, no diagnostic weights leave this artifact namespace.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time
from copy import deepcopy
import numpy as np
import torch
import yaml
from .data.cifar_io import prepare_experiment
from .data.partition import iid_partition
from .eval.offline import OfflineEvaluator
from .eval.metrics import classification_metrics
from .models.resnet_cifar import resnet18_cifar
from .training.dataset import ClientDataset
from .training.losses import MixedSupervisionLoss
from .training.baselines import FineOnlyLoss
from .training.diagnostic import supervised_epoch
from .training.fedavg import train_fedavg
from .training.seeds import seed_everything


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def gradient_audit(hierarchy):
    seed_everything(112)
    z = torch.randn(100, 100, requires_grad=True)
    fine = torch.full((100,), -1, dtype=torch.long)
    fine[:2] = torch.tensor([0, 1])
    coarse = torch.tensor([hierarchy.fine_to_coarse[i % 100] for i in range(100)])
    records = {}
    for mode in ("sample_mean", "group_mean"):
        criterion = MixedSupervisionLoss(hierarchy, aggregation=mode)
        loss = criterion(z, coarse, fine)
        grad, = torch.autograd.grad(loss, z)
        records[mode] = {"loss": float(loss.detach()), "fine_gradient_norm": float(grad[:2].norm()),
                         "coarse_gradient_norm": float(grad[2:].norm()), "finite": bool(torch.isfinite(grad).all())}
    records["fine_norm_ratio"] = records["group_mean"]["fine_gradient_norm"] / records["sample_mean"]["fine_gradient_norm"]
    records["coarse_norm_ratio"] = records["group_mean"]["coarse_gradient_norm"] / records["sample_mean"]["coarse_gradient_norm"]
    return records


def audit_p2(cfg, output):
    """Reevaluate existing checkpoints ONLY on their original validation subset."""
    sim = prepare_experiment(cfg["dataset_root"], split_seed=20261011, train_samples=1000,
                             validation_size=5000, validation_samples=100, test_samples=0)
    truth = np.arange(100)
    coarse = np.array(sim.hierarchy.fine_to_coarse)
    good = torch.full((100, 100), -20.)
    good[torch.arange(100), torch.arange(100)] = 20
    bad = good.roll(1, dims=1)
    result = {"validation_distribution": sim.validation.label_distribution(),
              "hierarchy": sim.hierarchy.coarse_to_fine,
              "metric_controls": {"correct_logits": classification_metrics(truth, good.argmax(1), coarse),
                                  "wrong_logits": classification_metrics(truth, bad.argmax(1), coarse)},
              "gradient_audit": gradient_audit(sim.hierarchy), "baselines": {}}
    full_validation = prepare_experiment(cfg["dataset_root"], split_seed=20261011, train_samples=1000,
                                        validation_size=5000, validation_samples=None, test_samples=0)
    result["full_validation_distribution"] = full_validation.validation.label_distribution()
    del full_validation
    for baseline in ("B0", "B1", "B2", "B3", "B4"):
        path = Path(cfg["p2_artifacts"]) / baseline / "checkpoint.pt"
        if not path.exists():
            result["baselines"][baseline] = {"status": "checkpoint_missing"}
            continue
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = resnet18_cifar().to(cfg["device"])
        model.load_state_dict(saved["model"])
        actual = sim.validation.evaluate(model, cfg["device"], 32, diagnostics=True)
        original = saved["stages"][-1]["validation"]
        assert actual["accuracy"] == original["accuracy"]
        assert actual["macro_f1"] == original["macro_f1"]
        result["baselines"][baseline] = {"validation": actual,
            "fine_examples_seen": sum(c["fine_examples_seen"] for r in saved["rounds"] for c in r["clients"]),
            "coarse_examples_seen": sum(c["coarse_examples_seen"] for r in saved["rounds"] for c in r["clients"]),
            "gradient_steps": sum(r["gradient_steps"] for r in saved["rounds"]),
            "bn_counters": sorted(set(int(v) for k, v in saved["model"].items() if k.endswith("num_batches_tracked"))),
            "training_losses": [r["train_loss"] for r in saved["rounds"]],
            "bn_variance_min": min(float(v.min()) for k,v in saved["model"].items() if k.endswith("running_var")),
            "bn_variance_max": max(float(v.max()) for k,v in saved["model"].items() if k.endswith("running_var")),
            "finite_state": all(bool(torch.isfinite(v).all()) for v in saved["model"].values() if v.is_floating_point())}
        del model
    write_json(output / "p2_audit.json", result)
    return result


def authorize_diagnostic(cfg, samples):
    sim = prepare_experiment(cfg["dataset_root"], split_seed=cfg["split_seed"], train_samples=samples,
                             validation_size=5000, validation_samples=cfg["validation_samples"], test_samples=0)
    oracle = sim.new_oracle(samples)
    oracle.query(list(range(samples)))
    purchased = dict(oracle.purchased_labels())
    ds = ClientDataset(sim.pool, range(samples), purchased, sim.hierarchy)
    images = np.stack([sim.pool.sample(i).image for i in range(samples)])
    training_eval = OfflineEvaluator(images, [purchased[i] for i in range(samples)], sim.pool.coarse_labels())
    assert set(sim.manifest["train_original_ids"]).isdisjoint(sim.manifest["validation_original_ids"])
    assert sim.test is None
    return sim, purchased, ds, training_eval


def audit_bn(cfg, output):
    """Training-only BN counterfactual, never applied to formal baselines."""
    from torch.utils.data import DataLoader
    sim, _purchased, ds, training_eval = authorize_diagnostic(cfg, 256)
    records = {}
    for lr in (.05, .02):
        seed_everything(cfg["init_seed"])
        model = resnet18_cifar().to(cfg["device"])
        initial_conv = model.conv1.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)
        record = supervised_epoch(model, ds, optimizer, seed=cfg["training_seed"], batch_size=16,device=cfg["device"])
        record["eval_train"] = training_eval.evaluate(model,cfg["device"],32,diagnostics=True)
        record["eval_validation"] = sim.validation.evaluate(model,cfg["device"],32,diagnostics=True)
        record["conv1_update_norm"] = float((model.conv1.weight.detach()-initial_conv).norm())
        record["bn_variance_min"] = min(float(v.min()) for k,v in model.state_dict().items() if k.endswith("running_var"))
        record["bn_variance_max"] = max(float(v.max()) for k,v in model.state_dict().items() if k.endswith("running_var"))
        clone = deepcopy(model).train()
        correct = total_ce = 0
        with torch.inference_mode():
            for x,_c,f,_ids in DataLoader(ds,batch_size=16):
                z=clone(x.to(cfg["device"]))
                y=f.to(cfg["device"])
                correct += int((z.argmax(1)==y).sum())
                total_ce += float(torch.nn.functional.cross_entropy(z,y,reduction="sum"))
        record["train_mode_train"] = {"accuracy":correct/len(ds),"fine_ce":total_ce/len(ds)}
        del clone
        clone = deepcopy(model).eval()
        for module in clone.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
                module.momentum=None
                module.train()
        with torch.inference_mode():
            for x,_c,_f,_ids in DataLoader(ds,batch_size=16):
                clone(x.to(cfg["device"]))
        assert all(torch.equal(p,q) for p,q in zip(model.parameters(),clone.parameters()))
        record["training_only_bn_recalibration"] = {"train":training_eval.evaluate(clone,cfg["device"],32,diagnostics=True),
            "validation":sim.validation.evaluate(clone,cfg["device"],32,diagnostics=True),"parameters_unchanged":True}
        records[str(lr)] = record
        del clone,model,optimizer
    write_json(output/"bn_audit.json",records)
    return records


def run_diagnostic(cfg, name, output):
    t = cfg[name]
    sim, purchased, ds, training_eval = authorize_diagnostic(cfg, t["samples"])
    path = output / name
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "authorization.json", {"purpose": "diagnostic_full_fine", "spent": len(purchased),
               "authorization": "User authorized independent full-fine engineering diagnostics; not a baseline budget",
               "manifest": sim.manifest, "config": t})
    seed_everything(cfg["init_seed"])
    model = resnet18_cifar().to(cfg["device"])
    if name != "fedavg":
        optimizer = (torch.optim.Adam(model.parameters(), lr=t["learning_rate"]) if t["optimizer"] == "adam" else
                     torch.optim.SGD(model.parameters(), lr=t["learning_rate"], momentum=t["momentum"]))
    else:
        parts = iid_partition(len(ds), t["clients"], cfg["partition_seed"])
        clients = {cid: ClientDataset(sim.pool, ids, purchased, sim.hierarchy) for cid, ids in parts.items()}
        write_json(path / "partitions.json", {str(k): v.tolist() for k, v in parts.items()})
    start = time.perf_counter()
    history = [{"iteration": 0, "elapsed_seconds": 0., "train": training_eval.evaluate(model, cfg["device"], 32, diagnostics=True),
                "validation": sim.validation.evaluate(model, cfg["device"], 32, diagnostics=True)}]
    for iteration in range(t["iterations"]):
        tick = time.perf_counter()
        if name == "fedavg":
            record = train_fedavg(model, clients, FineOnlyLoss(), rounds=1, start_round=iteration,
                clients_per_round=t["clients"], local_epochs=1, batch_size=t["batch_size"],
                learning_rate=t["learning_rate"], momentum=t["momentum"], max_steps=t["max_steps"],
                sampling_seed=cfg["sampling_seed"], training_seed=cfg["training_seed"], device=cfg["device"], fixed_steps=True)[0]
            examples = sum(c["examples_seen"] for c in record["clients"])
            record["train_loss"] = sum(c["mean_loss"] * c["examples_seen"] for c in record["clients"]) / examples
        else:
            record = supervised_epoch(model, ds, optimizer, seed=cfg["training_seed"] + iteration,
                                      batch_size=t["batch_size"], device=cfg["device"])
        if cfg["device"] == "cuda":
            torch.cuda.synchronize()
        record.update({"iteration": iteration + 1, "training_seconds": time.perf_counter() - tick,
                       "train": training_eval.evaluate(model, cfg["device"], 32, diagnostics=True),
                       "validation": sim.validation.evaluate(model, cfg["device"], 32, diagnostics=True),
                       "elapsed_seconds": time.perf_counter() - start})
        history.append(record)
        write_json(path / "history.json", history)
        print(f"{name} {iteration+1} loss={record['train_loss']:.4f} train_acc={record['train']['accuracy']:.4f} val_acc={record['validation']['accuracy']:.4f}", flush=True)
        # Only authorized TRAIN accuracy stops overfit; validation never controls training.
        if name == "overfit" and record["train"]["accuracy"] >= t["target_accuracy"]:
            break
    payload = {"purpose": "diagnostic_full_fine", "model": model.state_dict(), "config": cfg,
               "name": name, "iteration": len(history)-1, "purchased": purchased,
               "optimizer": optimizer.state_dict() if name != "fedavg" else None}
    torch.save(payload, path / "checkpoint.pt")
    restored = resnet18_cifar().to(cfg["device"])
    state = torch.load(path / "checkpoint.pt", map_location=cfg["device"], weights_only=True)
    restored.load_state_dict(state["model"])
    restored_metrics = training_eval.evaluate(restored, cfg["device"], 32, diagnostics=True)
    assert restored_metrics == history[-1]["train"]
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    summary = {"purpose": "diagnostic_full_fine", "samples": len(ds), "iterations": len(history)-1,
               "initial_train": history[0]["train"], "final_train": history[-1]["train"],
               "initial_validation": history[0]["validation"], "final_validation": history[-1]["validation"],
               "seconds": time.perf_counter()-start, "checkpoint_exact": True,
               "bn_counters": sorted(set(int(v) for k,v in model.state_dict().items() if k.endswith("num_batches_tracked"))),
               "target_met": history[-1]["train"]["accuracy"] >= t["target_accuracy"] if name == "overfit" else None}
    write_json(path / "summary.json", summary)
    del model, restored
    return summary


def validate_diagnostic_config(cfg):
    if cfg["device"] not in ("cpu", "cuda") or not 1 <= cfg["threads"] <= 8:
        raise ValueError("invalid diagnostic device/threads")
    if not 1 <= cfg["validation_samples"] <= 5000:
        raise ValueError("invalid diagnostic validation size")
    for name in ("overfit", "central", "fedavg"):
        t = cfg[name]
        if not 2 <= t["samples"] <= 1024 or not 1 <= t["iterations"] <= 100 or not 2 <= t["batch_size"] <= 64:
            raise ValueError("local diagnostic limits exceeded")
        if not np.isfinite(t["learning_rate"]) or t["learning_rate"] <= 0 or not 0 <= t["momentum"] < 1:
            raise ValueError("invalid diagnostic optimizer")
    if cfg["overfit"]["optimizer"] not in ("adam", "sgd") or cfg["central"]["optimizer"] not in ("adam", "sgd"):
        raise ValueError("invalid diagnostic optimizer")
    if not .95 <= cfg["overfit"]["target_accuracy"] <= 1:
        raise ValueError("overfit target must be at least 95%")
    f = cfg["fedavg"]
    if not 1 <= f["clients"] <= f["samples"] or not 1 <= f["max_steps"] <= 50:
        raise ValueError("invalid diagnostic federation")
    if cfg["device"] == "cpu" and (cfg["overfit"]["samples"] > 128 or cfg["overfit"]["iterations"] > 40 or
            cfg["central"]["samples"] > 256 or cfg["central"]["iterations"] > 10 or
            f["samples"] > 128 or f["iterations"] > 10 or f["max_steps"] > 4 or
            f["clients"] * f["iterations"] * f["max_steps"] * f["batch_size"] > 5120 or
            cfg["validation_samples"] > 500):
        raise ValueError("CPU diagnostics limited to tiny authorized subsets; full training prohibited")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p25_local_diagnostic.yaml")
    parser.add_argument("--only", choices=["audit", "bn", "overfit", "central", "fedavg"])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_diagnostic_config(cfg)
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: select the explicit bounded CPU diagnostic config")
    torch.set_num_threads(cfg["threads"])
    seed_everything(cfg["init_seed"])
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", {"config": cfg, "python": platform.python_version(), "torch": str(torch.__version__),
        "hardware": torch.cuda.get_device_name() if cfg["device"] == "cuda" else platform.processor(), "cuda": torch.version.cuda,
        "memory_bytes": torch.cuda.get_device_properties(0).total_memory if cfg["device"] == "cuda" else None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())})
    if args.only in (None, "audit"):
        audit_p2(cfg, output)
    if args.only in (None, "bn"):
        audit_bn(cfg, output)
    for name in ("overfit", "central", "fedavg"):
        if args.only in (None, name):
            run_diagnostic(cfg, name, output)


if __name__ == "__main__":
    main()
