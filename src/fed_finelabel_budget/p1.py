"""Trusted P1 controller. Fine truth stays here; training receives snapshots."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
import torchvision
import yaml
from .data.budget import cumulative_targets, initial_random_ids
from .data.cifar_io import bootstrap_train
from .data.hierarchy import Cifar100Hierarchy
from .data.label_oracle import FineLabelOracle
from .data.partition import iid_partition, coarse_dirichlet_partition, validate_partition
from .data.public_pool import PublicPool
from .models.resnet_cifar import resnet18_cifar
from .training.dataset import ClientDataset
from .training.fedavg import train_fedavg
from .training.losses import MixedSupervisionLoss
from .training.seeds import seed_everything


def run_smoke(config):
    seeds, data, training = config["seeds"], config["dataset"], config["training"]
    if config["mode"] not in ("synthetic", "cifar100"):
        raise ValueError("mode must be synthetic or cifar100")
    # Hard smoke limits: this entry point must never start a full experiment.
    n = int(data["samples"] if config["mode"] == "synthetic" else data["real_smoke_samples"])
    if not 100 <= n <= 300 or not 2 <= config["n_clients"] <= 3:
        raise ValueError("smoke requires 100..300 samples and 2..3 clients")
    if not 1 <= training["rounds"] <= 2 or not 1 <= training["max_steps"] <= 2:
        raise ValueError("smoke requires 1..2 rounds and 1..2 steps per client")
    if not 1 <= config["threads"] <= 4:
        raise ValueError("smoke threads must be 1..4")
    torch.set_num_threads(config["threads"])
    budget = config["budget"]
    targets = cumulative_targets(n, budget["initial_fraction"], budget["additional_fractions"])
    if config["mode"] == "synthetic":
        fine = np.arange(n) % 100
        coarse = (fine * 7) % 20
        hierarchy = Cifar100Hierarchy.from_label_pairs(fine, coarse)
        images = np.random.default_rng(seeds["data"]).integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
        pool = PublicPool(images, coarse)
        oracle = FineLabelOracle(dict(enumerate(map(int, fine))), targets[-1])
    else:
        full_pool, full_oracle, hierarchy = bootstrap_train(data["root"], targets[-1])
        # Public ID subset, independent of fine labels. Keep original IDs for oracle.
        subset = np.random.default_rng(seeds["data"]).choice(len(full_pool), n, replace=False)
        samples = [full_pool.sample(int(i)) for i in subset]
        pool = PublicPool(np.stack([s.image for s in samples]), np.array([s.coarse_label for s in samples]))
        oracle = full_oracle
    pcfg = config["partition"]
    if pcfg["type"] == "iid":
        parts = iid_partition(n, config["n_clients"], seeds["partition"])
    elif pcfg["type"] == "dirichlet_coarse":
        parts = coarse_dirichlet_partition(pool.coarse_labels(), config["n_clients"], pcfg["alpha"],
                                          seeds["partition"], pcfg["min_client_samples"], pcfg["max_attempts"])
    else:
        raise ValueError("invalid partition type")
    validate_partition(parts, n)
    ids = initial_random_ids(parts, budget["initial_fraction"], seeds["query"])
    if config["mode"] == "synthetic":
        oracle.query(ids)
        purchased = oracle.purchased_labels()
    else:
        replies = oracle.query([int(subset[i]) for i in ids])
        purchased = {i: replies[int(subset[i])] for i in ids}
    if oracle.spent != targets[0]:
        raise AssertionError("initial budget mismatch")
    clients = {cid: ClientDataset(pool, indices, purchased, hierarchy) for cid, indices in parts.items()}
    seed_everything(seeds["init"])
    model = resnet18_cifar()
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    start = time.perf_counter()
    history = train_fedavg(model, clients, MixedSupervisionLoss(hierarchy), **training,
                           sampling_seed=seeds["client_sampling"], training_seed=seeds["training"],
                           device=config["device"])
    elapsed = time.perf_counter() - start
    changed = [k for k, p in model.named_parameters() if not torch.equal(before[k], p.detach().cpu())]
    if not changed:
        raise AssertionError("no model parameters changed")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    summary = {"status": "PASS", "mode": config["mode"] + "_P1_smoke_only",
               "config": config, "git_commit": git_commit, "git_dirty": git_dirty,
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "torchvision": torchvision.__version__, "numpy": np.__version__,
                               "platform": platform.platform()},
               "samples": n, "budget_targets": targets, "fine_queries_spent": oracle.spent,
               "fine_labels_available": len(purchased), "remaining_queries": oracle.remaining,
               "changed_parameter_tensors": len(changed), "elapsed_training_seconds": elapsed,
               "test_or_validation_read": False, "history": history}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    path = output / "checkpoint.pt"
    torch.save({"model": state, "config": config, "history": history,
                "hierarchy": hierarchy.fine_to_coarse, "git_commit": git_commit}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not all(torch.equal(v, loaded["model"][k]) for k, v in state.items()):
        raise AssertionError("checkpoint roundtrip mismatch")
    summary["checkpoint_verified"] = True
    (output / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Bounded P1 smoke training; no P2 strategies")
    parser.add_argument("--config", default="configs/p1_smoke.yaml")
    parser.add_argument("--mode", choices=["synthetic", "cifar100"])
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("mode", "device", "output"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    print(json.dumps(run_smoke(config), indent=2))


if __name__ == "__main__":
    main()
