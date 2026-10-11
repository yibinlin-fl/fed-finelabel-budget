"""Trusted simulation controller. Algorithms never receive this module's data bundle."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
import yaml
from .data.cifar_io import prepare_experiment
from .data.budget import cumulative_targets, initial_random_ids
from .data.partition import iid_partition, coarse_dirichlet_partition, validate_partition
from .data.public_pool import ClientPublicView
from .eval.metrics import client_budget_statistics
from .models.resnet_cifar import resnet18_cifar
from .query.strategies import BASELINES, allocate_quotas, local_query
from .training.baselines import build_clients
from .training.fedavg import train_fedavg
from .training.seeds import seed_everything


def validate_config(cfg):
    t, d = cfg["training"], cfg["dataset"]
    if cfg["stage_model_policy"] != "continue":
        raise ValueError("P2 implements consistent continuation between stages only")
    if not cfg["baselines"] or len(set(cfg["baselines"])) != len(cfg["baselines"]) or not set(cfg["baselines"]) <= set(BASELINES):
        raise ValueError("invalid baselines")
    if cfg["n_clients"] < 1 or not 1 <= t["clients_per_round"] <= cfg["n_clients"]:
        raise ValueError("invalid client configuration")
    if min(t["rounds_per_stage"], t["batch_size"], t["max_steps"]) < 1 or t["local_epochs"] != 1:
        raise ValueError("fixed-step protocol requires local_epochs=1 and positive limits")
    if not np.isfinite(t["learning_rate"]) or t["learning_rate"] <= 0:
        raise ValueError("invalid learning rate")
    if cfg["device"] not in ("cpu", "cuda") or not 1 <= cfg["threads"] <= 8:
        raise ValueError("invalid device/threads")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use the bounded CPU pilot config")
    if cfg["budget"] != {"initial_fraction": .02, "additional_fractions": [.02, .04]}:
        raise ValueError("P2 requires cumulative 2%, 4%, 8% budget")
    if cfg["device"] == "cpu":
        work = len(cfg["baselines"]) * 3 * t["rounds_per_stage"] * t["clients_per_round"] * t["max_steps"] * t["batch_size"]
        if (d["train_samples"] is None or not 100 <= d["train_samples"] <= 1000 or
            d["validation_samples"] is None or not 1 <= d["validation_samples"] <= 200 or
            d["test_samples"] is None or not 0 <= d["test_samples"] <= 200 or work > 1200):
            raise ValueError("CPU full experiment prohibited: use bounded subsets and <=1200 gradient examples")
    if cfg["query_batch_size"] < 1 or cfg["evaluation_batch_size"] < 1:
        raise ValueError("batch sizes must be positive")


def acquire_to_target(oracle, request, target):
    """Reject duplicates/overspend before oracle; repeat replay is controller-only."""
    if len(request) != len(set(request)) or set(request) & set(oracle.purchased_labels()):
        raise ValueError("query must contain distinct unpurchased IDs")
    if len(request) != target - oracle.spent:
        raise ValueError("query does not match current cumulative stage cap")
    oracle.query(request)
    if oracle.spent != target:
        raise AssertionError("budget accounting mismatch")


def query_increment(strategy, pool, parts, purchased, count, seed, model, hierarchy, device, batch_size):
    capacities = {cid: sum(int(i) not in purchased for i in ids) for cid, ids in parts.items()}
    quotas = allocate_quotas(capacities, count)
    selected = []
    for cid, ids in sorted(parts.items()):
        local_purchased = {int(i): purchased[int(i)] for i in ids if int(i) in purchased}
        local_public = ClientPublicView(pool, ids)
        selected.extend(local_query(strategy, local_public, ids, local_purchased, quotas[cid], seed + cid,
                                    model, hierarchy, device, batch_size))
    if len(selected) != count or len(set(selected)) != count:
        raise AssertionError("invalid query strategy response")
    return selected


def _protocol_hash(cfg, manifest, parts):
    # Output folder/baseline list are execution routing, not a baseline protocol.
    protocol = {k: v for k, v in cfg.items() if k not in ("output", "baselines")}
    protocol["manifest"] = manifest
    protocol["partitions"] = {str(k): v.tolist() for k, v in parts.items()}
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def _save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    # Single explicit file replacement; never directory cleanup.
    os.replace(temporary, path)


def run_baseline(cfg, simulation, parts, strategy, output, *, resume=None, stop_after_rounds=None,
                 model_factory=resnet18_cifar):
    pool, hierarchy = simulation.pool, simulation.hierarchy
    t, seeds = cfg["training"], cfg["seeds"]
    targets = cumulative_targets(len(pool), .02, [.02, .04])
    initial = initial_random_ids(parts, .02, seeds["query"])
    protocol_hash = _protocol_hash(cfg, simulation.manifest, parts)
    seed_everything(seeds["init"])
    model = model_factory().to(cfg["device"])
    oracle = simulation.new_oracle(targets[-1])
    rounds, stages, ledger = [], [], []
    next_round = 0
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["protocol_hash"] != protocol_hash or saved["strategy"] != strategy:
            raise ValueError("resume protocol/strategy mismatch")
        if saved["initial_ids"] != initial:
            raise ValueError("resume initial ID mismatch")
        # Only replay purchased IDs; hidden labels never persist in checkpoint.
        oracle.query(list(saved["purchased"]))
        if dict(oracle.purchased_labels()) != saved["purchased"]:
            raise ValueError("resume purchased labels mismatch")
        model.load_state_dict(saved["model"])
        rounds, stages, ledger = saved["rounds"], saved["stages"], saved["ledger"]
        next_round = saved["next_round"]
        if not 0 <= next_round <= len(targets) * t["rounds_per_stage"]:
            raise ValueError("invalid resume round")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    if resume is not None:
        _save_checkpoint(checkpoint, saved)
    invocation_rounds = 0
    for stage, target in enumerate(targets):
        stage_end = (stage + 1) * t["rounds_per_stage"]
        if next_round >= stage_end:
            continue
        query_time = 0.0
        if oracle.spent < target:
            start = time.perf_counter()
            ids = initial if stage == 0 else query_increment(strategy, pool, parts, oracle.purchased_labels(),
                target - oracle.spent, seeds["query"] + stage * 100003, model, hierarchy,
                cfg["device"], cfg["query_batch_size"])
            acquire_to_target(oracle, ids, target)
            query_time = time.perf_counter() - start
            ledger.append({"stage": stage, "target": target, "new_ids": ids, "new_count": len(ids),
                           "client_budget": client_budget_statistics(parts, oracle.purchased_labels(), pool.coarse_labels()),
                           "query_seconds": query_time})
        if oracle.spent != target:
            raise ValueError("checkpoint budget exceeds current stage")
        clients, criterion = build_clients(strategy, pool, parts, oracle.purchased_labels(), hierarchy)
        if any(not len(ds) for ds in clients.values()):
            raise ValueError("fine-only client has no labels: choose feasible public min_client_samples >=50")
        for round_index in range(next_round, stage_end):
            before = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
            start = time.perf_counter()
            history = train_fedavg(model, clients, criterion, rounds=1, start_round=round_index,
                clients_per_round=t["clients_per_round"], local_epochs=1, batch_size=t["batch_size"],
                learning_rate=t["learning_rate"], max_steps=t["max_steps"],
                sampling_seed=seeds["client_sampling"], training_seed=seeds["training"],
                device=cfg["device"], fixed_steps=True, aggregation_counts={cid: len(ids) for cid, ids in parts.items()})[0]
            if cfg["device"] == "cuda":
                torch.cuda.synchronize()
            train_time = time.perf_counter() - start
            changed = sum(not torch.equal(before[k], v.detach().cpu()) for k, v in model.named_parameters())
            if changed == 0 or any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
                raise RuntimeError("model did not update or has nonfinite state")
            validation = simulation.validation.evaluate(model, cfg["device"], cfg["evaluation_batch_size"])
            examples = sum(c["examples_seen"] for c in history["clients"])
            history.update({"strategy": strategy, "stage": stage, "budget_fraction": [.02, .04, .08][stage],
                "purchased": oracle.spent, "budget_denominator": len(pool),
                "actual_sampled_examples": examples, "gradient_steps": sum(c["steps"] for c in history["clients"]),
                "train_loss": sum(c["mean_loss"] * c["examples_seen"] for c in history["clients"]) / examples,
                "training_seconds": train_time, "changed_parameter_tensors": changed,
                "validation": validation, "model_policy": "continue", "communication_rounds": round_index + 1,
                "seeds": seeds, "protocol_hash": protocol_hash})
            rounds.append(history)
            next_round = round_index + 1
            if next_round == stage_end:
                stages.append({"stage": stage, "budget": target, "validation": validation,
                               "client_budget": client_budget_statistics(parts, oracle.purchased_labels(), pool.coarse_labels())})
            payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "purchased": dict(oracle.purchased_labels()), "initial_ids": initial,
                "next_round": next_round, "rounds": rounds, "stages": stages, "ledger": ledger,
                "protocol_hash": protocol_hash, "strategy": strategy, "config": cfg,
                "hierarchy": hierarchy.fine_to_coarse}
            _save_checkpoint(checkpoint, payload)
            (output / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rounds), encoding="utf-8")
            print(f"{strategy} round={next_round} budget={oracle.spent}/{len(pool)} loss={history['train_loss']:.6f} val_acc={validation['accuracy']:.4f}", flush=True)
            invocation_rounds += 1
            if stop_after_rounds is not None and invocation_rounds >= stop_after_rounds:
                return {"status": "PAUSED_AT_ROUND_BOUNDARY", "checkpoint": str(checkpoint), "rounds": rounds}
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_ok = all(torch.equal(v.detach().cpu(), restored["model"][k]) for k, v in model.state_dict().items())
    if not checkpoint_ok:
        raise AssertionError("checkpoint roundtrip mismatch")
    test_metrics = simulation.test.evaluate(model, cfg["device"], cfg["evaluation_batch_size"]) if simulation.test else None
    result = {"status": "PASS", "strategy": strategy, "initial_ids": initial,
              "targets": targets, "stages": stages, "rounds": rounds, "ledger": ledger,
              "checkpoint_verified": checkpoint_ok, "test_final_only": test_metrics,
              "protocol_hash": protocol_hash}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rounds), encoding="utf-8")
    return result


def run_pilot(cfg, *, resume=None):
    validate_config(cfg)
    if resume is not None and len(cfg["baselines"]) != 1:
        raise ValueError("resume requires exactly one --baseline")
    torch.set_num_threads(cfg["threads"])
    d, seeds, pcfg = cfg["dataset"], cfg["seeds"], cfg["partition"]
    simulation = prepare_experiment(d["root"], split_seed=seeds["split"],
        validation_size=d["validation_size"], train_samples=d["train_samples"],
        validation_samples=d["validation_samples"], test_samples=d["test_samples"])
    parts = (iid_partition(len(simulation.pool), cfg["n_clients"], seeds["partition"]) if pcfg["type"] == "iid"
             else coarse_dirichlet_partition(simulation.pool.coarse_labels(), cfg["n_clients"], pcfg["alpha"],
                    seeds["partition"], pcfg["min_client_samples"], pcfg["max_attempts"]))
    if pcfg["type"] not in ("iid", "dirichlet_coarse"):
        raise ValueError("invalid partition type")
    validate_partition(parts, len(simulation.pool))
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({**simulation.manifest,
        "partitions": {str(k): v.tolist() for k, v in parts.items()},
        "hierarchy": simulation.hierarchy.coarse_to_fine}, indent=2), encoding="utf-8")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {"config": cfg, "git_commit": git_commit,
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results = [run_baseline(cfg, simulation, parts, baseline, output / baseline, resume=resume) for baseline in cfg["baselines"]]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["strategy", "stage", "budget_fraction", "purchased", "communication_rounds", "train_loss",
                  "accuracy", "macro_f1", "actual_sampled_examples", "gradient_steps", "training_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for row in result["rounds"]:
                writer.writerow({**{k: row[k] for k in fields if k not in ("accuracy", "macro_f1")},
                                 "accuracy": row["validation"]["accuracy"], "macro_f1": row["validation"]["macro_f1"]})
    return results


def main():
    parser = argparse.ArgumentParser(description="P2 fixed-budget baselines, bounded CPU or explicit CUDA config")
    parser.add_argument("--config", default="configs/p2_cpu_pilot.yaml")
    parser.add_argument("--baseline", choices=BASELINES)
    parser.add_argument("--resume")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.baseline:
        cfg["baselines"] = [args.baseline]
    if args.output:
        cfg["output"] = args.output
    run_pilot(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
