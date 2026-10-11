"""P0-only offline smoke check and official data integrity commands."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from .data.partition import iid_partition, coarse_dirichlet_partition
from .data.label_oracle import FineLabelOracle
from .data.budget import initial_random_ids, additional_random_ids, cumulative_targets
from .data.cifar_io import inspect_dataset
from .data.hierarchy import Cifar100Hierarchy
from .data.public_pool import PublicPool


def run_preflight(config: dict) -> dict:
    """Synthetic protocol check, NOT a training experiment or accuracy result."""
    seed = int(config["seed"])
    n_clients = int(config["n_clients"])
    n_samples = 1000  # all twenty coarse groups, all hundred synthetic fine IDs
    # Synthetic data is deliberately not a meaningful training corpus.
    fine = np.arange(n_samples) % 100
    coarse = ((fine * 7) % 20).astype(np.int64)  # synthetic non-contiguous 20x5 mapping
    fake_images = np.zeros((n_samples, 32, 32, 3), dtype=np.uint8)
    pool = PublicPool(fake_images, coarse)
    hierarchy = Cifar100Hierarchy.from_label_pairs(fine, coarse)
    assert len(hierarchy.coarse_to_fine) == 20
    pcfg = config["partition"]
    if pcfg["type"] == "iid":
        parts = iid_partition(n_samples, n_clients, seed)
    elif pcfg["type"] == "dirichlet_coarse":
        parts = coarse_dirichlet_partition(pool.coarse_labels(), n_clients, float(pcfg["alpha"]), seed,
            min_client_samples=int(pcfg["min_client_samples"]), max_attempts=int(pcfg["max_attempts"]))
    else:
        raise ValueError("partition type must be iid or dirichlet_coarse")
    b = config["budget"]
    targets = cumulative_targets(n_samples, b["initial_fraction"], b["additional_fractions"])
    oracle = FineLabelOracle({i: int(fine[i]) for i in range(n_samples)}, targets[-1])
    initial = initial_random_ids(parts, b["initial_fraction"], seed + 101)
    oracle.query(initial)
    obtained = [oracle.spent]
    if oracle.spent != targets[0]:
        raise AssertionError("initial count != budget target")
    if initial:
        oracle.query([initial[0], initial[0]])
        assert oracle.spent == targets[0], "repeated query incorrectly charged"
    for k, target in enumerate(targets[1:], 1):
        request = additional_random_ids(parts, set(oracle.purchased_labels()), target - oracle.spent, seed + 101 + k)
        oracle.query(request)
        assert oracle.spent == target
        obtained.append(oracle.spent)
    return {
        "status": "PASS",
        "mode": "synthetic_P0_integrity_only_NO_training",
        "synthetic_samples": len(pool),
        "clients": len(parts),
        "client_min_size": min(map(len, parts.values())),
        "fine_query_targets": targets,
        "fine_queries_observed": obtained,
        "remaining_queries": oracle.remaining,
        "dataset_downloaded_or_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 protocol and data checks (no model training)")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="run fully offline synthetic integrity test")
    preflight.add_argument("--config", default="configs/p0.yaml")
    check = sub.add_parser("data-check", help="validate actual official CIFAR-100 files")
    check.add_argument("--root", default="data/cifar100")
    args = parser.parse_args()
    if args.command == "preflight":
        with Path(args.config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        report = run_preflight(config)
    else:
        report = {"status": "PASS", "mode": "official_CIFAR100_integrity", **inspect_dataset(args.root)}
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
