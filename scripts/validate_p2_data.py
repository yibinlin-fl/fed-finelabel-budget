"""Trusted real-data diagnostic; no query strategy or evaluation-based tuning."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from fed_finelabel_budget.data.cifar_io import bootstrap_train, inspect_dataset
from fed_finelabel_budget.data.partition import iid_partition, coarse_dirichlet_partition, validate_partition
from fed_finelabel_budget.models.resnet_cifar import resnet18_cifar
from fed_finelabel_budget.training.dataset import ClientDataset
from fed_finelabel_budget.training.losses import MixedSupervisionLoss
from fed_finelabel_budget.training.seeds import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/cifar100")
    parser.add_argument("--output", default="results/p2_data_validation.json")
    args = parser.parse_args()
    summary = inspect_dataset(args.root)
    pool, oracle, hierarchy = bootstrap_train(args.root, 8)
    partitions = {}
    for name in ("iid", "dirichlet_0.3", "dirichlet_1.0"):
        def make():
            return (iid_partition(len(pool), 10, 101) if name == "iid" else
                    coarse_dirichlet_partition(pool.coarse_labels(), 10, float(name.split("_")[1]), 101,
                                               min_client_samples=50))
        parts, replay = make(), make()
        validate_partition(parts, len(pool))
        assert all((parts[c] == replay[c]).all() for c in parts)
        partitions[name] = {"sizes": [len(parts[c]) for c in sorted(parts)], "exactly_once": True, "replayed": True}
    oracle.query(list(range(8)))
    dataset = ClientDataset(pool, range(16), oracle.purchased_labels(), hierarchy)
    torch.set_num_threads(1)
    seed_everything(102)
    model = resnet18_cifar()
    images, coarse, fine, _ = next(iter(DataLoader(dataset, batch_size=16)))
    losses = MixedSupervisionLoss(hierarchy)(model(images), coarse, fine, "none")
    losses.mean().backward()
    assert torch.isfinite(losses).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    summary.update({"status": "PASS", "hierarchy": hierarchy.coarse_to_fine,
                    "noncontiguous_mapping": any(c != f // 5 for f, c in enumerate(hierarchy.fine_to_coarse)),
                    "partitions": partitions, "fine_ce_mean": float(losses[:8].detach().mean()),
                    "coarse_marginal_mean": float(losses[8:].detach().mean()),
                    "finite_gradients": True, "purchased": oracle.spent,
                    "mode": "real_data_integrity_and_gradient_diagnostic_NO_accuracy_claim"})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
