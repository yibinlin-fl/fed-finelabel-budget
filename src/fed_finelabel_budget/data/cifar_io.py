"""Trusted CIFAR-100 bootstrap: ONLY this module decodes hidden fine labels.

The public split holds images+coarse labels. Raw fine labels enter the oracle,
not ClientPublicPool/partition/query APIs. Do not import this bootstrap from
client training or acquisition code. Read only locally sourced trusted pickle.
"""
from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
from .public_pool import PublicPool
from .label_oracle import FineLabelOracle
from .hierarchy import Cifar100Hierarchy


def _split_path(root: str | Path, split: str) -> Path:
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    root = Path(root)
    folder = root / "cifar-100-python" if (root / "cifar-100-python").is_dir() else root
    path = folder / split
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python scripts/download_cifar100.py --root {root}")
    return path


def _trusted_read(root: str | Path, split: str, strict: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with _split_path(root, split).open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")  # trusted official dataset only
    if not isinstance(payload, dict):
        raise ValueError("invalid CIFAR-100 pickle structure")
    pixels = np.asarray(payload[b"data"])
    fine = np.asarray(payload[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(payload[b"coarse_labels"], dtype=np.int64)
    expected = 50000 if split == "train" else 10000
    if pixels.dtype != np.uint8 or pixels.ndim != 2 or pixels.shape[1] != 3072:
        raise ValueError("invalid CIFAR-100 image array")
    if len(pixels) != len(fine) or len(pixels) != len(coarse):
        raise ValueError("inconsistent CIFAR-100 labels")
    if strict and len(pixels) != expected:
        raise ValueError(f"expected {expected} samples for {split}, got {len(pixels)}")
    images = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    return images, coarse, fine


def bootstrap_train(
    root: str | Path, max_queries: int, *, strict: bool = True
) -> tuple[PublicPool, FineLabelOracle, Cifar100Hierarchy]:
    """TRUSTED CONTROLLER ONLY: never pass oracle to acquisition/training code."""
    images, coarse, fine = _trusted_read(root, "train", strict)
    hierarchy = Cifar100Hierarchy.from_label_pairs(fine, coarse)
    pool = PublicPool(images, coarse)
    oracle = FineLabelOracle({i: int(fine[i]) for i in range(len(fine))}, max_queries)
    return pool, oracle, hierarchy


def inspect_dataset(root: str | Path, *, strict: bool = True) -> dict[str, int]:
    """Trusted offline integrity checker, not an algorithm-facing evaluator."""
    tr_images, tr_coarse, tr_fine = _trusted_read(root, "train", strict)
    hierarchy = Cifar100Hierarchy.from_label_pairs(tr_fine, tr_coarse)
    te_images, te_coarse, te_fine = _trusted_read(root, "test", strict)
    hierarchy.verify_pairs(te_fine, te_coarse)
    return {
        "train_samples": len(tr_images),
        "test_samples": len(te_images),
        "coarse_groups": len(hierarchy.coarse_to_fine),
        "fine_classes": len(hierarchy.fine_to_coarse),
    }
