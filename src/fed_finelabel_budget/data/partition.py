"""Seeded partitions using sample IDs or public coarse labels (never fine)."""
from __future__ import annotations
import numpy as np

def _validate(n_samples: int, n_clients: int) -> None:
    if not n_samples >= n_clients >= 1:
        raise ValueError("require n_samples >= n_clients >= 1")

def validate_partition(parts: dict[int, np.ndarray], n_samples: int) -> None:
    all_ids = np.concatenate([np.asarray(parts[k], dtype=np.int64) for k in sorted(parts)])
    if len(all_ids) != n_samples or not np.array_equal(np.sort(all_ids), np.arange(n_samples)):
        raise ValueError("partition must contain each sample ID exactly once")

def iid_partition(n_samples: int, n_clients: int, seed: int) -> dict[int, np.ndarray]:
    _validate(n_samples, n_clients)
    rng = np.random.default_rng(seed)
    out = {i: a.copy() for i, a in enumerate(np.array_split(rng.permutation(n_samples), n_clients))}
    validate_partition(out, n_samples)
    return out

def coarse_dirichlet_partition(
    coarse_labels: np.ndarray, n_clients: int, alpha: float, seed: int,
    min_client_samples: int = 1, max_attempts: int = 1000
) -> dict[int, np.ndarray]:
    coarse = np.asarray(coarse_labels, dtype=np.int64)
    _validate(len(coarse), n_clients)
    if coarse.ndim != 1 or np.any((coarse < 0) | (coarse >= 20)):
        raise ValueError("coarse labels must be 1-D within 0..19")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive finite")
    if min_client_samples < 1 or n_clients * min_client_samples > len(coarse):
        raise ValueError("infeasible minimum client size")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    rng = np.random.default_rng(seed)
    for _attempt in range(max_attempts):
        bins: list[list[int]] = [[] for _ in range(n_clients)]
        for c in np.unique(coarse):
            ids = rng.permutation(np.flatnonzero(coarse == c))
            weights = rng.dirichlet(np.full(n_clients, alpha))
            counts = rng.multinomial(len(ids), weights)
            pos = 0
            for j, count in enumerate(counts):
                bins[j].extend(map(int, ids[pos: pos + count]))
                pos += int(count)
        if min(map(len, bins)) >= min_client_samples:
            parts = {j: rng.permutation(ids).astype(np.int64) for j, ids in enumerate(bins)}
            validate_partition(parts, len(coarse))
            return parts
    raise RuntimeError("minimum sample requirement not achieved; lower min_client_samples")
