"""Exact global annotation counts and coarse-only random initial acquisition."""
from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR
import numpy as np

def fraction_to_count(n_samples: int, fraction: float | str) -> int:
    if n_samples < 0:
        raise ValueError("negative sample count")
    f = Decimal(str(fraction))
    if not Decimal(0) <= f <= Decimal(1):
        raise ValueError("fraction outside [0,1]")
    return int((n_samples * f).to_integral_value(rounding=ROUND_FLOOR))

def cumulative_targets(n_samples: int, initial: float, additional: list[float]) -> list[int]:
    fractions = [Decimal(str(v)) for v in [initial, *additional]]
    if any(f < 0 for f in fractions) or sum(fractions) > 1:
        raise ValueError("invalid schedule")
    return [fraction_to_count(n_samples, str(sum(fractions[:k]))) for k in range(1, len(fractions) + 1)]

def initial_random_ids(partitions: dict[int, np.ndarray], fraction: float, seed: int) -> list[int]:
    if not partitions:
        raise ValueError("no clients")
    rng = np.random.default_rng(seed)
    total = sum(map(len, partitions.values()))
    target = fraction_to_count(total, fraction)
    frac = Decimal(str(fraction))
    quotas: dict[int, int] = {}
    remainders = []
    for cid, ids in sorted(partitions.items()):
        exact = len(ids) * frac
        base = int(exact.to_integral_value(rounding=ROUND_FLOOR))
        quotas[cid] = base
        remainders.append((exact - base, cid))
    need = target - sum(quotas.values())
    if not 0 <= need <= len(remainders):
        raise AssertionError("quota remainder error")
    ties = {cid: float(rng.random()) for cid in partitions}
    for _frac, cid in sorted(remainders, key=lambda t: (-t[0], ties[t[1]]))[:need]:
        quotas[cid] += 1
    chosen = []
    for cid, ids in sorted(partitions.items()):
        chosen.extend(int(i) for i in rng.choice(ids, size=quotas[cid], replace=False))
    assert len(chosen) == len(set(chosen)) == target
    return chosen

def additional_random_ids(partitions: dict[int, np.ndarray], purchased_ids: set[int], count: int, seed: int) -> list[int]:
    if count < 0:
        raise ValueError("negative acquisition count")
    eligible = sorted({int(i) for ids in partitions.values() for i in ids} - purchased_ids)
    if count > len(eligible):
        raise ValueError("not enough eligible samples")
    return [int(i) for i in np.random.default_rng(seed).choice(eligible, size=count, replace=False)]
