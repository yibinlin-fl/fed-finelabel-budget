"""Derive CIFAR-100 hierarchy from labels, never from assumed contiguous IDs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class Cifar100Hierarchy:
    fine_to_coarse: tuple[int, ...]
    coarse_to_fine: tuple[tuple[int, ...], ...]

    @classmethod
    def from_label_pairs(cls, fine_labels: Iterable[int], coarse_labels: Iterable[int]) -> "Cifar100Hierarchy":
        fine, coarse = list(map(int, fine_labels)), list(map(int, coarse_labels))
        if len(fine) != len(coarse) or not fine:
            raise ValueError("fine/coarse label arrays must have equal nonzero length")
        mapping: dict[int, int] = {}
        for f, c in zip(fine, coarse):
            if not 0 <= f < 100 or not 0 <= c < 20:
                raise ValueError(f"out-of-range CIFAR-100 label pair: {(f, c)}")
            if f in mapping and mapping[f] != c:
                raise ValueError(f"fine label {f} has conflicting coarse labels")
            mapping[f] = c
        if len(mapping) != 100:
            raise ValueError(f"expected 100 distinct fine labels, got {len(mapping)}")
        groups: list[list[int]] = [[] for _ in range(20)]
        for f in range(100):
            groups[mapping[f]].append(f)
        if any(len(group) != 5 for group in groups):
            raise ValueError("official CIFAR-100 hierarchy must have 20 groups of 5 fine labels")
        return cls(tuple(mapping[f] for f in range(100)), tuple(tuple(sorted(g)) for g in groups))

    def fine_ids(self, coarse_label: int) -> tuple[int, ...]:
        if not 0 <= coarse_label < 20:
            raise ValueError("coarse label must be in 0..19")
        return self.coarse_to_fine[coarse_label]

    def verify_pairs(self, fine_labels: Iterable[int], coarse_labels: Iterable[int]) -> None:
        for fine, coarse in zip(fine_labels, coarse_labels, strict=True):
            if not 0 <= int(fine) < 100 or self.fine_to_coarse[int(fine)] != int(coarse):
                raise ValueError("observed pair does not match train hierarchy")
