"""Trusted fine-label simulator with atomic query accounting.

This API protects against accidental leakage, NOT malicious Python introspection
or reading the original CIFAR pickle files. Keep oracle in trusted controller;
never give it or the raw files to selection/training modules.
"""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from types import MappingProxyType

class FineLabelOracle:
    __slots__ = ("__truth", "__purchased", "__max_queries")

    def __init__(self, hidden_truth: Mapping[int, int], max_queries: int):
        truth = {int(i): int(y) for i, y in hidden_truth.items()}
        if not truth or min(truth) < 0 or any(not 0 <= y < 100 for y in truth.values()):
            raise ValueError("hidden_truth IDs must be nonnegative and labels within 0..99")
        if type(max_queries) is not int or not 0 <= max_queries <= len(truth):
            raise ValueError("invalid maximum query count")
        self.__truth = truth
        self.__purchased: dict[int, int] = {}
        self.__max_queries = max_queries

    @property
    def spent(self) -> int:
        return len(self.__purchased)

    @property
    def remaining(self) -> int:
        return self.__max_queries - self.spent

    @property
    def max_queries(self) -> int:
        return self.__max_queries

    def purchased_labels(self) -> Mapping[int, int]:
        return MappingProxyType(dict(self.__purchased))

    def query(self, sample_ids: Iterable[int]) -> Mapping[int, int]:
        requested = list(sample_ids)
        if any(type(i) is not int for i in requested):
            raise TypeError("sample IDs must be Python integers")
        unique = list(dict.fromkeys(requested))
        if any(i not in self.__truth for i in unique):
            raise KeyError("unknown sample ID; entire batch rejected")
        unseen = [i for i in unique if i not in self.__purchased]
        if len(unseen) > self.remaining:
            raise ValueError("insufficient budget; entire batch rejected")
        for i in unseen:
            self.__purchased[i] = self.__truth[i]
        return MappingProxyType({i: self.__purchased[i] for i in unique})
