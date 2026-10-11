"""One supervision term per sample, evaluated in log space."""
import torch
from torch import nn
from torch.nn import functional as F
from ..data.hierarchy import Cifar100Hierarchy


class MixedSupervisionLoss(nn.Module):
    def __init__(self, hierarchy: Cifar100Hierarchy):
        super().__init__()
        groups = hierarchy.coarse_to_fine
        if len(groups) != 20 or any(len(g) != 5 for g in groups):
            raise ValueError("require 20 groups of 5 fine classes")
        if sorted(f for g in groups for f in g) != list(range(100)):
            raise ValueError("hierarchy must partition 100 fine classes")
        if any(hierarchy.fine_to_coarse[f] != c for c, g in enumerate(groups) for f in g):
            raise ValueError("inconsistent hierarchy")
        self.register_buffer("groups", torch.tensor(groups, dtype=torch.long))
        self.register_buffer("mapping", torch.tensor(hierarchy.fine_to_coarse, dtype=torch.long))

    def forward(self, logits, coarse, fine, reduction="mean"):
        if logits.ndim != 2 or logits.shape[1] != 100 or not len(logits):
            raise ValueError("require nonempty (N,100) logits")
        if coarse.shape != (len(logits),) or fine.shape != coarse.shape:
            raise ValueError("label shape mismatch")
        if coarse.dtype != torch.long or fine.dtype != torch.long:
            raise ValueError("labels must be int64")
        if ((coarse < 0) | (coarse >= 20)).any() or ((fine < -1) | (fine >= 100)).any():
            raise ValueError("label outside valid range")
        known = fine >= 0
        if (self.mapping[fine[known]] != coarse[known]).any():
            raise ValueError("fine/coarse conflict")
        losses = logits.new_empty(len(logits))
        losses[known] = F.cross_entropy(logits[known], fine[known], reduction="none")
        unknown = ~known
        z = logits[unknown]
        selected = z.gather(1, self.groups[coarse[unknown]])
        losses[unknown] = torch.logsumexp(z, 1) - torch.logsumexp(selected, 1)
        if reduction == "none":
            return losses
        if reduction != "mean":
            raise ValueError("reduction must be mean or none")
        return losses.mean()
