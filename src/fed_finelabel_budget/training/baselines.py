from torch import nn
from torch.nn import functional as F
from .dataset import ClientDataset
from .losses import MixedSupervisionLoss


class FineOnlyLoss(nn.Module):
    def forward(self, logits, coarse, fine):
        if (fine < 0).any():
            raise ValueError("B0 training requires purchased fine labels only")
        return F.cross_entropy(logits, fine)


def build_clients(strategy, pool, partitions, purchased, hierarchy):
    if strategy not in ("B0", "B1", "B2", "B3", "B4"):
        raise ValueError("unknown baseline")
    clients = {}
    for cid, ids in partitions.items():
        selected = [int(i) for i in ids if int(i) in purchased] if strategy == "B0" else ids
        clients[cid] = ClientDataset(pool, selected, purchased, hierarchy)
    return clients, FineOnlyLoss() if strategy == "B0" else MixedSupervisionLoss(hierarchy)
