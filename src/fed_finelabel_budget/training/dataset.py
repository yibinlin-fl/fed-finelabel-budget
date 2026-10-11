from collections.abc import Mapping, Iterable
import numpy as np
import torch
from torch.utils.data import Dataset
from ..data.public_pool import PublicPool
from ..data.hierarchy import Cifar100Hierarchy


class ClientDataset(Dataset):
    """Freeze only this client's purchased labels; -1 marks unknown fine."""
    def __init__(self, pool: PublicPool, ids: Iterable[int], purchased: Mapping[int, int],
                 hierarchy: Cifar100Hierarchy):
        self.ids = tuple(map(int, ids))
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate client IDs")
        # Materialize only the client's public samples, not other clients' images.
        self.samples = tuple(pool.sample(i) for i in self.ids)
        self.fine = tuple(int(purchased.get(i, -1)) for i in self.ids)
        for sample, fine in zip(self.samples, self.fine):
            if fine != -1 and (not 0 <= fine < 100 or
                              hierarchy.fine_to_coarse[fine] != sample.coarse_label):
                raise ValueError("purchased fine label conflicts with public hierarchy")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = torch.from_numpy(np.array(sample.image.transpose(2, 0, 1), copy=True)).float()
        return image.div(255).sub(0.5).div(0.5), sample.coarse_label, self.fine[index], sample.sample_id
