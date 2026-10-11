"""Image and public coarse label access only: fine labels are never included."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class PublicSample:
    sample_id: int
    coarse_label: int
    image: np.ndarray

class PublicPool:
    __slots__ = ("__images", "__coarse")

    def __init__(self, images: np.ndarray, coarse_labels: np.ndarray):
        images, coarse = np.asarray(images), np.asarray(coarse_labels, dtype=np.int64)
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError("images must be uint8 (N,32,32,3)")
        if coarse.shape != (len(images),) or not len(images):
            raise ValueError("coarse labels must match nonempty image count")
        if np.any((coarse < 0) | (coarse >= 20)):
            raise ValueError("coarse labels must be in 0..19")
        self.__images = images.copy()
        self.__images.flags.writeable = False
        self.__coarse = coarse.copy()
        self.__coarse.flags.writeable = False

    def __len__(self) -> int:
        return len(self.__coarse)

    def coarse_labels(self) -> np.ndarray:
        return self.__coarse.copy()

    def sample(self, sample_id: int) -> PublicSample:
        i = int(sample_id)
        if not 0 <= i < len(self):
            raise IndexError("sample ID outside public pool")
        return PublicSample(i, int(self.__coarse[i]), self.__images[i].copy())


class ClientPublicView:
    """Controller materializes only this client's public samples, preserving IDs."""
    __slots__ = ("__samples",)

    def __init__(self, pool: PublicPool, ids):
        self.__samples = {int(i): pool.sample(int(i)) for i in ids}

    def sample(self, sample_id: int) -> PublicSample:
        sample = self.__samples[int(sample_id)]  # other-client IDs raise KeyError
        return PublicSample(sample.sample_id, sample.coarse_label, sample.image.copy())

    def coarse_labels(self):
        return {i: sample.coarse_label for i, sample in self.__samples.items()}

    def __len__(self):
        return len(self.__samples)
