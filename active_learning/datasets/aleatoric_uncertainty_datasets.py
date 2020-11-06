import numpy as np

from datasets.synbols import Synbols


def _shuffle_subset(data: np.ndarray, shuffle_prop: float, rng) -> np.ndarray:
    to_shuffle = np.nonzero(rng.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[rng.permutation(to_shuffle), ...]
    return data


class AleatoricSynbols(Synbols):
    def __init__(self, path, split, key='font', transform=None, p=0.0,
                 seed=None, n_classes=None):
        super().__init__(path=path, split=split, key=key, transform=transform)
        self.p = p
        self.seed = seed
        self.noise_classes = n_classes
        self.rng = np.random.RandomState(self.seed)
        if self.p > 0:
            # Label noise
            self.y = self._shuffle_label()

    def get_splits(self, source):
        if self.split == 'train':
            start = 0
            end = int(0.7 * len(source))
        elif self.split == 'calib':
            start = int(0.7 * len(source))
            end = int(0.8 * len(source))
        elif self.split == 'val':
            start = int(0.8 * len(source))
            end = int(0.9 * len(source))
        elif self.split == 'test':
            start = int(0.9 * len(source))
            end = len(source)
        return start, end

    def get_values_split(self, y):
        start, end = self.get_splits(source=y)
        return y[self.indices[start:end]]

    def _shuffle_label(self):
        return _shuffle_subset(self.y, self.p, self.rng)

    def __getitem__(self, item):
        return self.transform(self.x[item]), self.y[item]

    def __len__(self):
        return len(self.x)
