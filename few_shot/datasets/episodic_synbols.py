from torch.utils.data import Dataset
from .episodic_dataset import EpisodicDataset
import numpy as np
import json
import os
from torchvision import transforms as tt

import sys ; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# from generation.datasets import SynbolsHDF5
from classification.datasets import SynbolsHDF5


class EpisodicSynbols(EpisodicDataset):
    def __init__(self, path, split, sampler, size, key='font', transform=None,
                    mask=None, trim_size=None, task=None):
        if 'npz' in path:
            dataset = SynbolsNpz(path, split, key, transform)
        elif 'h5py' in path:
            # dataset = cls_dataset.SynbolsHDF5(path, task,
            dataset = SynbolsHDF5(path, task,
                    mask=mask, trim_size=trim_size, raw_labels=False)
        else:
            Exception('not implemented')
        self.x = dataset.x
        self.name = "synbols"
        ## make sure we have enough data per class:
        unique, counts = np.unique(dataset.y, return_counts=True)
        #TODO: dont harcode this
        low_data_classes = unique[counts < 15] # 5-shot 5-query 
        low_data_classes_idx = np.isin(dataset.y, low_data_classes)
        self.x = self.x[~low_data_classes_idx]
        dataset.y = dataset.y[~low_data_classes_idx] 

        
        super().__init__(dataset.y, sampler, size, transform)
    
    def sample_images(self, indices):
        return [self.transforms(self.x[i]) for i in indices]

if __name__ == '__main__':
    synbols = SynbolsHDF5('default_n=100000_2020-May-20.h5py', 'val')
