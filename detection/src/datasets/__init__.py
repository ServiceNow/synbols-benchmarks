import torchvision
import torch
import numpy as np
from torchvision.transforms import transforms
from sklearn.utils import shuffle
from PIL import Image
from . import synbols
from src import utils as ut
import os
import os
import numpy as np

import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


def get_dataset(dataset_dict, split, datadir, exp_dict, dataset_size=None):
    name = dataset_dict['name']


    if name == "synbols":
        dataset = synbols.Synbols(split=split, datadir=datadir, exp_dict=exp_dict)
        if dataset_size is not None and dataset_size.get(split, 'all') != 'all':
            # dataset = synbols.Synbols(split='train', exp_dict=exp_dict)
            dataset.meta_list = dataset.meta_list[:dataset_size[split]]
        # else:
        
    else:
        raise ValueError('dataset %s not found' % name)

    print(split, ':', len(dataset))
    return dataset