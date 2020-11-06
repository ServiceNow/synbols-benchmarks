from torch.utils.data import Dataset
import numpy as np
import json
import os
from torchvision import transforms as tt
from torchvision.datasets import MNIST, SVHN
from PIL import Image
import torch
import h5py
import multiprocessing
import sys
from synbols import stratified_splits

import copy
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.download_dataset import get_data_path_or_download

def get_dataset(splits, data_root, exp_dict):
    dataset_dict = exp_dict["dataset"]
    if dataset_dict["backend"] == "synbols_hdf5":
        full_path = get_data_path_or_download(exp_dict["dataset"]["name"],
                                              data_root=data_root)
        
        # For debugging purposes, reference_dataset should be None for most usages
        reference_dataset = dataset_dict.get("reference_dataset", None)
        if reference_dataset is not None:
            reference_data = SynbolsHDF5(reference_dataset,
                                         dataset_dict["task"],
                                         mask=dataset_dict["mask"],
                                         trim_size=dataset_dict.get(
                                             "trim_size", None),
                                         reference_mask=dataset_dict["reference_mask"])

        data = SynbolsHDF5(full_path,
                           dataset_dict["task"],
                           mask=dataset_dict["mask"],
                           trim_size=dataset_dict.get("trim_size", None),
                           raw_labels=dataset_dict.get("raw_labels", False))
        ret = []
        for split in splits:
            transform = [tt.ToPILImage()]
            if dataset_dict["augmentation"] and split == "train":
                transform += [tt.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-2, 2, -2, 2), resample=Image.BICUBIC, fillcolor=0),
                              # tt.RandomHorizontalFlip(),
                              tt.ColorJitter(0.4, 0.4, 0.4, 0.4)]
            transform += [tt.ToTensor(),
                          tt.Normalize([0.5] * dataset_dict["channels"],
                                       [0.5] * dataset_dict["channels"])]
            transform = tt.Compose(transform)
            if reference_dataset is not None and split in ["train", "val"]:
                dataset = SynbolsSplit(
                    reference_data, split, transform=transform)
            else:
                dataset = SynbolsSplit(data, split, transform=transform)
            ret.append(dataset)
        exp_dict["num_classes"] = len(ret[0].labelset)  # FIXME: this is hacky
        return ret
    elif dataset_dict["backend"] == "mnist":
        ret = []
        exp_dict["num_classes"] = 10  # FIXME: this is hacky
        for split in splits:
            transform = []
            if dataset_dict["augmentation"] and split == "train":
                transform += [tt.Resize(dataset_dict["height"]),
                              tt.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-2, 2, -2, 2), resample=Image.BICUBIC, fillcolor=0)]
                #   tt.RandomHorizontalFlip()]
            else:
                transform += [tt.Resize(dataset_dict["height"])]
            transform += [tt.ToTensor(),
                          tt.Normalize([0.5], [0.5])]
            transform = tt.Compose(transform)
            ret.append(MNIST(data_root,
                             train=(split == "train"), transform=transform, download=True))
        return ret
    elif dataset_dict["backend"] == "svhn":
        ret = []
        exp_dict["num_classes"] = 10  # FIXME: this is hacky
        for split in splits:
            transform = []
            if dataset_dict["augmentation"] and split == "train":
                transform += [tt.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-2, 2, -2, 2), resample=Image.BICUBIC, fillcolor=0),
                              # tt.RandomHorizontalFlip(),
                              tt.ColorJitter(0.4, 0.4, 0.4, 0.4)]
            transform += [tt.ToTensor(),
                          tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            transform = tt.Compose(transform)
            split_dict = {'train': 'train', 'test': 'test', 'val': 'test'}
            ret += [SVHN(data_root,
                         split=split_dict[split], transform=transform, download=True)]
        return ret
    else:
        raise ValueError


def _read_json_key(args):
    string, key = args
    return json.loads(string)[key]

def get_stratified(values, fn, ratios=[0.6, 0.2, 0.2], tomap=True):
    vfield = list(map(fn, values))
    if isinstance(vfield[0], float):
        pmap = stratified_splits.percentile_partition(vfield, ratios)
    else:
        pmap = stratified_splits.unique_class_based_partition(vfield, ratios)
    if tomap:
        return stratified_splits.partition_map_to_mask(pmap)
    else:
        return pmap

class SynbolsHDF5:
    """HDF5 Backend Class"""
    def __init__(self, path, task, ratios=[0.6, 0.2, 0.2], mask=None, trim_size=None, raw_labels=False, reference_mask=None):
        """Constructor: loads data and parses labels.

        Args:
            path (str): path where the data is stored (see full_path above)
            task (str): 'char', 'font', or the field of choice 
            ratios (list, optional): The train/val/test split ratios. Defaults to [0.6, 0.2, 0.2].
            mask (ndarray, optional): Mask with the data partition. Defaults to None.
            trim_size (int, optional): Trim the dataset to a smaller size, for debugging speed. Defaults to None.
            raw_labels (bool, optional): Whether to include all the attributes of the synbols for each batch. Defaults to False.
            reference_mask (ndarray, optional): If train and validation are done with two different datasets, the 
                                                reference mask specifies the partition of the training data. Defaults to None.

        Raises:
            ValueError: Error message
        """
        self.path = path
        self.task = task
        self.ratios = ratios
        print("Loading hdf5...")
        with h5py.File(path, 'r') as data:
            self.x = data['x'][...]
            y = data['y'][...]
            print("Converting json strings to labels...")
            with multiprocessing.Pool(8) as pool:
                self.y = pool.map(json.loads, y)
            print("Done converting.")
            if isinstance(mask, str):
                if "split" in data:
                    if mask in data['split'] and mask == "random":
                        self.mask = data["split"][mask][...]
                    else:
                        self.mask = self.parse_mask(mask, ratios=ratios)
                else:
                    raise ValueError
            else:
                self.mask = mask

            self.y = np.array([_y[task] for _y in self.y])

            if raw_labels:
                print("Parsing raw labels...")
                raw_labels = copy.deepcopy(self.y)
                self.raw_labels = []
                self.raw_labelset = {k: [] for k in raw_labels[0].keys()}
                for item in raw_labels:
                    ret = {}
                    for key in item.keys():
                        if isinstance(item[key], str) or isinstance(item[key], int):
                            self.raw_labelset[key] = []
                        ret[key] = item[key]

                    self.raw_labels.append(ret)
                str2int = {}
                for k in self.raw_labelset.keys():
                    v = self.raw_labelset[k]
                    if len(v) > 0:
                        v = list(sorted(set(v)))
                        self.raw_labelset[k] = v
                        str2int[k] = {k: i for i, k in enumerate(v)}
                for item in self.raw_labels:
                    for k in str2int.keys():
                        item[k] = str2int[k][item[k]]

                print("Done parsing raw labels.")
            else:
                self.raw_labels = None

            self.trim_size = trim_size
            if trim_size is not None and len(self.x) > self.trim_size:
                self.mask = self.trim_dataset(self.mask)
            self.reference_mask = reference_mask
            if self.reference_mask is not None:
                self.mask[:, [0, 1]] = np.load(self.reference_mask)[...]
            print("Done reading hdf5.")

    def trim_dataset(self, mask, train_size=60000, val_test_size=20000):
        labelset = np.sort(np.unique(self.y))
        counts = np.array([np.count_nonzero(self.y == y) for y in labelset])
        imxclass_train = int(np.ceil(train_size / len(labelset)))
        imxclass_val_test = int(np.ceil(val_test_size / len(labelset)))
        ind_train = np.arange(mask.shape[0])[mask[:, 0]]
        y_train = self.y[ind_train]
        ind_train = np.concatenate([np.random.permutation(ind_train[y_train == y])[
                                   :imxclass_train] for y in labelset], 0)
        ind_val = np.arange(mask.shape[0])[mask[:, 1]]
        y_val = self.y[ind_val]
        ind_val = np.concatenate([np.random.permutation(ind_val[y_val == y])[
                                 :imxclass_val_test] for y in labelset], 0)
        ind_test = np.arange(mask.shape[0])[mask[:, 2]]
        y_test = self.y[ind_test]
        ind_test = np.concatenate([np.random.permutation(ind_test[y_test == y])[
                                  :imxclass_val_test] for y in labelset], 0)
        current_mask = np.zeros_like(mask)
        current_mask[ind_train, 0] = True
        current_mask[ind_val, 1] = True
        current_mask[ind_test, 2] = True
        return current_mask

    def parse_mask(self, mask, ratios):
        args = mask.split("_")[1:]
        if "stratified" in mask:
            mask = 1
            for arg in args:
                if arg == 'translation-x':
                    def fn(x): return x['translation'][0]
                elif arg == 'translation-y':
                    def fn(x): return x['translation'][1]
                else:
                    def fn(x): return x[arg]
                mask *= get_stratified(self.y, fn,
                                       ratios=[ratios[1], ratios[0], ratios[2]])
            mask = mask[:, [1, 0, 2]]
        elif "compositional" in mask:
            partition_map = None
            if len(args) != 2:
                raise RuntimeError(
                    "Compositional splits must contain two fields to compose")
            for arg in args:
                if arg == 'translation-x':
                    def fn(x): return x['translation'][0]
                elif arg == 'translation-y':
                    def fn(x): return x['translation'][1]
                else:
                    def fn(x): return x[arg]
                if partition_map is None:
                    partition_map = get_stratified(self.y, fn, tomap=False)
                else:
                    _partition_map = get_stratified(self.y, fn, tomap=False)
                    partition_map = stratified_splits.compositional_split(
                        _partition_map, partition_map)
            partition_map = partition_map.astype(bool)
            mask = np.zeros_like(partition_map)
            for i, split in enumerate(np.argsort(partition_map.astype(int).sum(0))[::-1]):
                mask[:, i] = partition_map[:, split]
        else:
            raise ValueError
        return mask.astype(bool)


class SynbolsSplit(Dataset):
    def __init__(self, dataset, split, transform=None):
        """Given a Backend (dataset), it splits the data in train, val, and test.


        Args:
            dataset (object): backend to load, it should contain the following attributes:
                - x, y, mask, ratios, path, task, mask
            split (str): train, val, or test
            transform (torchvision.transforms, optional): A composition of torchvision transforms. Defaults to None.
        """
        self.path = dataset.path
        self.task = dataset.task
        self.mask = dataset.mask
        if dataset.raw_labels is not None:
            self.raw_labelset = dataset.raw_labelset
        self.raw_labels = dataset.raw_labels
        self.ratios = dataset.ratios
        self.split = split
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        self.split_data(dataset.x, dataset.y, dataset.mask, dataset.ratios)

    def split_data(self, x, y, mask, ratios, rng=np.random.RandomState(42)):
        if mask is None:
            if self.split == 'train':
                start = 0
                end = int(ratios[0] * len(x))
            elif self.split == 'val':
                start = int(ratios[0] * len(x))
                end = int((ratios[0] + ratios[1]) * len(x))
            elif self.split == 'test':
                start = int((ratios[0] + ratios[1]) * len(x))
                end = len(x)
            indices = rng.permutation(len(x))
            indices = indices[start:end]
        else:
            mask = mask[:, ["train", "val", "test"].index(self.split)]
            indices = np.arange(len(y))  # 0....nsamples
            indices = indices[mask]
        self.labelset = list(sorted(set(y)))
        self.y = np.array([self.labelset.index(y) for y in y])
        self.x = x[indices]
        self.y = self.y[indices]
        if self.raw_labels is not None:
            self.raw_labels = np.array(self.raw_labels)[indices]

    def __getitem__(self, item):
        if self.raw_labels is None:
            return self.transform(self.x[item]), self.y[item]
        else:
            return self.transform(self.x[item]), self.y[item], self.raw_labels[item]

    def __len__(self):
        return len(self.x)
