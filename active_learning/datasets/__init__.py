import os

DATA_ROOT = os.environ.get('DATA_ROOT', '/tmp')

from baal.active import ActiveLearningDataset
from torchvision import transforms as tt
from tools.download_dataset import get_data_path_or_download
from .aleatoric_uncertainty_datasets import AleatoricSynbols
from .synbols import Synbols


def get_dataset(split, dataset_dict):
    if dataset_dict["name"] == "active_learning":
        transform = tt.Compose([tt.ToPILImage(), tt.ToTensor()])
        n_classes = dataset_dict.get('n_classes')
        n_classes = n_classes or (52 if dataset_dict['task'] == 'char' else 1002)
        path = get_data_path_or_download(dataset_dict["path"], DATA_ROOT)
        dataset = AleatoricSynbols(path=path,
                                   split=split,
                                   key=dataset_dict["task"],
                                   transform=transform,
                                   p=dataset_dict.get('p', 0.0),
                                   seed=dataset_dict.get('seed', 666),
                                   n_classes=n_classes,
                                   )
        if split == 'train':
            # Make an AL dataset and label randomly.
            dataset = ActiveLearningDataset(dataset, pool_specifics={'transform': transform})
            dataset.label_randomly(dataset_dict['initial_pool'])
        return dataset
    else:
        raise ValueError("Dataset %s not found" % dataset_dict["name"])
