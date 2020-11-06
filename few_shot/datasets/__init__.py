from torchvision import transforms as tt
from .synbols_loaders import SynbolsFolder, SynbolsNpz
from .episodic_dataset import FewShotSampler # should be imported here because it changes the dataloader to be episodic
from .episodic_synbols import EpisodicSynbols 
from .episodic_miniimagenet import EpisodicMiniImagenet

import os, sys ; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.download_dataset import get_data_path_or_download


def get_dataset(split, data_root, exp_dict):
    dataset_dict = exp_dict["dataset"]
    if dataset_dict["backbone"] == "fewshot_synbols":
        full_path = get_data_path_or_download(exp_dict["dataset"]["name"],
                                        data_root=data_root)
        _split = 'test' if split == 'ood' else split
        transform = [tt.ToPILImage()]
        transform += [tt.ToTensor(),
                    tt.Normalize([0.5] * dataset_dict["channels"], 
                                    [0.5] * dataset_dict["channels"])]
        transform = tt.Compose(transform)
        
        sampler = FewShotSampler(nclasses=dataset_dict["nclasses_%s" %_split],
                                 support_size=dataset_dict["support_size_%s" %_split],
                                 query_size=dataset_dict["query_size_%s" %_split],
                                 unlabeled_size=0)
        return EpisodicSynbols(full_path, 
                                split=_split, 
                                sampler=sampler, 
                                size=dataset_dict["%s_iters" %_split], 
                                key=dataset_dict["task"][split], 
                                transform=transform,
                                mask=exp_dict['dataset']['mask'],
                                # task=dataset_dict["task"]
                                task=dataset_dict["task"][split])
    elif dataset_dict["backbone"] == "miniimagenet":
        # transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
        transform = tt.Compose([tt.ToPILImage(),
                                            tt.Resize((84,84)),
                                            tt.ToTensor()])
        sampler = FewShotSampler(nclasses=dataset_dict["nclasses_%s" %split],
                                 support_size=dataset_dict["support_size_%s" %split],
                                 query_size=dataset_dict["query_size_%s" %split],
                                 unlabeled_size=0)
        return EpisodicMiniImagenet(dataset_dict["name"],
                                split=split, 
                                sampler=sampler, 
                                size=dataset_dict["%s_iters" %split], 
                                transforms=transform)
    else:
        raise ValueError("Dataset %s not found" % dataset_dict["backbone"])
