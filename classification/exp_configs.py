from haven import haven_utils as hu

baselines = []
mlp = {"name": "mlp", "depth": 3, "hidden_size": 256}
wrn = {"name": "wrn", "width": 4, "depth": 28}
resnet12 = {"name": "resnet12"}
conv4_gap = {"name": "conv4", "gap": True}
conv4 = {"name": "conv4", "gap": False}

augmentation = False
for seed in [3, 42, 123]:
    mnist = {
        "backend": "mnist",
        "width": 32,
        "height": 32,
        "channels": 1,
        "augmentation": False,
        "mask": None,
        "task": "char"
    }
    svhn = {
        "backend": "svhn",
        "width": 32,
        "height": 32,
        "channels": 3,
        "augmentation": False,
        "mask": None,
        "task": "char"
    }
    default_1M = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "default_n=1000000_2020-Oct-19.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random",
    }
    default_100K = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "default_n=100000_2020-Oct-19.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random",
    }
    natural_100K = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "natural-patterns_n=100000_2020-Oct-20.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random",
    }
    compositional_rotation_scale = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "default_n=100000_2020-Oct-19.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "compositional_rotation_scale",
        "trim_size": 100000
    }
    stratified_scale = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "default_n=100000_2020-Oct-19.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "stratified_scale",
    }
    stratified_rotation = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "default_n=100000_2020-Oct-19.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "stratified_rotation",
    }
    translation_x = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "default_n=100000_2020-Oct-19.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "stratified_translation-x",
    }
    stratified_char = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "less-variations_n=100000_2020-Oct-19.h5py",
        "task": "font",
        "mask": "stratified_char",
        "augmentation": augmentation
    }
    default_font_100K = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "less-variations_n=100000_2020-Oct-19.h5py",
        "task": "font",
        "mask": "random",
        "augmentation": augmentation
    }
    default_font_1M = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "less-variations_n=100000_2020-Oct-19.h5py",
        "task": "font",
        "mask": "random",
        "augmentation": augmentation
    }
    stratified_font = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "default_n=100000_2020-Oct-19.h5py",
        "task": "char",
        "mask": "stratified_font",
        "augmentation": augmentation
    }
    camouflage = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "camouflage_n=100000_2020-Oct-19.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random",
    }
    korean = {
        "backend": "synbols_hdf5",
        "width": 32,
        "height": 32,
        "channels": 3,
        "name": "korean-1k_n=100000_2020-Oct-19.h5py",
        "task": "char",
        "augmentation": augmentation,
        "mask": "random",
    }

    datasets = [
                camouflage,
                compositional_rotation_scale,
                default_100K,
                default_1M,
                default_font_100K,
                # default_font_1M,
                natural_100K,
                korean,
                # mnist,
                stratified_char,
                stratified_font,
                stratified_rotation,
                # svhn
                stratified_scale,
                translation_x
                ]
    
    for lr in [0.0004]:
        for dataset in datasets:
            for backbone in [mlp, conv4, conv4_gap]:
                baselines += [{'lr':lr,
                            'batch_size': 512,
                            'min_lr_decay': 1e-3,
                            'amp': 2,
                            "seed": seed,
                            'model': "classification",
                            'backbone': backbone,
                            'max_epoch': 200,
                            'episodic': False,
                            'dataset': dataset}]
                
            for _augmentation in [True, False]:
                for backbone in [resnet12]:
                    baselines += [{'lr':lr,
                                'batch_size': 512,
                                'min_lr_decay': 1e-3,
                                'amp': 2,
                                "seed": seed,
                                'model': "classification",
                                'backbone': backbone,
                                'max_epoch': 200,
                                'episodic': False,
                                'dataset': dataset.copy()}]
                    baselines[-1]["dataset"]['augmentation'] = _augmentation 

                for backbone in [wrn]:
                    baselines += [{'lr':lr / 4,
                                'batch_size': 128,
                                'min_lr_decay': 1e-3,
                                'amp': 2,
                                "seed": seed,
                                'model': "classification",
                                'backbone': backbone,
                                'max_epoch': 200,
                                'episodic': False,
                                'dataset': dataset.copy()}]
                    baselines[-1]["dataset"]['augmentation'] = _augmentation 

EXP_GROUPS = {}            
EXP_GROUPS["baselines"] = baselines
