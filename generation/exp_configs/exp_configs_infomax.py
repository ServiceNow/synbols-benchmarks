from haven import haven_utils as hu

biggan = {
    "name": "biggan",
    "mlp_width": 4,
    "mlp_depth": 2,
    "channels_width": 4,
    "dp_prob": 0.3,
    "feature_extractor": "resnet"
}
infomax = {
    "name": "infomax",
    "mlp_width": 4,
    "mlp_depth": 2,
    "channels_width": 4,
    "dp_prob": 0.3,
    "feature_extractor": "deepinfomax"
}
augmentation = False
default = {
    "backend": "synbols_hdf5",
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "non-camou-shade_n=100000_2020-Oct-19.h5py",
    "task": "char",
    "augmentation": augmentation,
    "mask": "random",
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
solid = {
    "backend": "synbols_hdf5",
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "non-camou-bw_n=100000_2020-Oct-19.h5py",
    "task": "char",
    "augmentation": augmentation,
    "mask": "random",
}
natural = {
    "backend": "synbols_hdf5",
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "natural-patterns_n=100000_2020-Oct-20.h5py",
    "task": "char",
    "augmentation": augmentation,
    "mask": "random",
}
EXP_GROUPS = {}            
EXP_GROUPS["deepinfomax"] = hu.cartesian_exp_group({'lr': 0.0001,
                        'beta_annealing': True,
                        'ngpu': 1,
                        'batch_size': 256,
                        'seed': [2, 42, 128],
                        'amp': 0,
                        'min_lr_decay': 1e-3,
                        'model': "deepinfomax",
                        'backbone': [infomax],
                        'z_dim': [64],
                        'max_epoch': 200,
                        'episodic': False,
                        'dataset': [solid, camouflage, default, natural]})