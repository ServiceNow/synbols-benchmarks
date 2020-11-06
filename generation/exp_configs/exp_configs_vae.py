from haven import haven_utils as hu

default = {
    "backend": "synbols_hdf5",
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "non-camou-shade_n=100000_2020-Oct-19.h5py",
    "task": "char",
    "augmentation": False,
    "mask": "random",
}
camouflage = {
    "backend": "synbols_hdf5",
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "camouflage_n=100000_2020-Oct-19.h5py",
    "task": "char",
    "augmentation": False,
    "mask": "random",
}
solid = {
    "backend": "synbols_hdf5",
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "non-camou-bw_n=100000_2020-Oct-19.h5py",
    "task": "char",
    "augmentation": False,
    "mask": "random",
}
natural = {
    "backend": "synbols_hdf5",
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "natural-patterns_n=100000_2020-Oct-20.h5py",
    "task": "char",
    "augmentation": False,
    "mask": "random",
}
biggan_biggan = {
    "name": "biggan",
    "mlp_width": 2,
    "mlp_depth": 2,
    "channels_width": 4,
    "dp_prob": 0.3,
    "feature_extractor": "resnet"
}
biggan_infomax = {
    "name": "biggan",
    "mlp_width": 2,
    "mlp_depth": 2,
    "channels_width": 4,
    "dp_prob": 0.3,
    "feature_extractor": "deepinfomax"
}
EXP_GROUPS = {}            
EXP_GROUPS["vae"] = hu.cartesian_exp_group({'lr': 0.0001,
                        'beta_annealing': True,
                        'ngpu': 1,
                        'beta': [0.01],
                        'batch_size': 256,
                        'seed': [3, 42, 123],
                        'amp': 0,
                        'min_lr_decay': 1e-3,
                        'model': "vae",
                        'backbone': [biggan_biggan, biggan_infomax],
                        'z_dim': [64],
                        'max_epoch': 200,
                        'episodic': False,
                        'hierarchical': [False, True],
                        'dataset': [camouflage, default, solid, natural]})