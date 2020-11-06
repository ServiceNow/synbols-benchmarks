from haven import haven_utils as hu

# Define exp groups for parameter search
model_cfg = {
    'lr': [0.001],
    'batch_size': [32],
    'model': "active_learning",
    'seed': [1337, 1338, 1339, 1340],
    'mu': 1e-3,
    'reg_factor': 1e-4,
    'backbone': "vgg16",
    'num_classes': 96,
    'query_size': [100],
    'learning_epoch': 10,
    'heuristic': ['bald', 'random', 'entropy'],
    'iterations': [20],
    'max_epoch': 200,
    'imagenet_pretraining': [True],
}

EXP_GROUPS = {
    'active_char_missing_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
                'path': 'missing-symbol_n=100000_2020-Oct-19.h5py',
                'name': 'active_learning',
                'task': 'char',
                'initial_pool': 2000,
                'seed': 1337})),
    'active_char_label_noise_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
                'path': 'default_n=100000_2020-Oct-19.h5py',
                'name': 'active_learning',
                'task': 'char',
                'initial_pool': 2000,
                'seed': 1337,
                'p': 0.15})),
    'active_char_pixel_noise_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
                'path': 'pixel-noise_n=100000_2020-Oct-22.h5py',
                'name': 'active_learning',
                'task': 'char',
                'initial_pool': 2000,
                'seed': 1337})),
    'active_char_default_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
                'path': 'default_n=100000_2020-Oct-19.h5py',
                'name': 'active_learning',
                'task': 'char',
                'initial_pool': 2000,
                'seed': 1337})),
    'active_char_large_trans_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
                'path': 'large-translation_n=100000_2020-Oct-19.h5py',
                'name': 'active_learning',
                'task': 'char',
                'initial_pool': 2000,
                'seed': 1337})),
    'active_char_partly_occluded_calibrated':
        hu.cartesian_exp_group(dict({**model_cfg, **dict(model='calibrated_active_learning')},
                                    calibrate=[True, False], dataset={
                'path': 'some-large-occlusion_n=100000_2020-Oct-19.h5py',
                'name': 'active_learning',
                'task': 'char',
                'initial_pool': 2000,
                'seed': 1337})),
}
