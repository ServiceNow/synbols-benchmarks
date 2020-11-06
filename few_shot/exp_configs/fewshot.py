from haven import haven_utils as hu
import numpy as np

# -------------------------------- SETTINGS -----------------------------------%

fewshot_boilerplate = {
    'benchmark':'fewshot',
    'episodic': True,
    'max_epoch':5000,
    'patience':[10, 25, 50],
    'dataset': hu.cartesian_exp_group({
        'name':'balanced-font-chars_n=1000000_2020-Oct-19.h5py',
        # 'name':'default_n=100000_2020-Oct-19.h5py', NOTE: for debugging
        'backbone': 'fewshot_synbols',
        "width": 32,
        "height": 32,
        "channels": 3,
        'task': [
                {'train':'char', 'val':'char', 'test':'char', 'ood':'font'},
                {'train':'font', 'val':'font', 'test':'font', 'ood':'char'}
        ],
        'mask': ['stratified_char', 'stratified_font'], 
        'trim_size': [None],
        "z_dim_multiplier": 2*2,
        # NOTE: start 5-way 5-shot 15-query
        'nclasses_train': 5, 
        'nclasses_val': 5,
        'nclasses_test': 5,
        'support_size_train': 5,
        'support_size_val': 5,
        'support_size_test': 5,
        'query_size_train': 15,
        'query_size_val': 15,
        'query_size_test':15,
        # NOTE: end 5-way 5-shot 5-query
        # NOTE: for debugging, only 5 iters:
        # 'train_iters': 5,
        # 'val_iters': 5,
        # 'test_iters': 5,
        # 'ood_iters': 5,
        'train_iters': 500,
        'val_iters': 500,
        'test_iters': 500,
        'ood_iters': 500,
    })
}


# -------------------------------- BACKBONES ----------------------------------%

conv4_backbone = {
    'lr':[0.005, 0.002, 0.001, 0.0005, 0.002, 0.0001],
    'batch_size':[1],
    'optimizer':'adam',
    'backbone': hu.cartesian_exp_group({ 
        'name':'conv4',
        # 'hidden_size': [256, 64],
        'hidden_size': [64],
    })
}

resnet18_backbone = {
    'backbone': hu.cartesian_exp_group({ 
        'name':'resnet18',
        'hidden_size': [64],
        'imagenet_pretraining': [False, True],
    })
}

# -------------------------------- METHODS ------------------------------------%

ProtoNet = {
    'model': "ProtoNet",
}


MAML = {
    'model': "MAML",
    'inner_lr':[0.5, 0.1, 0.01, 0.001, 0.0001],
    'n_inner_iter':[1, 2, 4, 8, 16],
}

RelationNet = {
    'model': "RelationNet",
}

# ------------------------------ EXPERIMENTS ----------------------------------%

# n_trials = 14 # Proto- and RelationNet
n_trials = 30 # MAML
n_runs = 3

def random_search(hp_lists, n_trials, n_runs=1):
    for i in range(len(hp_lists)):
        if i ==0:
            out = np.random.choice(hu.cartesian_exp_group(
                    hp_lists[i]), n_trials, replace=True).tolist()
        if i == len(hp_lists) - 1:
            out = hu.ignore_duplicates(out)
            print('remove {} duplicates'.format(n_trials-len(out)))
            break
        to_add = np.random.choice(hu.cartesian_exp_group(
                    hp_lists[i+1]), n_trials, replace=True).tolist()
        out = [dict(out[i], **to_add[i]) for i in range(n_trials)]
    ## running multiple 
    if n_runs == 1:
        return out
    else:
        out_n_runs = []
        for i in range(n_runs):
            out_n_runs += [dict(out[j], **{'seed':i}) for j in range(len(out))]
        return out_n_runs

EXP_GROUPS = {}

EXP_GROUPS['fewshot_ProtoNet'] = random_search(
    [fewshot_boilerplate, ProtoNet, conv4_backbone], n_trials, n_runs)
 
EXP_GROUPS['fewshot_MAML'] = random_search(
    [fewshot_boilerplate, MAML, conv4_backbone], n_trials, n_runs)

EXP_GROUPS['fewshot_RelationNet'] = random_search(
    [fewshot_boilerplate, RelationNet, conv4_backbone], n_trials, n_runs)


# ------------------------------ MINIIMAGENET ----------------------------------%


fewshot_mini_boilerplate = {
    'benchmark':'fewshot',
    'episodic': True,
    'max_epoch':[1000, 1500],
    'dataset': hu.cartesian_exp_group({
        'path':'PAT/TO/MINIIMAGENET',
        'name': 'miniimagenet',
        "width": 84,
        "height": 84,
        "channels": 3,
        "z_dim_multiplier": 5*5,
        ## start 5-way 5-shot 15-query
        'nclasses_train': 5, 
        'nclasses_val': 5,
        'nclasses_test': 5,
        'support_size_train': 5,
        'support_size_val': 5,
        'support_size_test': 5,
        'query_size_train': 15,
        'query_size_val': 15,
        'query_size_test': 15,
        ## end 5-way 5-shot 15-query
        'train_iters': 500,
        'val_iters': 500,
        'test_iters': 500,
    })
}


EXP_GROUPS['fewshot_mini_ProtoNet'] = random_search(
    [fewshot_mini_boilerplate, ProtoNet, conv4_backbone], n_trials)
 
EXP_GROUPS['fewshot_mini_MAML'] = random_search(
    [fewshot_mini_boilerplate, MAML, conv4_backbone], n_trials)

EXP_GROUPS['fewshot_mini_RelationNet'] = random_search(
    [fewshot_mini_boilerplate, RelationNet, conv4_backbone], n_trials)