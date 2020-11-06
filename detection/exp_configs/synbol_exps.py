from haven import haven_utils as hu
import itertools, copy
EXP_GROUPS = {}

count_list = []
for network in ['fcn8_vgg16']:
        for loss in ['density','lcfcn',   ]:
                count_list += [{'name':'semseg_counting',
                        # 'clip_grad':True,
                        'n_classes':2,
                                                'base':network, 'n_channels':3, 
                                                'loss':loss}]
                # count_list += [{'name':'semseg_counting',
                # 'n_classes':1,
                #                  'base':network, 'n_channels':3, 
                #                  'loss':loss}]
        
                


     
EXP_GROUPS["synbols_count"] = hu.cartesian_exp_group({
        'batch_size': 1,
        'num_channels':1,
        'dataset': [
                {'name':'synbols', 'mode': 'crowded', 'n_classes':2, 'transform':'basic', 'transform_mode':None},
                 {'name':'synbols', 'mode': 'fixed_scale', 'n_classes':2, 'transform':'basic', 'transform_mode':None},
                {'name':'synbols', 'mode': 'no_overlap', 'n_classes':2, 'transform':'basic', 'transform_mode':None},
                {'name':'synbols', 'mode': 'overlap', 'n_classes':2, 'transform':'basic', 'transform_mode':None},
                    ],
        'dataset_size':[
                #  {'train':10, 'val':1, 'test':1},
                {'train':'all', 'val':'all'}
                ],
        'runs':[0],
        'max_epoch': [100],
        'optimizer': [ "adam"], 
        'lr': [ 1e-3,],
        'model': count_list,
        })

semseg_list = []
for network in [ 'fcn8_vgg16']:
        for loss in [ 'joint_cross_entropy', 'cross_entropy',]:
                semseg_list += [{'name':'semseg', 
                                 'base':network,
                                  'n_channels':3,
                                   'loss':loss}]

EXP_GROUPS["synbols_seg"] = hu.cartesian_exp_group({
        'batch_size': 1,
        'num_channels':1,
        'dataset': [
                {'name':'synbols', 'overlap': 0, 'n_classes':3, 'transform':'basic', 'transform_mode':None},
                {'name':'synbols', 'overlap': 1, 'n_classes':3, 'transform':'basic', 'transform_mode':None},
               
                    ],
        'dataset_size':[
                 {'train':1, 'val':1, 'test':1},
                {'train':'all', 'val':'all'}
                ],
        'max_epoch': [100],
        'optimizer': [ "adam"], 
        'lr': [ 1e-3,],
        'model': semseg_list,
        })

