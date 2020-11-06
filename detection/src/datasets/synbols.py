from torch.utils.data import Dataset
import numpy as np
import json
import os
from haven import haven_utils as hu
from torchvision import transforms as tt
from PIL import Image
import glob, torch
import h5py, json
import torchvision.transforms as transforms

class Synbols(torch.utils.data.Dataset):
    def __init__(self, split, datadir, exp_dict, mode='counting'):
        
        self.exp_dict = exp_dict
        
        if self.exp_dict['dataset']['mode'] == 'crowded':
            self.path = os.path.join(datadir, 'counting-crowded_n=100000_2020-Oct-19.h5py')

        elif self.exp_dict['dataset']['mode'] == 'fixed_scale':
            self.path = os.path.join(datadir, 'counting-fix-scale_n=100000_2020-Oct-19.h5py')

        elif ['no_overlap', 'overlap']:
            self.path = os.path.join(datadir, 'counting_n=100000_2020-Oct-19.h5py')

        else:
            stop
        
        path_id = hu.hash_str(self.path)
        train_meta_fname = os.path.join(datadir, 'train_meta_list_v1_%s.pkl' %path_id)
        val_meta_fname = os.path.join(datadir, 'val_meta_list_v1_%s.pkl' % path_id)

        if not os.path.exists(train_meta_fname):
            meta = load_attributes_h5(self.path)
            meta_list, splits = meta
            for i, m in enumerate(meta_list):
                meta_list[i] = json.loads(m)
            for i, m in enumerate(meta_list):
                meta_list[i]['index'] = i
            meta_list = np.array(meta_list)
            train_split = splits['stratified_char'][:, 0]
            val_split = splits['stratified_char'][:, 1]
            # test_split = splits['stratified_char'][:, 2]

            train_meta_list = meta_list[train_split][:10000]
            val_meta_list = meta_list[val_split][:10000]

            hu.save_pkl(train_meta_fname, train_meta_list)
            hu.save_pkl(val_meta_fname, val_meta_list)
        # self.transform = None
        # load_minibatch_h5(self.path, [indices])
        # self.img_list = glob.glob(self.path+"/*.jpeg")
        self.split = split
        if split == 'train':
            self.meta_list = np.array(hu.load_pkl(train_meta_fname))
            n = int(0.9*len(self.meta_list))
            self.meta_list = self.meta_list[:n]

        elif split == 'val':
            self.meta_list = np.array(hu.load_pkl(train_meta_fname))
            n = int(0.9*len(self.meta_list))
            self.meta_list = self.meta_list[n:]

        elif split == 'test':
            self.meta_list = np.array(hu.load_pkl(val_meta_fname))

        if self.exp_dict['dataset']['mode'] == 'no_overlap':
            self.meta_list = [m for m in self.meta_list if m['overlap_score']==0]

        elif self.exp_dict['dataset']['mode'] == 'overlap':
            self.meta_list = [m for m in self.meta_list if m['overlap_score']> 0]

        elif self.exp_dict['dataset']['mode'] in ['crowded', 'fixed_scale']:
            self.meta_list = self.meta_list
        else:
            stop

        symbol_dict = {}
        for i in range (len(self.meta_list)):
            meta =  self.meta_list[i]
            for s in meta['symbols']:
                if s['char'] not in symbol_dict:
                    symbol_dict[s['char']] = []
                symbol_dict[s['char']] += [i]

        self.n_classes = 2
        self.symbol_dict = symbol_dict
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        
        # self.meta_list = self.meta_list[np.unique(symbol_dict['k'])]
        # self.absolute_indices = self.absolute_indices[np.unique(symbol_dict['k'])][:10]

    def __getitem__(self, index):
        meta=  self.meta_list[index]
        img, mask = load_minibatch_h5(self.path, [meta['index']])
        points = get_point_list(mask)
        if 1: 
            char_id_list = [i+1 for i, s in enumerate(meta['symbols']) if s['char'] in ['a']]
            # char_id_list = [i+1 for i, s in enumerate(meta['symbols'])]
            mask_out = np.zeros(mask.shape)

            for char_id in char_id_list:
                mask_out[mask == char_id] = 1

            # hu.save_image('tmp.png', mask_out)
        points = points*mask_out

        assert(abs(len(char_id_list) -  points.sum()) <= 2)
        img_float = self.img_transform(img[0])
        meta['hash'] = hu.hash_dict({'id':meta['index']})
        meta['shape'] = mask_out.squeeze().shape
        meta['split'] = self.split
        return {'images': img_float, 
                'points':torch.FloatTensor(points),
                'masks': torch.LongTensor(mask_out), 'meta':meta}
    def __len__(self):
        return len(self.meta_list)

def load_minibatch_h5(file_path, indices):
    with h5py.File(file_path, 'r') as fd:
        x = np.array(fd['x'][indices])
        mask = np.array(fd['mask'][indices])
    return x, mask

def load_attributes_h5(file_path):
    with h5py.File(file_path, 'r') as fd:
        # y = [json.loads(attr) for attr in fd['y']]
        y = list(fd['y'])
        splits = {}
        if 'split' in fd.keys():
            for key in fd['split'].keys():
                splits[key] = np.array(fd['split'][key])

        return y, splits

def get_point_list(mask_inst):
    from scipy.ndimage.morphology import distance_transform_edt
    mask_inst = mask_inst.squeeze()
    points = np.zeros(mask_inst.shape)

    point_list = []

    for i, u in enumerate(np.unique(mask_inst)):
        if u == 0:
            continue
        seg_ind = mask_inst==u
        dist = distance_transform_edt(seg_ind)
        yx = np.unravel_index(dist.argmax(), dist.shape)
        class_id = 1
        if class_id == 255:
            continue

        points[yx[0], yx[1]] = class_id

        point_list += [{'y':yx[0], 'x':yx[1], 'cls':class_id}]

    return points