# from . import semseg_cost
import torch
import os
import tqdm 
from . import semseg, semseg_counting
import torch


def get_model(model_dict, exp_dict=None, train_set=None):

    if model_dict['name'] in ["semseg_counting"]:
        model =  semseg_counting.SemSegCounting(exp_dict)

    if model_dict['name'] in ["semseg"]:
        model =  semseg.SemSeg(exp_dict)

        # load pretrained
        if 'pretrained' in model_dict:
            model.load_state_dict(torch.load(model_dict['pretrained']))
 
    return model





