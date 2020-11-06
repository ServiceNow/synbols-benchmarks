from . import fcn8_vgg16, unet2d, unet_resnet, fcn8_resnet, deeplab

from torchvision import models
import torch
import torch.nn as nn



def get_base(base_name, exp_dict, n_classes):
    if base_name == "deeplab":
        base = deeplab.DeepLab(pretrained_backbone=True)
        

    if base_name == "fcn8_vgg16":
        base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes)

    if base_name == "fcn8_resnet50":
        base = fcn8_resnet.FCN8(n_classes=n_classes)

    if base_name == "unet2d":
        base = unet2d.UNet(n_channels=exp_dict['model'].get('n_channels', 1), n_classes=n_classes)

    if base_name == "unet_resnet":
        base = unet_resnet.ResNetUNet(n_classes=n_classes)


    if base_name == 'pspnet':
      
        if exp_dict['model']['base'] == 'pspnet':
            net_fn = smp.PSPNet

        assert net_fn is not None

        base = smp.PSPNet(encoder_name=exp_dict['model']['encoder'],
                          in_channels= exp_dict['model']['n_channels'],
                          encoder_weights= None,  # ignore error. it still works.
                           classes= n_classes,
                           psp_use_batchnorm=False,
                           psp_dropout=0.5)

    return base

