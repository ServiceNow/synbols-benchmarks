# -*- coding: utf-8 -*-

import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu 
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_networks import infnet, fcn8_vgg16, unet_resnet
from src import utils as ut
from src.modules.lcfcn import lcfcn_loss

class SemSeg(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.exp_dict = exp_dict
        self.train_hashes = set()
        n_classes = self.n_classes = self.exp_dict['model'].get('n_classes', 1)
        self.network_name = self.exp_dict['model'].get('base', 'unet2d')
        if self.network_name == 'infnet':
            self.model_base = infnet.InfNet(n_classes=1,
                        loss=self.exp_dict['model']['loss'])

        if self.network_name == 'fcn8_vgg16':
            self.model_base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes)

        if self.network_name == "unet_resnet":
            self.model_base = unet_resnet.ResNetUNet(n_class=n_classes)

        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(
                self.model_base.parameters(), lr=self.exp_dict["lr"], betas=(0.99, 0.999))

        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(
                self.model_base.parameters(), lr=self.exp_dict["lr"])
        self.epoch = 0

    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict(),
                      'epoch':self.epoch}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        if 'opt' not in state_dict:
            return
        self.opt.load_state_dict(state_dict["opt"])
        self.epoch = state_dict['epoch']

    def train_on_loader(self, train_loader):
        
        self.train()
        self.epoch += 1
        n_batches = len(train_loader)

        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        train_monitor = TrainMonitor()
    
        for batch in train_loader:
            score_dict = self.train_on_batch(batch)
            train_monitor.add(score_dict)
            msg = ' '.join(["%s: %.3f" % (k, v) for k,v in train_monitor.get_avg_score().items()])
            pbar.set_description('Training - %s' % msg)
            pbar.update(1)
            
        pbar.close()

        if self.network_name == 'infnet':
            infnet.adjust_lr(self.opt,
                                self.epoch, decay_rate=0.1, decay_epoch=30)

        return train_monitor.get_avg_score()


    def train_on_batch(self, batch):
        # add to seen images
        for m in batch['meta']:
            self.train_hashes.add(m['hash'])

        if self.network_name == 'infnet':
            loss = self.model_base.train_on_batch(batch, self.opt)

        else:
            self.opt.zero_grad()

            images, labels = batch["images"], batch["masks"]
            images, labels = images.cuda(), labels.cuda()
            
            logits = self.model_base(images)

            # compute loss
            loss_name = self.exp_dict['model']['loss']
            if loss_name == 'cross_entropy':
                if self.n_classes == 1:
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='mean')
                else:
                    probs = F.log_softmax(logits, dim=1)
                    loss = F.nll_loss(probs, labels, reduction='mean', ignore_index=255)

            elif loss_name == 'joint_cross_entropy':
                loss = ut.joint_loss(logits, labels.float())
            
            elif loss_name == 'point_loss':
                points = batch['points'][:,None]
                ind = points!=255
                # self.vis_on_batch(batch, savedir_image='tmp.png')

                # POINT LOSS
                # loss = ut.joint_loss(logits, points[:,None].float().cuda(), ignore_index=255)
                # print(points[ind].sum())
                if ind.sum() == 0:
                    loss = 0.
                else:
                    loss = F.binary_cross_entropy_with_logits(logits[ind], 
                                            points[ind].float().cuda(), 
                                            reduction='mean')
                                            
                # print(points[ind].sum().item(), float(loss))
            
            elif loss_name == 'seam_loss':
                # (Alzayat) compute seam loss from https://github.com/YudeWang/SEAM/blob/master/train_SEAM.py
                x = (batch['masks'].cuda()*logits)
                n, c, h, w = x.size()
                k = h * w // 4
                x = torch.max(x, dim=1)[0]
                y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
                y = F.relu(y, inplace=False)
                loss = torch.sum(y) / (k * n)

            elif loss_name == 'single_stage_loss':
                # (Issam) compute single stage loss https://github.com/visinf/1-stage-wseg
                pass

            if loss != 0:
                loss.backward()
                if self.exp_dict['model'].get('clip_grad'):
                    ut.clip_gradient(self.opt, 0.5)
                self.opt.step()

            

            

        return {'train_loss': float(loss)}

    @torch.no_grad()
    def predict_on_batch(self, batch):
        self.eval()
        image = batch['images'].cuda()

        if self.network_name == 'infnet':
            s5, s4, s3, s2, se = self.model_base.forward(image)
            res = s2
            res = F.upsample(res, size=batch['meta'][0]['shape'],              
                         mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res > 0.5
            
        elif self.n_classes == 1:
            res = self.model_base.forward(image)
            res = F.upsample(res, size=batch['meta'][0]['shape'],              
                         mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy() > 0.5 
        else:
            self.eval()
            logits = self.model_base.forward(image)
            res = logits.argmax(dim=1).data.cpu().numpy()
        return res 

    def vis_on_batch(self, batch, savedir_image):
        image = batch['images']
        gt = np.asarray(batch['masks'], np.float32)
        gt /= (gt.max() + 1e-8)
        res = self.predict_on_batch(batch)

        image = F.interpolate(image, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        img_res = hu.save_image(savedir_image,
                     hu.denormalize(image, mode='rgb')[0],
                      mask=res[0], return_image=True)

        img_gt = hu.save_image(savedir_image,
                     hu.denormalize(image, mode='rgb')[0],
                      mask=gt[0], return_image=True)
        img_gt = text_on_image( 'Groundtruth', np.array(img_gt), color=(0,0,0))
        img_res = text_on_image( 'Prediction', np.array(img_res), color=(0,0,0))
        
        if 'points' in batch:
            img_gt = np.array(hu.save_image(savedir_image, img_gt/255.,
                                points=batch['points'][0].numpy()!=255, radius=2, return_image=True))
        img_list = [np.array(img_gt), np.array(img_res)]
        hu.save_image(savedir_image, np.hstack(img_list))

    def val_on_loader(self, loader, savedir_images=None, n_images=0):
       
        self.eval()
        val_list = []
        for i, batch in enumerate(tqdm.tqdm(loader)):
            if batch['masks'].sum()==0:
                continue
            val_list += [self.val_on_batch(batch)]
            if i < n_images:
                self.vis_on_batch(batch, savedir_image=os.path.join(savedir_images, 
                    '%d.png' % batch['meta'][0]['index']))
            
        return pd.DataFrame(val_list).mean().to_dict()
        
    def val_on_batch(self, batch):
        # make sure it wasn't trained on
        for m in batch['meta']:
            assert(m['hash'] not in self.train_hashes)

        self.eval()
        image = batch['images']
        gt_mask = np.array(batch['masks'])
        prob_mask = self.predict_on_batch(batch)

        T = 0.1

        pred_mask = prob_mask > T
        
        # FP+TP
        NumRec = (pred_mask == 1).sum()
        # FN+TN
        NumNoRec = (pred_mask == 0).sum()
        # LabelAnd
        LabelAnd = pred_mask & gt_mask
        # TP
        NumAnd = (LabelAnd == 1).sum() 

        # TP + FN
        num_obj = gt_mask.sum()

        # FP + TP
        num_pred = pred_mask.sum()

        FN = num_obj-NumAnd;
        FP = NumRec-NumAnd;
        TN = NumNoRec-FN;

        val_dict = {}

        if NumAnd == 0:
            val_dict['PreFtem'] = 0
            val_dict['RecallFtem'] = 0
            val_dict['FmeasureF'] = 0
            val_dict['Dice'] = 0
            val_dict['SpecifTem'] = 0
        else:
            val_dict['PreFtem'] = NumAnd/NumRec
            val_dict['RecallFtem'] = NumAnd/num_obj
            val_dict['SpecifTem'] = TN/(TN+FP)
            val_dict['Dice'] = 2 * NumAnd/(num_obj+num_pred) 
            val_dict['FmeasureF'] = (( 2.0 * val_dict['PreFtem'] * val_dict['RecallFtem'] ) /
                                         (val_dict['PreFtem'] + val_dict['RecallFtem']))

        val_dict['%s_score' % batch['meta'][0]['split']] = val_dict['Dice']
        # pprint.pprint(val_dict)
        return val_dict
        """
        NumRec = length( find( Label3==1 ) ); %FP+TP
        NumNoRec = length(find(Label3==0)); % FN+TN
        LabelAnd = Label3 & gtMap;
        NumAnd = length( find ( LabelAnd==1 ) ); %TP
        num_obj = sum(sum(gtMap));  %TP+FN
        num_pred = sum(sum(Label3)); % FP+TP

        FN = num_obj-NumAnd;
        FP = NumRec-NumAnd;
        TN = NumNoRec-FN;
        %SpecifTem = TN/(TN+FP)
        %Precision = TP/(TP+FP)

        if NumAnd == 0
            PreFtem = 0;
            RecallFtem = 0;
            FmeasureF = 0;
            Dice = 0;
            SpecifTem = 0;
        else
            PreFtem = NumAnd/NumRec;
            RecallFtem = NumAnd/num_obj;
            SpecifTem = TN/(TN+FP);
            Dice = 2 * NumAnd/(num_obj+num_pred);
        %     FmeasureF = ( ( 1.3* PreFtem * RecallFtem ) / ( .3 * PreFtem + RecallFtem ) ); % beta = 0.3
            FmeasureF = (( 2.0 * PreFtem * RecallFtem ) / (PreFtem + RecallFtem)); % beta = 1.0
        end
        """


def text_on_image(text, image, color=None):
    """Adds test on the image
    
    Parameters
    ----------
    text : [type]
        [description]
    image : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,40)
    fontScale              = 0.8
    if color is None:
        fontColor              = (1,1,1)
    else:
        fontColor              = color
    lineType               = 1
    # img_mask = skimage.transform.rescale(np.array(img_mask), 1.0)
    # img_np = skimage.transform.rescale(np.array(img_points), 1.0)
    img_np = cv2.putText(image, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness=2
        # lineType
        )
    return img_np


class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}
