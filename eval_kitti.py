#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import torch
import torch.nn as nn
from options import MVS2DOptions, EvalCfg
import networks
from torch.utils.data import DataLoader
from datasets.kitti import DDAD_kitti
from hybrid_evaluate_depth import evaluate_depth_maps, compute_errors,compute_errors1,compute_errors_perimage
import torch.nn.functional as F
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def to_gpu(inputs, keys=None):
    if keys == None:
        keys = inputs.keys()
    for key in keys:
        if key not in inputs:
            continue
        ipt = inputs[key]
        if type(ipt) == torch.Tensor:
            inputs[key] = ipt.cuda()
        elif type(ipt) == list and type(ipt[0]) == torch.Tensor:
            inputs[key] = [
                x.cuda() for x in ipt
            ]
        elif type(ipt) == dict:
            for k in ipt.keys():
                if type(ipt[k]) == torch.Tensor:
                    ipt[k] = ipt[k].cuda()


options = MVS2DOptions()
opts = options.parse()

# opts.cfg = "./configs/kitti.conf"
dataset = DDAD_kitti(opts, False)
data_loader = DataLoader(dataset,
                           1,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True,
                           drop_last=False,
                           sampler=None)
model = networks.MVS2D(opt=opts).cuda()
pretrained_dict = torch.load("pretrained_model/kitti/model_kitti.pth")

model.load_state_dict(pretrained_dict)
model.eval()

min_depth = opts.EVAL_MIN_DEPTH
max_depth = opts.EVAL_MAX_DEPTH

index = 0
total_result_sum = {}
total_result_count = {}
with torch.no_grad():
    for batch_idx, inputs in enumerate(data_loader):
        to_gpu(inputs)

        imgs, proj_mats, pose_mats = [], [], []
        for i in range(inputs['num_frame'][0].item()):
            imgs.append(inputs[('color', i, 0)])
            proj_mats.append(inputs[('proj', i)])
            pose_mats.append(inputs[('pose', i)])

        depth_gt = inputs[("depth_gt", 0, 0)]
        depth_gt_np = depth_gt.cpu().detach().numpy().squeeze()
        mask = (depth_gt_np>min_depth) & (depth_gt_np < max_depth)

        if np.sum(mask.astype(np.float32)) > 5:

            outputs = model(imgs[0], imgs[1:], pose_mats[0], pose_mats[1:],
                                 inputs[('inv_K_pool', 0)])
            depth_pred_1_tensor = outputs[('depth_pred', 0)]
            depth_pred_2_tensor = outputs[('depth_pred_2', 0)]

            depth_pred_2 = depth_pred_2_tensor.cpu().detach().numpy().squeeze()
            depth_pred_1 = depth_pred_1_tensor.cpu().detach().numpy().squeeze()

            error_temp = compute_errors_perimage(depth_gt_np[mask], depth_pred_1[mask], min_depth, max_depth)
            error_temp_2_ = compute_errors_perimage(depth_gt_np[mask], depth_pred_2[mask], min_depth, max_depth)
            print('cur',index, error_temp)
            index = index + 1
            error_temp_2 = {}
            for k,v in error_temp_2_.items():
                new_k = k + '_2'
                error_temp_2[new_k] = error_temp_2_[k]

            error_temp_all = {}
            error_temp_all.update(error_temp)
            error_temp_all.update(error_temp_2)

            for k,v in error_temp_all.items():
                if not isinstance(v,float):
                    v=v.items()
                if k in total_result_sum:
                    total_result_sum[k] = total_result_sum[k] + v
                else:
                    total_result_sum[k] = v

    for k in total_result_sum.keys():
        total_result_count[k] = total_result_sum['valid_number']

    print('final####################################')

    for k in total_result_sum.keys():
        this_tensor = torch.tensor([total_result_sum[k], total_result_count[k]])
        this_list = [this_tensor]
        this_tensor = this_list[0].detach().cpu().numpy()
        reduce_sum = this_tensor[0].item()
        reduce_count = this_tensor[1].item()
        reduce_mean = reduce_sum / reduce_count
        print(k, reduce_mean)

