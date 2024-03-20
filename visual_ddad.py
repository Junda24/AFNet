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
from datasets.DDAD import DDAD
import torch.nn.functional as F
from utils import *

def resize_depth_preserve(depth, shape):
    """
    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """

    # Store dimensions and reshapes to single column
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h)).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w)).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return np.expand_dims(depth, axis=0)


def homo_warping_depth(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    #height, width = src_fea.shape[2], src_fea.shape[3]
    h_src, w_src = src_fea.shape[2], src_fea.shape[3]
    h_ref, w_ref = depth_values.shape[2], depth_values.shape[3]

    with torch.no_grad():

        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        
        y, x = torch.meshgrid([torch.arange(0, h_ref, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, w_ref, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h_ref * w_ref), x.view(h_ref * w_ref)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]

        rot_xyz = torch.matmul(rot, xyz)
        rot_depth_xyz = rot_xyz * depth_values.view(batch, 1, -1)

        proj_xyz = rot_depth_xyz + trans.view(batch,3,1)

        proj_xy = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :]  # [B, 2, Ndepth, H*W]
        z = proj_xyz[:, 2:3, :].view(batch, h_ref, w_ref)
        proj_x_normalized = proj_xy[:, 0, :] / ((w_src - 1) / 2.0) - 1
        proj_y_normalized = proj_xy[:, 1, :] / ((h_src - 1) / 2.0) - 1
        X_mask = ((proj_x_normalized > 1)+(proj_x_normalized < -1)).detach()
        proj_x_normalized[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((proj_y_normalized > 1)+(proj_y_normalized < -1)).detach()
        proj_y_normalized[Y_mask] = 2
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
        proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, h_ref, w_ref)
        proj_mask = (proj_mask + (z <= 0)) > 0

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, h_ref, w_ref, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, h_ref, w_ref)

    #return warped_src_fea , proj_mask
    return warped_src_fea


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

# opts.width = int(640)
# opts.height = int(480)
dataset = DDAD(opts, False)
data_loader = DataLoader(dataset,
                           1,
                           shuffle=False,
                           num_workers=1,
                           pin_memory=True,
                           drop_last=False,
                           sampler=None)
model = networks.MVS2D(opt=opts).cuda()
pretrained_dict = torch.load("/home/cjd/MVS2D/log/AFNet/models/weights_latest/model.pth")
model_dict = model.state_dict()
pretrained_dict = {
    k: v
    for k, v in pretrained_dict.items() if k in model_dict
    }
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model.eval()
root_path = '/data/cjd/AFnet/visual/ddad/'
with torch.no_grad():
    for batch_idx, inputs in enumerate(data_loader):
        print(batch_idx)
        to_gpu(inputs)

        imgs, proj_mats, pose_mats = [], [], []
        for i in range(inputs['num_frame'][0].item()):
            imgs.append(inputs[('color', i, 0)])
            proj_mats.append(inputs[('proj', i)])
            pose_mats.append(inputs[('pose', i)])

        pose_mats[0] = pose_mats[0]*0.75
        pose_mats[1] = pose_mats[1]*0.75
        pose_mats[2] = pose_mats[2]*0.75

        outputs = model(imgs[0], imgs[1:], pose_mats[0], pose_mats[1:], inputs[('inv_K_pool', 0)])

        depth_gt = inputs[("depth_gt", 0, 0)][0].cpu().detach().numpy().squeeze()
        depth_gt = resize_depth_preserve(depth_gt, (608,960))
        depth_gt_path = os.path.join(root_path,'depth_gt','{}.png'.format(batch_idx))
        depth_gt_np = gray_2_colormap_np_2(depth_gt ,max = 120)[:,:,::-1]

        img0 = imgs[0]


        depth_pred = outputs[('depth_pred', 0)][0]
        depth_pred_2 = outputs[('depth_pred_2', 0)][0]

        depth_pred_np = gray_2_colormap_np(depth_pred ,max = 120)[:,:,::-1]
        depth_pred_2_np = gray_2_colormap_np(depth_pred_2 ,max = 120)[:,:,::-1]
        img0_path = os.path.join(root_path,'img0', '{}.png'.format(batch_idx))
        depth_1_path = os.path.join(root_path,'depth_1','{}.png'.format(batch_idx))
        depth_2_path = os.path.join(root_path,'depth_2', '{}.png'.format(batch_idx))


        img0_np = img0[0].cpu().detach().numpy().squeeze().transpose(1,2,0)
        img0_np = (img0_np / img0_np.max() * 255).astype(np.uint8)
        cv2.imwrite(img0_path, img0_np)
        cv2.imwrite(depth_1_path, depth_pred_np)
        cv2.imwrite(depth_2_path, depth_pred_2_np)
        cv2.imwrite(depth_gt_path, depth_gt_np)






        # a = input('input some')
        # print(a)

        # break
