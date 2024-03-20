from __future__ import absolute_import, division, print_function
import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from time import time
import torch
import torch.utils.data as data
from torchvision import transforms
import cv2

cv2.setNumThreads(0)
import glob
import utils
import torch.nn.functional as F
from utils import npy
import json
from mmdet.apis import init_detector, inference_detector
import mmcv
from skimage.metrics import structural_similarity

import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
        print(rot_xyz.shape)
        print(depth_values.shape)
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

def main():
    json_path = "/home/cjd/tmp/DDAD_video.json"
    data_path_ori = '/data/cjd/ddad/ddad_train_val/'
    data_path_root = '/data/cjd/ddad/my_ddad/'
    data_path = os.path.join(data_path_root, 'val/')
    f = open(json_path, 'r')
    content_all = f.read()
    json_list_all = json.loads(content_all)
    f.close()
    file_names = json_list_all["val"]
    file_names = [x for x in file_names if 'timestamp' in x.keys() and 'timestamp_back' in x.keys() and 'timestamp_forward' in x.keys() and x['Camera'] == 'CAMERA_01']

    model = init_detector("/home/cjd/mmdetection3d-master/configs/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim.py", "/home/cjd/MVS2D/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_20201008_211222-0b16ac4b.pth", device = 'cuda:3')
    # print(model.CLASSES)
    class_all = model.CLASSES
    print(class_all)
    lenth = len(file_names)
    print(lenth)
    for index in range(lenth):
        inputs = {}
        # cur_img_path = data_path_ori + str(file_names[index]['video_num']) + '/rgb/' + file_names[index]['Camera'] +'/'+ str(file_names[index]['timestamp']) + '.png'
        cur_npz_path = data_path + str(file_names[index]['timestamp']) + '_' + file_names[index]['Camera'] + '.npz'
        pre_npz_path = data_path + str(file_names[index]['timestamp_back']) + '_' + file_names[index]['Camera'] + '.npz'
        next_npz_path = data_path + str(file_names[index]['timestamp_forward']) + '_' + file_names[index]['Camera'] + '.npz'

        file_cur = np.load(cur_npz_path)
        file_pre = np.load(pre_npz_path)
        file_next = np.load(next_npz_path)


        depth_cur_gt = file_cur['depth']
        depth_cur_gt = np.array(depth_cur_gt).astype(np.float32)

        inputs[("depth_gt", 0, 0)] = torch.from_numpy(depth_cur_gt)

        rgb_cur = file_cur['rgb']
        # print(rgb_cur.shape)
        rgb_cur_input = cv2.cvtColor(rgb_cur, cv2.COLOR_BGR2RGB)
        rgb_cur_input = torch.from_numpy(rgb_cur_input).permute(2, 0, 1) / 255.
        inputs[("color", 0, 0)] = rgb_cur_input
        # cv2.imwrite('img.png', rgb_cur)
        pose_cur = file_cur['pose']
        pose_cur = np.linalg.inv(pose_cur).astype('float32')
        inputs[("pose", 0)] = pose_cur
        rgb_pre = file_pre['rgb']
        rgb_pre_input = cv2.cvtColor(rgb_pre, cv2.COLOR_BGR2RGB)
        rgb_pre_input = torch.from_numpy(rgb_pre_input).permute(2, 0, 1) / 255.
        inputs[("color", 1, 0)] = rgb_pre_input
        pose_pre = file_pre['pose']
        pose_pre = np.linalg.inv(pose_pre).astype('float32')
        inputs[("pose", 1)] = pose_pre 
        rgb_next = file_next['rgb']
        rgb_next_input = cv2.cvtColor(rgb_next, cv2.COLOR_BGR2RGB)
        rgb_next_input = torch.from_numpy(rgb_next_input).permute(2, 0, 1) / 255.
        inputs[("color", 2, 0)] = rgb_next_input
        pose_next = file_next['pose']
        pose_next = np.linalg.inv(pose_next).astype('float32')  
        inputs[("pose", 2)] = pose_next 
        K = file_cur['intrinsics']

        inv_K = np.linalg.inv(K)

        K_pool = {}
        ho, wo, _ = rgb_cur.shape
        for i in range(6):
            K_pool[(ho // 2**i, wo // 2**i)] = K.copy().astype('float32')
            K_pool[(ho // 2**i, wo // 2**i)][:2, :] /= 2**i

        inputs['K_pool'] = K_pool

        inputs[("inv_K_pool", 0)] = {}
        for k, v in K_pool.items():
            K44 = np.eye(4)
            K44[:3, :3] = v
            inputs[("inv_K_pool", 0)][k] = np.linalg.inv(K44).astype('float32')

        inputs[("inv_K", 0)] = torch.from_numpy(inv_K.astype('float32'))

        inputs[("K", 0)] = torch.from_numpy(K.astype('float32'))

        for i in range(3):
            inputs[("proj", i)] = {}
            for k, v in inputs['K_pool'].items():
                K44 = np.eye(4)
                K44[:3, :3] = v
                inputs[("proj",
                        i)][k] = torch.from_numpy(np.matmul(K44, inputs[("pose",
                                                        i)]).astype('float32'))
        to_gpu(inputs)
        h, w, _ = rgb_cur.shape
        imgs, proj_mats, pose_mats = [], [], []
        for i in range(3):
            imgs.append(inputs[('color', i, 0)])
            proj_mats.append(inputs[('proj', i)])
            pose_mats.append(inputs[('pose', i)])

        depth_gt = inputs[("depth_gt", 0, 0)][None,None,:,:]
        img0 = imgs[0][None,:,:,:]
        img1 = imgs[1][None,:,:,:]

        proj_mats_0 = proj_mats[0][(h, w)][None,:,:]

        proj_mats_1 = proj_mats[1][(h, w)][None,:,:]
        # print(img1.shape)
        # print(proj_mats_0.shape)
        # print(depth_gt.shape)
        warped_img0 = homo_warping_depth(img1, proj_mats_1, proj_mats_0, depth_gt)
        img0_np = img0[0].cpu().detach().numpy().squeeze().transpose(1,2,0)
        warped_img0_np = warped_img0[0].cpu().detach().numpy().squeeze().transpose(1,2,0)
        depth_gt_np = depth_gt.cpu().detach().numpy().squeeze()
        # img0_np = (img0_np / img0_np.max() * 255).astype(np.uint8)
        # cv2.imwrite('img0.png', img0_np)

        # warped_img0_np = (warped_img0_np / warped_img0_np.max() * 255).astype(np.uint8)
        # cv2.imwrite('warped_img.png', warped_img0_np)
        # print(rgb_cur.shape)
        # img = mmcv.imread(cur_img_path)
        result = inference_detector(model, rgb_cur)

        index_list = [0,1,2,3,4,5,6,7]

        mask_all = np.zeros_like(rgb_cur[:,:,0], dtype = bool)
        for index_ in index_list:
            object_number = len(result[1][index_])
            for index_object in range(object_number):
                mask_now = result[1][index_][index_object] & (depth_gt_np > 0)
                # diff_now = warped_img0_np[mask_now] -  img0_np[mask_now]
                if np.sum(mask_now.astype(float)) > 50:
                    ssim = structural_similarity(img0_np[mask_now], warped_img0_np[mask_now], multichannel = True)
                else:
                    ssim = 0.3
                print(ssim)
                if result[0][index_][index_object][4] > 0.5 and ssim < 0.75:
                # print(result[0][index_][index_object])
                    mask_all = mask_all + result[1][index_][index_object]

        # # car_number = len(result[1][0])
        
        # # for car_index in range(car_number):
        # #     car_mask = car_mask + result[1][0][car_index]
        mask_all_vis = mask_all.astype(np.float)
        mask_all_vis = (mask_all_vis*255).astype(np.uint8)

        save_path = cur_npz_path.replace('.npz', '_dynamic.npz')
        print(save_path)

        np.savez(save_path, mask_all)

        # cv2.imwrite('seg_mask.png',mask_all_vis)
        # a = input('print something')
        # print(a)


if __name__ == "__main__":
    main()