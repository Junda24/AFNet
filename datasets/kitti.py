# dataloader for KITTI / when training & testing F-Net and MaGNet
import os
import random
import glob

import numpy as np
import torch
import torch.utils.data as data
import torch.utils.data.distributed
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pykitti

import cv2

import utils
import torch.nn.functional as F
from utils import npy
import json



class DDAD_kitti(data.Dataset):
    def __init__(self, opt, is_train):
        super(DDAD_kitti, self).__init__()
        self.opt = opt
        self.is_train = is_train
        if self.is_train:
            with open("./data_split/kitti_eigen_train.txt", 'r') as f:
                self.filenames = f.readlines()
        else:
            with open("./data_split/kitti_eigen_test.txt", 'r') as f:
                self.filenames = f.readlines()

        # self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.dataset_path = self.opt.data_path

        # local window
        self.window_radius = int(2)
        self.n_views = int(2)
        self.frame_interval = self.window_radius // (self.n_views // 2)
        self.img_idx_center = self.n_views // 2

        # window_idx_list
        self.window_idx_list = list(range(-self.n_views // 2, (self.n_views // 2) + 1))
        self.window_idx_list = [i * self.frame_interval for i in self.window_idx_list]

        # image resolution
        self.img_H = self.opt.height  # 352
        self.img_W = self.opt.width   # 1216

    def __len__(self):
        return len(self.filenames)

    # get camera intrinscs
    def get_cam_intrinsics(self, p_data):
        raw_img_size = p_data.get_cam2(0).size
        raw_W = int(raw_img_size[0])
        raw_H = int(raw_img_size[1])

        top_margin = int(raw_H - 352)
        left_margin = int((raw_W - 1216) / 2)

        # original intrinsic matrix (4X4)
        IntM_ = p_data.calib.K_cam2

        # updated intrinsic matrix
        IntM = np.zeros((3, 3))
        IntM[2, 2] = 1.
        IntM[0, 0] = IntM_[0, 0]
        IntM[1, 1] = IntM_[1, 1]
        IntM[0, 2] = (IntM_[0, 2] - left_margin) 
        IntM[1, 2] = (IntM_[1, 2] - top_margin) 

        IntM = IntM.astype(np.float32)
        return IntM

    def __getitem__(self, idx):
        inputs = {}
        date, drive, mode, img_idx = self.filenames[idx].split(' ')
        img_idx = int(img_idx)
        scene_name = '%s_drive_%s_sync' % (date, drive)

        # identify the neighbor views
        img_idx_list = [img_idx + i for i in self.window_idx_list]
        p_data = pykitti.raw(self.dataset_path + '/rawdata', date, drive, frames=img_idx_list)

        # cam intrinsics
        cam_intrins = self.get_cam_intrinsics(p_data)

        # color augmentation
        color_aug = False
        if self.is_train:
            if random.random() > 0.5:
                color_aug = True
                aug_gamma = random.uniform(0.9, 1.1)
                aug_brightness = random.uniform(0.9, 1.1)
                aug_colors = np.random.uniform(0.9, 1.1, size=3)

        # data array
        data_array = []
        for i in range(self.n_views + 1):
            cur_idx = img_idx_list[i]

            # read img
            img_name = '%010d.png' % cur_idx
            img_path = self.dataset_path + '/rawdata/{}/{}/image_02/data/{}'.format(date, scene_name, img_name)
            img = Image.open(img_path).convert("RGB")

            # kitti benchmark crop
            height = img.height
            width = img.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            img = img.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # to tensor
            img = np.array(img).astype(np.float32) / 255.0      # (H, W, 3)
            img_ori = torch.from_numpy(img).permute(2, 0, 1)
            if color_aug:
                img = self.augment_image(img, aug_gamma, aug_brightness, aug_colors)
            img = torch.from_numpy(img).permute(2, 0, 1)        # (3, H, W)
            img = self.normalize(img)

            # read dmap (only for the ref img)
            if i == self.img_idx_center:
                dmap_path = self.dataset_path + '/{}/{}/proj_depth/groundtruth/image_02/{}'.format(mode, scene_name,
                                                                                                   img_name)
                gt_dmap = Image.open(dmap_path).crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                gt_dmap = np.array(gt_dmap)[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
                gt_dmap = gt_dmap / 256.0
                gt_dmap = torch.from_numpy(gt_dmap).permute(2, 0, 1)  # (1, H, W)
            else:
                gt_dmap = 0.0

            # read extM
            pose = p_data.oxts[i].T_w_imu
            M_imu2cam = p_data.calib.T_cam2_imu
            extM = np.matmul(M_imu2cam, np.linalg.inv(pose))
            extM = extM.astype('float32')

            data_dict = {
                'img_ori': img_ori,
                'img': img,
                'gt_dmap': gt_dmap,
                'extM': extM,
                'scene_name': scene_name,
                'img_idx': str(img_idx),
            }
            data_array.append(data_dict)

        inputs[("color", 0, 0)] = data_array[1]['img']
        inputs[("img_ori", 0, 0)] = data_array[1]['img_ori']

        inputs[("depth_gt", 0, 0)] = data_array[1]['gt_dmap']
        inputs[("pose", 0)] = data_array[1]['extM']

        inputs[("color", 1, 0)] = data_array[0]['img']
        inputs[("pose", 1)] = data_array[0]['extM']
        inputs[("img_ori", 1, 0)] = data_array[0]['img_ori']

        inputs[("color", 2, 0)] = data_array[2]['img']
        inputs[("pose", 2)] = data_array[2]['extM']
        inputs[("img_ori", 2, 0)] = data_array[2]['img_ori']


        inputs = self.get_K(cam_intrins, inputs)

        inputs = self.compute_projection_matrix(inputs)

        inputs['num_frame'] = 3

        return inputs

    def augment_image(self, image, gamma, brightness, colors):
        # gamma augmentation
        image_aug = image ** gamma

        # brightness augmentation
        image_aug = image_aug * brightness

        # color augmentation
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def get_K(self, K, inputs):
        inv_K = np.linalg.inv(K)
        K_pool = {}
        ho, wo = self.opt.height, self.opt.width
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
    
        return inputs


    def compute_projection_matrix(self, inputs):
        for i in range(3):
            inputs[("proj", i)] = {}
            for k, v in inputs['K_pool'].items():
                K44 = np.eye(4)
                K44[:3, :3] = v
                inputs[("proj",
                        i)][k] = np.matmul(K44, inputs[("pose",
                                                        i)]).astype('float32')
        return inputs