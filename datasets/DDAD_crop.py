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



class DDAD(data.Dataset):
    def __init__(self, opt, is_train):
        super(DDAD, self).__init__()
        self.opt = opt
        self.json_path = "/home/cjd/tmp/DDAD_video.json"
        self.data_path_root = '/data/cjd/ddad/my_ddad/'
        self.is_train = is_train
        if self.is_train:
            self.data_path = os.path.join(self.data_path_root, 'train/')
        else:
            self.data_path = os.path.join(self.data_path_root, 'val/')

        f = open(self.json_path, 'r')
        content_all = f.read()
        json_list_all = json.loads(content_all)
        f.close()

        if self.is_train:
            self.file_names = json_list_all["train"]
            # self.file_names = self.file_names[:300]
            print('train', len(self.file_names))
        else:
            self.file_names = json_list_all["val"]
            # self.file_names = self.file_names[:300]

            print('val', len(self.file_names))

        print('filter_pre', len(self.file_names))
        self.file_names = [x for x in self.file_names if 'timestamp' in x.keys() and 'timestamp_back' in x.keys() and 'timestamp_forward' in x.keys() and x['Camera'] == 'CAMERA_01']
        print('filter_after', len(self.file_names))
        self.file_names = self.file_names[:50]



    def get_k_ori_randomcrop(self, k_raw, inputs, x1, y1):

        fx_ori = k_raw[0,0]
        fy_ori = k_raw[1,1]
        fx_virtual = 1060.0
        fx_scale = fx_ori / fx_virtual
        # print('fx_scale', fx_scale)
        inputs[("depth_gt", 0, 0)] = inputs[("depth_gt", 0, 0)] / fx_scale

        pose_cur = inputs[("pose", 0)]
        pose_cur[:3, 3] = pose_cur[:3, 3] / fx_scale
        inputs[("pose", 0)] = pose_cur
        inputs[("pose_inv", 0)] = np.linalg.inv(inputs[("pose", 0)])

        pose_pre = inputs[("pose", 1)]
        pose_pre[:3, 3] = pose_pre[:3, 3] / fx_scale
        inputs[("pose", 1)] = pose_pre
        inputs[("pose_inv", 1)] = np.linalg.inv(inputs[("pose", 1)])

        pose_next = inputs[("pose", 2)]
        pose_next[:3, 3] = pose_next[:3, 3] / fx_scale
        inputs[("pose", 2)] = pose_next
        inputs[("pose_inv", 2)] = np.linalg.inv(inputs[("pose", 2)])

        # inputs['focal_scale'] = float(fx_scale)

        K = np.zeros((3,3), dtype = float)
        K[0,0] = fx_ori
        K[1,1] = fy_ori
        K[2,2] = 1.0
        K[0,2] = k_raw[0,2]
        K[1,2] = k_raw[1,2]

        h_crop = y1 - self.opt.height
        w_crop = x1

        K[0,2] = K[0,2] - w_crop
        K[1,2] = K[1,2] - h_crop

        return K, inputs

    def get_k_ori_centercrop(self, k_raw, inputs, x_center, y_center):

        fx_ori = k_raw[0,0]
        fy_ori = k_raw[1,1]
        fx_virtual = 1060.0
        fx_scale = fx_ori / fx_virtual
        # print('fx_scale', fx_scale)
        inputs[("depth_gt", 0, 0)] = inputs[("depth_gt", 0, 0)] / fx_scale

        pose_cur = inputs[("pose", 0)]
        pose_cur[:3, 3] = pose_cur[:3, 3] / fx_scale
        inputs[("pose", 0)] = pose_cur
        inputs[("pose_inv", 0)] = np.linalg.inv(inputs[("pose", 0)])

        pose_pre = inputs[("pose", 1)]
        pose_pre[:3, 3] = pose_pre[:3, 3] / fx_scale
        inputs[("pose", 1)] = pose_pre
        inputs[("pose_inv", 1)] = np.linalg.inv(inputs[("pose", 1)])

        pose_next = inputs[("pose", 2)]
        pose_next[:3, 3] = pose_next[:3, 3] / fx_scale
        inputs[("pose", 2)] = pose_next
        inputs[("pose_inv", 2)] = np.linalg.inv(inputs[("pose", 2)])

        # inputs['focal_scale'] = float(fx_scale)

        K = np.zeros((3,3), dtype = float)
        K[0,0] = fx_ori
        K[1,1] = fy_ori
        K[2,2] = 1.0
        K[0,2] = k_raw[0,2]
        K[1,2] = k_raw[1,2]

        h_crop = int(y_center - 0.5*self.opt.height)
        w_crop = int(x_center - 0.5*self.opt.width)

        K[0,2] = K[0,2] - w_crop
        K[1,2] = K[1,2] - h_crop

        return K, inputs


    def __len__(self):
        return len(self.file_names)


    def __getitem__(self, index):
        inputs = {}         
        cur_npz_path = self.data_path + str(self.file_names[index]['timestamp']) + '_' + self.file_names[index]['Camera'] + '.npz'
        pre_npz_path = self.data_path + str(self.file_names[index]['timestamp_back']) + '_' + self.file_names[index]['Camera'] + '.npz'
        next_npz_path = self.data_path + str(self.file_names[index]['timestamp_forward']) + '_' + self.file_names[index]['Camera'] + '.npz'

        cur_mask_path = cur_npz_path.replace('.npz', '_dynamic.npz')
        inputs['dynamic_mask'] = cur_mask_path
        file_cur = np.load(cur_npz_path)
        file_pre = np.load(pre_npz_path)
        file_next = np.load(next_npz_path)

        depth_cur_gt = file_cur['depth']
        depth_cur_gt = np.array(depth_cur_gt).astype(np.float32)


        inputs[("depth_gt", 0, 0)] = depth_cur_gt


        if self.is_train:
            #h_ori=1216, w_ori=1936
            if random.randint(0, 10) < 8:
                y_center = int((1216 + self.opt.height)/2)
                y1 = random.randint(int(y_center - 70), int(y_center + 50))
            else:
                y1 = random.randint(self.opt.height, 1216)
            x1 = random.randint(0, 1930 - self.opt.width)
        # else:
        #     y1 = int((1216 + self.opt.height)/2)
        #     x1 = int((1936 - self.opt.width)/2)

            inputs[("depth_gt", 0, 0)] = inputs[("depth_gt", 0, 0)][y1 - int(self.opt.height):y1, x1:x1+int(self.opt.width)][None, :, :]

            rgb_cur = file_cur['rgb']
            rgb_cur = rgb_cur[y1 - int(self.opt.height):y1, x1:x1+int(self.opt.width)]
            rgb_cur = cv2.cvtColor(rgb_cur, cv2.COLOR_BGR2RGB)
            rgb_cur = torch.from_numpy(rgb_cur).permute(2, 0, 1) / 255.
            inputs[("color", 0, 0)] = rgb_cur



            rgb_pre = file_pre['rgb']
            rgb_pre = rgb_pre[y1 - int(self.opt.height):y1, x1:x1+int(self.opt.width)]
            rgb_pre = cv2.cvtColor(rgb_pre, cv2.COLOR_BGR2RGB)
            rgb_pre = torch.from_numpy(rgb_pre).permute(2, 0, 1) / 255.
            inputs[("color", 1, 0)] = rgb_pre


            rgb_next = file_next['rgb']
            rgb_next = rgb_next[y1 - int(self.opt.height):y1, x1:x1+int(self.opt.width)]
            rgb_next = cv2.cvtColor(rgb_next, cv2.COLOR_BGR2RGB)
            rgb_next = torch.from_numpy(rgb_next).permute(2, 0, 1) / 255.
            inputs[("color", 2, 0)] = rgb_next


            pose_cur = file_cur['pose']
            pose_cur = np.linalg.inv(pose_cur).astype('float32')
            inputs[("pose", 0)] = pose_cur

            pose_pre = file_pre['pose']
            pose_pre = np.linalg.inv(pose_pre).astype('float32')            
            inputs[("pose", 1)] = pose_pre

            pose_next = file_next['pose']
            pose_next = np.linalg.inv(pose_next).astype('float32')            
            inputs[("pose", 2)] = pose_next

            k_raw = file_cur['intrinsics']
            # print('k_raw',k_raw)
            k_crop, inputs = self.get_k_ori_randomcrop(k_raw, inputs, x1, y1)
            inputs = self.get_K(k_crop, inputs)


        else:
            x_center = int((1936 + self.opt.width)/2)
            y_center = int((1216 + self.opt.height)/2)

            inputs[("depth_gt", 0, 0)] = inputs[("depth_gt", 0, 0)][int(y_center - 0.5*self.opt.height):int(y_center + 0.5*self.opt.height), int(x_center - 0.5*self.opt.width):int(x_center + 0.5*self.opt.width)][None, :, :]
            print(inputs[("depth_gt", 0, 0)].shape)
            rgb_cur = file_cur['rgb']
            rgb_cur = rgb_cur[int(y_center - 0.5*self.opt.height):int(y_center + 0.5*self.opt.height), int(x_center - 0.5*self.opt.width):int(x_center + 0.5*self.opt.width)]
            rgb_cur = cv2.cvtColor(rgb_cur, cv2.COLOR_BGR2RGB)
            rgb_cur = torch.from_numpy(rgb_cur).permute(2, 0, 1) / 255.
            inputs[("color", 0, 0)] = rgb_cur



            rgb_pre = file_pre['rgb']
            rgb_pre = rgb_pre[int(y_center - 0.5*self.opt.height):int(y_center + 0.5*self.opt.height), int(x_center - 0.5*self.opt.width):int(x_center + 0.5*self.opt.width)]
            rgb_pre = cv2.cvtColor(rgb_pre, cv2.COLOR_BGR2RGB)
            rgb_pre = torch.from_numpy(rgb_pre).permute(2, 0, 1) / 255.
            inputs[("color", 1, 0)] = rgb_pre


            rgb_next = file_next['rgb']
            rgb_next = rgb_next[int(y_center - 0.5*self.opt.height):int(y_center + 0.5*self.opt.height), int(x_center - 0.5*self.opt.width):int(x_center + 0.5*self.opt.width)]
            rgb_next = cv2.cvtColor(rgb_next, cv2.COLOR_BGR2RGB)
            rgb_next = torch.from_numpy(rgb_next).permute(2, 0, 1) / 255.
            inputs[("color", 2, 0)] = rgb_next


            pose_cur = file_cur['pose']
            pose_cur = np.linalg.inv(pose_cur).astype('float32')
            inputs[("pose", 0)] = pose_cur

            pose_pre = file_pre['pose']
            pose_pre = np.linalg.inv(pose_pre).astype('float32')            
            inputs[("pose", 1)] = pose_pre

            pose_next = file_next['pose']
            pose_next = np.linalg.inv(pose_next).astype('float32')            
            inputs[("pose", 2)] = pose_next

            k_raw = file_cur['intrinsics']

            k_crop, inputs = self.get_k_ori_centercrop(k_raw, inputs, x_center, y_center)

            inputs = self.get_K_test(k_crop, inputs)

        inputs = self.compute_projection_matrix(inputs)


        inputs['num_frame'] = 3

        # for key, value in inputs.items():
        #     print(key, value.dtype)

        return inputs



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

    def get_K_test(self, K, inputs):
        inv_K = np.linalg.inv(K)
        K_pool = {}
        ho, wo = self.opt.eval_height, self.opt.eval_width
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
        for i in range(self.opt.num_frame):
            inputs[("proj", i)] = {}
            for k, v in inputs['K_pool'].items():
                K44 = np.eye(4)
                K44[:3, :3] = v
                inputs[("proj",
                        i)][k] = np.matmul(K44, inputs[("pose",
                                                        i)]).astype('float32')
        return inputs
