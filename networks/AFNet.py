import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import cv2
import time
import numpy as np
import torchvision.models as models
from utils import *
from .module import UNet
import math

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

class FeatureNet(nn.Module):
    def __init__(self, base_model):
        super(FeatureNet, self).__init__()

        self.base_model = base_model
        self.up_3 = nn.Sequential(
            nn.ConvTranspose2d(768,384,kernel_size=3,padding=1,output_padding=1,stride=2,bias=False), 
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))

        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(384,192,kernel_size=3,padding=1,output_padding=1,stride=2,bias=False), 
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True))
        self.conv_2 = ConvBnReLU(384*2, 384)
        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(192,96,kernel_size=3,padding=1,output_padding=1,stride=2,bias=False), 
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True))
        self.conv_1 = ConvBnReLU(192*2, 192)
        self.conv_0 = ConvBnReLU(96*2, 96)
        self.lastconv = nn.Sequential(convbn(96,64,3,1,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,32,kernel_size=1,padding=0,stride=1,bias=False))
        self.num_ch_dec = [64, 64, 128, 256]

    def forward(self, x):
        mono_0, mono_1, mono_2, mono_3 = self.base_model.forward_features(x)
        mono_2_up = self.conv_2(torch.cat((self.up_3(mono_3),mono_2),1)) + mono_2
        mono_1_up = self.conv_1(torch.cat((self.up_2(mono_2_up),mono_1),1)) + mono_1
        mono_0_up = self.conv_0(torch.cat((self.up_1(mono_1_up),mono_0),1)) + mono_0
        mvs_feature = self.lastconv(mono_0_up)
        return mvs_feature, mono_0, mono_1, mono_2, mono_3


class cost_multiscale(nn.Module):
    def __init__(self, opt, num_ch_dec):
        super(cost_multiscale, self).__init__()
        self.opt = opt
        self.num_ch_dec = num_ch_dec
        self.conv_fusion_1 = nn.Sequential(ConvBnReLU(32 + self.opt.nlabel, self.num_ch_dec[0], stride=1),ConvBnReLU(self.num_ch_dec[0], self.num_ch_dec[0]))
        self.down_sample_2 = ConvBnReLU(self.num_ch_dec[0], self.num_ch_dec[1], stride=2)
        self.down_sample_3 = ConvBnReLU(self.num_ch_dec[1], self.num_ch_dec[2], stride=2)
        self.down_sample_4 = ConvBnReLU(self.num_ch_dec[2], self.num_ch_dec[3], stride=2)

    def forward(self,x):
        mvs_cost_0 = self.conv_fusion_1(x)
        mvs_cost_1 = self.down_sample_2(mvs_cost_0)
        mvs_cost_2 = self.down_sample_3(mvs_cost_1)
        mvs_cost_3 = self.down_sample_4(mvs_cost_2)
        return mvs_cost_0, mvs_cost_1, mvs_cost_2, mvs_cost_3

class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVS2D(nn.Module):
    def __init__(self, opt):
        super(MVS2D, self).__init__()
        self.opt = opt
        self.iters = 3
        self.depth_values = self.get_bins()
        self.base_model = ConvNeXt(depths=[3,3,9,3], dims=[96,192,384,768])
        self.base_model.load_state_dict(torch.load('./pretrained_model/convnext/convnext_tiny_22k_1k_384.pth', 'cpu')['model'], strict=True)
        self.feature = FeatureNet(self.base_model)
        self.cost_downsample = nn.Sequential(
            nn.Conv3d(32,8,3,1,padding=1,bias=False),
            nn.Conv3d(8,1,3,1,padding=1,bias=False))

        depth_bins_after = np.linspace(math.log(self.opt.min_depth), math.log(self.opt.max_depth), self.opt.num_depth_regressor_anchor)
        depth_bins_after = np.array([math.exp(x) for x in depth_bins_after])
        depth_values_after = torch.from_numpy(depth_bins_after).float()
        self.register_buffer('depth_expectation_anchor', depth_values_after)


        self.feat_names = [
            'layer1',  # 1/4 resol
            'layer2',  # 1/8 resol
            'layer3',  # 1/16 resol
            'layer4',  # 1/32 resol
        ]
        self.feat_name2ch = {
            'layer1': 96,
            'layer2': 192,
            'layer3': 384,
            'layer4': 768
        }
        self.feat_channels = [self.feat_name2ch[x] for x in self.feat_names]

        self.cost_multiscale = cost_multiscale(self.opt, self.feat_channels)


        self.layers = {}
        ## ----------- Decoder ----------
        self.num_ch_dec = [64, 64, 128, 256]
        ch_cur = self.feat_channels[-1]
        for i in range(3, 0, -1):
            k = 1 if i == 3 else 3
            self.layers[("upconv", i, 0)] = ConvBlock(ch_cur,
                                                      self.num_ch_dec[i],
                                                      kernel_size=k)
            ch_mid = self.num_ch_dec[i]
            if self.opt.use_skip:
                ch_mid += self.feat_channels[i - 1]
            self.layers[("upconv", i, 1)] = ConvBlock_double(ch_mid,
                                                      self.num_ch_dec[i],
                                                      kernel_size=k)
            ch_cur = self.num_ch_dec[i]


        self.layers_2 = {}
        ## ----------- Decoder ----------
        self.num_ch_dec = [64, 64, 128, 256]
        ch_cur = self.feat_channels[-1]
        for i in range(3, 0, -1):
            k = 1 if i == 3 else 3
            self.layers_2[("upconv", i, 0)] = ConvBlock(ch_cur,
                                                      self.num_ch_dec[i],
                                                      kernel_size=k)
            ch_mid = self.num_ch_dec[i]
            if self.opt.use_skip:
                ch_mid += self.feat_channels[i - 1]
            self.layers_2[("upconv", i, 1)] = ConvBlock_double(ch_mid,
                                                      self.num_ch_dec[i],
                                                      kernel_size=k)
            ch_cur = self.num_ch_dec[i]

        self.temp = nn.ModuleList(list(self.layers.values()))
        self.temp_2 = nn.ModuleList(list(self.layers_2.values()))

        ## ----------- Depth Regressor ----------
        ch_cur = self.num_ch_dec[self.opt.output_scale - self.opt.input_scale -
                                 1]
        odim = 256
        output_chal = odim if not self.opt.pred_conf else odim + 1

        self.conv_out_1 = UNet(inp_ch=ch_cur,
                             output_chal=output_chal,
                             down_sample_times=3,
                             channel_mode=self.opt.unet_channel_mode)     

        self.conv_out_2 = UNet(inp_ch=ch_cur,
                             output_chal=output_chal,
                             down_sample_times=3,
                             channel_mode=self.opt.unet_channel_mode)      


        self.depth_regressor = nn.Sequential(
            nn.Conv2d(odim,
                      self.opt.num_depth_regressor_anchor,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(self.opt.num_depth_regressor_anchor),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.opt.num_depth_regressor_anchor,
                      self.opt.num_depth_regressor_anchor,
                      kernel_size=1),
        )

        self.depth_regressor_2 = nn.Sequential(
            nn.Conv2d(odim,
                      self.opt.num_depth_regressor_anchor,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(self.opt.num_depth_regressor_anchor),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.opt.num_depth_regressor_anchor,
                      self.opt.num_depth_regressor_anchor,
                      kernel_size=1),
        )

        self.conv_up = nn.Sequential(
            nn.Conv2d(1 + 2 + 256, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1, padding=0),
        )

        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(1 + 2 + 256, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1, padding=0),
        )

        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(1 + 2 + 96, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1, padding=0),
        )

        self.feature_residual_mono = nn.Sequential(
            nn.Conv2d(96+2*ch_cur, 128, 3, padding =1),
            nn.Conv2d(128, ch_cur, 1, padding = 0))

        self.feature_residual_mvs = nn.Sequential(
            nn.Conv2d(96+2*ch_cur, 128, 3, padding =1),
            nn.Conv2d(128, ch_cur, 1, padding = 0))

    def get_bins(self):
        depth_bins = np.linspace(math.log(self.opt.min_depth), math.log(self.opt.max_depth), self.opt.nlabel)
        depth_bins = np.array([math.exp(x) for x in depth_bins])
        depth_values = torch.from_numpy(depth_bins).float()
        return depth_values

    def depth_regression(self, p, depth_values):
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
        depth = torch.sum(p * depth_values, 1).unsqueeze(1)
        return depth

    def upsample(self, x, scale_factor=2):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=scale_factor, mode="nearest")


    def decoder(self, ref_feature):
        x = ref_feature[-1]
        for i in range(3, 0, -1):
            x = self.layers[("upconv", i, 0)](x)
            if i >= 1 - self.opt.input_scale:
                x = self.upsample(x)
                if self.opt.use_skip:
                    x = torch.cat((x, ref_feature[i - 1]), 1)
                x = self.layers[("upconv", i, 1)](x)
            else:
                break
        return x

    def decoder_2(self, ref_feature):
        x = ref_feature[-1]
        for i in range(3, 0, -1):
            x = self.layers_2[("upconv", i, 0)](x)
            if i >= 1 - self.opt.input_scale:
                x = self.upsample(x)
                if self.opt.use_skip:
                    x = torch.cat((x, ref_feature[i - 1]), 1)
                x = self.layers_2[("upconv", i, 1)](x)
            else:
                break
        return x

    def regress_depth(self, feature_map_d):
        x = self.depth_regressor(feature_map_d).softmax(dim=1)
        d = compute_depth_expectation(
            x,
            self.depth_expectation_anchor.unsqueeze(0).repeat(x.shape[0],
                                                              1)).unsqueeze(1)
        return d

    def regress_depth_2(self, feature_map_d):
        x = self.depth_regressor_2(feature_map_d).softmax(dim=1)
        d = compute_depth_expectation(
            x,
            self.depth_expectation_anchor.unsqueeze(0).repeat(x.shape[0],
                                                              1)).unsqueeze(1)
        return d

    def forward(self, ref_img, src_imgs, ref_proj, src_projs, inv_K):
        outputs = {}

        num_views = len(src_imgs) + 1

        if self.training:
            ref_feature_mvs, ref_feature_mono_0, ref_feature_mono_1, ref_feature_mono_2, ref_feature_mono_3 = self.feature(ref_img)
            src_features_mvs = []
            features_monos = []
            features_monos.append([ref_feature_mono_0, ref_feature_mono_1, ref_feature_mono_2, ref_feature_mono_3])
            for x in src_imgs:
                mvs_feature, mono_0, mono_1, mono_2, mono_3 = self.feature(x)
                src_features_mvs.append(mvs_feature)
                features_monos.append([mono_0, mono_1, mono_2, mono_3])
        
        else:
            images_all = ref_img
            for x in src_imgs:
                images_all = torch.cat((images_all, x), dim = 0)
            all_feature_mvs, all_feature_mono_0, all_feature_mono_1, all_feature_mono_2, all_feature_mono_3 = self.feature(images_all)
            ref_feature_mvs, ref_feature_mono_0, ref_feature_mono_1, ref_feature_mono_2, ref_feature_mono_3 = all_feature_mvs[0,:].unsqueeze(0), all_feature_mono_0[0,:].unsqueeze(0), all_feature_mono_1[0,:].unsqueeze(0), all_feature_mono_2[0,:].unsqueeze(0), all_feature_mono_3[0,:].unsqueeze(0)
            num_src = int(num_views -1)
            src_features_mvs = []
            features_monos = []
            for idx in range(num_src):
                src_features_mvs.append(all_feature_mvs[int(idx+1),:].unsqueeze(0))
                features_monos.append([all_feature_mono_0[int(idx+1),:].unsqueeze(0), all_feature_mono_1[int(idx+1),:].unsqueeze(0), all_feature_mono_2[int(idx+1),:].unsqueeze(0), all_feature_mono_3[int(idx+1),:].unsqueeze(0)])

        sz_ref = (ref_feature_mvs.shape[2], ref_feature_mvs.shape[3])
        sz_src = (src_features_mvs[0].shape[2], src_features_mvs[0].shape[3])

        depth_values = self.depth_values[None, :, None, None].repeat(ref_feature_mvs.shape[0],1,ref_feature_mvs.shape[2], ref_feature_mvs.shape[3]).to(ref_feature_mvs.device)
        num_depth = self.opt.nlabel
        ref_volume = ref_feature_mvs.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume

        ref_proj_temp = ref_proj
        src_projs_temp = [proj for proj in src_projs]
        for i, (src_fea, src_proj) in enumerate(zip(src_features_mvs, src_projs_temp)):
            T_ref2src = torch.matmul(src_proj, torch.inverse(ref_proj_temp))

            warped_volume, proj_mask, grid = homo_warping(src_fea, T_ref2src, depth_values, inv_K[sz_src])
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)
            del warped_volume

        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        del src_features_mvs

        cost_reg = self.cost_downsample(volume_variance)
        cost_reg = cost_reg.squeeze(1)

        ref_skip_feat = []
        ref_skip_feat2 = []

        cost_reg_all = torch.cat((ref_feature_mvs, cost_reg), 1)
        mvs_cost_0, mvs_cost_1, mvs_cost_2, mvs_cost_3 = self.cost_multiscale(cost_reg_all)

        ref_skip_feat = [mvs_cost_0, mvs_cost_1, mvs_cost_2, mvs_cost_3]
        ref_skip_feat2 = [ref_feature_mono_0, ref_feature_mono_1, ref_feature_mono_2, ref_feature_mono_3]

        feature_map_1 = self.decoder(ref_skip_feat)
        feature_map_2 = self.decoder_2(ref_skip_feat2)

        feature_mvs_residual = self.feature_residual_mvs(torch.cat((feature_map_1, feature_map_2, ref_feature_mono_0), 1))
        feature_mono_residual = self.feature_residual_mono(torch.cat((feature_map_1, feature_map_2, ref_feature_mono_0), 1))

        feature_map_1 = feature_map_1 + feature_mvs_residual
        feature_map_2 = feature_map_2 + feature_mono_residual

        feature_map_1 = self.conv_out_1(feature_map_1)
        confidence_map_1 = feature_map_1[:, -1:, :, :]
        feature_map_d_1 = feature_map_1[:, :-1, :, :]

        feature_map_2 = self.conv_out_2(feature_map_2)
        confidence_map_2 = feature_map_2[:, -1:, :, :]
        feature_map_d_2 = feature_map_2[:, :-1, :, :]

        depth_pred = self.regress_depth(feature_map_d_1)
        depth_pred_2 = self.regress_depth_2(feature_map_d_2)

        _, _, h, w = depth_pred.shape
        idv, idu = np.meshgrid(np.linspace(0, 1, h),
                               np.linspace(0, 1, w),
                               indexing='ij')
        self.meshgrid = torch.from_numpy(np.stack((idu, idv))).float()


        ## upsample depth map into input resolution
        depth_pred = self.upsample(
            depth_pred, scale_factor=4) + 1e-1 * self.conv_up(
                torch.cat((depth_pred, self.meshgrid.unsqueeze(0).repeat(
                    depth_pred.shape[0], 1, 1,
                    1).to(depth_pred), feature_map_d_1), 1))


        depth_pred_2 = self.upsample(
            depth_pred_2, scale_factor=4) + 1e-1 * self.conv_up(
                torch.cat((depth_pred_2, self.meshgrid.unsqueeze(0).repeat(
                    depth_pred_2.shape[0], 1, 1,
                    1).to(depth_pred_2), feature_map_d_2), 1))

        confidence_map_1 = self.upsample(confidence_map_1, scale_factor=4)
        confidence_map_2 = self.upsample(confidence_map_2, scale_factor=4)

        outputs[('depth_pred', 0)] = depth_pred
        outputs[('depth_pred_2', 0)] = depth_pred_2
        outputs[('log_conf_pred', 0)] = confidence_map_1
        outputs[('log_conf_pred_2', 0)] = confidence_map_2

        return outputs