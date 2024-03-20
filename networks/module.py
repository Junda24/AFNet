import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    # def forward_features(self, x):
    #     for i in range(4):
    #         x = self.downsample_layers[i](x)
    #         x = self.stages[i](x)
    #     return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward_features(self, x):
        out_0 = self.downsample_layers[0](x)
        out_0 = self.stages[0](out_0)
        out_1 = self.downsample_layers[1](out_0)
        out_1 = self.stages[1](out_1)
        out_2 = self.downsample_layers[2](out_1)
        out_2 = self.stages[2](out_2)
        out_3 = self.downsample_layers[3](out_2)
        out_3 = self.stages[3](out_3)
        return out_0, out_1, out_2, out_3


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))

def compute_depth_expectation(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)

    depth = torch.sum(p * depth_values, 1)
    return depth

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1)
    def forward(self, x):
        x = self.conv(x)
        return x

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()

        if kernel_size ==3:
            self.conv = Conv3x3(in_channels, out_channels)
        elif kernel_size == 1:
            self.conv = Conv1x1(in_channels, out_channels)

        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock_double(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock_double, self).__init__()

        if kernel_size ==3:
            self.conv = Conv3x3(in_channels, out_channels)
        elif kernel_size == 1:
            self.conv = Conv1x1(in_channels, out_channels)

        self.conv_1 = Conv1x1(out_channels, out_channels)

        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv_1(out)
        out = self.nonlin(out)
        return out

# def homo_warping(src_fea, src_proj, ref_proj, depth_values):
#     # src_fea: [B, C, H, W]
#     # src_proj: [B, 4, 4]
#     # ref_proj: [B, 4, 4]
#     # depth_values: [B, Ndepth, H, W]
#     # out: [B, C, Ndepth, H, W]
#     batch, channels = src_fea.shape[0], src_fea.shape[1]
#     num_depth = depth_values.shape[1]
#     #height, width = src_fea.shape[2], src_fea.shape[3]
#     h_src, w_src = src_fea.shape[2], src_fea.shape[3]
#     h_ref, w_ref = depth_values.shape[2], depth_values.shape[3]

#     with torch.no_grad():
#         proj = torch.matmul(src_proj, torch.inverse(ref_proj))

#         rot = proj[:, :3, :3]  # [B,3,3]
#         trans = proj[:, :3, 3:4]  # [B,3,1]

        
#         y, x = torch.meshgrid([torch.arange(0, h_ref, dtype=torch.float32, device=src_fea.device),
#                                torch.arange(0, w_ref, dtype=torch.float32, device=src_fea.device)])
#         y, x = y.contiguous(), x.contiguous()
#         y, x = y.view(h_ref * w_ref), x.view(h_ref * w_ref)
#         xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
#         xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
#         rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
#         rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
#                                                                                             -1)  # [B, 3, Ndepth, H*W]
#         proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
#         proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
#         z = proj_xyz[:, 2:3, :, :].view(batch, num_depth, h_ref, w_ref)
#         proj_x_normalized = proj_xy[:, 0, :,:] / ((w_src - 1) / 2.0) - 1
#         proj_y_normalized = proj_xy[:, 1, :,:] / ((h_src - 1) / 2.0) - 1
#         X_mask = ((proj_x_normalized > 1)+(proj_x_normalized < -1)).detach()
#         proj_x_normalized[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
#         Y_mask = ((proj_y_normalized > 1)+(proj_y_normalized < -1)).detach()
#         proj_y_normalized[Y_mask] = 2
#         proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
#         grid = proj_xy
#         proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, h_ref, w_ref)
#         proj_mask = (proj_mask + (z <= 0)) > 0

#     warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * h_ref, w_ref, 2), mode='bilinear',
#                                    padding_mode='zeros', align_corners=True)

#     warped_src_fea = warped_src_fea.view(batch, channels, num_depth, h_ref, w_ref)

#     #return warped_src_fea , proj_mask
#     return warped_src_fea , proj_mask, grid.view(batch, num_depth, h_ref, w_ref, 2)


def homo_warping(src_fea, T_ref2src, depth_values, inv_K):
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

        inv_k_33 = inv_K[:, :3, :3]
        rot = T_ref2src[:, :3, :3]
        trans = T_ref2src[:, :3, 3:4]

        
        y, x = torch.meshgrid([torch.arange(0, h_ref, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, w_ref, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h_ref * w_ref), x.view(h_ref * w_ref)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(inv_k_33, xyz)
        depth_xyz = xyz.unsqueeze(2).repeat(1,1,num_depth,1) * depth_values.view(batch, 1, num_depth, -1)
        depth_xyz = depth_xyz.permute(0,2,1,3)

        rot_depth_xyz = torch.matmul(rot.unsqueeze(1).repeat(1, num_depth, 1, 1), depth_xyz) + trans.unsqueeze(1)
        proj_xyz = torch.matmul(torch.inverse(inv_k_33).unsqueeze(1).repeat(1,num_depth,1,1), rot_depth_xyz)
        proj_xyz = proj_xyz.permute(0,2,1,3)
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        z = proj_xyz[:, 2:3, :, :].view(batch, num_depth, h_ref, w_ref)
        proj_x_normalized = proj_xy[:, 0, :,:] / ((w_src - 1) / 2.0) - 1
        proj_y_normalized = proj_xy[:, 1, :,:] / ((h_src - 1) / 2.0) - 1
        X_mask = ((proj_x_normalized > 1)+(proj_x_normalized < -1)).detach()
        proj_x_normalized[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((proj_y_normalized > 1)+(proj_y_normalized < -1)).detach()
        proj_y_normalized[Y_mask] = 2
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
        proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, h_ref, w_ref)
        proj_mask = (proj_mask + (z <= 0)) > 0

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * h_ref, w_ref, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, h_ref, w_ref)

    #return warped_src_fea , proj_mask
    return warped_src_fea , proj_mask, grid.view(batch, num_depth, h_ref, w_ref, 2)


def homo_warping_depth(src_fea, T_ref2src, depth_values, inv_K):
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

        inv_k_33 = inv_K[:, :3, :3]
        rot = T_ref2src[:, :3, :3]
        trans = T_ref2src[:, :3, 3:4]

        
        y, x = torch.meshgrid([torch.arange(0, h_ref, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, w_ref, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h_ref * w_ref), x.view(h_ref * w_ref)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(inv_k_33, xyz)
        depth_xyz = xyz.unsqueeze(2).repeat(1,1,num_depth,1) * depth_values.view(batch, 1, num_depth, -1)
        depth_xyz = depth_xyz.permute(0,2,1,3)

        rot_depth_xyz = torch.matmul(rot.unsqueeze(1).repeat(1, num_depth, 1, 1), depth_xyz) + trans.unsqueeze(1)
        proj_xyz = torch.matmul(torch.inverse(inv_k_33).unsqueeze(1).repeat(1,num_depth,1,1), rot_depth_xyz)
        proj_xyz = proj_xyz.permute(0,2,1,3)
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        z = proj_xyz[:, 2:3, :, :].view(batch, num_depth, h_ref, w_ref)
        proj_x_normalized = proj_xy[:, 0, :,:] / ((w_src - 1) / 2.0) - 1
        proj_y_normalized = proj_xy[:, 1, :,:] / ((h_src - 1) / 2.0) - 1
        X_mask = ((proj_x_normalized > 1)+(proj_x_normalized < -1)).detach()
        proj_x_normalized[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((proj_y_normalized > 1)+(proj_y_normalized < -1)).detach()
        proj_y_normalized[Y_mask] = 2
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
        proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, h_ref, w_ref)
        proj_mask = (proj_mask + (z <= 0)) > 0

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * h_ref, w_ref, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, h_ref, w_ref)

    #return warped_src_fea , proj_mask
    return warped_src_fea


# def homo_warping_depth(src_fea, T_ref2src, depth_values):
#     # # src_fea: [B, C, H, W]
#     # # src_proj: [B, 4, 4]
#     # # ref_proj: [B, 4, 4]
#     # # depth_values: [B, Ndepth, H, W]
#     # # out: [B, C, Ndepth, H, W]
#     # batch, channels = src_fea.shape[0], src_fea.shape[1]
#     # num_depth = depth_values.shape[1]
#     # #height, width = src_fea.shape[2], src_fea.shape[3]
#     # h_src, w_src = src_fea.shape[2], src_fea.shape[3]
#     # h_ref, w_ref = depth_values.shape[2], depth_values.shape[3]

#     # with torch.no_grad():

#     #     proj = torch.matmul(src_proj, torch.inverse(ref_proj))
#     #     rot = proj[:, :3, :3]  # [B,3,3]
#     #     trans = proj[:, :3, 3:4]  # [B,3,1]

        
#     #     y, x = torch.meshgrid([torch.arange(0, h_ref, dtype=torch.float32, device=src_fea.device),
#     #                            torch.arange(0, w_ref, dtype=torch.float32, device=src_fea.device)])
#     #     y, x = y.contiguous(), x.contiguous()
#     #     y, x = y.view(h_ref * w_ref), x.view(h_ref * w_ref)
#     #     xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
#     #     xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]

#     #     rot_xyz = torch.matmul(rot, xyz)
#     #     rot_depth_xyz = rot_xyz * depth_values.view(batch, 1, -1)

#     #     proj_xyz = rot_depth_xyz + trans.view(batch,3,1)

#     #     proj_xy = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :]  # [B, 2, Ndepth, H*W]
#     #     z = proj_xyz[:, 2:3, :].view(batch, h_ref, w_ref)
#     #     proj_x_normalized = proj_xy[:, 0, :] / ((w_src - 1) / 2.0) - 1
#     #     proj_y_normalized = proj_xy[:, 1, :] / ((h_src - 1) / 2.0) - 1
#     #     X_mask = ((proj_x_normalized > 1)+(proj_x_normalized < -1)).detach()
#     #     proj_x_normalized[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
#     #     Y_mask = ((proj_y_normalized > 1)+(proj_y_normalized < -1)).detach()
#     #     proj_y_normalized[Y_mask] = 2
#     #     proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
#     #     grid = proj_xy
#     #     proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, h_ref, w_ref)
#     #     proj_mask = (proj_mask + (z <= 0)) > 0

#     # warped_src_fea = F.grid_sample(src_fea, grid.view(batch, h_ref, w_ref, 2), mode='bilinear',
#     #                                padding_mode='zeros', align_corners=True)

#     # warped_src_fea = warped_src_fea.view(batch, channels, num_depth, h_ref, w_ref)

#     # #return warped_src_fea , proj_mask
#     # return warped_src_fea

#     # src_fea: [B, C, H, W]
#     # src_proj: [B, 4, 4]
#     # ref_proj: [B, 4, 4]
#     # depth_values: [B, Ndepth, H, W]
#     # out: [B, C, Ndepth, H, W]
#     batch, channels = src_fea.shape[0], src_fea.shape[1]
#     num_depth = depth_values.shape[1]
#     #height, width = src_fea.shape[2], src_fea.shape[3]
#     h_src, w_src = src_fea.shape[2], src_fea.shape[3]
#     h_ref, w_ref = depth_values.shape[2], depth_values.shape[3]

#     with torch.no_grad():

#         # proj = torch.matmul(src_proj, torch.inverse(ref_proj))
#         # rot = proj[:, :3, :3]  # [B,3,3]
#         # trans = proj[:, :3, 3:4]  # [B,3,1]
#         rot = T_ref2src[:, :3, :3]  # [B,3,3]
#         trans = T_ref2src[:, :3, 3:4]  # [B,3,1]

        
#         y, x = torch.meshgrid([torch.arange(0, h_ref, dtype=torch.float32, device=src_fea.device),
#                                torch.arange(0, w_ref, dtype=torch.float32, device=src_fea.device)])
#         y, x = y.contiguous(), x.contiguous()
#         y, x = y.view(h_ref * w_ref), x.view(h_ref * w_ref)
#         xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
#         xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]

#         rot_xyz = torch.matmul(rot, xyz)
#         rot_depth_xyz = rot_xyz * depth_values.view(batch, 1, -1)

#         proj_xyz = rot_depth_xyz + trans.view(batch,3,1)

#         proj_xy = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :]  # [B, 2, Ndepth, H*W]
#         z = proj_xyz[:, 2:3, :].view(batch, h_ref, w_ref)
#         proj_x_normalized = proj_xy[:, 0, :] / ((w_src - 1) / 2.0) - 1
#         proj_y_normalized = proj_xy[:, 1, :] / ((h_src - 1) / 2.0) - 1
#         X_mask = ((proj_x_normalized > 1)+(proj_x_normalized < -1)).detach()
#         proj_x_normalized[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
#         Y_mask = ((proj_y_normalized > 1)+(proj_y_normalized < -1)).detach()
#         proj_y_normalized[Y_mask] = 2
#         proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, Ndepth, H*W, 2]
#         grid = proj_xy
#         proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, h_ref, w_ref)
#         proj_mask = (proj_mask + (z <= 0)) > 0

#     warped_src_fea = F.grid_sample(src_fea, grid.view(batch, h_ref, w_ref, 2), mode='bilinear',
#                                    padding_mode='zeros', align_corners=True)

#     warped_src_fea = warped_src_fea.view(batch, channels, num_depth, h_ref, w_ref)

#     #return warped_src_fea , proj_mask
#     return warped_src_fea


class LinearEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super(LinearEmbedding, self).__init__()
        weight = torch.randn(1, dim)
        depth = torch.arange(max_len).float()
        weight = weight * depth.unsqueeze(-1)
        self.register_buffer('weight', weight)

class UniformEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super(UniformEmbedding, self).__init__()
        weight = torch.randn(1, dim)
        weight = weight.repeat(max_len, 1)
        self.register_buffer('weight', weight)

class CosineEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super(CosineEmbedding, self).__init__()
        weight = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('weight', weight)

class encoder(nn.Module):
    def __init__(self, model):
        super(encoder, self).__init__()
        import torchvision.models as models
        if model == 'densenet121':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif model == 'densenet161':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif model == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnet101':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnext50':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnext101':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif model == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        else:
            print('Not supported encoder: {}'.format(model))


    def forward(self, x):
        features = [x]
        skip_feat = [x]
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
        
        return skip_feat

class UNet(nn.Module):
    def __init__(self, inp_ch=32, output_chal=1, down_sample_times=1,channel_mode='v0'):
        super(UNet, self).__init__()
        basic_block = ConvBnReLU
        num_depth = 128

        self.conv0 = basic_block(inp_ch, num_depth)
        if channel_mode == 'v0':
            channels = [num_depth, num_depth//2, num_depth//4, num_depth//8, num_depth // 8]
        elif channel_mode == 'v1':
            channels = [num_depth, num_depth, num_depth, num_depth, num_depth,num_depth]
        self.down_sample_times = down_sample_times
        for i in range(down_sample_times):
            setattr(self, 'conv_%d' % i, 
                    nn.Sequential(basic_block(channels[i], channels[i+1], stride=2), 
                        basic_block(channels[i+1], channels[i+1])))

        for i in range(down_sample_times-1,-1,-1):
            setattr(self, 'deconv_%d' % i, 
                    nn.Sequential(
                nn.ConvTranspose2d(channels[i+1],
                                   channels[i],
                                   kernel_size=3,
                                   padding=1,
                                   output_padding=1,
                                   stride=2,
                                   bias=False), nn.BatchNorm2d(channels[i]),
                nn.ReLU(inplace=True))
)

        self.prob = nn.Conv2d(num_depth, output_chal, 1, stride=1, padding=0)

    def forward(self, x):
        features = {}
        conv0 = self.conv0(x)
        x = conv0
        features[0] = conv0
        for i in range(self.down_sample_times):
            x = getattr(self, 'conv_%d' % i)(x)
            features[i+1] = x
        for i in range(self.down_sample_times-1,-1,-1):
            x = features[i] + getattr(self, 'deconv_%d' % i)(x)
        x = self.prob(x)
        return x
