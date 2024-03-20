from __future__ import absolute_import, division, print_function
import open3d as o3d
from collections import defaultdict
import os
import random
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import time
import torch.nn.functional as F
import re
import collections.abc as container_abcs
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
import matplotlib
import matplotlib.pyplot as plt

def gray_2_colormap_np_2(img, cmap = 'rainbow', max = None):
    img = img.squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap

def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


def plot(xs,
         ys,
         stds=None,
         xlabel='',
         ylabel='',
         title='',
         legends=None,
         save_fn='test.png',
         marker=None,
         marker_size=12):
    MARKERS = ["o", "X", "D", "^", "<", "v", ">"]
    if marker is None:
        marker = MARKERS[3]

    nline = len(ys)
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(figsize=(7, 7))
    grid = plt.GridSpec(nrows, ncols, figure=fig)

    ax1 = plt.subplot(grid[0, 0])
    lh = []
    for i in range(nline):
        if stds is not None:
            #l, _, _= ax1.errorbar(xs, ys[i], yerr=stds[i], linewidth=4, marker=MARKERS[0], markersize=1, )
            l, = ax1.plot(
                xs,
                ys[i],
                linewidth=4,
                marker=marker,
                markersize=marker_size,
            )
            color = l.get_color()
            low = [x[0] for x in stds[i]]
            high = [x[1] for x in stds[i]]
            ax1.fill_between(xs, low, high, color=color, alpha=.1)

        else:
            l, = ax1.plot(
                xs,
                ys[i],
                linewidth=4,
                marker=marker,
                markersize=marker_size,
            )
        lh.append(l)

    ax1.set_xlabel(xlabel, fontsize=25)
    ax1.set_ylabel(ylabel, fontsize=25)
    ax1.set_title(title, fontsize=25)
    if legends is not None:
        lgnd = ax1.legend(lh, legends, fontsize=15)
    plt.savefig(save_fn)


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:
    Returns:
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(backend=backend,
                            init_method='tcp://127.0.0.1:%d' % tcp_port,
                            rank=local_rank,
                            world_size=num_gpus)
    rank = dist.get_rank()
    return num_gpus, rank


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def randomRotation(epsilon):
    axis = (np.random.rand(3) - 0.5)
    axis /= np.linalg.norm(axis)
    dtheta = np.random.randn(1) * np.pi * epsilon
    K = np.array(
        [0, -axis[2], axis[1], axis[2], 0, -axis[0], -axis[1], axis[0],
         0]).reshape(3, 3)
    dR = np.eye(3) + np.sin(dtheta) * K + (1 - np.cos(dtheta)) * np.matmul(
        K, K)
    return dR


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(
            data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(batch, list) and isinstance(elem, tuple):
        #data = torch.cat((x[0] for x in batch))
        return [x[0] for x in batch]
    if type(elem) == tuple and elem[1] == 'varlen':
        return [x[0] for x in batch]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except:
            import ipdb
            ipdb.set_trace()

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    #elif isinstance(elem, int_classes):
    elif isinstance(elem, int):
        return torch.tensor(batch)
    #elif isinstance(elem, string_classes):
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


_use_shared_memory = False

np_str_obj_array_pattern = re.compile(r'[SaUO]')

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collatev1_1(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(batch[0])
    if isinstance(batch, list) and isinstance(elem, tuple):
        #data = torch.cat((x[0] for x in batch))
        return [x[0] for x in batch]
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except:
            import ipdb
            ipdb.set_trace()
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        try:
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(error_msg_fmt.format(elem.dtype))

                return default_collatev1_1(
                    [torch.from_numpy(b) for b in batch])
        except:
            import ipdb
            ipdb.set_trace()
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    #elif isinstance(batch[0], int_classes):
    elif isinstance(batch[0], int):
        return torch.tensor(batch)
    #elif isinstance(batch[0], string_classes):
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {
            key: default_collatev1_1([d[key] for d in batch])
            for key in batch[0]
        }
    elif isinstance(batch[0], tuple) and hasattr(batch[0],
                                                 '_fields'):  # namedtuple
        return type(batch[0])(*(default_collatev1_1(samples)
                                for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collatev1_1(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))


def backproject_depth_th(depth, inv_K, mask=False, device='cuda'):
    h, w = depth.shape
    idu, idv = np.meshgrid(range(w), range(h))
    grid = np.stack((idu.flatten(), idv.flatten(), np.ones([w * h])))
    grid = torch.from_numpy(grid).float().to(device)
    x = torch.matmul(inv_K[:3, :3], grid)
    x = x * depth.flatten()[None, :]
    x = x.t()
    if mask:
        x = x[depth.flatten() > 0]
    return x


def backproject_depth(depth, inv_K, mask=False):
    h, w = depth.shape
    idu, idv = np.meshgrid(range(w), range(h))
    grid = np.stack((idu.flatten(), idv.flatten(), np.ones([w * h])))
    x = np.matmul(inv_K[:3, :3], grid)
    x = x * depth.flatten()[None, :]
    x = x.T
    if mask:
        x = x[depth.flatten() > 0]
    return x


def parameters_count(net, name, do_print=True):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    if do_print:
        print('#params %s: %.3f M' % (name, params / 1e6))
    return params


def cuda_time():
    torch.cuda.synchronize()
    return time.time()


def transform3x3(pc, T):
    # T: [4,4]
    # pc: [n, 3]
    # return: [n, 3]
    return (np.matmul(T[:3, :3], pc.T)).T


def transform4x4(pc, T):
    # T: [4,4]
    # pc: [n, 3]
    # return: [n, 3]
    return (np.matmul(T[:3, :3], pc.T) + T[:3, 3:4]).T


def transform4x4_th(pc, T):
    # T: [4,4]
    # pc: [n, 3]
    # return: [n, 3]
    return (torch.matmul(T[:3, :3], pc.t()) + T[:3, 3:4]).t()


def v(var, cuda=True, volatile=False):
    if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
        res = Variable(var.float(), volatile=volatile)
    elif type(var) == np.ndarray:
        res = Variable(torch.from_numpy(var).float(), volatile=volatile)
    if cuda:
        res = res.cuda()
    return res


def npy(var):
    return var.data.cpu().numpy()


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def write_ply(fn, point, normal=None, color=None):

    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(point)
    if color is not None:
        ply.colors = o3d.utility.Vector3dVector(color)
    if normal is not None:
        ply.normals = o3d.utility.Vector3dVector(normal)
    o3d.io.write_point_cloud(fn, ply)


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def Thres_metrics(pred, gt, mask, interval, thre):
    abs_diff = (pred - gt).abs() / interval
    metric = (mask * (abs_diff < thre).float()).sum() / mask.sum()
    return metric


def Thres_metrics_np(pred, gt, mask, interval, thre):
    abs_diff = np.abs(pred - gt) / interval
    metric = (mask * (abs_diff < thre)).sum() / mask.sum()
    return metric
