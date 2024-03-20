import json
from pathlib import Path

import numpy as np
import pykitti
import torch
import torchvision
from PIL import Image
from scipy import sparse
# from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms

# from utils import map_fn
# ["00","04","05","07"]
# (256, 512)
# "00","01","02","04","05","06","07","08","09","10"

class KittiOdometryDataset(Dataset):

    def __init__(self, dataset_dir = "/data/cjd/kitti_odometry/dataset/", frame_count=2, sequences=["10"], depth_folder="image_depth_annotated",
                 target_image_size=(480, 640), max_length=3000, dilation=1, offset_d=0, use_color=True, use_dso_poses=True,
                 use_color_augmentation=False, lidar_depth=True, dso_depth=False, annotated_lidar=True, return_stereo=False, return_mvobj_mask=False, use_index_mask=()):
        """
        Dataset implementation for KITTI Odometry.
        :param dataset_dir: Top level folder for KITTI Odometry (should contain folders sequences, poses, poses_dvso (if available)
        :param frame_count: Number of frames used per sample (excluding the keyframe). By default, the keyframe is in the middle of those frames. (Default=2)
        :param sequences: Which sequences to use. Should be tuple of strings, e.g. ("00", "01", ...)
        :param depth_folder: The folder within the sequence folder that contains the depth information (e.g. sequences/00/{depth_folder})
        :param target_image_size: Desired image size (correct processing of depths is only guaranteed for default value). (Default=(256, 512))
        :param max_length: Maximum length per sequence. Useful for splitting up sequences and testing. (Default=None)
        :param dilation: Spacing between the frames (Default 1)
        :param offset_d: Index offset for frames (offset_d=0 means keyframe is centered). (Default=0)
        :param use_color: Use color (camera 2) or greyscale (camera 0) images (default=True)
        :param use_dso_poses: Use poses provided by d(v)so instead of KITTI poses. Requires poses_dvso folder. (Default=True)
        :param use_color_augmentation: Use color jitter augmentation. The same transformation is applied to all frames in a sample. (Default=False)
        :param lidar_depth: Use depth information from (annotated) velodyne data. (Default=False)
        :param dso_depth: Use depth information from d(v)so. (Default=True)
        :param annotated_lidar: If lidar_depth=True, then this determines whether to use annotated or non-annotated depth maps. (Default=True)
        :param return_stereo: Return additional stereo frame. Only used during training. (Default=False)
        :param return_mvobj_mask: Return additional moving object mask. Only used during training. If return_mvobj_mask=2, then the mask is returned as target instead of the depthmap. (Default=False)
        :param use_index_mask: Use the listed index masks (if a sample is listed in one of the masks, it is not used). (Default=())
        """
        self.dataset_dir = Path(dataset_dir)
        self.frame_count = frame_count
        self.sequences = sequences
        self.depth_folder = depth_folder
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.lidar_depth = lidar_depth
        self.annotated_lidar = annotated_lidar
        self.dso_depth = dso_depth
        self.target_image_size = target_image_size
        self.use_index_mask = use_index_mask
        self.offset_d = offset_d
        if self.sequences is None:
            self.sequences = [f"{i:02d}" for i in range(11)]
        self._datasets = [pykitti.odometry(dataset_dir, sequence) for sequence in self.sequences]
        self._offset = (frame_count // 2) * dilation
        extra_frames = frame_count * dilation
        if self.annotated_lidar and self.lidar_depth:
            extra_frames = max(extra_frames, 10)
            self._offset = max(self._offset, 5)
        self._dataset_sizes = [
            len((dataset.cam0_files if not use_color else dataset.cam2_files)) - (extra_frames if self.use_index_mask is None else 0) for dataset in
            self._datasets]
        if self.use_index_mask is not None:
            index_masks = []
            for sequence_length, sequence in zip(self._dataset_sizes, self.sequences):
                index_mask = {i:True for i in range(sequence_length)}
                for index_mask_name in self.use_index_mask:
                    with open(self.dataset_dir / "sequences" / sequence / (index_mask_name + ".json")) as f:
                        m = json.load(f)
                        for k in list(index_mask.keys()):
                            if not str(k) in m or not m[str(k)]:
                                del index_mask[k]
                index_masks.append(index_mask)
            self._indices = [
                list(sorted([int(k) for k in sorted(index_mask.keys()) if index_mask[k] and int(k) >= self._offset and int(k) < dataset_size + self._offset - extra_frames]))
                for index_mask, dataset_size in zip(index_masks, self._dataset_sizes)
            ]
            self._dataset_sizes = [len(indices) for indices in self._indices]
        if max_length is not None:
            self._dataset_sizes = [min(s, max_length) for s in self._dataset_sizes]
        self.length = sum(self._dataset_sizes)

        # intrinsics_box = [self.compute_target_intrinsics(dataset, target_image_size, use_color) for dataset in
        #                   self._datasets]
        # self._crop_boxes = [b for _, b in intrinsics_box]
        if self.dso_depth:
            self.dso_depth_parameters = [self.get_dso_depth_parameters(dataset) for dataset in self._datasets]
        elif not self.lidar_depth:
            self._depth_crop_boxes = [
                self.compute_depth_crop(self.dataset_dir / "sequences" / s / depth_folder) for s in
                self.sequences]
        # self._intrinsics = [format_intrinsics(i, self.target_image_size) for i, _ in intrinsics_box]
        self.dilation = dilation
        self.use_color = use_color
        self.use_dso_poses = use_dso_poses
        self.use_color_augmentation = use_color_augmentation
        if self.use_dso_poses:
            for dataset in self._datasets:
                dataset.pose_path = self.dataset_dir / "poses"
                dataset._load_poses()
        # if self.use_color_augmentation:
        #     self.color_transform = ColorJitterMulti(brightness=.2, contrast=.2, saturation=.2, hue=.1)
        self.return_stereo = return_stereo
        if self.return_stereo:
            self._stereo_transform = []
            for d in self._datasets:
                st = torch.eye(4, dtype=torch.float32)
                st[0, 3] = d.calib.b_rgb if self.use_color else d.calib.b_gray
                self._stereo_transform.append(st)

        self.return_mvobj_mask = return_mvobj_mask

        self.dpv_W = int(self.target_image_size[1]/4)
        self.dpv_H = int(self.target_image_size[0]/4)

        self.ray_array = self.get_ray_array()

    def get_ray_array(self):
        ray_array = np.ones((self.dpv_H, self.dpv_W, 3))
        x_range = np.arange(self.dpv_W)
        y_range = np.arange(self.dpv_H)
        x_range = np.concatenate([x_range.reshape(1, self.dpv_W)] * self.dpv_H, axis=0)
        y_range = np.concatenate([y_range.reshape(self.dpv_H, 1)] * self.dpv_W, axis=1)
        ray_array[:, :, 0] = x_range + 0.5
        ray_array[:, :, 1] = y_range + 0.5
        return ray_array

    def get_dataset_index(self, index: int):
        for dataset_index, dataset_size in enumerate(self._dataset_sizes):
            if index >= dataset_size:
                index = index - dataset_size
            else:
                return dataset_index, index
        return None, None

    def preprocess_image(self, img: Image.Image):
        height = int(376)
        width = int(1241)
        top_margin = int(height - self.target_image_size[0]) 
        # top_margin = int((height - self.target_image_size[0]) / 2)
        left_margin = int((width - self.target_image_size[1]) / 2)
        img = img.crop((left_margin, top_margin, left_margin + self.target_image_size[1], top_margin + self.target_image_size[0]))
        img = np.array(img).astype(np.float32) / 255.0 
        img = torch.from_numpy(img).permute(2, 0, 1)        # (3, H, W)
        img = self.normalize(img)
        return img

    # def preprocess_depth(self, depth: np.ndarray, crop_box=None):
    #     if crop_box:
    #         if crop_box[1] >= 0 and crop_box[3] <= depth.shape[0]:
    #             depth = depth[int(crop_box[1]):int(crop_box[3]), :]
    #         else:
    #             depth_ = np.ones((crop_box[3] - crop_box[1], depth.shape[1]))
    #             depth_[-crop_box[1]:-crop_box[1]+depth.shape[0], :] = depth
    #             depth = depth_
    #         if crop_box[0] >= 0 and crop_box[2] <= depth.shape[1]:
    #             depth = depth[:, int(crop_box[0]):int(crop_box[2])]
    #         else:
    #             depth_ = np.ones((depth.shape[0], crop_box[2] - crop_box[0]))
    #             depth_[:, -crop_box[0]:-crop_box[0]+depth.shape[1]] = depth
    #             depth = depth_
    #     if self.target_image_size:
    #         depth = resize(depth, self.target_image_size, order=0)
    #     return torch.tensor(1 / depth)

    def preprocess_depth_dso(self, depth: Image.Image, dso_depth_parameters, crop_box=None):
        h, w, f_x = dso_depth_parameters
        depth = np.array(depth, dtype=np.float)
        indices = np.array(np.nonzero(depth), dtype=np.float)
        indices[0] = np.clip(indices[0] / depth.shape[0] * h, 0, h-1)
        indices[1] = np.clip(indices[1] / depth.shape[1] * w, 0, w-1)

        depth = depth[depth > 0]
        depth = (w * depth / (0.54 * f_x * 65535))

        data = np.concatenate([indices, np.expand_dims(depth, axis=0)], axis=0)

        if crop_box:
            data = data[:, (crop_box[1] <= data[0, :]) & (data[0, :] < crop_box[3]) & (crop_box[0] <= data[1, :]) & (data[1, :] < crop_box[2])]
            data[0, :] -= crop_box[1]
            data[1, :] -= crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            crop_width = crop_box[2] - crop_box[0]
        else:
            crop_height = h
            crop_width = w

        data[0] = np.clip(data[0] / crop_height * self.target_image_size[0], 0, self.target_image_size[0]-1)
        data[1] = np.clip(data[1] / crop_width * self.target_image_size[1], 0, self.target_image_size[1]-1)

        depth = np.zeros(self.target_image_size)
        depth[np.around(data[0]).astype(np.int), np.around(data[1]).astype(np.int)] = data[2]

        return torch.tensor(depth, dtype=torch.float32)

    # def preprocess_depth_annotated_lidar(self, depth: Image.Image, crop_box=None):
    #     depth = np.array(depth, dtype=np.float)
    #     h, w = depth.shape
    #     indices = np.array(np.nonzero(depth), dtype=np.float)

    #     depth = depth[depth > 0]
    #     depth = 256.0 / depth

    #     depth = 1/depth

    #     data = np.concatenate([indices, np.expand_dims(depth, axis=0)], axis=0)

    #     if crop_box:
    #         data = data[:, (crop_box[1] <= data[0, :]) & (data[0, :] < crop_box[3]) & (crop_box[0] <= data[1, :]) & (
    #                     data[1, :] < crop_box[2])]
    #         data[0, :] -= crop_box[1]
    #         data[1, :] -= crop_box[0]
    #         crop_height = crop_box[3] - crop_box[1]
    #         crop_width = crop_box[2] - crop_box[0]
    #     else:
    #         crop_height = h
    #         crop_width = w

    #     data[0] = np.clip(data[0] / crop_height * self.target_image_size[0], 0, self.target_image_size[0] - 1)
    #     data[1] = np.clip(data[1] / crop_width * self.target_image_size[1], 0, self.target_image_size[1] - 1)

    #     depth = np.zeros(self.target_image_size)
    #     depth[np.around(data[0]).astype(np.int), np.around(data[1]).astype(np.int)] = data[2]

    #     return torch.tensor(depth, dtype=torch.float32)

    def preprocess_depth_annotated_lidar(self, depth: Image.Image, left_margin, top_margin):
        gt_dmap = depth.crop((left_margin, top_margin, left_margin + self.target_image_size[1], top_margin + self.target_image_size[0]))
        gt_dmap = np.array(gt_dmap)[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
        gt_dmap = gt_dmap / 256.0
        gt_dmap = torch.from_numpy(gt_dmap).permute(2, 0, 1)  # (1, H, W)

        return gt_dmap

    def __getitem__(self, index: int):
        inputs = {}
        dataset_index, index = self.get_dataset_index(index)
        if dataset_index is None:
            raise IndexError()

        if self.use_index_mask is not None:
            index = self._indices[dataset_index][index] - self._offset

        sequence_folder = self.dataset_dir / "sequences" / self.sequences[dataset_index]
        depth_folder = sequence_folder / self.depth_folder

        if self.use_color_augmentation:
            self.color_transform.fix_transform()

        dataset = self._datasets[dataset_index]

        IntM_ = dataset.calib.P_rect_00 if not self.use_color else dataset.calib.P_rect_20

        raw_W = int(1241)
        raw_H = int(376)

        # top_margin = int((raw_H - self.target_image_size[0]) / 2)
        top_margin = int(raw_H - self.target_image_size[0])
        left_margin = int((raw_W - self.target_image_size[1]) / 2)

        IntM = np.zeros((3, 3))
        IntM[2, 2] = 1.
        IntM[0, 0] = IntM_[0, 0]
        IntM[1, 1] = IntM_[1, 1]
        IntM[0, 2] = (IntM_[0, 2] - left_margin)
        IntM[1, 2] = (IntM_[1, 2] - top_margin) 
        IntM = IntM.astype(np.float32)

        keyframe_depth = self.preprocess_depth_annotated_lidar(Image.open(depth_folder / f"{(index + self._offset):06d}.png"), left_margin, top_margin)

        keyframe = self.preprocess_image(
            (dataset.get_cam0 if not self.use_color else dataset.get_cam2)(index + self._offset))
        keyframe_pose = torch.tensor(dataset.poses[index + self._offset], dtype=torch.float32)

        frames = [self.preprocess_image((dataset.get_cam0 if not self.use_color else dataset.get_cam2)(index + self._offset + i + self.offset_d)) for i in
                  range(-(self.frame_count // 2) * self.dilation, ((self.frame_count + 1) // 2) * self.dilation + 1, self.dilation) if i != 0]
        # intrinsics = [self._intrinsics[dataset_index] for _ in range(self.frame_count)]
        poses = [torch.tensor(dataset.poses[index + self._offset + i + self.offset_d], dtype=torch.float32) for i in
                 range(-(self.frame_count // 2) * self.dilation, ((self.frame_count + 1) // 2) * self.dilation + 1, self.dilation) if i != 0]

        # print('pose_shape',poses[0].shape)
        # print('frames',len(frames))
        # print('poses',len(poses))        
        # print('intrinsics', len(intrinsics))
        data_array = []

        data_dict_preframe = {
            'img': frames[0],
            'gt_dmap': 0.0,
            'extM': np.linalg.inv(poses[0].numpy()),
            # 'extM': poses[0].numpy(),
            'scene_name': torch.tensor([int(self.sequences[dataset_index])], dtype=torch.int32),
            'img_idx': torch.tensor([int(index - self._offset)], dtype=torch.int32)
        }

        data_array.append(data_dict_preframe)

        data_dict_keyframe = {
            'img': keyframe,
            'gt_dmap': keyframe_depth,
            'extM': np.linalg.inv(keyframe_pose.numpy()),
            # 'extM': keyframe_pose.numpy(),            
            'scene_name': torch.tensor([int(self.sequences[dataset_index])], dtype=torch.int32),
            'img_idx': torch.tensor([int(index + self._offset)], dtype=torch.int32)
        }
        data_array.append(data_dict_keyframe)

        data_dict_nextframe = {
            'img': frames[1],
            'gt_dmap': 0.0,
            'extM': np.linalg.inv(poses[1].numpy()),
            # 'extM': poses[1].numpy(),
            'scene_name': torch.tensor([int(self.sequences[dataset_index])], dtype=torch.int32),
            'img_idx': torch.tensor([int(index + 2*self._offset)], dtype=torch.int32)
        }

        data_array.append(data_dict_nextframe)

        inputs[("color", 0, 0)] = data_array[1]['img']
        inputs[("depth_gt", 0, 0)] = data_array[1]['gt_dmap']
        inputs[("pose", 0)] = data_array[1]['extM']

        inputs[("color", 1, 0)] = data_array[0]['img']
        inputs[("pose", 1)] = data_array[0]['extM']


        inputs[("color", 2, 0)] = data_array[2]['img']
        inputs[("pose", 2)] = data_array[2]['extM']

        inputs = self.get_K(IntM, inputs)

        inputs = self.compute_projection_matrix(inputs)

        inputs['num_frame'] = 3

        return inputs

    def __len__(self) -> int:
        return self.length

    def compute_depth_crop(self, depth_folder):
        # This function is only used for dense gt depth maps.
        example_dm = np.load(depth_folder / "000000.npy")
        ry = example_dm.shape[0] / self.target_image_size[0]
        rx = example_dm.shape[1] / self.target_image_size[1]
        if ry < 1 or rx < 1:
            if ry >= rx:
                o_w = example_dm.shape[1]
                w = int(np.ceil(ry * self.target_image_size[1]))
                h = example_dm.shape[0]
                return ((o_w - w) // 2, 0, (o_w - w) // 2 + w, h)
            else:
                o_h = example_dm.shape[0]
                h = int(np.ceil(rx * self.target_image_size[0]))
                w = example_dm.shape[1]
                return (0, (o_h - h) // 2, w, (o_h - h) // 2 + h)
        if ry >= rx:
            o_h = example_dm.shape[0]
            h = rx * self.target_image_size[0]
            w = example_dm.shape[1]
            return (0, (o_h - h) // 2, w, (o_h - h) // 2 + h)
        else:
            o_w = example_dm.shape[1]
            w = ry * self.target_image_size[1]
            h = example_dm.shape[0]
            return ((o_w - w) // 2, 0, (o_w - w) // 2 + w, h)

    def compute_target_intrinsics(self, dataset, target_image_size, use_color):
        # Because of cropping and resizing of the frames, we need to recompute the intrinsics
        P_cam = dataset.calib.P_rect_00 if not use_color else dataset.calib.P_rect_20
        orig_size = tuple(reversed((dataset.cam0 if not use_color else dataset.cam2).__next__().size))

        r_orig = orig_size[0] / orig_size[1]
        r_target = target_image_size[0] / target_image_size[1]

        if r_orig >= r_target:
            new_height = r_target * orig_size[1]
            box = (0, (orig_size[0] - new_height) // 2, orig_size[1], orig_size[0] - (orig_size[0] - new_height) // 2)

            c_x = P_cam[0, 2] / orig_size[1]
            c_y = (P_cam[1, 2] - (orig_size[0] - new_height) / 2) / new_height

            rescale = orig_size[1] / target_image_size[1]

        else:
            new_width = orig_size[0] / r_target
            box = ((orig_size[1] - new_width) // 2, 0, orig_size[1] - (orig_size[1] - new_width) // 2, orig_size[0])

            c_x = (P_cam[0, 2] - (orig_size[1] - new_width) / 2) / new_width
            c_y = P_cam[1, 2] / orig_size[0]

            rescale = orig_size[0] / target_image_size[0]

        f_x = P_cam[0, 0] / target_image_size[1] / rescale
        f_y = P_cam[0, 0] / target_image_size[0] / rescale

        intrinsics = (f_x, f_y, c_x, c_y)

        return intrinsics, box

    def get_dso_depth_parameters(self, dataset):
        # Info required to process d(v)so depths
        P_cam =  dataset.calib.P_rect_20
        orig_size = tuple(reversed(dataset.cam2.__next__().size))
        return orig_size[0], orig_size[1], P_cam[0, 0]

    def get_index(self, sequence, index):
        for i in range(len(self.sequences)):
            if int(self.sequences[i]) != sequence:
                index += self._dataset_sizes[i]
            else:
                break
        return index


    def get_K(self, K, inputs):
        inv_K = np.linalg.inv(K)
        K_pool = {}
        ho, wo = self.target_image_size[0], self.target_image_size[1]
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