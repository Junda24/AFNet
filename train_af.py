from __future__ import absolute_import, division, print_function
from open3d import *
import numpy as np
import torch
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import json
from utils import *
import networks
import os
import glob
import random
import torch.optim as optim
from options import MVS2DOptions, EvalCfg
from trainer_base_af import BaseTrainer
from hybrid_evaluate_depth import evaluate_depth_maps, compute_errors,compute_errors1,compute_errors_perimage
from dtu_pyeval import dtu_pyeval
import pprint
import torch.distributed as dist


class Trainer(BaseTrainer):
    def __init__(self, options):
        super(Trainer, self).__init__(options)

    def build_model(self):
        self.parameters_to_train = []
        self.model = networks.MVS2D(opt=self.opt).cuda()
        self.parameters_to_train += list(self.model.parameters())
        parameters_count(self.model, 'MVS2D')

    # def build_optimizer(self):
    #     if self.opt.optimizer.lower() == 'adam':
    #         self.model_optimizer = optim.Adam(
    #             self.model.parameters(),
    #             lr=self.opt.LR,
    #             weight_decay=self.opt.WEIGHT_DECAY)
    #     elif self.opt.optimizer.lower() == 'sgd':
    #         self.model_optimizer = optim.SGD(
    #             self.model.parameters(),
    #             lr=self.opt.LR,
    #             weight_decay=self.opt.WEIGHT_DECAY)

    # def val_epoch(self):
    #     print("Validation")
    #     writer = self.writers['val']
    #     self.set_eval()
    #     results_depth = []
    #     val_loss = []
    #     config = EvalCfg(
    #         eigen_crop=False,
    #         garg_crop=False,
    #         min_depth=self.opt.EVAL_MIN_DEPTH,
    #         max_depth=self.opt.EVAL_MAX_DEPTH,
    #         vis=self.epoch % 10 == 0 and self.opt.eval_vis,
    #         disable_median_scaling=self.opt.disable_median_scaling,
    #         print_per_dataset_stats=self.opt.dataset == 'DeMoN',
    #         save_dir=os.path.join(self.log_path, 'eval_%03d' % self.epoch))
    #     if not os.path.exists(config.save_dir):
    #         os.makedirs(config.save_dir)
    #     print('evaluation results save to folder %s' % config.save_dir)
    #     times = []
    #     val_stats = defaultdict(list)
    #     dict_pred = {}
    #     dict_pred_2 = {}
    #     total_result_count = {}
    #     total_result_count_2 = {}

    #     with torch.no_grad():
    #         for batch_idx, inputs in enumerate(self.val_loader):
    #             if self.opt.val_epoch_size != -1 and batch_idx >= self.opt.val_epoch_size:
    #                 break
    #             if batch_idx % 100 == 0:
    #                 print(batch_idx, len(self.val_loader))
    #             # filenames = inputs["filenames"]
    #             losses, outputs = self.process_batch(inputs, 'val')
    #             # b = len(inputs["filenames"])

    #             s = 0
    #             pred_depth = npy(outputs[('depth_pred', s)])
    #             pred_depth_2 = npy(outputs[('depth_pred_2', s)])
    #             depth_gt = npy(inputs[('depth_gt', 0, s)])
    #             mask = np.logical_and(depth_gt > config.MIN_DEPTH,
    #                           depth_gt < config.MAX_DEPTH)
    #             interval = (935 - 425) / (128 - 1)  # Interval value used by MVSNet
    #             # dict_pred_temp = compute_errors(depth_gt[mask], pred_depth[mask], config.disable_median_scaling, config.MIN_DEPTH, config.MAX_DEPTH, interval)
    #             dict_pred_temp = compute_errors(depth_gt, pred_depth, config.disable_median_scaling, config.MIN_DEPTH, config.MAX_DEPTH, interval)

    #             dict_pred_temp_2 = compute_errors1(depth_gt[mask], pred_depth_2[mask], config.disable_median_scaling, config.MIN_DEPTH, config.MAX_DEPTH, interval)

    #             for k, v in dict_pred_temp.items():
    #                 # print(k,v)
    #                 if k in dict_pred:
    #                     dict_pred[k] = dict_pred[k] + v
    #                     # dict_pred['total_count'] = dict_pred['total_count'] + 1.0
    #                 else:
    #                     dict_pred[k] = v
    #                     # dict_pred['total_count'] = 1.0

    #             for k, v in dict_pred_temp_2.items():
    #                 k = k + '_2'
    #                 if k in dict_pred_2:
    #                     dict_pred_2[k] = dict_pred_2[k] + v
    #                     # dict_pred_2['total_count_2'] = dict_pred_2['total_count_2'] + 1.0
    #                 else:
    #                     dict_pred_2[k] = v
    #                     # dict_pred_2['total_count_2'] = 1.0
    #             if batch_idx % 80 == 0:
    #                 writer.add_image('image0', inputs[("color", 0, 0)][0], global_step=self.step, walltime=None, dataformats='CHW')
    #                 writer.add_image('image1', inputs[("color", 1, 0)][0], global_step=self.step, walltime=None, dataformats='CHW')
    #                 writer.add_image('image2', inputs[("color", 2, 0)][0], global_step=self.step, walltime=None, dataformats='CHW')
    #                 depth_gt = gray_2_colormap_np(inputs[("depth_gt", 0, 0)][0][0])
    #                 writer.add_image('depth_gt', depth_gt, global_step=self.step, walltime=None, dataformats='HWC')
    #                 depth_pred = gray_2_colormap_np(outputs[('depth_pred', 0)][0][0])
    #                 writer.add_image('depth_pred', depth_pred, global_step=self.step, walltime=None, dataformats='HWC')

    #     # print('total_count', dict_pred['total_count'])
    #     # print('total_count_2', dict_pred_2['total_count_2'])
    #     # print('abs_rel', dict_pred['abs_rel'])

    #     for k in dict_pred.keys():
    #         total_result_count[k] = dict_pred['total_count']

    #     for k in dict_pred_2.keys():
    #         total_result_count_2[k] = dict_pred_2['total_count_2']


    #     for k in dict_pred.keys():
    #         this_tensor = torch.tensor([dict_pred[k], total_result_count[k]]).to(self.device)
    #         this_list = [this_tensor]
    #         torch.distributed.all_reduce_multigpu(this_list)
    #         this_tensor = this_list[0].detach().cpu().numpy()
    #         reduce_sum = this_tensor[0].item()
    #         reduce_count = this_tensor[1].item()
    #         reduce_mean = reduce_sum / reduce_count
    #         if self.is_master:
    #             writer.add_scalar(k, reduce_mean, self.step)

    #     for k in dict_pred_2.keys():
    #         this_tensor = torch.tensor([dict_pred_2[k], total_result_count_2[k]]).to(self.device)
    #         this_list = [this_tensor]
    #         torch.distributed.all_reduce_multigpu(this_list)
    #         this_tensor = this_list[0].detach().cpu().numpy()
    #         reduce_sum = this_tensor[0].item()
    #         reduce_count = this_tensor[1].item()
    #         reduce_mean = reduce_sum / reduce_count
    #         if self.is_master:
    #             writer.add_scalar(k, reduce_mean, self.step)

    #     self.set_train()

    def val_epoch(self):
        print("Validation")
        writer = self.writers['val']
        self.set_eval()
        results_depth = []
        val_loss = []
        config = EvalCfg(
            eigen_crop=False,
            garg_crop=False,
            min_depth=self.opt.EVAL_MIN_DEPTH,
            max_depth=self.opt.EVAL_MAX_DEPTH,
            vis=self.epoch % 10 == 0 and self.opt.eval_vis,
            disable_median_scaling=self.opt.disable_median_scaling,
            print_per_dataset_stats=self.opt.dataset == 'DeMoN',
            save_dir=os.path.join(self.log_path, 'eval_%03d' % self.epoch))
        if not os.path.exists(config.save_dir) and self.is_master:
            os.makedirs(config.save_dir)
        print('evaluation results save to folder %s' % config.save_dir)
        times = []
        val_stats = defaultdict(list)
        total_result_sum = {}
        total_result_count = {}

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                if self.opt.val_epoch_size != -1 and batch_idx >= self.opt.val_epoch_size:
                    break
                if batch_idx % 100 == 0:
                    print(batch_idx, len(self.val_loader))
                # filenames = inputs["filenames"]
                losses, outputs = self.process_batch(inputs, 'val')
                # b = len(inputs["filenames"])

                s = 0
                pred_depth = npy(outputs[('depth_pred', s)])
                pred_depth_2 = npy(outputs[('depth_pred_2', s)])
                depth_gt = npy(inputs[('depth_gt', 0, s)])
                mask = np.logical_and(depth_gt > config.MIN_DEPTH,
                              depth_gt < config.MAX_DEPTH)
                error_temp = compute_errors_perimage(depth_gt[mask], pred_depth[mask], config.MIN_DEPTH, config.MAX_DEPTH)
                error_temp_2_ = compute_errors_perimage(depth_gt[mask], pred_depth_2[mask], config.MIN_DEPTH, config.MAX_DEPTH)

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
                self.eval_step += 1

                if self.eval_step % 80 == 0 and self.is_master:
                    writer.add_image('image0', inputs[("color", 0, 0)][0], global_step=self.eval_step, walltime=None, dataformats='CHW')
                    writer.add_image('image1', inputs[("color", 1, 0)][0], global_step=self.eval_step, walltime=None, dataformats='CHW')
                    writer.add_image('image2', inputs[("color", 2, 0)][0], global_step=self.eval_step, walltime=None, dataformats='CHW')
                    depth_gt = gray_2_colormap_np(inputs[("depth_gt", 0, 0)][0][0])
                    writer.add_image('depth_gt', depth_gt, global_step=self.eval_step, walltime=None, dataformats='HWC')
                    depth_pred = gray_2_colormap_np(outputs[('depth_pred', 0)][0][0])
                    writer.add_image('depth_pred', depth_pred, global_step=self.eval_step, walltime=None, dataformats='HWC')
                    depth_pred_2 = gray_2_colormap_np(outputs[('depth_pred_2', 0)][0][0])
                    writer.add_image('depth_pred_2', depth_pred_2, global_step=self.eval_step, walltime=None, dataformats='HWC')

        # print('total_count', dict_pred['total_count'])
        # print('total_count_2', dict_pred_2['total_count_2'])
        # print('abs_rel', dict_pred['abs_rel'])

            for k in total_result_sum.keys():
                total_result_count[k] = total_result_sum['valid_number']


            for k in total_result_sum.keys():
                this_tensor = torch.tensor([total_result_sum[k], total_result_count[k]]).to(self.device)
                this_list = [this_tensor]
                torch.distributed.all_reduce_multigpu(this_list)
                torch.distributed.barrier()
                this_tensor = this_list[0].detach().cpu().numpy()
                reduce_sum = this_tensor[0].item()
                reduce_count = this_tensor[1].item()
                reduce_mean = reduce_sum / reduce_count
                if self.is_master:
                    writer.add_scalar(k, reduce_mean, self.eval_step)

        self.set_train()

    def process_batch(self, inputs, mode):
        self.to_gpu(inputs)

        imgs, proj_mats, pose_mats = [], [], []
        for i in range(inputs['num_frame'][0].item()):
            imgs.append(inputs[('color', i, self.opt.input_scale)])
            proj_mats.append(inputs[('proj', i)])
            pose_mats.append(inputs[('pose', i)])

        # outputs = self.model(imgs[0], imgs[1:], proj_mats[0], proj_mats[1:],
        #                      inputs[('inv_K_pool', 0)])
        outputs = self.model(imgs[0], imgs[1:], pose_mats[0], pose_mats[1:],
                             inputs[('inv_K_pool', 0)])
        losses = self.compute_losses(inputs, outputs)
        return losses, outputs

    def compute_losses(self, inputs, outputs):
        losses, loss, s = {}, 0, 0
        depth_pred = outputs[('depth_pred', s)]
        depth_pred_2 = outputs[('depth_pred_2', s)]
        depth_gt = inputs[('depth_gt', 0, s)]


        valid_depth = (depth_gt > 0)

        # if self.opt.pred_conf:
        #     log_conf_pred = outputs[('log_conf_pred', s)]
        #     conf_pred = torch.exp(log_conf_pred)
        #     min_conf = self.opt.min_conf
        #     max_conf = self.opt.max_conf if self.opt.max_conf != -1 else None
        #     conf_pred = conf_pred.clamp(min_conf, max_conf)
        #     loss_depth = ((depth_pred - depth_gt).abs() / conf_pred +
        #                   log_conf_pred)[valid_depth].mean()
        # else:
        loss_depth = (depth_pred[valid_depth] - depth_gt[valid_depth]).abs().mean()

        loss_depth_2 = (depth_pred_2[valid_depth] - depth_gt[valid_depth]).abs().mean()

        losses["depth"] = loss_depth
        losses["depth_2"] = loss_depth_2

        loss += loss_depth + loss_depth_2
        losses["loss"] = loss

        return losses


def run_fusion(dense_folder, out_folder, opts):
    cmd = f"python patchmatch_fusion.py \
                --dense_folder {dense_folder} \
                --outdir {out_folder} \
                --n_proc 4 \
                --conf_thres {opts.conf_thres} \
                --att_thres {opts.att_thres} \
                --use_conf_thres {opts.pred_conf} \
                --geo_depth_thres {opts.geo_depth_thres} \
                --geo_pixel_thres {opts.geo_pixel_thres} \
                --num_consistent {opts.num_consistent} \
                "

    os.system(cmd)


if __name__ == "__main__":
    options = MVS2DOptions()
    opts = options.parse()

    set_random_seed(666)

    if torch.cuda.device_count() > 1 and not opts.multiprocessing_distributed:
        raise Exception(
            "Detected more than 1 GPU. Please set multiprocessing_distributed=1 or set CUDA_VISIBLE_DEVICES"
        )

    opts.distributed = opts.world_size > 1 or opts.multiprocessing_distributed
    if opts.multiprocessing_distributed:
        # total_gpus, opts.rank = init_dist_pytorch(opts.tcp_port,
        #                                           opts.local_rank,
        #                                           backend='nccl')
        print('opts.local_rank', opts.local_rank)
        torch.cuda.set_device(opts.local_rank)
        dist.init_process_group("nccl", rank=opts.local_rank, world_size=3)
        opts.ngpus_per_node = 3
        opts.gpu = opts.local_rank
        print("Use GPU: {}/{} for training".format(opts.gpu,
                                                   opts.ngpus_per_node))
    else:
        opts.gpu = 0

    if opts.mode == 'train':
        trainer = Trainer(opts)
        trainer.train()

    elif opts.mode == 'test':
        trainer = Trainer(opts)
        trainer.val()

    elif opts.mode == 'full_test':
        ##  save depth prediction
        opts.mode = 'test'
        trainer = Trainer(opts)
        trainer.val()

        ## fuse dense prediction into final point cloud
        dense_folder = f"{opts.log_dir}/{opts.model_name}/eval_000/prediction"
        out_folder = f"{opts.log_dir}/{opts.model_name}/recon"
        run_fusion(dense_folder, out_folder, opts)

        ## eval point cloud
        MeanData, MeanStl, MeanAvg = dtu_pyeval(
            f"{out_folder}",
            gt_dir='./data/SampleSet/MVS Data/',
            voxel_down_sample=False,
            fn=f"{out_folder}/result.txt")
