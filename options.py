from __future__ import absolute_import, division, print_function
import os
import argparse
from pyhocon import ConfigFactory


class MVS2DOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MVS2D options")

        # PATH OPTIONS
        self.parser.add_argument("--log_dir", type=str, help="", default=None)
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="",
                                 default=None)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 default=None,
                                 help="")
        self.parser.add_argument("--overwrite",
                                 type=int,
                                 default=None,
                                 help="")
        self.parser.add_argument('--note', type=str, help="", default=None)
        self.parser.add_argument("--DECAY_STEP_LIST",
                                 nargs="+",
                                 type=int,
                                 help="",
                                 default=None)
        # DATA OPTIONS
        self.parser.add_argument(
            "--mode",
            type=str,
            default=None,
            choices=["train", "test", "train+test", "full_test", "recon"])
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="",
                                 default=None)
        self.parser.add_argument("--use_test", type=int, default=None, help="")
        self.parser.add_argument("--robust", type=int, default=None, help="")
        self.parser.add_argument("--perturb_pose",
                                 type=int,
                                 default=None,
                                 help="")
        self.parser.add_argument('--num_frame',
                                 type=int,
                                 help="",
                                 default=None)
        self.parser.add_argument('--fullsize_eval',
                                 type=int,
                                 help="",
                                 default=None)
        self.parser.add_argument('--filter', nargs="+", type=str, default=None)
        # MODEL OPTIONS
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="",
                                 default=None)

        # TRAINING OPTIONS

        # OPTIMIZATION OPTIONS

        # MULTI-GPU OPTIONS
        self.parser.add_argument("--world_size", type=int, default=1, help="")
        self.parser.add_argument("--multiprocessing_distributed",
                                 type=int,
                                 default=None,
                                 help="")
        self.parser.add_argument('--rank', type=int, help="", default=0)
        self.parser.add_argument('--gpu', type=int, help="", default=None)
        self.parser.add_argument('--local_rank', type=int, help="", default=0)
        self.parser.add_argument('--tcp_port', type=int, default=None, help="")

        # OTHERS
        self.parser.add_argument('--save_prediction',
                                 type=int,
                                 help="",
                                 default=None)
        self.parser.add_argument("--debug", help="", action="store_true")
        # self.parser.add_argument('--cfg', type=str, default="./configs/DDAD_kitti.conf")
        self.parser.add_argument('--cfg', type=str)

        self.parser.add_argument('--epoch_size', type=int, default=None)
        self.parser.add_argument('--val_epoch_size', type=int, default=None)

    def parse(self):
        self.options = self.parser.parse_args()
        cfg = ConfigFactory.parse_file(self.options.cfg)
        for k in cfg.keys():
            if k not in self.options:
                setattr(self.options, k, cfg[k])
            else:
                if getattr(self.options, k) is None:
                    setattr(self.options, k, cfg[k])

        return self.options


class EvalCfg(object):
    def __init__(self,
                 save_dir,
                 min_depth=1e-3,
                 max_depth=10.0,
                 vis=False,
                 disable_median_scaling=True,
                 eigen_crop=True,
                 print_per_dataset_stats=False,
                 garg_crop=False):
        self.save_dir = save_dir
        self.vis = vis
        self.MIN_DEPTH = min_depth
        self.MAX_DEPTH = max_depth
        self.eigen_crop = eigen_crop
        self.garg_crop = garg_crop
        self.disable_median_scaling = disable_median_scaling
        self.print_per_dataset_stats = print_per_dataset_stats
