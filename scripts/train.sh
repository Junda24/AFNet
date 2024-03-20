#!/bin/bash
# cfg=./configs/DDAD.conf
cfg=./configs/DDAD_kitti.conf


CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 train_kitti.py  --num_epochs=60 --DECAY_STEP_LIST 30 40 --cfg $cfg --load_weights_folder=/home/cjd/MVS2D/pretrained_model/kitti/ --fullsize_eval=1 --use_test=0
