#!/bin/bash
BASE_DIR='./'
EXP_DIR="${BASE_DIR}/experiments/release/ScanNet/exp0"
cfg=./configs/scannet/release.conf

## test 
CUDA_VISIBLE_DEVICES=0  python  train.py  --model_name=config0_test --mode=test --cfg $cfg --load_weights_folder=./pretrained_model/scannet/MVS2D --use_test=1 --fullsize_eval=1 
