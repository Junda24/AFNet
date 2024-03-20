<div align="center">
<h2>AFNet: Adaptive Fusion of Single-View and Multi-View Depth for Autonomous Driving</h2>
<h2>**CVPR 2024**<h2>

[Paper](https://arxiv.org/pdf/2403.07535.pdf)
</div>

This work presents AFNet, a new multi-view and singleview depth fusion network AFNet for alleviating the defects of the existing multi-view methods, which will fail under noisy poses in real-world autonomous driving scenarios.

![teaser](assets/pointcloud2.png)


## ✏️ Changelog
### March 20 2024
* Initial release. 

## ⚙️ Installation

The code is tested with CUDA11.7. Please use following commands to install dependencies: 

```
conda create --name AFNet python=3.7
conda activate AFNet

pip install -r requirements.txt
```

## 🎬 Demo
![teaser](assets/visual_compare.png)


## ⏳ Training & Testing

We use 4 Nvidia 3090 GPU for training. You may need to modify 'CUDA_VISIBLE_DEVICES' and batch size to accomodate your GPU resources.

#### Training
First download and extract DDAD and KITTI data and split. Then run following command to train our model. 
```
bash scripts/train.sh
```

#### Testing 
First download and extract data, split and pretrained models.

Then run:
```
bash scripts/test.sh
```


#### Acknowledgement
Thanks to Zhenpei Yang for opening source of his excellent works [MVS2D](https://github.com/zhenpeiyang/MVS2D?tab=readme-ov-file#nov-27-2021)


