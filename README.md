
# Cartoongan_pytorch
=============================================
## Introduction
Implementation of [CartoonGan](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) with Pytorch. The dataset I used is download from flickr and https://archive.org/details/animationandcartoons.

## Prerequisite
Pytorch(0.4.1)

Visdom(if loss visualization  required)

Opencv(3.4.1)

Numpy

## Usage
To use this code, you should make a folder, which include two sub-folder : real_image and ani_images. you also need to make  "samples" and "wights" 
to save generated  images and weights.
    
    git clone https://github.com/Septembit/Cartoongan_pytorch
    python train.py --ngpu 1 --imgdir "it is your data folder path,which include two sub-folder : real_image and ani_images"

## Results
To be continued..


## Future works
1. The images in animation dataset I used do not include edges. So, I didn't use edge-smoothing in my code. 
