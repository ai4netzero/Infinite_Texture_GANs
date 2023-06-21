# Infinite_Texture_GANs
Infinite Generation of single texture images using patch-by-patch GAN


![241](https://github.com/ai4netzero/Infinite_Texture_GANs/assets/48333146/d3716401-153e-4972-b6b4-998b746381a4)
[241_7x21_ex.pdf](https://github.com/ai4netzero/Infinite_Texture_GANs/files/11818623/241_7x21_ex.pdf)

This repository contains the codebase for the paper "Generating Infinite-Resolution Texture using GANs with Patch-by-Patch Paradigm" 
The aim of this project is to generate high-quality texture patterns with infinite resolution using GANs.

## Requirements

* Python 3.8.10
* PyTorch 1.12.1
* NumPy
* matplotlib
* torchvision

## Dataset

To train the GAN, you will need a single texture image. The image should be organized in the following specific directory structure:

```
datasets/
        image1.jpg
```

## Usage
To train the GANs model, run the following command:

```
nohup python train.py --data_path datasets/241.jpg --data single_image --sampling 8000 --img_ch 3 --data_ext jpg --spec_norm_D --D_model patch_GAN --att --D_ch 64 --G_ch 52
G_patch_2D --n_layers_G 6 --n_layers_D 4 --leak_G 0.02 --G_upsampling nearest --zdim 128 --base_res 4 --n_cl 1 --x_fake_GD --G_cond_method conv3x3 --num_patches_w 3 
--num_patches_h 3   --batch_size 80 --random_crop 192 --epochs 600 --save_rate 50 --ema --smooth --dev_num 0 --ngpu 1 --fname ../Exps/wall_v2/241_run1 > ../Exps/logs/24_run1.out &
```
Make sure to replace datasets/241.jpg with the path to your iamge. Adjust other paramters according to your requirements.


## Acknowledgements
Please cite the original paper if you use this code for your research






