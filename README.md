# Infinite_Texture_GANs
Infinite Generation of single texture images using patch-by-patch GAN

This repository contains the codebase for the paper "Generating Infinite-Size Textures using GANs with Patch-by-Patch Paradigm", https://arxiv.org/abs/2309.02340,
The aim of this project is to generate high-quality texture patterns with infinite resolution using GANs.

Below is an example of input flower image (614x440) and generated image of size 7808x7808.

<p align="center">
  <img width="400" height="300" src="https://github.com/ai4netzero/Infinite_Texture_GANs/blob/main/examples/241.jpg">
</p>

![alt text](https://github.com/ai4netzero/Infinite_Texture_GANs/blob/main/examples/241_61x61.jpeg)



## Requirements

* Python 3.8.10
* PyTorch 1.12.1
* NumPy
* matplotlib
* torchvision

## Dataset

To train the models, you will need a single texture image. The image should be organized in the following specific directory structure:

```
datasets/
        241.jpg
```

## Usage
To train the GANs model, run the following command 
(Note that itt is recommended to use BN instead of SSM for most texture as it is much faster and don't produce artefacts):

```
python train.py --data_path datasets/241.jpg --attention --leak_G 0.02 --sampling 8000 --spec_norm_D --padding_mode local --outer_padding replicate   --random_crop 192 --saving_rate 50  --epochs 2 --type_norm bn  --smooth --ema  --num_gpus 4 --gpu_list 0 1 2 3 --fname results/241_lp_bn_replicate_outerpadding
```

To run the model with SSM set  ``` --type_norm SSM ```

Make sure to replace datasets/241.jpg with the path to your image. Adjust other paramters according to your requirements (e.g., you can try to reduce training time by smaller model capacity by setting --n_layers_G 5 --n_layers_D 3).

After training you can use the example notebook "generate_example.ipynb" to generate large arbitrary size texture images using the saved model. 

## Acknowledgements
Please cite the original paper if you use this code for your research






