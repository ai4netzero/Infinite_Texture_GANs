# Infinite_Texture_GANs
Infinite Generation of single texture images using patch-by-patch GAN

This repository contains the codebase for the paper "Local Padding in Patch-Based GANs for Seamless Infinite-Sized Texture Synthesis", https://arxiv.org/abs/2309.02340,
The aim of this project is to generate high-quality texture patterns with infinite resolution using GANs. This local padding also eliminate tiling in super-resolution images please see https://github.com/Alhasan-Abdellatif/Real-ESRGAN-lp


![alt text](https://github.com/ai4netzero/Infinite_Texture_GANs/blob/main/examples/res_diff_examples.png)



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

## Training
To train the GANs model, run the following command 
(Note that it is recommended to use BN instead of SSM for most texture as it is much faster and don't produce artefacts):

```
python train.py --data_path datasets/241.jpg --attention --leak_G 0.02 --sampling 8000 --spec_norm_D --padding_mode local --outer_padding replicate   --random_crop 192 --saving_rate 50  --epochs 300 --type_norm BN  --smooth --ema  --num_gpus 4 --gpu_list 0 1 2 3 --fname results/241_lp_bn_replicate_outerpadding
```

To run the model with SSM set  ``` --type_norm SSM ```

Make sure to replace datasets/241.jpg with the path to your image. Adjust other hyper-parameters according to your requirements. Below is some of hyper-parameters we found to work in some experiments. For most experiments setting num_images =8 ,num_workers=0 works well.

 Experiment | Image     | Random Crop | n_layers_G | n_layers_D | 
|------------|-----------|-------------|------------|------------|
| Experiment 1 | 241.jpg   | 192       | 6          | 4          |
| Experiment 2 | 34.jpg    | 128       | 5          | 4          |
| Experiment 3 | 12.jpg    | 128       | 5          | 3          |
| Experiment 4 | 73.jpg    | 128       | 5          | 3          |
| Experiment 5 | 417.jpg    | 48       | 4          | 3          |


## Inference
After training you can use the example notebook "generate_example.ipynb" to generate large arbitrary size texture images using the saved model. 
Or you can run the test_sample.py file:

```
python test_sample.py --model_path results/241_lp_BN_replicate_outerpadding_nlg6_nld4_padzstochastic_v2/300_150.pth  --output_resolution_height 1024  --output_resolution_width 1024
```

## Acknowledgements
Please cite the original paper if you use this code for your research






