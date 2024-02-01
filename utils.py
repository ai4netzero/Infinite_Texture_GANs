import torch.nn.functional as F
import time
import argparse
from collections import OrderedDict
import os
import random
import torch.nn as nn
import torch.utils.data
import numpy as np
from math import ceil
from models.generators import ResidualPatchGenerator
from models.discriminators import PatchDiscriminator

def prepare_parser():
    parser = argparse.ArgumentParser()
                  
    # data settings     
    parser.add_argument('--data', type=str, default='single_image'
                       ,help = 'type of data')
    parser.add_argument('--data_path', type=str, default='datasets/241.jpg'
                       ,help = 'data path')
    parser.add_argument('--data_ext', type=str, default='jpg'
                       ,help = 'data extension txt, png')
    parser.add_argument('--center_crop', type=int, default=None
                       ,help = 'center cropping')
    parser.add_argument('--random_crop', type=int, default=None
                       ,help = 'random cropping')
    parser.add_argument('--resize_h', type=int, default=None
                       ,help = 'resize for h ')
    parser.add_argument('--resize_w', type=int, default=None
                       ,help = 'resize for w')                                                                          
    parser.add_argument('--sampling', type=int, default=8000
                       ,help = 'randomly sample --sampling instances from the training data if not None')
    
    # models settings
    parser.add_argument('--D_model', type=str, default='patch_GAN'
                        ,help = 'Discriminator Model can be residual_GAN, dcgan, sngan or patch_GAN')
    parser.add_argument('--attention',action='store_true',default=False
                        ,help = 'Use Attention in the generator if True  (only implmented in residual_GAN)')
    parser.add_argument('--img_ch', type=int, default=3
                        ,help = 'the number of image channels 1 for grayscale 3 for RGB')
    parser.add_argument('--G_ch', type=int, default=52
                        ,help = 'base multiplier for the Generator (for cnn_GAN should be large 512/1024) , (for ')
    parser.add_argument('--D_ch', type=int, default=64
                        ,help = 'base multiplier for the discriminator')
    parser.add_argument('--leak_G', type=float, default=0
                        ,help = 'use leaky relu activation for generator with leak= leak_G,zero value will use RELU')
    parser.add_argument('--leak_D', type=float, default=0
                        ,help = 'use leaky relu activation for discriminator with leak= leak_G,zero value will use RELU')
    parser.add_argument('--z_dim', type=int, default=128
                        ,help ='dimenstion of the latent input')
    parser.add_argument('--map_dim', type=int, default=1
                        ,help ='dimenstion of the modulation map if SSM is used')
    parser.add_argument('--spec_norm_D', default=False,action='store_true'
                       ,help = 'apply spectral normalization in discriminator')
    parser.add_argument('--spec_norm_G', default=False,action='store_true'
                       ,help = 'apply spectral normalization in generator')
    parser.add_argument('--n_layers_D', type=int, default=4
                       ,help = 'number of layers used in discriminator of dcgan,patchGAN')
    parser.add_argument('--n_layers_G', type=int, default=6
                       ,help = 'number of layers used in generator') 
    parser.add_argument('--norm_layer_D', type=str, default=None
                       ,help = 'normalization layer in patchGAN')
    parser.add_argument('--base_res', type=int, default=4
                       ,help = 'base resolution for G') 
    parser.add_argument('--padding_mode', type=str, default='zeros'
                       ,help = 'padding used in G either zeros or local')
    parser.add_argument('--type_norm_G', type=str, default='BN'
                       ,help = 'type normalization used in G either BN or SSM')

    # optimizers settings
    parser.add_argument('--lr_G', type=float, default=2e-4
                        ,help = 'Generator learning rate')
    parser.add_argument('--lr_D', type=float, default=2e-4
                        ,help = 'discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0
                        ,help = 'first momentum value for ADAM optimizer')
    parser.add_argument('--beta2', type=float, default=0.999
                        ,help = 'second momentum value for ADAM optimizer')
    parser.add_argument('--batch_size', type=int, default=64
                        ,help = 'discriminator batch size')
    
    #training settings
    parser.add_argument('--loss', type=str, default='standard'
                        ,help = 'Loss function can be standard,hinge or wgan')
    parser.add_argument('--disc_iters', type=int, default=1
                        ,help = ' no. discriminator updates per one generator update')
    parser.add_argument('--epochs', type=int, default=1
                        ,help ='number of epochs')
    parser.add_argument('--saving_rate', type=int, default=30
                        ,help = 'save checkpoints every 30 epcohs')
    parser.add_argument('--ema',action='store_true' , default=False
                        ,help = 'keep EMA of G weights')
    parser.add_argument('--ema_decay',type = float, default=0.999
                        ,help = 'EMA decay rate')
    parser.add_argument('--decay_lr',type=str,default=None
                        ,help = 'if not None decay the learning rates (exp,step)')
    parser.add_argument('--seed', type=int, default=None
                       ,help = 'None to use random seed can be fixed for reporoduction')
    parser.add_argument('--smooth',default=False,action='store_true'
                       , help = 'Use smooth labeling if True')
    
    # patch generation parameters
    parser.add_argument('--num_images', type=int, default=8
                        ,help ='number of images generated by the generator')
    parser.add_argument('--num_patches_width', type=int, default=3
                        ,help ='Number of patches along the width dimension of the image')
    parser.add_argument('--num_patches_height', type=int, default=3
                        ,help ='Number of patches along the height dimension of the image')                      
    parser.add_argument('--outer_padding', type=str, default='replicate'
                        ,help='padding used in the borders of the outer patches either replicate or constant for zero padding')
    parser.add_argument('--padding_size', type=int, default=1
                        ,help ='padding size used in local padding')
    parser.add_argument('--conv_reduction', type=int, default=2
                        ,help ='reduction after the convolution operator')
    
    
    # GPU settings
    parser.add_argument('--num_gpus', type=int, default=1
                        ,help = 'number of gpus to be used')                                     
    parser.add_argument('--dev_num', type=int, default=0
                        ,help = 'the index of a gpu to be used if --ngpu is 1 ')
    parser.add_argument('--gpu_list', nargs='+', default=None,type=int
                        ,help='list of devices to used in parallizatation if ngpu > 1')
                        
    # folder name             
    parser.add_argument('--fname', type=str, default='models_cp',help='folder name to save cp')

    return parser

def prepare_device(args):
    # Device
    ngpu = args.num_gpus
    print("Device: ",args.dev_num)

    if ngpu==1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.dev_num)
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    else:
        device = torch.device("cuda:"+str(args.dev_num)) 
    return device

def prepare_seed(args):
    #Seeds
    if args.seed is None:
        # use if you want new results
        seed = random.randint(1, 10000) 
    else : 
        seed = args.seed 

    print("Random Seed: ", seed)
    return seed
   
def prepare_data(args):
    print(" laoding " +args.data +" ...")

    if args.resize_h is None and args.resize_w is None:
        resize = None
    else:
        resize = (args.resize_h,args.resize_w)

    if args.data == 'single_image':
        from datasets import datasets_classes
        train_data = datasets_classes.single_image(path = args.data_path
                                                ,ext = args.data_ext
                                                ,sampling = args.sampling
                                                ,random_crop = args.random_crop
                                                ,center_crop = args.center_crop)
    elif args.data == 'multiple_images':
        from datasets import datasets_classes
        train_data = datasets_classes.multiple_images(path = args.data_path
                                                ,ext = args.data_ext
                                                ,sampling = args.sampling
                                                ,random_crop = args.random_crop
                                                ,center_crop = args.center_crop
                                                ,resize=resize)

    else:
        print('no data named :',args.data)
        exit()

    dataloader = torch.utils.data.DataLoader(train_data,
                       shuffle=True, batch_size=args.batch_size,
                       num_workers=1)

    print("Finished data loading")    
    return dataloader,train_data


            
def prepare_models(args,device = 'cpu'):           
    #model
    netG = ResidualPatchGenerator(z_dim = args.z_dim,G_ch = args.G_ch,base_res=args.base_res,n_layers_G = args.n_layers_G,attention=args.attention,
                                        img_ch= args.img_ch,leak = args.leak_G,SN = args.spec_norm_G,type_norm = args.type_norm_G,map_dim = args.map_dim,
                                        padding_mode = args.padding_mode,outer_padding = args.outer_padding,
                                        num_patches_h = args.num_patches_height,num_patches_w=args.num_patches_width,
                                        padding_size = args.padding_size,conv_reduction = args.conv_reduction).to(device)
        
        

    if args.D_model == 'patch_GAN':
        netD = PatchDiscriminator(img_ch=args.img_ch,base_ch = args.D_ch,n_layers_D=args.n_layers_D,kw = 4
                                    ,SN= args.spec_norm_D,norm_layer = args.norm_layer_D).to(device)
    return netG,netD


def prepare_filename(args):
    filename = str(args.epochs) + "_"

    if args.fname is not None:
        if not os.path.exists(args.fname):
            os.makedirs(args.fname)
        filename = args.fname+"/" + filename
    return filename


def build_z(num_images=1,z_dim=128,base_res=4,num_patches_height=3, num_patches_width=3,total_num_patches_height=3, total_num_patches_width=3,device='cpu'):
    
    pad_size = 2
    image_base_height = total_num_patches_height*base_res+pad_size
    image_base_width = total_num_patches_width*base_res+pad_size
    
    # Build the spatial latent input z for the full image
    z_full_image = torch.randn(num_images,z_dim,image_base_height,image_base_width).to(device)
    
    # Crop the z input into overlapping sub-image each has resolution of (num_patches_height*base_res,num_patches_width*base_res)
    # The overlap is of size base_res and is needed to regenerate the outer patches in the next steps with appopriate padding
    z_sub_image = crop_images(z_full_image,num_patches_height*base_res+pad_size,num_patches_width*base_res+pad_size,(num_patches_width-1)*base_res,device = device)
    
    return z_sub_image


def build_maps(num_images=1,map_dim=1,n_layers_G=4,base_res=4,num_patches_height=3, num_patches_width=3,total_num_patches_height=3, total_num_patches_width=3,device='cpu'):
    
    maps_sub_image = []
    pad_size_maps = 4
    for i in range(0,n_layers_G):
        res_layer = (2**i)*base_res
        
        # Build the spatial map for the full image at this layer
        # Add padding of 4 because the maps are passed to a 2 consecutive conv. layers.
        maps_full_image =  torch.randn(num_images,map_dim,total_num_patches_height*res_layer+pad_size_maps,total_num_patches_width*res_layer+pad_size_maps).to(device)
        
        # Crop the map input into overlapping sub-image each has resolution of (num_patches_height*res_layer+pad_size_maps,num_patches_width*res_layer+pad_size_maps)
        # The overlap is of size res_layer and is needed to regenerate the outer patches in the next steps with appopriate padding
        map_res_with_padding_h = num_patches_height*res_layer+pad_size_maps
        map_res_with_padding_w = num_patches_width*res_layer+pad_size_maps
        maps_layer_i = crop_images(maps_full_image,map_res_with_padding_h,map_res_with_padding_w,(num_patches_width-1)*res_layer,device = device)
        
        maps_sub_image.append(maps_layer_i)

    return maps_sub_image
    
def sample_from_gen_PatchByPatch_test(netG,z_dim=128,base_res=4,map_dim = 1,num_images=1,
                                      num_patches_height=3, num_patches_width=3,device ='cpu',output_resolution_height = 384,output_resolution_width = 384): 
    """
    Generate a large image using a Patch-by-Patch sampling approach from a PyTorch generator network.

    This function generates the large image in steps, where each step involves generating a sub-image.
    The outer patches in each sub_image is padded using outer_padding (zeros or replicate padding).
    These patches are re-generated in the next sub-image with local padding and the previously generated outer patches are dropped.
    The sub-images are concatenated first vertically to form rows and then generation continues row by row.
    Finally, the rows are concatenated to form the complete large image.

    Parameters:
    - netG (nn.Module): The PyTorch generator network used for image generation.
    - z_dim (int): Dimension of the input latent vector (default is 128).
    - base_res (int): Base resolution of the generated image (default is 4).
    - map_dim (int): Dimension of the maps used in case of SSM (default is 1).
    - num_images (int): Number of images to generate (default is 1).
    - num_patches_height (int): Number of patches along the height dimension (default is 3).
    - num_patches_width (int): Number of patches along the width dimension (default is 3).
    - device (str): Device on which to perform the generation (default is 'cpu').
    - output_resolution_height (int): Desired height of the output large image in pixels (default is 384).
    - output_resolution_width (int): Desired width of the output large image in pixels  (default is 384).

    Returns:
    torch.Tensor: The generated large image tensor of shape (num_images, 3, output_resolution_height, output_resolution_width).
    """


    if isinstance(netG, nn.DataParallel): 
        n_layers_G =  netG.module.n_layers_G
        type_norm = netG.module.type_norm
    else:
        n_layers_G =  netG.n_layers_G
        type_norm = netG.type_norm
    
    # The image patch resolution generated by the generator
    generator_patch_resolution = (2**(n_layers_G-1))*base_res

    
    # calculate the number of steps in both dimensions required to iterate through the generator to generate the full image
    steps_h = ceil((output_resolution_height/generator_patch_resolution - 1)/(num_patches_height-1))
    steps_w = ceil((output_resolution_width/generator_patch_resolution - 1)/(num_patches_width-1))
    
    # calculate how many patches needed to be generated
    total_num_patches_height = steps_h*(num_patches_height-1)+1
    total_num_patches_width = steps_w*(num_patches_width-1)+1
    
    # Build the inputs to the generator z and maps
    z_sub_images = build_z(num_images,z_dim,base_res,num_patches_height, num_patches_width,total_num_patches_height, total_num_patches_width,device='cpu')
    if type_norm == 'SSM':
        map_sub_images = build_maps(num_images,map_dim,n_layers_G,base_res,num_patches_height, num_patches_width,total_num_patches_height, total_num_patches_width,device)
    
    # Iterate through the generator with to generate the sub_images in sequence 
    # and concatente the sub_images to form the full image
    
    last_row_ind = steps_h-1
    last_column_ind = steps_w-1

    sub_image_ind = 0
    for ind_h in range(steps_h):
        for ind_w in range(steps_w):
            
            # Update the location of the generated image. Note that the first row could also be the last column
            if last_row_ind ==0:
                image_location = '1st_row_last_row'
            elif ind_h == 0:
                image_location = '1st_row'
            elif ind_h == last_row_ind :
                image_location = 'last_row'
            else:
                image_location = 'inter_row'
            
            if last_column_ind ==0:
                image_location += '_1st_col_last_col'
            elif ind_w ==0:
                image_location += '_1st_col'
            elif ind_w == last_column_ind :
                image_location += '_last_col'
            else:
                image_location += '_inter_col'
            
            
            # Get z input for the current sub_image and crop it into patches
            z_sub_image = z_sub_images[[sub_image_ind]].to(device)
            #z_patches = crop_images(z_sub_image,base_res,base_res,base_res,device = device)
            
            # Get map input for the current sub_image and crop it into patches
            if type_norm == 'SSM':
                map_pacthes = []
                for i in range(0,n_layers_G):
                    res_layer = (2**i)*base_res
                    pad_size = 4
                    maps = crop_images(map_sub_images[i][[sub_image_ind]],res_layer+pad_size,res_layer+pad_size,res_layer,device = device)
                    map_pacthes.append(maps)
            else:
                map_pacthes =  [None] * n_layers_G
                    
            # Pass the input to the model to get patch_i
            with torch.no_grad():
                patches_i = netG(z_sub_image,map_pacthes,image_location)
            
            # Concatenate the patches to form a a sub_image
            sub_image_i = merge_patches_into_image(patches_i,num_patches_height,num_patches_width,device).cpu()
            
            # Drop the re-generated patches
            # Crop the left and bottom patches in the sub_image if it is not in the last row or last column
            if ind_h != last_row_ind and ind_w !=last_column_ind:
                sub_image_i_cropped = sub_image_i[:,:,0:generator_patch_resolution*(num_patches_height-1),0:generator_patch_resolution*(num_patches_width-1)]
                
            # Crop only the bottom patches in the sub_image if it is in the last column and not in the last row
            elif ind_h != last_row_ind:
                sub_image_i_cropped = sub_image_i[:,:,0:generator_patch_resolution*(num_patches_height-1),:]
                
            # Crop only the left patches in the sub_image if it is in the last row and not in the last column
            elif ind_w !=last_column_ind:
                sub_image_i_cropped = sub_image_i[:,:,:,0:generator_patch_resolution*(num_patches_width-1)]
                
            # Otherwise do not crop the image
            else:
                sub_image_i_cropped = sub_image_i
                    
            # Update the index for the next sub_image
            sub_image_ind = sub_image_ind+1
                        
            # Concatenate the sub images together to form a row
            if '1st_col' in image_location:
                image_row = sub_image_i_cropped
            else:
                image_row = torch.cat((image_row,sub_image_i_cropped),-1)
                
        # Concatenate the rows together to form the full image
        if '1st_row' in image_location:
            full_image = image_row
        else:
            full_image = torch.cat((full_image,image_row),-2)
            
    # Adjust the generated image to match the target size if it is larger
    full_image = full_image[:,:,:output_resolution_height,:output_resolution_width] 
    
    return full_image
    
    
    

def sample_from_gen_PatchByPatch_train(netG,z_dim=128,base_res=4,map_dim = 1,num_images=1, num_patches_height=3, num_patches_width=3,device ='cpu'): 
    """  
        Generate images using the generator network netG in a patch-by-patch fashion during training.

        Args:
            generator (torch.nn.Module): The generator network used for image synthesis.
            z_dim (int): Dimension of the input latent vector (default is 128).
            base_res (int): Base resolution of the generated image (default is 4).
            map_dim (int): Dimension of the maps used in case of SSM (default is 1).
            num_images (int): Number of synthetic images to generate (default is 1).
            num_patches_height (int): Number of patches along the height dimension of the image (default is 3).
            num_patches_width (int): Number of patches along the width dimension of the image (default is 3).
            device (str): Device on which to perform the generation (default is 'cpu').

        Returns:
            torch.Tensor: Tensor containing the generated images with shape (num_images, C, H, W), 
            where H = num_patches_height*generator.base_res, W=num_patches_width*generator.base_res
        """
        
    if isinstance(netG, nn.DataParallel): 
        n_layers_G =  netG.module.n_layers_G
        type_norm = netG.module.type_norm
    else:
        n_layers_G =  netG.n_layers_G
        type_norm = netG.type_norm
                
    #Build the spatial latent input z 
    pad_size = 2
    z_images =  torch.randn(num_images,z_dim,num_patches_height*base_res+pad_size,num_patches_width*base_res+pad_size).to(device)

    #Build the second input M for stochastic spatial modulation
    if type_norm == 'SSM':
        maps_per_layers = []
        pad_size = 4
        for i in range(0,n_layers_G):
            res_layer = (2**i)*base_res
            res_with_pad = res_layer+pad_size
            
            # Build the spatial map input with a pad of size 4, since we use two 3x3 conv layers, for num_images N : (N,mdim ,Tot_layer_res_h +2, Tot_layer_res_w+2)
            maps_images =  torch.randn(num_images,map_dim,num_patches_height*res_layer+pad_size,num_patches_width*res_layer+pad_size).to(device)
            
            # Crop the maps_images input into smaller overlapping patches of size res_layer + 4 with an overlap size of 4:  (N*num_patches_per_img,mdim ,res_layer +4, res_layer+4)
            maps_patches = crop_images(maps_images,res_with_pad,res_with_pad,res_layer,device = device)
            
            maps_per_layers.append((maps_patches))
    else:
        maps_per_layers = [None]*n_layers_G
    
    fake_images_patches = netG(z_images, maps_per_layers,image_location = '1st_row_1st_col')
    
    fake_images = merge_patches_into_image(fake_images_patches,num_patches_height,num_patches_width,device)
    
    return fake_images


def merge_patches_into_image(patches, num_rows=3, num_cols=3, device='cpu'):
    """
    Merge 2D patches into complete images.

    Args:
        patches (torch.Tensor): Input tensor of patches with shape (batch_size, channels, patch_height, patch_width).
        num_rows (int): Number of rows of patches in each image (default is 3).
        num_cols (int): Number of columns of patches in each image (default is 3).
        device (str): Device on which to perform the merging (default is 'cpu').

    Returns:
        torch.Tensor: Tensor containing merged images with shape (batch_size//num_patches_per_img, channels, height, width).
    """
    batch_size, channels, patch_height, patch_width = patches.size()
    num_patches_per_img = num_rows * num_cols

    # Calculate the dimensions of the merged images
    merged_height = patch_height * num_rows
    merged_width = patch_width * num_cols

    # Initialize an empty tensor to store the merged images
    merged_images = torch.empty((batch_size // num_patches_per_img, channels, merged_height, merged_width), device=device)

    for k in range(batch_size // num_patches_per_img):  # for each image
        merged_image = torch.tensor([]).to(device=device)

        for r in range(num_rows):  # each row in each image
            img_row = patches[k * num_patches_per_img + r * num_cols]

            for c in range(1, num_cols):
                img_row = torch.cat((img_row, patches[k * num_patches_per_img + r * num_cols + c]), dim=-1)

            merged_image = torch.cat((merged_image, img_row), dim=-2)  # concatenate rows

        merged_images[k] = merged_image

    return merged_images.to(device=device)
            

def load_netG(netG,checkpointname = None,add_module=False):
    checkpoint = torch.load(checkpointname)
    state_dict_G = checkpoint['netG_state_dict']    
    new_state_dict_G = OrderedDict()
    if add_module:
        for k, v in state_dict_G.items():
            #if 'module' in k:
            #    k = k.replace('module.','')
            k = 'module.'+k
            new_state_dict_G[k] = v
    else:
        for k, v in state_dict_G.items():
            if 'module' in k:
                k = k.replace('module.','')
            new_state_dict_G[k] = v
    netG.load_state_dict(new_state_dict_G)
    netG.eval()

    return netG

def get_trun_noise(truncated,z_dim,b_size,device):
    flag = True
    while flag:
        z = np.random.randn(100*b_size*z_dim) 
        z = z[np.where(abs(z)<truncated)]
        if len(z)>=64*z_dim:
            flag=False
    z = torch.from_numpy(z[:b_size*z_dim]).view(b_size,z_dim)
    z = z.float().to(device)
    return z

def elapsed_time(start_time):
    return time.time() - start_time

def calc_ralsloss_G(real, fake, margin=1.0):
    loss_real = torch.mean((real - fake.mean() + margin) ** 2)
    loss_fake = torch.mean((fake - real.mean() - margin) ** 2)
    loss = (loss_real + loss_fake)
    
    return loss


def crop_images(img, cropping_size_h=256, cropping_size_w=256, stride=256, device='cpu'):
    """
    Crop input images in a PyTorch tensor into smaller patches and concatenate them together.

    Args:
        img (torch.Tensor): Input tensor containing a batch of images with shape (N, C, H, W).
        cropping_size_h (int): Height of the cropped patches (default is 256).
        cropping_size_w (int): Width of the cropped patches (default is 256).
        stride (int): Stride for sliding the cropping window (default is 256).
        device (str): Device on which to perform the cropping and concatenation (default is 'cpu').

    Returns:
        torch.Tensor: Tensor containing concatenated patches with shape (N * P, C, cropping_size_h, cropping_size_w),
                     where P is the number of patches per image.
    """
    
    # Clone the input tensor to avoid modifying the original data
    img = img.clone()

    # Get the number of images in the batch
    N = img.shape[0]

    # Initialize an empty tensor to store the concatenated patches
    batch_patches = torch.tensor([]).to(device)

    # Loop through each image in the batch
    for l in range(N):
        # Crop the image into smaller patches
        crops = crop_image(img[l, :, :, :], cropping_size_h=cropping_size_h, cropping_size_w=cropping_size_w, stride=stride, device=device)

        # Concatenate the patches to the batch_patches tensor
        batch_patches = torch.cat((batch_patches, crops), 0)

    return batch_patches
   

def crop_image(img, cropping_size_h=256, cropping_size_w=256, stride=256, device='cpu'):
    """
    Crop input image tensor into smaller patches.

    Args:
        img (torch.Tensor): Input tensor representing an image with shape (C, H, W).
        cropping_size_h (int): Height of the cropped patches (default is 256).
        cropping_size_w (int): Width of the cropped patches (default is 256).
        stride (int): Stride for sliding the cropping window (default is 256).
        device (str): Device on which to perform the cropping (default is 'cpu').

    Returns:
        torch.Tensor: Tensor containing cropped patches with shape (P, C, cropping_size_h, cropping_size_w),
                     where P is the number of patches.
    """
    # Get the height and width of the input image
    img_h = img.shape[1]
    img_w = img.shape[2]

    # Initialize an empty tensor to store the cropped patches
    crops = torch.tensor([]).to(device)

    # Initialize starting and ending indices for height
    start_h = 0
    end_h = cropping_size_h

    # Iterate over height with a sliding window
    while end_h <= img_h:
        # Initialize starting and ending indices for width
        start_w = 0
        end_w = cropping_size_w

        # Iterate over width with a sliding window
        while end_w <= img_w:
            # Crop the image
            crop = img[:, start_h:end_h, start_w:end_w]

            # Concatenate the crop to the crops tensor
            crops = torch.cat((crops, crop.unsqueeze(0)))

            # Update the width indices
            start_w += stride
            end_w += stride

        # Update the height indices
        start_h += stride
        end_h += stride

    return crops


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
            
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
       