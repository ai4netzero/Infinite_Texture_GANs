import torch.nn.functional as F
import time
import argparse
from collections import OrderedDict
import os
import random
import torch.nn as nn
import torch.utils.data
import numpy as np

from models.generators import ResidualPatchGenerator
from models.discriminators import Patch_Discriminator





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
    parser.add_argument('--sampling', type=int, default=None
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
                       ,help = 'type normalization used in G either bn or SSM')

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
    parser.add_argument('--outer_padding', type=str, default='constant'
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
        netD = Patch_Discriminator(img_ch=args.img_ch,base_ch = args.D_ch,n_layers_D=args.n_layers_D,kw = 4
                                    ,SN= args.spec_norm_D,norm_layer = args.norm_layer_D).to(device)
    return netG,netD

def load_from_saved(args,netG,netD,optimizerG,optimizerD):
    checkpoint = torch.load(args.saved_cp)
    #load G
    state_dict_G = checkpoint['netG_state_dict']
    new_state_dict_G = OrderedDict()
    for k, v in state_dict_G.items():
        if 'module' in k:
            k = k.replace('module.','')
        new_state_dict_G[k] = v
    netG.load_state_dict(new_state_dict_G)
    
    #Load D
    state_dict_D = checkpoint['netD_state_dict']
    new_state_dict_D = OrderedDict()
    for k, v in state_dict_D.items():
        if 'module' in k:
            k = k.replace('module.','')
        new_state_dict_D[k] = v
    netD.load_state_dict(new_state_dict_D)
    #load optimizer
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

    st_epoch = checkpoint['epoch']+1
    G_losses = checkpoint['Gloss']
    D_losses = checkpoint['Dloss']
    #args = checkpoint['args']
    return netG,netD,optimizerG,optimizerD,st_epoch,G_losses,D_losses
    
def prepare_filename(args):
    filename = str(args.epochs) + "_"

    if args.fname is not None:
        if not os.path.exists(args.fname):
            os.makedirs(args.fname)
        filename = args.fname+"/" + filename
    return filename


def sample_from_gen_PatchByPatch(netG,z_dim=128,base_res=4,map_dim = 4,num_images=1, num_patches_height=3, num_patches_width=3,device ='cpu'): 
        """  
            Generate images using the generator network netG in a patch-by-patch fashion.

            Args:
                generator (torch.nn.Module): The generator network used for image synthesis.
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
            
        num_patches_per_image = num_patches_height*num_patches_width
        generator_batch_size = num_patches_per_image*num_images
            
        #Build the spatial latent input z 
        z = torch.randn(generator_batch_size,z_dim,base_res,base_res).to(device)
    
        #Build the second input M for stochastic spatial modulation
        if type_norm == 'SSM':
            maps_per_layers = []
            pad_size = 4
            for i in range(0,n_layers_G):
                res_layer = (2**i)*base_res
                res_with_pad = res_layer+pad_size
                
                # Build the spatial map input with a pad of size 4, since we use two 3x3 conv layers, for num_images N : (N,mdim ,Tot_layer_res_h +2, Tot_layer_res_w+2)
                MAPS =  torch.randn(num_images,map_dim,num_patches_height*res_layer+pad_size,num_patches_width*res_layer+pad_size).to(device)
                
                # Crop the MAPS input into smaller overlapping patches of size res_layer + 4 with an overlap size of 4:  (N*num_patches_per_img,mdim ,res_layer +4, res_layer+4)
                maps = crop_images(MAPS,res_with_pad,res_with_pad,res_layer,device = device)
                
                maps_per_layers.append((maps))
        else:
            maps_per_layers = [None]*n_layers_G
        
        # During training set the input padding variable to None for all layers in the generator
        if netG.training:
            padding_variable_in = [None]*(n_layers_G+2)

        fake_images_patches,_ = netG(z, maps_per_layers,padding_variable_in,padding_location = None)
        
        fake_images = merge_patches_into_image(fake_images_patches,num_patches_height,num_patches_width,device)
        
        return fake_images

def scale_1D(args,netG,n_imgs = 1,h=None,w=None,device ='cpu'): 
    
    if h is None or w is None:
        h = args.num_patches_h
        w = args.num_patches_w
        
    #num_patches_per_img = h*w
    #n_imgs = b_size//num_patches_per_img
    #b_size =n_imgs*num_patches_per_img
    #if args.z_dist == 'normal': 
    
    pad_size_z = 2
    z_merged =  torch.randn(n_imgs,args.z_dim,h*args.base_res+pad_size_z,w*args.base_res+pad_size_z).to(device)
    res_withpadd_h = args.num_patches_h*args.base_res+pad_size_z
    res_withpadd_w = args.num_patches_w*args.base_res+pad_size_z
    z_local = crop_images(z_merged,res_withpadd_h,res_withpadd_w,(args.num_patches_w-1)*args.base_res,device = device)
        

    #print(z_local.shape)
    maps_local_per_res = []
    #pad_sizes = [4,4,4,4,4,4]    
    pad_size_maps = 4
    for i in range(0,args.n_layers_G):
    
        res1 = (2**i)*args.base_res
        #res1,res2 = resols[i]
        #pad_size = pad_sizes[i]
        maps_merged =  torch.randn(n_imgs,args.n_cl,h*res1+pad_size_maps,w*res1+pad_size_maps).to(device)
        res_withpadd_h = args.num_patches_h*res1+pad_size_maps
        res_withpadd_w = args.num_patches_w*res1+pad_size_maps
        maps_local = crop_images(maps_merged,res_withpadd_h,res_withpadd_w,(args.num_patches_w-1)*res1,device = device)
        maps_local_per_res.append(maps_local)
    
    gen_res = (2**(args.n_layers_G-1))*args.base_res
    #print(maps_local.shape)

    padding_variable = None
    for j in range(maps_local.size(0)): # num of iteration through netG
        res_withpadd = args.base_res + pad_size_z
        z = crop_images(z_local[[j]],res_withpadd,res_withpadd,args.base_res,device = device)
        maps_per_res = []
        #print(z.shape)
        for i in range(0,args.n_layers_G):
            res1 = (2**i)*args.base_res
            res_withpadd = res1 + pad_size_maps
            maps = crop_images(maps_local_per_res[i][[j]],res_withpadd,res_withpadd,res1,device = device)
            maps_per_res.append(maps)
            #print(maps.shape)
        y_G = maps_per_res 
        with torch.no_grad():
            fake,padding_variable = netG(z, y_G,args.num_patches_h,args.num_patches_w,padding_variable)
            #print(fake.shape)
        fake = fake.cpu() # (9,_,_,_)
        
        img_merged = merge_patches_into_image(fake,args.num_patches_h,args.num_patches_w,'cpu') # (1,_,3*,3*)
        #torch.save(img_merged, str(j)+'img.pt')
        #print(img_merged.shape)

        if j != maps_local.size(0)-1: # last patch
            img_merged = img_merged[:,:,:,0:gen_res*2]
        
        #print(img_merged.shape)

        if j ==0:
            full_img = img_merged
        else:
            full_img = torch.cat((full_img,img_merged),-1)

    return full_img

def scale_2D(args,netG,n_imgs = 1,h=None,w=None,device ='cpu'): 
    
    if h is None or w is None:
        h = args.num_patches_h
        w = args.num_patches_w
        
    #steps_h = int(np.ceil(h/args.num_patches_h))
    #steps_w = int(np.ceil(w/args.num_patches_w))
    
    # (h-1) should be mutiple of (num_patches_h-1)
    steps_h = int((h-1)/(args.num_patches_h-1))
    steps_w = int((w-1)/(args.num_patches_w-1))

    # generate z
    pad_size_z = 2
    z_merged =  torch.randn(n_imgs,args.z_dim,h*args.base_res+pad_size_z,w*args.base_res+pad_size_z).to(device)
    res_withpadd_h = args.num_patches_h*args.base_res+pad_size_z
    res_withpadd_w = args.num_patches_w*args.base_res+pad_size_z
    z_local = crop_images(z_merged,res_withpadd_h,res_withpadd_w,(args.num_patches_w-1)*args.base_res,device = device)
        


    # generate maps
    maps_local_per_res = []
    pad_size_maps = 4
    for i in range(0,args.n_layers_G):
    
        res1 = (2**i)*args.base_res
        maps_merged =  torch.randn(n_imgs,args.n_cl,h*res1+pad_size_maps,w*res1+pad_size_maps).to(device)
        res_withpadd_h = args.num_patches_h*res1+pad_size_maps
        res_withpadd_w = args.num_patches_w*res1+pad_size_maps
        maps_local = crop_images(maps_merged,res_withpadd_h,res_withpadd_w,(args.num_patches_w-1)*res1,device = device)
        maps_local_per_res.append(maps_local)
    
    gen_res = (2**(args.n_layers_G-1))*args.base_res
    #print(maps_local.shape)

    #padding_variable_h,padding_variable_v = None
    #for j in range(maps_local.size(0)): # num of iteration through netG
    
    n = 0
    padding_variable_h_in = None
    padding_variable_h_out_row = None
    padding_variable_h_in_row = None
    full_img_row =None
    for s_i in range(steps_h):
        
        padding_variable_v_in = None
        padding_variable_h_out_conv_all =[]
        # crop the padding_variable_h_out_row to form padding_variable_h_in
        if padding_variable_h_out_row is not None:
            for ind_layer,layer_out in enumerate(padding_variable_h_out_row):
                conv_list = []
                for ind_conv,conv_out in enumerate(layer_out):
                    # replicate padding             
                    padding_variable_h_out_row[ind_layer][ind_conv] = F.pad(padding_variable_h_out_row[ind_layer][ind_conv], (1,1,0,0), "replicate")
                    #padding_variable_v_out_row[ind_layer][ind_conv] = padding_variable_v_out_row[ind_layer][ind_conv].to(device)
                    i = min(ind_layer,args.n_layers_G-1)
                    res = (2**i)*args.base_res

                    res_withpadd_w = args.num_patches_w*res +2
                    res_withpadd_h = 1
                    #print(res_withpadd_w)
                    #print(padding_variable_h_out_row[ind_layer][ind_conv].shape)
                    padding_variable_h_out_conv = crop_images(padding_variable_h_out_row[ind_layer][ind_conv],
                                                                   res_withpadd_h,res_withpadd_w,(args.num_patches_w-1)*res,device = 'cpu')
                    #print(padding_variable_h_out_conv.shape)
                    #for instance in padding_variable_v_out_conv:
                    conv_list.append(padding_variable_h_out_conv)
                padding_variable_h_out_conv_all.append(conv_list)
        
        
            #print(padding_variable_h_out_conv.shape)            
            N_patches = padding_variable_h_out_conv.shape[0]
            #padding_variable_h_out_conv_all is a list of N_blocks = 7 each has N_patches instances of the same size
            N_blocks = len(padding_variable_h_out_conv_all) 
            #print(len(padding_variable_h_out_conv_all[0]))
            #print(N_patches,N_blocks)
            padding_variable_h_in_row = [] # list of padding_variable_h_in for every iteration in steps_w
            for j in range(N_patches):
                L1 = []
                for i in range(N_blocks):
                    conv_list = []
                    #print(len(padding_variable_h_out_conv_all[i]))
                    for l_i in padding_variable_h_out_conv_all[i]:
                        conv_list.append(l_i[[j]])
                    L1.append(conv_list)
                padding_variable_h_in_row.append(L1)

            #print(len(padding_variable_h_in_row[0][0]))
            #print(padding_variable_h_in_row[0][0].shape)
            
        for s_j in range(steps_w):
            
            
            # Get z
            res_withpadd = args.base_res + pad_size_z
            z = crop_images(z_local[[n]],res_withpadd,res_withpadd,args.base_res,device = device)
            
            
            # Get map
            maps_per_res = []
            for i in range(0,args.n_layers_G):
                res1 = (2**i)*args.base_res
                res_withpadd = res1 + pad_size_maps
                maps = crop_images(maps_local_per_res[i][[n]],res_withpadd,res_withpadd,res1,device = device)
                maps_per_res.append(maps)
            y_G = maps_per_res 
            
            # check if the patch is the last one in the row
            last = True if s_j == steps_w-1 else False
            
            if padding_variable_h_in_row is not None:
                #print(s_i,s_j)
                padding_variable_h_in = padding_variable_h_in_row[s_j]
                
                for ind_layer,layer_out in enumerate(padding_variable_h_in):
                    for ind_conv,conv_out in enumerate(layer_out):
                        padding_variable_h_in[ind_layer][ind_conv] = padding_variable_h_in[ind_layer][ind_conv].to(device)
                    
            
            with torch.no_grad():
                fake,padding_variable_v_out,padding_variable_h_out = netG(z, y_G,args.num_patches_h,args.num_patches_w
                                                                    ,padding_variable_h= padding_variable_h_in,padding_variable_v= padding_variable_v_in
                                                                    ,last = last)
            
            # for the next iteration padding_variable_v_in = padding_variable_v_out
            padding_variable_v_in = padding_variable_v_out
            #print(len(padding_variable_v_in[0][0].shape))
            
            #padding_variable_h_out = padding_variable_h_out.cpu()
            #torch.cuda.empty_cache()
            
            for ind_layer,layer_out in enumerate(padding_variable_h_out):
                    for ind_conv,conv_out in enumerate(layer_out):
                        padding_variable_h_out[ind_layer][ind_conv] = padding_variable_h_out[ind_layer][ind_conv].cpu()
                        torch.cuda.empty_cache()

            
            # concatenate the padding_variable_h_out to form padding_variable_h_out_row
            if s_j == 0:
                padding_variable_h_out_row = padding_variable_h_out
            else:
                for ind_layer,layer_out in enumerate(padding_variable_h_out):
                    for ind_conv,conv_out in enumerate(layer_out):
                        #print(padding_variable_h_out_row[ind_layer][ind_conv].shape)
                        #padding_variable_v_out_row[ind_layer][ind_conv] = padding_variable_v_out_row[ind_layer][ind_conv].cpu()
                        padding_variable_h_out_row[ind_layer][ind_conv] = torch.cat((padding_variable_h_out_row[ind_layer][ind_conv]
                                                                                     ,conv_out.cpu()),-1)
                        #print(padding_variable_h_out_row[ind_layer][ind_conv].shape)

                        
            fake = fake.cpu() # (9,_,_,_)
            torch.cuda.empty_cache()
            
            # concatenate the generated patches
            img_merged = merge_patches_into_image(fake,args.num_patches_h,args.num_patches_w,'cpu') # (1,_,3*,3*)
            #torch.save(img_merged, str(j)+'img.pt')
            #print(img_merged.shape)

            # drop the patched to be re-generated
            
            if s_j != steps_w-1 and s_i != steps_h-1: # not last in row or column
                img_merged = img_merged[:,:,0:gen_res*(args.num_patches_h-1),0:gen_res*(args.num_patches_w-1)]
            elif s_i != steps_h-1: # last column not last row
                img_merged = img_merged[:,:,0:gen_res*(args.num_patches_h-1),:]
            elif s_j != steps_w-1: # last row not last column
                img_merged = img_merged[:,:,:,0:gen_res*(args.num_patches_w-1)]
                

            
            if s_j ==0:
                full_img_row = img_merged
            else:
                full_img_row = torch.cat((full_img_row,img_merged),-1)
                
            n = n+1
        
        if s_i == 0:
            full_img = full_img_row
        else:
            full_img = torch.cat((full_img,full_img_row),-2)
            
    del padding_variable_h_in,padding_variable_v_in,padding_variable_v_out,padding_variable_h_out
    
    torch.cuda.empty_cache()

    return full_img


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
       
class _CustomDataParallel(nn.DataParallel):
    def __init__(self,model,gpu_ids):
        super(nn.DataParallel, self).__init__()
        self.model = nn.DataParallel(model,gpu_ids).cuda()
        
        
    def forward(self, *input):
        return self.model(*input)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)
