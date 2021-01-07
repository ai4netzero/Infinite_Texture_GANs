import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torchvision
import xml.etree.ElementTree as ET
from tqdm import tqdm_notebook as tqdm
import time
import argparse
from collections import OrderedDict
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import cv2
import matplotlib.animation as animation
from IPython.display import HTML
import sys
import config_file
from models import generators,discriminators

config = {}

config.update(config_file.config)



def prepare_parser():
    parser = argparse.ArgumentParser()
                  
    # data settings
    parser.add_argument('--data_path', type=str, default=None
                        ,help = 'path to the training set if None the default path will be used')        
    parser.add_argument('--data', type=str, default='channels'
                       ,help = 'type of data {channels,propchannels(channels with 3 diffiernt proportions)'
                        'orchannel (channels with 3 diffiernt orientation)'
                        'rgbchannels (channels with multiple facies')
                        
    # models settings
    parser.add_argument('--G_model', type=str, default='residual_GAN'
                        ,help = 'Generator Model can be residual_GAN, cnn_GAN, ...')
    parser.add_argument('--D_model', type=str, default='residual_GAN'
                        ,help = 'Discriminator Model can be residual_GAN, cnn_GAN, ...')
    parser.add_argument('--cgan',action='store_true',default=False
                        ,help = 'Use conditional GAN if True')
    parser.add_argument('--att',action='store_true',default=False
                        ,help = 'Use Attention if True')
    parser.add_argument('--img_ch', type=int, default=3
                        ,help = 'the number of image channels 1 for grayscale 3 for RGB')
    parser.add_argument('--G_ch', type=int, default=52
                        ,help = 'base multiplier for the Generator (for cnn_GAN should be large 512/1024) , (for ')
    parser.add_argument('--D_ch', type=int, default=32
                        ,help = 'base multiplier for the discriminator')
    parser.add_argument('--leak_G', type=float, default=0
                        ,help = 'use leaky relu activation for generator with leak= leak_G,zero value will use RELU')
    parser.add_argument('--leak_D', type=float, default=0
                        ,help = 'use leaky relu activation for discriminator with leak= leak_G,zero value will use RELU')
    parser.add_argument('--zdim', type=int, default=128
                        ,help ='dimenstion of latent vector')
    parser.add_argument('--spec_norm_D', default=True,action='store_false'
                       ,help = 'apply spectral normalization in discriminator')
    parser.add_argument('--spec_norm_G', default=False,action='store_true'
                       ,help = 'apply spectral normalization in generator')
    #parser.add_argument('--Patch_GAN', default=False,action='store_true'
    #                   ,help = 'Use Patch based discriminator')
    #parser.add_argument('--Double_D', default=False,action='store_true'
    #                   ,help = 'Use double discriminators')
    parser.add_argument('--y_layers', type=int, default=1
                       ,help = 'number condition mlp layers before concatentation ')

    parser.add_argument('--cbn_fixed_Var', default=False,action='store_true'
                       ,help = 'apply cbn with fixed variance in the generator')
    # Double discriminators parameters

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
    parser.add_argument('--G_batch_size', type=int, default=None
                        ,help = 'generator batch size if None it will be set to batch_size')
    
    #training settings
    parser.add_argument('--loss', type=str, default='standard'
                        ,help = 'Loss function can be standard,hinge or wgan')
    parser.add_argument('--separate_loss', default=False,action='store_true'
                        ,help = 'apply loss function seprately on D outpouts')
    parser.add_argument('--disc_iters', type=int, default=1
                        ,help = ' no. discriminator updates per one generator update')
    parser.add_argument('--epochs', type=int, default=1
                        ,help ='no. of epochs')
    parser.add_argument('--limit', type=float, default=None
                        ,help = 'if not None will limit training to --limit seconds')
    parser.add_argument('--save_rate', type=int, default=30
                        ,help = 'save checkpoints every 30 epcohs')
    parser.add_argument('--ema',action='store_true' , default=False
                        ,help = 'keep EMA of G weights')
    parser.add_argument('--ema_decay',type = float, default=0.999
                        ,help = 'EMA decay rate')
    parser.add_argument('--decay_lr',type=str,default=None
                        ,help = 'if not None decay the learning rates (exp,step)')
    parser.add_argument('--saved_cp', type=str, default=None
                        ,help='if not None start training from a loaded cp with path --saved_cp') 
    parser.add_argument('--seed', type=int, default=None
                       ,help = 'None to use random seed can be fixed for reporoduction')
    parser.add_argument('--z_dist', type=str, default='normal'
                       , help = ' distribution of latent space normal/uniform')
    parser.add_argument('--smooth',default=True,action='store_false'
                       , help = 'Use smooth labeling if True')
     
    parser.add_argument('--print_acc_t',action='store_true' , default=False
                       , help = 'print the area of conn. comp. in the channels')
    parser.add_argument('--acc_cond_list', nargs='+', default=None,type=float
                        ,help='List of conditions values for acc_t computation')          
    # GPU settings
    parser.add_argument('--ngpu', type=int, default=1
                        ,help = 'number of gpus to be used')                                     
    parser.add_argument('--dev_num', type=int, default=0
                        ,help = 'the index of a gpu to be used if --ngpu is 1 ')
                        
    # folder name             
    parser.add_argument('--fname', type=str, default='models_cp',help='folder name to save cp')
                        
    # Conditional GAN settings
    parser.add_argument('--D_cond_method', type=str, default='concat'
                        ,help='conditiong method concat for concatentation/proj for projection')
    parser.add_argument('--G_cond_method', type=str, default='cbn'
                        ,help='conditiong method concat for concatentation/cbn for conditional batch normalization')
    parser.add_argument('--n_cl', type=int, default=0
                       ,help = 'number of classes, 1 for continious conditioning')
    parser.add_argument('--n_cl1', type=int, default=0
                       ,help = 'number of classes in the second condition if there is')
    parser.add_argument('--real_cond_list', nargs='+', default=None,type=float
                        ,help='List of conditions values for the real samples e.g. 0.25 0.30 0.35 if not provided it will be  0 1 2 ..')   
    parser.add_argument('--discrete',action='store_true' , default=False
                       ,help = 'if True Sample only discrete labels from min_label to max_label')  
    parser.add_argument('--c_list', nargs='+', default=None,type=float
                        ,help='list of conditions to be sampled,if provided')                                                                                   
    parser.add_argument('--min_label', type=float, default=0
                        ,help='minimum label for conditions')
    parser.add_argument('--max_label', type=float, default=None
                        ,help='maximum label for conditions, if None will be set num of classes')
    parser.add_argument('--ohe',action='store_true' , default=False
                       ,help = 'use one hot encoding for conditioning')
    parser.add_argument('--img_cond',action='store_true' , default=False
                       ,help = 'use images for conditioning')
    parser.add_argument('--SN_y', action='store_true' , default=False
                        ,help='apply SN to the condition linear layer')
    #parser.add_argument('--min_v', type=int, default=0
    #                    ,help='minimum index for ohe conditions')
    #parser.add_argument('--max_v', type=int, default=None
    #                    ,help='maximum index for ohe conditions, if None it will be set to the number of classes')
    return parser


# these arguments apply only when running sample.py file to generate images.                        
def add_sample_parser(parser):
                        
    # paths                    
    parser.add_argument('--G_cp', type=str, default=None
                        ,help='path of generator checkpoint .pth file ')
    parser.add_argument('--out_path', type=str, default='out'
                       ,help = 'path to save images')
    parser.add_argument('--many', type=str, default=None
                        ,help='dir of the folder that contains .pth files to generate from multiple checkpoints')                 
    parser.add_argument('--truncated', type=float, default=0
                        ,help = 'if greater than 0 it will apply a truncation to normal dist. with --truncated value')
    parser.add_argument('--num_imgs', type=int, default=1000
                       , help = 'number of images to be generated')
    parser.add_argument('--img_name', type=str, default=''
                       ,help = 'append a string to the generate images numbers')
                                          
    # Genertared images configuration                    
    parser.add_argument('--figure', type=str, default='grid'
                        ,help='grid to save a grid of generated images or images to save them in --out_path')
    parser.add_argument('--grid_rows', type=int, default=3
                        ,help='num of rows in the grid')
    parser.add_argument('--gray2rgb', default=True,action='store_false'
                        ,help='If True save single-channel images as 3 channels')
    return parser



def prepare_device(args):
    # Device
    ngpu = args.ngpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.dev_num)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Device: ",device)
    return device

def prepare_seed(args):
    #Seeds
    if args.seed is None:
        seed = random.randint(1, 10000) # use if you want new results
    else : 
        seed = args.seed #random.randint(1, 10000) # use if you want new results

    print("Random Seed: ", seed)
    return seed
   
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#dataset
def prepare_data(args):
    print(" laoding " +args.data +" ...")
    if args.data == 'dogs':
        from datasets import Dogs_labels
        train_data = Dogs_labels.DogsDataset()
        dataloader = torch.utils.data.DataLoader(train_data,
                               shuffle=True, batch_size=args.batch_size,
                               num_workers=12,pin_memory=True)

    elif  args.data == 'cifar':
        train_data = dset.CIFAR10(root='./data', train=True, download=True,
                                   transform=stransforms.Compose([
                                       # transforms.Resize(image_size),
                                       # transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=2)

    elif args.data == 'channels':
        from datasets import Channels
        train_data = Channels.Channels(path = args.data_path)
        dataloader = torch.utils.data.DataLoader(train_data,
                               shuffle=True, batch_size=args.batch_size,
                               num_workers=4,pin_memory=True)
    elif args.data == 'rgbchannels':
        from datasets import RGBChannels
        train_data = RGBChannels.RGBChannels(path = args.data_path)
        dataloader = torch.utils.data.DataLoader(train_data,
                               shuffle=True, batch_size=args.batch_size,
                               num_workers=4,pin_memory=True)

    elif args.data == 'propchannels':
        from datasets import propchannels
        train_data = propchannels.propchannels(path = args.data_path)
        dataloader = torch.utils.data.DataLoader(train_data,
                               shuffle=True, batch_size=args.batch_size,
                               num_workers=16,pin_memory=True)

    elif args.data == 'orchannels':
        from datasets import orchannels
        train_data = orchannels.orchannels(path = args.data_path)
        dataloader = torch.utils.data.DataLoader(train_data,
                               shuffle=True, batch_size=args.batch_size,
                               num_workers=16,pin_memory=True)
    else:
        print('no data named :',args.data)
        exit()
    print("Finished data loading")    
    return dataloader,train_data


            
def prepare_models(args,n_cl = 0,device = 'cpu',only_G = False):           
    #model
    if args.G_model == 'cnn_GAN':
        netG = generators.CNN_Generator(args.zdim,img_ch=args.img_ch,base_ch= args.G_ch).to(device)
        netG.apply(weihts_init)

    elif args.G_model == 'residual_GAN':
        netG = generators.Res_Generator(args.zdim,img_ch=args.img_ch,n_classes = n_cl
                                        ,ch = args.G_ch,leak = args.leak_G,att = args.att
                                        ,SN = args.spec_norm_G
                                        ,cond_method = args.G_cond_method
                                        ,cbn_fixed_Var = args.cbn_fixed_Var).to(device)

    elif args.G_model == 'SPADE_GAN':
        netG = generators.SPADE_Generator(args.zdim,img_ch=args.img_ch,n_classes = n_cl
                                        ,ch = args.G_ch,leak = args.leak_G,att = args.att
                                        ,SN = args.spec_norm_G).to(device)
    if only_G:
        return netG

    if args.D_model == 'cnn_GAN':
        netD = discriminators.CNN_Discriminator(img_ch=args.img_ch,base_ch= args.D_ch,spectral_norm = args.spec_norm_D
                                     ,leak = args.leak_D).to(device)  
        netD.apply(weights_init)                    

    elif args.D_model == 'residual_GAN':
        netD = discriminators.Res_Discriminator(img_ch=args.img_ch,n_classes = n_cl,ch = args.D_ch
                                    ,leak = args.leak_D,att = args.att
                                    ,cond_method = args.D_cond_method
                                    ,SN = args.spec_norm_D
                                    ,y_layers = args.y_layers
                                    ,SN_y = args.SN_y).to(device)   

    elif  args.D_model == 'cnn_PatchGAN':
        netD = discriminators.Patch_CNN_Discriminator(img_ch = args.img_ch,ch =args.D_ch,n_classes = n_cl).to(device)                                            

    elif  args.D_model == 'Res_PatchGAN':
        netD = discriminators.Res_Discriminator(img_ch=args.img_ch,n_classes = n_cl,ch = args.D_ch
                                    ,leak = args.leak_D,att = args.att
                                    ,cond_method = args.D_cond_method
                                    ,SN = args.spec_norm_D
                                    ,y_layers = args.y_layers
                                    ,SN_y = args.SN_y
                                    ,patch = True).to(device)   

    elif args.D_model == 'Double_D':
        netD = discriminators.Double_Discriminator(config,img_ch=args.img_ch,n_classes = n_cl
                                    ,leak = args.leak_D
                                    ,cond_method = args.D_cond_method
                                    ,SN = args.spec_norm_D
                                    ,SN_y = args.SN_y
                                    ,separate_loss = args.separate_loss).to(device)    

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

    
#generate random labels
def sample_pseudo_labels(args,num_classes,batch_size,device):
    #returns labels used in D and G respectively.
    if args.max_label is None:
        max_value = num_classes
    else:
        max_value = args.max_label     
    #if num_classes > 1:

    # list of  conditions
    if args.c_list is not None:
        c = torch.tensor(args.c_list)
        y_ind =  torch.randint(low=0, high=len(c), size=(batch_size,1))
        y = c[y_ind].to(device)
        return y,y

    # discrete conditions
    if args.discrete:
        y =  torch.randint(low=int(args.min_label), high=int(max_value), size=(batch_size,1)).to(device)
        if args.ohe:
            y_ohe = torch.eye(num_classes)[y].to(device)
            return y_ohe,y_ohe
        else:
            y = y.type(torch.long)
            return y,y

    else: # continious conditions
        y = (args.max_label - args.min_label) * torch.rand(batch_size,1) + args.min_label
        if args.ohe:
            y = cont_2_ohe(y,num_classes)
        y = y.to(device)
        return y,y


def rotate_cv(image, angle, scale = 1.0):
    (h, w) = image.shape[:2]

    center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def prepeare_img_condition():
    v = torch.zeros(64,64)
    v[:,8:12] = 1
    v[:,23:27] = 1
    v[:,39:43] = 1
    v[:,55:59] = 1
    h = torch.zeros(64,64)
    h[8:12,:] = 1
    h[23:27,:] = 1
    h[39:43,:] = 1
    h[55:59,:] = 1
    ##h = rotate_cv(v.numpy(),-90,1.1)
    #h = np.round(h)
    #h = torch.tensor(h)
    i = rotate_cv(v.numpy(),-45,1.3)
    i = np.round(i)
    i = torch.tensor(i)
    return torch.cat((v.unsqueeze(0),i.unsqueeze(0),h.unsqueeze(0)),0)


def disc_2_img(y,device):
    angles = prepeare_img_condition().to(device)
    y_angles = torch.empty(y.size(0),1,64,64).to(device)
    y_angles[y==0] = angles[0].unsqueeze(0)
    y_angles[y==1] = angles[1].unsqueeze(0)
    y_angles[y==2] = angles[2].unsqueeze(0)
    y_angles[y_angles==0] = -1
    return y_angles

def cont_2_img_hv(y,device,pre =1,new = 1.3):
    v = torch.zeros(64,64)
    v[:,8:12] = 1
    v[:,23:27] = 1
    v[:,39:43] = 1
    v[:,55:59] = 1
    h = torch.zeros(64,64)
    h[8:12,:] = 1
    h[23:27,:] = 1
    h[39:43,:] = 1
    h[55:59,:] = 1
    y_angles = torch.empty(y.size(0),1,64,64).to(device)
    for i,angle in enumerate(y):
        angle = angle.item()
        if angle>=0 and angle<=45:
            o = rotate_cv(v.numpy(),-angle,((pre-new)/(-45))*angle + pre)
        elif  angle>45 and angle<=90:
            o = rotate_cv(h.numpy(),90-angle,((pre-new)/(45))*(angle-90) + pre)
        o = np.round(o)
        o = torch.tensor(o).to(device)
        y_angles[i] = o.unsqueeze(0)
    y_angles[y_angles==0] = -1
    return y_angles


def cont_2_ohe(y,num_classes,device='cpu'):
    y_c = torch.ceil(y)
    y_f = torch.floor(y)
    y_ohe_c = torch.eye(num_classes)[y_c.long()]
    y_ohe_f = torch.eye(num_classes)[y_f.long()]
    alpha = y-y_f
    y_ohe = y_ohe_c * alpha[:, None]+y_ohe_f * (1-alpha[:, None])
    return y_ohe.to(device)


def disc_2_ohe(y,num_classes,device):
    y_ohe = torch.eye(num_classes)[y].to(device)
    return y_ohe

def disc_2_cont(y,c_list,device):
    for i,v in enumerate(c_list):
        y[y==i] = v
    y = y.unsqueeze(1)
    return y    

def adaptive_adv_labels(args,y_vector,device,sigma = 0.05,pre_value = 1,new_value = 0.85):
    max_label = args.max_label
    min_label = args.min_label
    mid_label = (max_label+min_label)/2
    mid_min = (min_label+mid_label)/2
    mid_max = (max_label+mid_label)/2
    b_size = y_vector.size(0)
    adv_labels = torch.full((b_size, 1), pre_value, device=device)
    for i,v in enumerate(y_vector):
        if v >= min_label and v < mid_min:
            adv_labels[i] = ((pre_value-new_value)/(min_label-mid_min))*(v-min_label) +pre_value
        elif v >=  mid_min and v < mid_label:
            adv_labels[i] = ((pre_value-new_value)/(mid_label-mid_min))*(v-mid_label) +pre_value
        elif v >= mid_label and v < mid_max:
            adv_labels[i] = ((pre_value-new_value)/(mid_label-mid_max))*(v-mid_label) +pre_value
        elif v >= mid_max and v <= max_label:
            adv_labels[i] = ((pre_value-new_value)/(max_label-mid_max))*(v-max_label) +pre_value
    return adv_labelss
    
#generate fake images
def sample_from_gen(args,b_size, zdim, num_classes,netG,device ='cpu',truncated = 0): 
    # latent z
    if args.z_dist == 'normal': 
        z = torch.randn(b_size, zdim).to(device=device)
    elif args.z_dist =='uniform':
        z =2*torch.rand(b_size, zdim).to(device=device) -1
        
    if truncated > 0:
        z = get_trun_noise(truncated,zdim,b_size,device)
    #labels
    if num_classes>0:
        y_D,y_G = sample_pseudo_labels(args,num_classes,b_size,device)
        if args.img_cond:
            #print(y_D)
            if args.discrete:
                y_D = disc_2_img(y_D,device) # 0 to img with 0 deg,1 to img with 45 degs,..
            else:
                y_D = cont_2_img_hv(y_D,device) # img with y_D degs   
            #save_image(y_D[0],'1.png')
            #save_image(y_D[1],'2.png')         
            #exit()
            y_G = y_D
            #exit()
    else:
        y_D,y_G = None,None   
    fake = netG(z, y_G)
    
    return fake, y_D

def load_netG(netG,checkpointname = None):
    checkpoint = torch.load(checkpointname)
    state_dict_G = checkpoint['netG_state_dict']    
    new_state_dict_G = OrderedDict()
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

def save_images(args,netG,device,out_path):
    n_images= args.num_imgs
    truncated=args.truncated

    if not os.path.exists(out_path):
        os.mkdir(out_path)
        
    im_batch_size = 50
    
    if n_images<im_batch_size:
        im_batch_size = n_images
        
    n_batches = n_images//im_batch_size
        
    for i_batch in range(0, n_images, im_batch_size):
        if i_batch ==  n_batches*im_batch_size:
            im_batch_size = n_images - i_batch
            
        gen_images,_ = sample_from_gen(args,im_batch_size,args.zdim,args.n_cl,netG,device,truncated = args.truncated)
        gen_images = gen_images.cpu().detach()
        #shape=(*,ch=3,h,w), torch.Tensor
        
        #denormalize
        gen_images = gen_images*0.5 + 0.5
        
        if args.gray2rgb:
            for i_image in range(gen_images.size(0)):
                save_image(gen_images[i_image, :, :, :],
                           os.path.join(out_path,args.img_name+ f'image_{i_batch+i_image:05d}.png'))
        else:
            for i_image in range(gen_images.size(0)):
                x_pil = transforms.ToPILImage()(gen_images[i_image, :, :, :])
                x_pil.save(os.path.join(out_path,args.img_name+ f'image_{i_batch+i_image:05d}.png'))

    #shutil.make_archive(f'images', 'zip', out_path)
    

def save_grid(args,netG,device,nrows=3,ncol=3,out_path = 'plot'):
    b_size = nrows*ncol
    gen_images,_ = sample_from_gen(args,b_size,args.zdim,args.n_cl,netG,device,truncated = args.truncated)
    gen_images = gen_images.cpu().detach()
                        
    if args.img_ch == 1:
        gen_images = gen_images.squeeze()
        plt.close('all')
        fig,axes = plt.subplots(nrows,ncol,figsize=[10,10])
        #fake = fake.permute((0,3, 2, 1))
        for i,iax in enumerate( axes.flatten() ):
            iax.imshow(gen_images[i,:,:])
            iax.set_xticks([])
            iax.set_yticks([])
        fig.show()
    else:
        plt.close('all')
        fig = plt.figure(figsize=(10,10))
        l = vutils.make_grid(gen_images,nrow=nrows, padding=2, normalize=True).numpy()
        plt.imshow(np.transpose(l,(1,2,0)))
        plt.xticks([])
        plt.yticks([])
    fig.savefig(out_path+'.png')

def acc_t(netG,args,t_a=100,N=200,y_list =None ,device = 'cpu'):
    if y_list is None:
        y_list = args.acc_cond_list 
    netG.eval()
    z = torch.randn(N, 128).to(device=device,non_blocking=True)
    acc_t_y = []
    for i,x in enumerate(y_list):
        y = (x - x) * torch.rand(N) + x
        y =  y.unsqueeze(1)
        #y = cont_2_img_hv(y,device,new=1.3)
        with torch.no_grad():
            imgs = netG(z,y.to(device=device)).cpu().squeeze(1)
            AC= []
            for k in range(N):
                img = imgs[k].numpy()
                AC_img = area_of_components(img)
                AC.extend(AC_img)
        if len(AC)==0:
            print(AC_img)
            acc_t_y.append(None)
        else:         
            acc_t = sum(i <t_a for i in AC)/len(AC)*100
            acc_t_y.append(acc_t)

    return acc_t_y

def label_components(img,face = 1):
    c_l = 0
    l = np.zeros_like(img)
    q = []
    for i in range(np.size(img,0)):
        for j in range(np.size(img,1)):
            p = img[i,j]
            if p ==face and l[i,j] == 0:
                c_l += 1
                l[i,j] = c_l
                q.append((i,j))
            while len(q) != 0:
                #print(q)    
                #return q
                e = q.pop()
                n = get_distant_pixels(e[0],e[1],1,high = np.size(img,1)-1)
                #print(n)
                #return n
                for pix in n:
                    if img[pix[0],pix[1]] ==face and l[pix[0],pix[1]] ==0:
                        l[pix[0],pix[1]] = c_l
                        q.append((pix[0],pix[1]))
                        
                #return q
    return l

def get_distant_pixels(i,j,d,low = 0,high = 63):
    l = set()
    for dis in range(-d,d+1):
        if i+dis>=low and i+dis<=high and j+d>=low and j+d<=high:
            l.add((i+dis,j+d))
    for dis in range(-d,d+1):
        if i+dis>=low and i+dis<=high and j-d>=low and j-d<=high:
            l.add((i+dis,j-d))
    for dis in range(-d,d+1):
        if i+d>=low and i+d<=high and j+dis>=low and j+dis<=high:
            l.add((i+d,j+dis))
    for dis in range(-d,d+1):
        if i-d>=low and i-d<=high and j+dis>=low and j+dis<=high:
            l.add((i-d,j+dis))
    return l

def area_of_components(img,face = 1):
    l_c = label_components(img,face = face)
    no_comp = len(np.unique(l_c))-1
    areas = []
    for comp in range(1,no_comp+1):
        areas.append(np.count_nonzero(l_c == comp))
    return areas

def elapsed_time(start_time):
    return time.time() - start_time

def calc_ralsloss_G(real, fake, margin=1.0):
    loss_real = torch.mean((real - fake.mean() + margin) ** 2)
    loss_fake = torch.mean((fake - real.mean() - margin) ** 2)
    loss = (loss_real + loss_fake)
    
    return loss
