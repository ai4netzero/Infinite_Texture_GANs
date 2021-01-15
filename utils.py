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
from models import generators,discriminators



def prepare_parser():
    parser = argparse.ArgumentParser()
                  
    # data settings     
    parser.add_argument('--data', type=str, default='channels'
                       ,help = 'type of data')
    parser.add_argument('--data_path', type=str, default='datasets/prop_channels_train/'
                       ,help = 'data path')
    parser.add_argument('--labels_path', type=str, default= None
                       ,help = 'labels path')
    parser.add_argument('--data_ext', type=str, default='txt'
                       ,help = 'data extension txt, png')

    # models settings
    parser.add_argument('--G_model', type=str, default='residual_GAN'
                        ,help = 'Generator Model can be residual_GAN, dcgan, ...')
    parser.add_argument('--D_model', type=str, default='residual_GAN'
                        ,help = 'Discriminator Model can be residual_GAN, dcgan, sngan,...')
    parser.add_argument('--cgan',action='store_true',default=False
                        ,help = 'Use conditional GAN if True (only implmented in residual_GAN)')
    parser.add_argument('--att',action='store_true',default=False
                        ,help = 'Use Attention if True  (only implmented in residual_GAN)')
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
    parser.add_argument('--n_layers_D', type=int, default=3
                       ,help = 'number of layers used in discriminator of dcgan')

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
    parser.add_argument('--real_cond_list', nargs='+', default=None,type=float
                        ,help='List of conditions values for the real samples e.g. 0.25 0.30 0.35 if not provided it will be  0 1 2 ..')   
    parser.add_argument('--discrete',action='store_true' , default=False
                       ,help = 'if True Sample only discrete labels from min_label to max_label')  
    parser.add_argument('--c_list', nargs='+', default=None,type=float
                        ,help='list of conditions to be sampled, if provided')                                                                                   
    parser.add_argument('--min_label', type=float, default=0
                        ,help='minimum label for conditions')
    parser.add_argument('--max_label', type=float, default=None
                        ,help='maximum label for conditions, if None will be set num of classes')
    parser.add_argument('--ohe',action='store_true' , default=False
                       ,help = 'use one hot encoding for conditioning')
    parser.add_argument('--SN_y', action='store_true' , default=False
                        ,help='apply SN to the condition linear layer')
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

    elif args.data == 'channels':
        from datasets import channels_datasets
        train_data = channels_datasets.Channels(path = args.data_path,labels_path=args.labels_path,ext = args.data_ext)

    else:
        print('no data named :',args.data)
        exit()

    dataloader = torch.utils.data.DataLoader(train_data,
                       shuffle=True, batch_size=args.batch_size,
                       num_workers=4)

    print("Finished data loading")    
    return dataloader,train_data


            
def prepare_models(args,n_cl = 0,device = 'cpu',only_G = False):           
    #model
    if args.G_model == 'dcgan':
        netG = generators.DC_Generator(args.zdim,img_ch=args.img_ch,base_ch= args.G_ch).to(device)
        netG.apply(weihts_init)

    elif args.G_model == 'residual_GAN':
        netG = generators.Res_Generator(args.zdim,img_ch=args.img_ch,n_classes = n_cl
                                        ,base_ch = args.G_ch,leak = args.leak_G,att = args.att
                                        ,SN = args.spec_norm_G
                                        ,cond_method = args.G_cond_method).to(device)

    if only_G:
        return netG

    if args.D_model == 'dcgan':
        netD = discriminators.DC_Discriminator(img_ch=args.img_ch
                                             ,base_ch= args.D_ch
                                             ,leak = args.leak_D
                                             ,n_layers = args.n_layers_D).to(device)  
        netD.apply(weights_init)

    elif args.D_model == 'cnn_sngan':
        netD = discriminators.SN_Discriminator(img_ch=args.img_ch
                                            ,base_ch= args.D_ch
                                            ,leak = args.leak_D
                                            ,SN=args.spec_norm_D).to(device)  
        netD.apply(weights_init)                            

    elif args.D_model == 'residual_GAN':
        netD = discriminators.Res_Discriminator(img_ch=args.img_ch,n_classes = n_cl,base_ch = args.D_ch
                                    ,leak = args.leak_D,att = args.att
                                    ,cond_method = args.D_cond_method
                                    ,SN = args.spec_norm_D
                                    ,SN_y = args.SN_y).to(device)                                            

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
    elif args.discrete:
        y =  torch.randint(low=int(args.min_label), high=int(max_value), size=(batch_size,1)).to(device)
        return y,y
    elif args.ohe:
        y =  torch.randint(low=int(args.min_label), high=int(max_value), size=(batch_size,)).to(device)
        y_ohe = torch.eye(num_classes)[y].to(device)
        return y_ohe,y_ohe

    else: # continious conditions
        y = (args.max_label - args.min_label) * torch.rand(batch_size,1) + args.min_label
        y = y.to(device)
        return y,y

def disc_2_ohe(y,num_classes,device):
    y_ohe = torch.eye(num_classes)[y].to(device)
    return y_ohe

def disc_2_cont(y,c_list,device): # convert discrete label to label in c_list
    for i,v in enumerate(c_list):
        y[y==i] = v
    y = y.unsqueeze(1)
    return y    
    
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


def elapsed_time(start_time):
    return time.time() - start_time

def calc_ralsloss_G(real, fake, margin=1.0):
    loss_real = torch.mean((real - fake.mean() + margin) ** 2)
    loss_fake = torch.mean((fake - real.mean() - margin) ** 2)
    loss = (loss_real + loss_fake)
    
    return loss
