''' Sample
   This script loads a pretrained generator and sample imgaes '''
import random
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import shutil
import glob

from utils import *


# configurations
parser = prepare_parser()
parser = add_sample_parser(parser)
args = parser.parse_args()

# Device
device = prepare_device(args)

#Seeds
seed  = prepare_seed(args)
   
random.seed(seed)
torch.manual_seed(seed)

# parameters
n_cl = args.n_cl
n_cl1 = args.n_cl1
n_images= args.num_imgs


netG = prepare_models(args,n_cl,device,only_G = True)
   
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
                
                
if args.many is None:  # generate images for a single checkpoint args.G_cp of the model 
    netG = load_netG(netG,args.G_cp)
    if args.figure == 'grid':
        save_grid(args,netG,device,nrows=args.grid_rows,ncol=args.grid_rows,out_path = args.out_path)
    else:
        save_images(args,netG,device,out_path = args.out_path)
        
else: # generate images for each checkpoint saved in the path args.many 
    l = [x for x in os.listdir(args.many+"/") if x.endswith(".pth")]
    if not os.path.exists(args.many+"/"+args.out_path):
        os.mkdir(args.many+"/"+args.out_path)
    for checkname in l:
        print(checkname)
        netG = load_netG(netG,args.many+"/"+checkname)
        if args.figure == 'grid':
            save_grid(args,netG,device,nrows=args.grid_rows,ncol=args.grid_rows,out_path = args.many+"/"+args.out_path+"/"+ checkname)
        else:
            save_images(args,netG,device,os.path.splitext(args.many+"/"+args.out_path+"/"+checkname)[0])