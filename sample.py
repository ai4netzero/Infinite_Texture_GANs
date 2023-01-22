''' Sample
   This script loads a pretrained generator and sample imgaes '''
import random
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import shutil
import glob
from tqdm.notebook import trange, tqdm

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
n_images= args.num_imgs

if args.many is not None:
    l = [x for x in os.listdir(args.many+"/") if x.endswith(".pth")]
    if not os.path.exists(args.many+"/"+args.out_path):
        os.mkdir(args.many+"/"+args.out_path)
    args_model = torch.load(os.path.join(args.many,l[-1]),map_location='cpu')['args']
    
    
netG,_ = prepare_models(args_model,args_model.n_cl,device)

if (device.type == 'cuda') and (args.ngpu > 1):
    add_module = True
    netG = nn.DataParallel(netG, args.gpu_list)
else:
    add_module = False
   
def save_images(args,netG,device,out_path):
    n_images= args.num_imgs
    truncated=args.truncated

    if not os.path.exists(out_path):
        os.mkdir(out_path)
        
    im_batch_size = 20
    
    if n_images<im_batch_size:
        im_batch_size = n_images
        
    n_batches = n_images//im_batch_size
        
    for i_batch in tqdm(range(0, n_images, im_batch_size)):
        if i_batch ==  n_batches*im_batch_size:
            im_batch_size = n_images - i_batch
        
        h = args_model.num_patches_h
        w = args_model.num_patches_w   
        b_size = im_batch_size*h*w
        #print()
        #print(b_size)
            
        #gen_images,_ = sample_from_gen(args,im_batch_size,args.zdim,args.n_cl,netG,device,truncated = args.truncated)
        fake,_ = sample_patches_from_gen_2D(args_model,b_size,netG,None,device)
        img_patches = fake.cpu()
        imgs_merged = merge_patches_2D(img_patches,h,w,'cpu')
        imgs_merged = imgs_merged*0.5+0.5
        gen_images = imgs_merged
        gen_images = gen_images.cpu().detach()
        #shape=(*,ch=3,h,w), torch.Tensor
        
        #denormalize
        #gen_images = gen_images*0.5 + 0.5
        
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
    for checkname in l:
        print(checkname)
        netG = load_netG(netG,args.many+"/"+checkname,add_module)
        if args.figure == 'grid':
            save_grid(args,netG,device,nrows=args.grid_rows,ncol=args.grid_rows,out_path = args.many+"/"+args.out_path+"/"+ checkname)
        else:
            save_images(args,netG,device,os.path.splitext(args.many+"/"+args.out_path+"/"+checkname)[0])