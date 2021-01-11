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

print(args.c_list)
   
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

