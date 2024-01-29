import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import time
from collections import OrderedDict
import os 
from torchvision.utils import save_image

from  utils import sample_from_gen_PatchByPatch_test
from models import generators
#sys.path.append("models/")


parser = argparse.ArgumentParser()
                  
# data settings     
parser.add_argument('--output_resolution_height', type=int, default=384,help = 'output_resolution_height')
parser.add_argument('--output_resolution_width', type=int, default= 384,help = 'output_resolution_width')
parser.add_argument('--output_name', type=str, default= '241_generated.jpg',help = 'name of the generated image ')
parser.add_argument('--model_path', type=str, default= 'results/241_lp_bn_outerpadRepl/300_200.pth',help = 'path of the generator network')

args_sample = parser.parse_args()          

def load_G(state_dict_G,netG):
    new_state_dict_G = OrderedDict()

    for k, v in state_dict_G.items():
        if 'module' in k:
            k = k.replace('module.','')
        new_state_dict_G[k] = v
        
    netG.load_state_dict(new_state_dict_G)
    _=netG.eval()
    return netG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#filename = '../Exps/wall_v2/241_D_patch_dch64_nld4_G_patch2D_gch52_nlg6_npatches3x3_randomcrop192_n_cl1_originalspade_overlappad4_indmaps_overlappadconv_residual_FCG'
#filename = '241_lp_bn_clean/'
filename = args_sample.model_path
checkpoint = torch.load(filename,map_location='cpu')

args = checkpoint['args']
state_dict_G = checkpoint['netG_state_dict']   

#print(args)

netG = generators.ResidualPatchGenerator(z_dim = args.z_dim,G_ch = args.G_ch,base_res=args.base_res,n_layers_G = args.n_layers_G,attention=args.attention,
                                         img_ch= args.img_ch,leak = args.leak_G,SN = False,type_norm = args.type_norm_G,map_dim = 1,
                                         padding_mode = args.padding_mode,outer_padding = args.outer_padding,
                                         num_patches_h = 3,num_patches_w=3,padding_size = 1,conv_reduction = 2).to(device)



netG = load_G(state_dict_G,netG)


with torch.no_grad():
    #img = sample_from_gen_PatchByPatch(netG,num_images=1,device=device).cpu()
    img = sample_from_gen_PatchByPatch_test(netG,z_dim = args.z_dim,num_images=1,output_resolution_height=args_sample.output_resolution_height
                                            ,output_resolution_width=args_sample.output_resolution_width,device=device).cpu()

    
#im_np = np.round(im_np)
print('The image is saved as:', args_sample.output_name)
save_image(img*0.5+0.5,args_sample.output_name)

