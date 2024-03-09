import argparse
import os
import torch
from collections import OrderedDict
from torchvision.utils import save_image

from  utils import sample_from_gen_PatchByPatch_test,sample_from_gen
from models import generators


parser = argparse.ArgumentParser()
                  
# settings     
parser.add_argument('--output_resolution_height', type=int, default=384,help = 'output_resolution_height')
parser.add_argument('--output_resolution_width', type=int, default= 384,help = 'output_resolution_width')
parser.add_argument('--output_name', type=str, default= '241_generated.jpg',help = 'name of the generated image ')
parser.add_argument('--model_path', type=str, default= 'results/241_lp_bn_outerpadRepl/300__ema.pth',help = 'path of the generator network')
parser.add_argument('--tiles',default= False,action='store_true',help = 'use tiling of the input')

args_sample = parser.parse_args()          

def split_string_at_last_slash(input_string):
    last_slash_index = input_string.rfind('/')
    if last_slash_index == -1:
        return input_string, ''
    else:
        first_part = input_string[:last_slash_index]
        second_part = input_string[last_slash_index + 1:]
        return first_part, second_part


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


filename = args_sample.model_path
folder,file = split_string_at_last_slash(filename)

checkpoint = torch.load(filename,map_location='cpu')

args = checkpoint['args']
state_dict_G = checkpoint['netG_state_dict']   


netG = generators.ResidualPatchGenerator(z_dim = args.z_dim,G_ch = args.G_ch,base_res=args.base_res,n_layers_G = args.n_layers_G,attention=args.attention,
                                         img_ch= args.img_ch,leak = args.leak_G,SN = False,type_norm = args.type_norm_G,map_dim = 1,
                                         padding_mode = args.padding_mode,outer_padding = args.outer_padding,
                                         num_patches_h = 3,num_patches_w=3,padding_size = 1,conv_reduction = 2).to(device)



netG = load_G(state_dict_G,netG)

print(args)

with torch.no_grad():
    if args.padding_mode == 'local':
        img = sample_from_gen_PatchByPatch_test(netG,z_dim = args.z_dim,num_images=1,output_resolution_height=args_sample.output_resolution_height
                                            ,output_resolution_width=args_sample.output_resolution_width,device=device).cpu()
    else:
        scale = (2**(netG.n_layers_G-1))
        new_base_res = args_sample.output_resolution_height//scale
        img = sample_from_gen(netG,z_dim = args.z_dim,base_res=new_base_res,num_images=1,tiles=args_sample.tiles,device=device).cpu()

saving_path = os.path.join(folder, args_sample.output_name)

#im_np = np.round(im_np)
print('The image is saved as:', saving_path)
save_image(img*0.5+0.5,saving_path)

