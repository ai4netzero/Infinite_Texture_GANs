import torch.nn as nn
import sys
import torch.nn.utils.spectral_norm as SpectralNorm
import numpy as np
import torch
import torch.nn.functional as F
from models.layers import *

#from utils import *
import utils
        

class Res_Generator(nn.Module):
    def __init__(self,args,n_classes = 0):
        super(Res_Generator, self).__init__()

        self.z_dim = args.zdim
        self.base_ch = args.G_ch
        self.n_classes = n_classes
        self.att = args.att
        self.cond_method = cond_method = args.G_cond_method
        self.n_layers_G =n_layers_G= args.n_layers_G
        self.base_res =base_res =  args.base_res
        self.img_ch = args.img_ch
        self.leak = leak =  args.leak_G
        self.SN =SN =  args.spec_norm_G
        self.padding_mode = args.G_padding
        self.upsampling_mode = args.G_upsampling
        self.num_patches_h = args.num_patches_h
        self.num_patches_w = args.num_patches_w

        #

        self.up = nn.Upsample(scale_factor=2,mode = args.G_upsampling)

        if self.cond_method == 'concat':
            self.z_dim = self.z_dim+n_classes
            n_classes = 0

        if self.leak >0:
            self.activation = nn.LeakyReLU(self.leak)
        else:
            self.activation = nn.ReLU()  
        
        #self.dense = Linear(self.z_dim, base_res * base_res * self.base_ch*8,SN=SN)
        self.start = conv3x3(self.z_dim,self.base_ch*8,SN = SN,padding_mode=self.padding_mode,p=0).apply(init_weight)
        
        self.block1 = ResBlockGenerator(args,self.base_ch*8, self.base_ch*8,n_classes = n_classes,G_cond_method = 'conv3x3')
        self.block2 = ResBlockGenerator(args,self.base_ch*8, self.base_ch*4,n_classes = n_classes,G_cond_method = 'conv3x3')
        self.block3 = ResBlockGenerator(args,self.base_ch*4, self.base_ch*2,n_classes = n_classes,G_cond_method = 'conv3x3')
        if self.att:
            self.attention = Attention(self.base_ch*2,SN=SN)
        self.block4 = ResBlockGenerator(args,self.base_ch*2, self.base_ch,n_classes = n_classes,G_cond_method = 'conv3x3')
        if n_layers_G>=5:
            final_chin = self.base_ch//2
            self.block5 = ResBlockGenerator(args,self.base_ch, self.base_ch//2,n_classes = n_classes,G_cond_method = 'conv3x3')
            if n_layers_G == 6:
                final_chin = self.base_ch//4
                self.block6 = ResBlockGenerator(args,self.base_ch//2, self.base_ch//4,n_classes = n_classes,G_cond_method = 'conv3x3')
        else:
            final_chin = self.base_ch
        #self.bn = nn.BatchNorm2d(final_chin)

        self.final = conv3x3(final_chin,self.img_ch,SN = SN,padding_mode=self.padding_mode,p=0).apply(init_weight)
        


    def forward(self, z,y=None,num_patches_h=None,num_patches_w=None,padding_variable_h= None,padding_variable_v= None,last = False):
        if self.cond_method =='concat':
            z = torch.cat((z,y),1)
            y = None
        if num_patches_h is None or num_patches_w is None:
            num_patches_h = self.num_patches_h
            num_patches_w = self.num_patches_w
            
        if padding_variable_h is None:
            padding_variable_h = [[None,None]]*self.n_layers_G
            padding_variable_h.append([None])
        if padding_variable_v is None:
            padding_variable_v = [[None,None]]*self.n_layers_G
            padding_variable_v.append([None])

        padding_variable_out_v = []
        padding_variable_out_h = []

        h = self.start(z)
        #print(padding_variable)
        h,pad_var_out_v1,pad_var_out_h1,pad_var_out_v2,pad_var_out_h2 = self.block1(h,y[0]
                                            ,num_patches_h=num_patches_h,num_patches_w=num_patches_w
                                            ,padding_variable_h = padding_variable_h[0]
                                            ,padding_variable_v = padding_variable_v[0]
                                            ,last = last)
        padding_variable_out_v.append([pad_var_out_v1,pad_var_out_v2])
        padding_variable_out_h.append([pad_var_out_h1,pad_var_out_h2])

        
        h = self.up(h) 
        h,pad_var_out_v1,pad_var_out_h1,pad_var_out_v2,pad_var_out_h2 = self.block2(h, y[1]
                                                                                    ,num_patches_h=num_patches_h,num_patches_w=num_patches_w
                                                                                    ,padding_variable_h = padding_variable_h[1]
                                                                                    ,padding_variable_v = padding_variable_v[1]
                                                                                    ,last = last)
        padding_variable_out_v.append([pad_var_out_v1,pad_var_out_v2])
        padding_variable_out_h.append([pad_var_out_h1,pad_var_out_h2])
        
        h = self.up(h) 
        h,pad_var_out_v1,pad_var_out_h1,pad_var_out_v2,pad_var_out_h2 = self.block3(h, y[2]
                                                                                    ,num_patches_h=num_patches_h,num_patches_w=num_patches_w
                                                                                    ,padding_variable_h = padding_variable_h[2]
                                                                                    ,padding_variable_v = padding_variable_v[2]
                                                                                    ,last = last)
        padding_variable_out_v.append([pad_var_out_v1,pad_var_out_v2])
        padding_variable_out_h.append([pad_var_out_h1,pad_var_out_h2])
        
        if self.att:
            h = self.attention(h)
            
        h = self.up(h) 
        h,pad_var_out_v1,pad_var_out_h1,pad_var_out_v2,pad_var_out_h2 = self.block4(h,y[3]
                                                                                    ,num_patches_h=num_patches_h,num_patches_w=num_patches_w
                                                                                    ,padding_variable_h = padding_variable_h[3]
                                                                                    ,padding_variable_v = padding_variable_v[3]
                                                                                    ,last = last)
        padding_variable_out_v.append([pad_var_out_v1,pad_var_out_v2])
        padding_variable_out_h.append([pad_var_out_h1,pad_var_out_h2])
        
        if self.n_layers_G >=5:
            h = self.up(h) 
            h,pad_var_out_v1,pad_var_out_h1,pad_var_out_v2,pad_var_out_h2 = self.block5(h,y[4]
                                                                                        ,num_patches_h=num_patches_h,num_patches_w=num_patches_w
                                                                                        ,padding_variable_h = padding_variable_h[4]
                                                                                        ,padding_variable_v = padding_variable_v[4]
                                                                                        ,last = last)
            padding_variable_out_v.append([pad_var_out_v1,pad_var_out_v2])
            padding_variable_out_h.append([pad_var_out_h1,pad_var_out_h2])
            
        if self.n_layers_G == 6:
            h = self.up(h)
            h,pad_var_out_v1,pad_var_out_h1,pad_var_out_v2,pad_var_out_h2 = self.block6(h,y[5]
                                                                                        ,num_patches_h=num_patches_h,num_patches_w=num_patches_w
                                                                                        ,padding_variable_h = padding_variable_h[5]
                                                                                        ,padding_variable_v = padding_variable_v[5]
                                                                                        ,last = last)
            padding_variable_out_v.append([pad_var_out_v1,pad_var_out_v2])
            padding_variable_out_h.append([pad_var_out_h1,pad_var_out_h2])
        
        h,pad_var_out_vf,pad_var_out_hf = utils.overlap_padding(h,pad_size = 1,h=num_patches_h,w=num_patches_w
                                            ,padding_variable_h = padding_variable_h[-1][0]
                                            ,padding_variable_v = padding_variable_v[-1][0]
                                            ,last = last)
        if self. training :
            pad_var_out_vf = pad_var_out_hf =  None  
        
        padding_variable_out_v.append([pad_var_out_vf])
        padding_variable_out_h.append([pad_var_out_hf])
        
        #h = self.bn(h)
        h = self.activation(h)
        h = self.final(h)
        img = nn.Tanh()(h)
        
        if self.training:
            return img
        else:
            return img, padding_variable_out_v,padding_variable_out_h

class DC_Generator(nn.Module): # papers DCGAN or SNGAN
    def __init__(self,z_dim=128,img_ch=3,base_ch = 64,n_layers=4):
        super(DC_Generator, self).__init__()
        self.z_dim = z_dim

        sequence = [nn.ConvTranspose2d(z_dim, self.base_ch*8, 4, stride=1, bias=False),  # 4x4 (dense)
        nn.BatchNorm2d(self.base_ch*8),
        nn.ReLU()]

        ch_in = self.base_ch*8
        for n in range(0, n_layers):
            ch_out = ch_in//2
            sequence += [nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=(1, 1), bias=False),
                            nn.BatchNorm2d(ch_out),
                            nn.ReLU()]
            ch_in = ch_out

        sequence +=[nn.ConvTranspose2d(ch_out, img_ch, 3, stride=1, padding=(1, 1)),  # 1x (conv)
                nn.Tanh()]

        self.model =  nn.Sequential(*sequence)


    def forward(self, input,y = None):
        return self.model(input.view(-1,self.z_dim, 1, 1))


# Testing architecture
'''
noise = torch.randn(1, 128)
fake = G(noise)
print(fake.size())'''
