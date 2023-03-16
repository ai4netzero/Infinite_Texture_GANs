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
        
        self.dense = Linear(self.z_dim, base_res * base_res * self.base_ch*8,SN=SN)
        
        self.block1 = ResBlockGenerator(args,self.base_ch*8, self.base_ch*8,upsample=True,n_classes = n_classes,G_cond_method = 'conv3x3')
        self.block2 = ResBlockGenerator(args,self.base_ch*8, self.base_ch*4,upsample=True,n_classes = n_classes,G_cond_method = 'conv3x3')
        self.block3 = ResBlockGenerator(args,self.base_ch*4, self.base_ch*2,upsample=True,n_classes = n_classes,G_cond_method = 'conv3x3')
        if self.att:
            self.attention = Attention(self.base_ch*2,SN=SN)
        self.block4 = ResBlockGenerator(args,self.base_ch*2, self.base_ch,upsample=True,n_classes = n_classes,G_cond_method = 'conv3x3')
        if n_layers_G>=5:
            final_chin = self.base_ch//2
            self.block5 = ResBlockGenerator(args,self.base_ch, self.base_ch//2,upsample=True,n_classes = n_classes,G_cond_method = 'conv3x3')
            if n_layers_G == 6:
                final_chin = self.base_ch//4
                self.block6 = ResBlockGenerator(args,self.base_ch//2, self.base_ch//4,upsample=True,n_classes = n_classes,G_cond_method = 'conv3x3')
        else:
            final_chin = self.base_ch
        #self.bn = nn.BatchNorm2d(final_chin)

        self.final = conv3x3(final_chin,self.img_ch,SN = SN,padding_mode=self.padding_mode,p=0).apply(init_weight)
        
    def overlap_padding(self,input,pad_size =2,conv_red = 2):
        _,_,dx,dy = input.size()
        merged_input = utils.merge_patches_2D(input,h = self.num_patches_h,w = self.num_patches_w,device = input.device)
        merged_input = F.pad(merged_input, (pad_size,pad_size,pad_size,pad_size), "replicate", 0) 
        res_withpadd = dx +pad_size*conv_red
        padded_input = utils.crop_fun_(merged_input,res_withpadd,res_withpadd,dx,device = input.device)
        return padded_input



    def forward(self, z,y=None):
        if self.cond_method =='concat':
            z = torch.cat((z,y),1)
            y = None
        h = self.dense(z).view(-1,self.base_ch*8, self.base_res, self.base_res)
        #print(h.shape)
        #if coord_grids is None:
        #    coord_grids = [coord_grids]*self.n_layers_G
        h = self.block1(h,y[0])
        #print(h[:9,0])
        h = self.overlap_padding(h)
        #print(h.shape)
        #print(h[:9,0])

        #exit()
        h = self.up(h) # 8x8
        h = self.block2(h, y[1])
        h = self.overlap_padding(h)
        h = self.up(h) # 16x16
        h = self.block3(h, y[2])
        h = self.overlap_padding(h)
        if self.att:
            h = self.attention(h)
        h = self.up(h) #32x32
        h = self.block4(h,y[3])
        if self.n_layers_G >=5:
            h = self.overlap_padding(h)
            h = self.up(h) # 64x64
            h = self.block5(h,y[4])
        if self.n_layers_G == 6:
            h = self.overlap_padding(h)
            h = self.up(h) # 128x128
            h = self.block6(h,y[5])
        
        h = self.overlap_padding(h,pad_size = 1)
        #h = self.bn(h)
        h = self.activation(h)
        h = self.final(h)
        return nn.Tanh()(h)


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
