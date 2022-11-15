import torch.nn as nn
import sys
import torch.nn.utils.spectral_norm as SpectralNorm
import numpy as np
import torch
import torch.nn.functional as F
from models.layers import *

from utils import *

        

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
        #

        self.up = nn.Upsample(scale_factor=2)

        if self.cond_method == 'concat':
            self.z_dim = self.z_dim+n_classes
            n_classes = 0

        if args.use_coord is False:
            self.coord_emb_dim =0
        else:
            self.coord_emb_dim = args.coord_emb_dim

        if self.leak >0:
            self.activation = nn.LeakyReLU(self.leak)
        else:
            self.activation = nn.ReLU()  
        
        self.dense = Linear(self.z_dim, base_res * base_res * self.base_ch*8,SN=SN)
        
        self.block1 = ResBlockGenerator(args,self.base_ch*8, self.base_ch*8,upsample=True,n_classes = n_classes)
        self.block2 = ResBlockGenerator(args,self.base_ch*8, self.base_ch*4,upsample=True,n_classes = n_classes)
        self.block3 = ResBlockGenerator(args,self.base_ch*4, self.base_ch*2,upsample=True,n_classes = n_classes)
        if self.att:
            self.attention = Attention(self.base_ch*2,SN=SN)
        self.block4 = ResBlockGenerator(args,self.base_ch*2, self.base_ch,upsample=True,n_classes = n_classes)
        if n_layers_G==5:
            final_chin = self.base_ch//2
            self.block5 = ResBlockGenerator(args,self.base_ch, self.base_ch//2,upsample=True,n_classes = n_classes,coord_emb_dim = self.coord_emb_dim)
        else:
            final_chin = self.base_ch
        #self.bn = nn.BatchNorm2d(final_chin)

        self.final = conv3x3(final_chin+self.coord_emb_dim,self.img_ch,SN = SN).apply(init_weight)


    def forward(self, z,y=None,coord_grids = None):
        if self.cond_method =='concat':
            z = torch.cat((z,y),1)
            y = None
        h = self.dense(z).view(-1,self.base_ch*8, self.base_res, self.base_res)
        #if coord_grids is None:
        #    coord_grids = [coord_grids]*self.n_layers_G
        h = self.block1(h,y)
        h = self.up(h)
        h = self.block2(h, y)
        h = self.up(h)
        h = self.block3(h, y)
        #if self.att:
        #    h = self.attention(h)
        h = self.block4(h,y)
        if self.n_layers_G ==5:
            h = self.block5(h,y,coord_grids)
        #h = self.bn(h)
        h = self.activation(h)

        if coord_grids is not None:
            h = torch.cat((h,coord_grids),1)
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
