import torch.nn as nn
import sys
import torch.nn.utils.spectral_norm as SpectralNorm
import numpy as np
import torch
import torch.nn.functional as F
from models.layers import *
        

class Res_Generator(nn.Module):
    def __init__(self,z_dim =128,img_ch=3,base_ch =64,n_classes = 0,leak = 0,att = False,SN=False,cond_method = 'cbn'):
        super(Res_Generator, self).__init__()

        self.base_ch = base_ch
        self.n_classes = n_classes
        self.att = att
        self.cond_method = cond_method
        if self.cond_method == 'concat':
            z_dim = z_dim+n_classes
            n_classes = 0


        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()  
        
        self.dense = Linear(z_dim, 4 * 4 * base_ch*8,SN=SN)
        
        self.block1 = ResBlockGenerator(base_ch*8, base_ch*8,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        self.block2 = ResBlockGenerator(base_ch*8, base_ch*4,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        self.block3 = ResBlockGenerator(base_ch*4, base_ch*2,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        if att:
            self.attention = Attention(base_ch*2,SN=SN)
        self.block4 = ResBlockGenerator(base_ch*2, base_ch,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        
        self.bn = nn.BatchNorm2d(base_ch)
        self.final = conv3x3(base_ch,img_ch,SN = SN).apply(init_weight)

    def forward(self, z,y=None):
        if self.cond_method =='concat':
            z = torch.cat((z,y),1)
            y = None
        h = self.dense(z).view(-1,self.base_ch*8, 4, 4)
        h = self.block1(h,y)
        h = self.block2(h, y)
        h = self.block3(h, y)
        if self.att:
            h = self.attention(h)
        h = self.block4(h,y)
        h = self.bn(h)
        h = self.activation(h)
        h = self.final(h)
        return nn.Tanh()(h)


class DC_Generator(nn.Module): # papers DCGAN or SNGAN
    def __init__(self,z_dim=128,img_ch=3,base_ch = 64,n_layers=4):
        super(DC_Generator, self).__init__()
        self.z_dim = z_dim

        sequence = [nn.ConvTranspose2d(z_dim, base_ch*8, 4, stride=1, bias=False),  # 4x4 (dense)
        nn.BatchNorm2d(base_ch*8),
        nn.ReLU()]

        ch_in = base_ch*8
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
