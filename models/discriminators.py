import torch.nn as nn
import sys
import torch.nn.utils.spectral_norm as SpectralNorm
import numpy as np
import torch
import torch.nn.functional as F
from models.layers import *
        


class Res_Discriminator(nn.Module):
    def __init__(self, img_ch=3,base_ch = 32,n_classes=0,leak =0,att = False
        ,cond_method = 'concat',SN= True,SN_y = False):
        super(Res_Discriminator, self).__init__()
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()
            
        self.base_ch = base_ch
        self.att = att
        self.n_classes = n_classes # num_classes
        self.cond_method = cond_method
         
        
        #method of conditioning
        if n_classes !=0 :
            # conditioning by concatenation 
            if self.cond_method =='concat': 
                # concatenate after the 3rd layer
                self.embed_y = Linear(n_classes,base_ch * 2*8*8,SN=SN_y)
                    
            # conditioning by projection    
            elif self.cond_method =='proj': 
                self.embed_y = Linear(n_classes,base_ch * 16,SN=SN_y)
                #self.embed_y = nn.Embedding(n_classes,ch * 16).apply(init_weight)
            
            
        self.block1=OptimizedBlock(img_ch, base_ch,leak = leak,SN=SN)  #x/2
        if att:
            self.attention = Attention(base_ch,SN=SN)
        self.block2=ResBlockDiscriminator(base_ch, base_ch*2, downsample=True,leak = leak,SN=SN) #x/2
        
        if n_classes > 0 and self.cond_method =='concat':
            self.block3=ResBlockDiscriminator(base_ch*2 , base_ch*2,downsample=True,leak = leak,SN=SN)  #x/2
        else:    
            self.block3=ResBlockDiscriminator(base_ch*2 , base_ch*4,downsample=True,leak = leak,SN=SN)  #x/2
            
        self.block4=ResBlockDiscriminator(base_ch*4, base_ch*8,downsample=True,leak = leak,SN=SN)  #x/2
        self.block5=ResBlockDiscriminator(base_ch* 8, base_ch*16,leak = leak,SN=SN) #x
        
     
        self.fc =  Linear(self.base_ch*16, 1,SN=SN)

        

    def forward(self,x,y=None):
        h = self.block1(x)
        if self.att:
            h = self.attention(h)
        h = self.block2(h)
        h = self.block3(h)
        if y is not None and self.cond_method =='concat':
            h_y = self.embed_y(y)
            h_y = h_y.view(-1,self.base_ch*2,8,8)
            h = torch.cat((h,h_y),1)
        #print(h.shape)    
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)

        h = torch.sum(h,dim = (2,3))
        h = h.view(-1, self.base_ch*16)
        output = self.fc(h)

        if y is not None and self.cond_method =='proj': # use projection
            output += torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)    
        
           
        return output #,psi,self.embed_y(y),h,torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)


class DC_Discriminator(nn.Module): # paper DCGAN 
    def __init__(self,img_ch=3,base_ch = 64,n_layers=3):
        super(DC_Discriminator, self).__init__()
        self.z_dim = z_dim

        sequence = [ conv4x4(img_ch, base_ch, bias=False),
        nn.LeakyReLU(0.2, inplace=True)]

        ch_in = base_ch
        for n in range(0, n_layers):
            ch_out = ch_in*2
            sequence += [conv4x4(ch_in, ch_out, bias=False),
                        nn.BatchNorm2d(ch_out),
                        nn.LeakyReLU(0.2, inplace=True)]
            ch_in = ch_out


        self.model =  nn.Sequential(*sequence)
        self.final = conv4x4(ch_out, img_ch,s=1,p=0,bias=False)

    def forward(self, input,y = None):
        h =  self.model(input)
        o = self.final(h)
        return o.view(-1)




class SN_Discriminator(nn.Module): # paper SNGAN
    def __init__(self,img_ch=3,base_ch = 64,spectral_norm = False,leak =0.1):
        super(CNN_Discriminator, self).__init__()
        self.leak = leak
        self.base_ch = base_ch
        self.conv1 = conv3x3(img_ch, base_ch,SN=spectral_norm)
        self.conv2 = conv4x4(base_ch, base_ch,SN=spectral_norm)  # x/2

        self.conv3 = conv3x3(base_ch, base_ch*2,SN=spectral_norm)
        self.conv4 = conv4x4(base_ch*2, base_ch*2, SN=spectral_norm)  # x/2

        self.conv5 = conv3x3(base_ch*2, base_ch*4,SN=spectral_norm)
        self.conv6 = conv4x4(base_ch*4, base_ch*4,SN=spectral_norm)  # x/2

        self.conv7 = conv3x3(base_ch*4, base_ch*8,SN=spectral_norm)  # x
        
        if spectral_norm:
            self.fc = SpectralNorm(nn.Linear(8 * 8 * base_ch*8, 1))  # dens
        else:
            self.fc = nn.Linear(8 * 8 * base_ch*8, 1)  # dens 
           

    def forward(self, input,y=None):
        m = input
        m = nn.LeakyReLU(self.leak)(self.conv1(m))
        m = nn.LeakyReLU(self.leak)(self.conv2(m))
        m = nn.LeakyReLU(self.leak)(self.conv3(m))
        m = nn.LeakyReLU(self.leak)(self.conv4(m))
        m = nn.LeakyReLU(self.leak)(self.conv5(m))
        m = nn.LeakyReLU(self.leak)(self.conv6(m))
        m = nn.LeakyReLU(self.leak)(self.conv7(m))

        return self.fc(m.view(-1, 8 * 8  * self.base_ch*8))
