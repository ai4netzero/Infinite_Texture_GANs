import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm
import torch
from models.layers import OptimizedBlock,Attention,Linear,ResBlockDiscriminator,conv1x1,conv3x3,conv4x4
        


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
            
            elif self.cond_method =='conv1x1':
                self.embed_y = conv1x1(1,base_ch * 4,SN=SN_y)
            elif self.cond_method =='conv3x3':
                self.embed_y = conv3x3(1,base_ch * 4,SN=SN_y)
                        
            
        self.block1=OptimizedBlock(img_ch, base_ch,leak = leak,SN=SN)  #x/2
        if att:
            self.attention = Attention(base_ch,SN=SN)
        self.block2=ResBlockDiscriminator(base_ch, base_ch*2, downsample=True,leak = leak,SN=SN) #x/2
        
        if n_classes > 0 and self.cond_method =='concat':
            self.block3=ResBlockDiscriminator(base_ch*2 , base_ch*2,downsample=True,leak = leak,SN=SN)  #x/2
        else:
            self.block3=ResBlockDiscriminator(base_ch*2 , base_ch*4,downsample=True,leak = leak,SN=SN)  #x/2

        if n_classes > 0 and self.cond_method !='proj':
            self.block4=ResBlockDiscriminator(base_ch*4, base_ch*4,downsample=True,leak = leak,SN=SN)  #x/2
        else:    
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
        if y is not None and 'conv' in self.cond_method:
            w = h.size(-1)
            y = y.view(-1,1,w,w)
            h_y = self.embed_y(y)
            h = torch.cat((h,h_y),1)
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
        super(SN_Discriminator, self).__init__()
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


# paper pix2pix,sinGAN(batchnorm), SPADE,pix2pixHD(instancenorm)
class Patch_Discriminator(nn.Module):
    
    """PatchGAN discriminator that classifies patches of the image instead of the whole input image.

    The discriminator is used in pix2pix, sinGAN(batchnorm), SPADE, pix2pixHD(instancenorm).

    Args:
        img_ch (int): Channel number of inputs.
        base_ch (int): Base channels of the network layer.
        n_layers_D (int): Number of the discriminator layers . Default: 4
        kw (int): Kernel width. Defaults: 4
        SN (int): Apply spectral normalization to the network weights. Default: 32.
        norm_layer(str): The type of normalization layer used in the network: (batch,instance, None). Default: None
    """
    
    def __init__(self, img_ch=1,base_ch = 64,n_layers_D=4,kw = 4,SN= False,norm_layer = None):
        super(Patch_Discriminator, self).__init__()   
        nf = base_ch
        self.img_ch = img_ch
        padw = 1
        if kw == 4:
            conv_fun =  conv4x4   
        elif kw==3:
            conv_fun =  conv3x3  

        if norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
            affine = True
        elif norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
            affine = False
            
        sequence = [conv_fun(img_ch, base_ch,SN=SN,bias = True),
                nn.LeakyReLU(0.2, False)]

        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            if norm_layer:
                sequence += [conv_fun(nf_prev, nf,s = stride, SN=SN,bias = True),
                            norm_layer(nf,affine=affine),
                            nn.LeakyReLU(0.2, False)
                            ]
            else:
                sequence += [conv_fun(nf_prev, nf,s = stride, SN=SN,bias = True),
                            nn.LeakyReLU(0.2, False)
                            ]                

        sequence += [conv_fun(nf, 1,s = 1, SN=SN,bias = True)]

        self.model = nn.Sequential(*sequence)
        
    def forward(self, x):
        out = self.model(x)
        return out