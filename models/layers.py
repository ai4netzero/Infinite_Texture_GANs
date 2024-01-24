import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm
import torch
import torch.nn.functional as F
#from utils import utils.init_weight,crop_images,merge_patches_into_image
import utils


class conv2d_lp(nn.Module):
    """2D Conv supports local padding.

    Used in RRDB block in ESRGAN with local padding.

    Args:
        ch_in (int): number of input channels.
        ch_out (int): number of output channels
        local_padder (nn.Module): Local padding class or None if no local padding is used (in this case zero padding is used)
    """
    def __init__(self, ch_in, ch_out,local_padder= None,SN = False):
        super(conv2d_lp, self).__init__()
        
        self.local_padder = local_padder
        
        if self.local_padder is None:
            self.conv = conv3x3(ch_in,ch_out,SN,1,1)
        else:
            self.conv = conv3x3(ch_in,ch_out,SN,1,0)

    def forward(self, x,padding_variable_in =None, padding_location= None):
        
        if self.local_padder:
            x,padding_variable_out= self.local_padder(x,padding_variable_in,padding_location)
        else:
            padding_variable_out = None
        
        x = self.conv(x)
        
        return x, padding_variable_out

class LocalPadder(nn.Module):
    """
        PyTorch implementation of a Local Padder, which performs local padding based on convolutional settings.
        
        First the module merges the small-size patches together, perform outer padding and finally crop the patches with
        the specified overlapping padding size

        Args:
            num_patches_h (int): Number of patches along the height dimension (default is 3).
            num_patches_w (int): Number of patches along the width dimension (default is 3).
            outer_padding (str): Padding mode for outer patches (default is 'constant').
            padding_size (int): Padding size for each patch (default is 1).
            conv_reduction (int): Reduction factor in spatial size after convolution (default is 2 for 3x3 conv).
        """

    def __init__(self,num_patches_h = 3,num_patches_w =3,outer_padding = 'constant',padding_size =1,conv_reduction = 2):
        super(LocalPadder, self).__init__()
        
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.outer_padding = outer_padding
        self.padding_size = padding_size
        self.conv_reduction = conv_reduction
        
    def forward(self, input,padding_variable_in = None,padding_location=None):
        
        if padding_variable_in is not None:
            padding_variable_in_vertical,padding_variable_in_horizontal = padding_variable_in
            
        _,_,dx,dy = input.size()
        merged_input = utils.merge_patches_into_image(input,self.num_patches_h,self.num_patches_w,input.device)
           
        # During inference only
        # Get the last rows and columns to be used as an output padding variables for the next generation step 
        if self.eval():
            padding_variable_out_vertical = merged_input[:,:,:,[dx*(self.num_patches_h-1)-1]] # vertical slice
            if padding_location == 'horizontal':
                padding_variable_out_horizontal = merged_input[:,:,[dy*(self.num_patches_w-1)-1],:] 
            else:
                padding_variable_out_horizontal = merged_input[:,:,[dy*(self.num_patches_w-1)-1],:dx*(self.num_patches_h-1)]
                
            padding_variable_out = [padding_variable_out_vertical,padding_variable_out_horizontal]
        else:
            padding_variable_out = None

                  
        # Outer padding during training or if they are the first patches to be generated during inference  
        if padding_location == None:
            merged_input = F.pad(merged_input, (self.padding_size,self.padding_size,self.padding_size,self.padding_size), self.outer_padding) 

        # Pad from left vertically if thery are the intermediate patches to be generated in the first row
        elif padding_location == 'vertical': 
            merged_input = torch.cat((padding_variable_in_vertical,merged_input),-1)
            merged_input = F.pad(merged_input, (0,self.padding_size,self.padding_size,self.padding_size), self.outer_padding) 

        # Pad from top horizontally if thery are the final patches to be generated in the final rows
        elif padding_location == 'horizontal':
            merged_input = F.pad(merged_input, (self.padding_size,self.padding_size,0,self.padding_size), self.outer_padding) 
            merged_input = torch.cat((padding_variable_in_horizontal,merged_input),-2)

        # Pad from left and top if thery are the intermediate patches to be generated in an intermediate row
        elif padding_location == 'both': 
            merged_input = torch.cat((padding_variable_in_vertical,merged_input),-1)
            merged_input = F.pad(merged_input, (0,self.padding_size,0,self.padding_size), self.outer_padding) 
            merged_input = torch.cat((padding_variable_in_horizontal,merged_input),-2) # extended


        # Perform cropping after padding to get the patches back
        res_with_padding = dx +self.padding_size*self.conv_reduction
        padded_output = utils.crop_images(merged_input,res_with_padding,res_with_padding,dx,device = input.device)


        return padded_output,padding_variable_out

 


def conv3x3(ch_in,ch_out,SN = False,s = 1,p=1,bias = True,padding_mode='zeros'):
    if SN:
        return SpectralNorm(nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=p, stride=s,bias = bias,padding_mode =padding_mode).apply(utils.init_weight))
    else:
        return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=p,stride=s,bias = bias,padding_mode =padding_mode).apply(utils.init_weight)

def Linear(ch_in,ch_out,SN = False,bias = True):
    if SN:
        return SpectralNorm(nn.Linear(ch_in,ch_out,bias = bias).apply(utils.init_weight))
    else:
        return nn.Linear(ch_in,ch_out,bias = bias).apply(utils.init_weight)

def conv4x4(ch_in,ch_out,SN = False,s = 2,p=1,bias = True):
    if SN:
        return SpectralNorm(nn.Conv2d(ch_in, ch_out, kernel_size=4, padding=p,stride=s,bias = bias).apply(utils.init_weight))
    else:
        return nn.Conv2d(ch_in, ch_out, kernel_size=4, padding=p,stride=s,bias = bias).apply(utils.init_weight)    
    
def conv1x1(ch_in,ch_out,SN = False,s = 1,p=0,bias = True):
    if SN:
        return SpectralNorm(nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=p,stride=s,bias = bias).apply(utils.init_weight))
    else:
        return nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=p,stride=s,bias = bias).apply(utils.init_weight)
 
        
class StochasticSpatialModulation(nn.Module):
    def __init__(self,in_channel, map_dim,SN=False):
        super().__init__()
        
        self.in_channel = in_channel
        
        # The output number of channels is double in the input to account fot both the bias and variance
        self.out_channels = in_channel*2


        self.bn = nn.BatchNorm2d(in_channel, affine=False)  # no learning parameters
        
        self.mlp_shared = nn.Sequential(conv3x3(map_dim, 128,SN=SN,p=0).apply(utils.init_weight),
                                        nn.ReLU()
                                        )
        
        self.embed = conv3x3(128, self.out_channels,bias = True,SN=SN,p=0).apply(utils.init_weight)       
        nn.init.orthogonal_(self.embed.weight.data[:, :in_channel], gain=1)
        self.embed.weight.data[:, in_channel:].zero_()

    def forward(self, inputs, maps):
        out = self.bn(inputs)
        actv  = self.mlp_shared(maps.float())
        embed = self.embed(actv)
        gamma, beta = embed.chunk(2, dim=1)
        out = (1+gamma) * out + beta
        return out
 
class Attention(nn.Module):
    def __init__(self, channels,SN=False):
        super().__init__()
        self.channels = channels
        self.theta    = conv1x1(channels, channels//8,SN=SN).apply(utils.init_weight)
        self.phi      = conv1x1(channels, channels//8,SN=SN).apply(utils.init_weight)
        self.g        = conv1x1(channels, channels//2,SN=SN).apply(utils.init_weight)
        self.o        = conv1x1(channels//2, channels,SN=SN).apply(utils.init_weight)
        self.gamma    = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def forward(self, inputs):
        batch,c,h,w = inputs.size()
        theta = self.theta(inputs) #->(*,c/8,h,w)
        phi   = F.max_pool2d(self.phi(inputs), [2,2]) #->(*,c/8,h/2,w/2)
        g     = F.max_pool2d(self.g(inputs), [2,2]) #->(*,c/2,h/2,w/2)
        
        theta = theta.view(batch, self.channels//8, -1) #->(*,c/8,h*w)
        phi   = phi.view(batch, self.channels//8, -1) #->(*,c/8,h*w/4)
        g     = g.view(batch, self.channels//2, -1) #->(*,c/2,h*w/4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1) #->(*,h*w,h*w/4)
        o    = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch,self.channels//2,h,w)) #->(*,c,h,w)
        return self.gamma*o + inputs
   




    
class ResBlockGenerator(nn.Module):
    def __init__(self,generator,in_channels, out_channels,hidden_channels=None):
        super(ResBlockGenerator, self).__init__()
        
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.learnable_sc = (in_channels != out_channels) 
        self.type_norm = generator.type_norm
        self.leak = generator.leak
        self.map_dim = generator.map_dim
        self.SN = generator.SN
        
        self.conv1 = conv2d_lp(in_channels,hidden_channels,generator.local_padder,self.SN).apply(utils.init_weight)
        self.conv2 = conv2d_lp(hidden_channels,out_channels,generator.local_padder,self.SN).apply(utils.init_weight)
        
        if self.learnable_sc:
            self.conv3 = conv1x1(in_channels,out_channels,SN=self.SN).apply(utils.init_weight)

        if self.type_norm == 'BN': 
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(hidden_channels)
        elif self.type_norm == 'SSM':                
            self.bn1 = StochasticSpatialModulation(in_channels,self.map_dim,self.SN)
            self.bn2 = StochasticSpatialModulation(hidden_channels,self.map_dim,self.SN)
            if self.learnable_sc:
                self.bn3 = StochasticSpatialModulation(in_channels,self.map_dim,self.SN)
        # TODO 
        # throw an exception if the type_norm is not correct

        if self.leak > 0:
            self.activation = nn.LeakyReLU(self.leak)
        else:
            self.activation = nn.ReLU() 

    def shortcut(self, x,map=None):
        if self.learnable_sc:
            if self.type_norm == 'SSM':
                x = self.bn3(x,map)
            x = self.conv3(x)
        return x

    def forward(self, x,map=None,pad_variable_in = None,padding_location = None):
        
        if pad_variable_in == None:
            pad_variable_in=[None]*2
            
        if self.type_norm == 'SSM':
            out = self.activation(self.bn1(x,map))
        else:
            out = self.activation(self.bn1(x))
        
        out,pad_var1 = self.conv1(out,pad_variable_in[0],padding_location)
        
        if self.type_norm == 'SSM':
            out = self.activation(self.bn2(out,map))
        else:
            out = self.activation(self.bn2(out))


        
        out,pad_var2 = self.conv2(out,pad_variable_in[1],padding_location)
        
        out_res = self.shortcut(x,map)
        out = out+out_res
        
        return out, [pad_var1,pad_var2]
                
    
class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False,hidden_channels=None,leak = 0,SN=True,BN=False):
        super(ResBlockDiscriminator, self).__init__()
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.conv1 = conv3x3(in_channels, hidden_channels,SN = SN).apply(utils.init_weight)
        self.conv2 = conv3x3(hidden_channels, out_channels,SN = SN).apply(utils.init_weight)
        self.conv3 = conv1x1(in_channels, out_channels,SN = SN).apply(utils.init_weight)

        self.learnable_sc = (in_channels != out_channels) or downsample
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 

        self.downsampling = nn.AvgPool2d(2)
        self.downsample = downsample

        self.BN = BN
        if BN:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(hidden_channels)

    def residual(self, x):
        h = x
        if self.BN:
            h = self.bn1(h)
        h = self.activation(h)
        h = self.conv1(h)
        if self.BN:
            h = self.bn2(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.downsampling(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv3(x)
            if self.downsample:
                return self.downsampling(x)
            else:
                return x
        else:
            return x

    def forward (self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(nn.Module):

    def __init__(self, in_channels, out_channels,leak =0,SN=True):
        super(OptimizedBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels,SN=SN).apply(utils.init_weight)
        self.conv2 = conv3x3(out_channels, out_channels,SN=SN).apply(utils.init_weight)
        self.conv3 = conv1x1(in_channels, out_channels,SN=SN).apply(utils.init_weight)
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 
        
        self.model = nn.Sequential(
            self.conv1,
            self.activation,
            self.conv2,
            nn.AvgPool2d(2)  # stride = 2 ( default = kernel size)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.conv3
        )
    def forward(self, x):
        return self.model(x) + self.bypass(x)     
    
