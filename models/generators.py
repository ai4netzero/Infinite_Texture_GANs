import torch.nn as nn
from models.layers import LocalPadder,Attention,conv2d_lp,ResBlockGenerator

class ResidualPatchGenerator(nn.Module):
    """
        PyTorch implementation of a Residual Patch Generator.

        Args:
            z_dim (int): Dimension of the input latent vector (default is 128).
            G_ch (int): Number of channels in the generator's first layer (default is 64).
            base_res (int): Resolution at the first layer (default is 4).
            n_layers_G (int): Number of layers in the generator (default is 4).
            att (bool): Whether to use attention module in the generator (default is True).
            img_ch (int): Number of channels in the generated image (default is 3 for RGB).
            leak (float): Leaky ReLU negative slope (default is 0).
            SN (bool): Whether to use spectral normalization (default is False).
            padding_mode (str): Padding mode for convolution layers (default is 'local').
            outer_padding (str): Padding mode for outer patches (default is 'replicate').
            num_patches_h (int): Number of patches along the height dimension (default is 3).
            num_patches_w (int): Number of patches along the width dimension (default is 3).
        """
    
    def __init__(self,z_dim = 128,G_ch = 64,base_res=4,n_layers_G = 4,attention=True,img_ch= 3
                 ,leak = 0,SN = False,type_norm = 'BN',map_dim = 1,
                 padding_mode = 'local',outer_padding = 'replicate',
                 num_patches_h = 3,num_patches_w=3,padding_size = 1,conv_reduction = 2):
        
        super(ResidualPatchGenerator, self).__init__()

        self.z_dim = z_dim
        self.base_ch = G_ch
        self.base_res =  base_res
        self.n_layers_G = n_layers_G
        self.attention = attention
        self.img_ch = img_ch        
        self.leak  =  leak
        self.SN  =  SN
        self.type_norm = type_norm
        self.map_dim = map_dim
        self.padding_mode = padding_mode
        self.outer_padding = outer_padding
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.padding_size = padding_size
        self.conv_reduction = conv_reduction
        
        LocalPadder.set_attributes(num_patches_h =num_patches_h ,num_patches_w=num_patches_w,outer_padding = outer_padding
                                            ,padding_size = padding_size,conv_reduction = conv_reduction)
        
        self.up = nn.Upsample(scale_factor=2,mode = 'nearest')

        if self.leak >0:
            self.activation = nn.LeakyReLU(self.leak)
        else:
            self.activation = nn.ReLU()  
        
        self.start = conv2d_lp(self.z_dim,self.base_ch*8,SN,padding_mode)
        
        self.block1 = ResBlockGenerator(self,self.base_ch*8, self.base_ch*8,padding_mode=padding_mode)
        self.block2 = ResBlockGenerator(self,self.base_ch*8, self.base_ch*4,padding_mode=padding_mode)
        self.block3 = ResBlockGenerator(self,self.base_ch*4, self.base_ch*2,padding_mode=padding_mode)
        self.block4 = ResBlockGenerator(self,self.base_ch*2, self.base_ch,padding_mode=padding_mode)   
             
        # To generate sizes of 16x the base_res 
        if self.n_layers_G>=5:
            final_chin = self.base_ch//2
            self.block5 = ResBlockGenerator(self,self.base_ch, self.base_ch//2,padding_mode=padding_mode)
            # To generate sizes of 32x the base_res 
            if self.n_layers_G == 6:
                final_chin = self.base_ch//4
                self.block6 = ResBlockGenerator(self,self.base_ch//2, self.base_ch//4,padding_mode=padding_mode)
        else:
            final_chin = self.base_ch

        if self.type_norm == 'BN':
            self.bn = nn.BatchNorm2d(final_chin)
            
        if self.attention:
            self.attention = Attention(self.base_ch*2,SN=SN)
    
        self.final = conv2d_lp(final_chin,self.img_ch,SN,padding_mode)
        

    def forward(self, z,maps=None,image_location = '1st_row_1st_col'):
            
        h = self.start(z,image_location) # x
        
        h = self.block1(h,maps[0],image_location)
        
        h = self.up(h) # 2x
        h = self.block2(h, maps[1],image_location)
        
        h = self.up(h) # 4x
        h = self.block3(h, maps[2],image_location)
        
        if self.attention:
            h = self.attention(h)
            
        h = self.up(h) # 8x
        h = self.block4(h, maps[3],image_location)
        
        # Form the output padding variable to be used for the next iteration during inference 
        #padding_variable_out = [pad_var_start,pad_var_block1,pad_var_block2,pad_var_block3,pad_var_block4]
        
        if self.n_layers_G >=5:
            h = self.up(h) # 16x
            h = self.block5(h, maps[4],image_location)
        if self.n_layers_G == 6:
            h = self.up(h) # 32x
            h = self.block6(h, maps[5],image_location)

        if self.type_norm == 'BN':
            h = self.bn(h)
            
        h = self.activation(h)
        
        h = self.final(h,image_location)
        
        out = nn.Tanh()(h)
        
        
        return out

    