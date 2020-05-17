import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

def conv3d_pad(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
               dilation=(1,1,1), padding=(1,1,1), bias=False):
    # the size of the padding should be a 6-tuple        
    padding = tuple([x for x in padding for _ in range(2)][::-1])
    return  nn.Sequential(
                nn.ReplicationPad3d(padding),
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias))     

def conv3d_bn_non(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes))              

def conv3d_bn_elu(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes),
            nn.ELU(inplace=True))                                   

class residual_block_2d(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True):
        super(residual_block_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_bn_elu( in_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1)),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1))
        )
        self.projector = conv3d_bn_non(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y  

class residual_block_3d(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True):
        super(residual_block_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_bn_elu( in_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1)),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1))
        )
        self.projector = conv3d_bn_non(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y       

class bottleneck_dilated(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True):
        super(bottleneck_dilated, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_bn_elu( in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0)),
            conv3d_bn_elu(out_planes, out_planes, kernel_size=(3,3,3), dilation=(1,2,2), padding=(1,2,2)),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        )
        self.projector = conv3d_bn_non(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y        

class unet_3d(nn.Module):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,256], activation='sigmoid'):
        super(unet_3d, self).__init__()
        self.activation = activation
        print('final activation function: '+self.activation)

        # Encoding Path
        self.layer1_E = nn.Sequential(
            residual_block_2d(in_num, filters[0], projection=True),
            residual_block_2d(filters[0], filters[0], projection=False),
            residual_block_2d(filters[0], filters[0], projection=True),
            SELayer(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )

        self.layer2_E = nn.Sequential(
            residual_block_2d(filters[0], filters[1], projection=True),
            residual_block_2d(filters[1], filters[1], projection=False),
            residual_block_2d(filters[1], filters[1], projection=True),
            SELayer(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3_E = nn.Sequential(
            residual_block_3d(filters[1], filters[2], projection=True),
            residual_block_3d(filters[2], filters[2], projection=False),
            residual_block_3d(filters[2], filters[2], projection=True),
            SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        # Center Block
        self.center = nn.Sequential(
            bottleneck_dilated(filters[2], filters[3], projection=True),
            bottleneck_dilated(filters[3], filters[3], projection=False),
            bottleneck_dilated(filters[3], filters[3], projection=True),
            SELayer(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # Decoding Path
        self.layer1_D = nn.Sequential(
            residual_block_2d(filters[0], filters[0], projection=True),
            residual_block_2d(filters[0], filters[0], projection=False),
            residual_block_2d(filters[0], filters[0], projection=True),
            SELayer(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )

        self.layer2_D = nn.Sequential(
            residual_block_2d(filters[1], filters[1], projection=True),
            residual_block_2d(filters[1], filters[1], projection=False),
            residual_block_2d(filters[1], filters[1], projection=True),
            SELayer(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3_D = nn.Sequential(
            residual_block_3d(filters[2], filters[2], projection=True),
            residual_block_3d(filters[2], filters[2], projection=False),
            residual_block_3d(filters[2], filters[2], projection=True),
            SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        # down & up sampling
        self.down = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)

        # convert to probability
        self.conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv2 = conv3d_bn_elu(filters[2], filters[1], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0))
        self.fconv = conv3d_bn_non(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

    def forward(self, x):

        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)

        x = self.center(x)

        # Decoding Path
        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = self.fconv(x)
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 
        return x

class unet_3d_simple(nn.Module):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,128,256,256], activation='sigmoid'):
        super(unet_3d_simple, self).__init__()
        self.activation = activation
        print('final activation function: '+self.activation)

        # Encoding Path
        self.layer1_E = residual_block_2d(in_num, filters[0], projection=True)
        self.layer2_E = residual_block_2d(filters[0], filters[1], projection=True)
        self.layer3_E = residual_block_3d(filters[1], filters[2], projection=True)
        self.layer4_E = residual_block_3d(filters[2], filters[3], projection=True)
        self.layer5_E = bottleneck_dilated(filters[3], filters[4], projection=True)

        # Center Block
        self.center = bottleneck_dilated(filters[4], filters[5], projection=True)

        # Decoding Path
        self.layer1_D = residual_block_2d(filters[0], filters[0], projection=True)
        self.layer2_D = residual_block_2d(filters[1], filters[1], projection=True)
        self.layer3_D = residual_block_3d(filters[2], filters[2], projection=True)
        self.layer4_D = residual_block_3d(filters[3], filters[3], projection=True)
        self.layer5_D = bottleneck_dilated(filters[4], filters[4], projection=True)

        # down & up sampling
        self.down = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)

        # convert to probability
        self.conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv2 = conv3d_bn_elu(filters[2], filters[1], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv4 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv5 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0))

        self.so1_conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so1_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so2_conv1 = conv3d_bn_elu(filters[2], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so2_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so3_conv1 = conv3d_bn_elu(filters[3], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so3_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so4_conv1 = conv3d_bn_elu(filters[4], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so4_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so5_conv1 = conv3d_bn_elu(filters[5], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so5_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.map_conv = conv3d_bn_elu(filters[0], out_num, kernel_size=(1,1,1), padding=(0,0,0))
        self.fconv = conv3d_bn_non(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.map_fconv = conv3d_bn_non(6, out_num, kernel_size=(3,3,3), padding=(1,1,1))  

    def forward(self, x):
        # Encoding path
        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)
        z4 = self.layer4_E(x)
        x = self.down(z4)
        z5 = self.layer5_E(x)
        x = self.down(z5)

        x = self.center(x)
        # side output 5
        so5_add = self.so5_conv1(x)
        so5 = self.so5_fconv(so5_add)

        x = self.up(self.conv5(x))
        x = x + z5
        x = self.layer5_D(x)

        # side output 4
        so4_add = self.so4_conv1(x) + self.up(so5_add)
        so4 = self.so4_fconv(so4_add)

        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

        # side output 3
        so3_add = self.so3_conv1(x) + self.up(so4_add)
        so3 = self.so3_fconv(so3_add)
        so3 = torch.sigmoid(so3)

        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)
    
        # side output 2
        so2_add =  self.so2_conv1(x) + self.up(so3_add)
        so2 = self.so2_fconv(so2_add)
        so2 = torch.sigmoid(so2)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        # side output 1
        so1_add =  self.so1_conv1(x) + self.up(so2_add)
        so1 = self.so1_fconv(so1_add)
        so1 = torch.sigmoid(so1)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = x + self.up(so1_add)
        x = self.fconv(x)
        x = F.elu(x)

        x = torch.cat((x, 
                       self.up(so1), 
                       self.up(self.up(so2)),
                       self.up(self.up(self.up(so3))),
                       self.up(self.up(self.up(self.up(so4)))),
                       self.up(self.up(self.up(self.up(self.up(so5)))))   
                       ), dim=1)
        x = self.map_fconv(x)
        if self.activation == 'sigmoid':
            x = 2.0 * (torch.sigmoid(x) - 0.5)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 

        return x, so1, so2, so3

class SELayer(nn.Module):
    # Squeeze-and-excitation layer
    def __init__(self, channel, channel_reduction=4, spatial_reduction=4, z_reduction=1):
        super(SELayer, self).__init__()
        self.pool_size = (z_reduction, spatial_reduction, spatial_reduction)
        self.se = nn.Sequential(
                nn.AvgPool3d(kernel_size=self.pool_size, stride=self.pool_size),
                nn.Conv3d(channel, channel // channel_reduction, kernel_size=1),
                SynchronizedBatchNorm3d(channel // channel_reduction),
                nn.ELU(inplace=True),
                nn.Conv3d(channel // channel_reduction, channel, kernel_size=1),
                SynchronizedBatchNorm3d(channel),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=self.pool_size, mode='trilinear', align_corners=True),
                )     

    def forward(self, x):
        y = self.se(x)
        z = x + y*x
        return z 

class unet_3d_sk(nn.Module):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,128,256,256], activation='sigmoid'):
        super(unet_3d_sk, self).__init__()
        self.activation = activation
        print('final activation function: '+self.activation)

        # Encoding Path
        self.layer1_E = residual_block_2d(in_num, filters[0], projection=True)
        self.layer2_E = residual_block_2d(filters[0], filters[1], projection=True)
        self.layer3_E = residual_block_3d(filters[1], filters[2], projection=True)
        self.layer4_E = residual_block_3d(filters[2], filters[3], projection=True)
        self.layer5_E = bottleneck_dilated(filters[3], filters[4], projection=True)

        # Center Block
        self.center = bottleneck_dilated(filters[4], filters[5], projection=True)

        # Decoding Path
        self.layer1_D = residual_block_2d(filters[0], filters[0], projection=True)
        self.layer2_D = residual_block_2d(filters[1], filters[1], projection=True)
        self.layer3_D = residual_block_3d(filters[2], filters[2], projection=True)
        self.layer4_D = residual_block_3d(filters[3], filters[3], projection=True)
        self.layer5_D = bottleneck_dilated(filters[4], filters[4], projection=True)

        # down & up sampling
        self.down = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)

        # convert to probability
        self.conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv2 = conv3d_bn_elu(filters[2], filters[1], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv4 = conv3d_bn_elu(filters[4], filters[3], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv5 = conv3d_bn_elu(filters[5], filters[4], kernel_size=(1,1,1), padding=(0,0,0))

        self.so0_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so1_conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so1_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so2_conv1 = conv3d_bn_elu(filters[2], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so2_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so3_conv1 = conv3d_bn_elu(filters[3], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so3_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so4_conv1 = conv3d_bn_elu(filters[4], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so5_conv1 = conv3d_bn_elu(filters[5], filters[0], kernel_size=(3,3,3), padding=(1,1,1))

        self.map_conv1 = conv3d_bn_elu(filters[0], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.map_fconv = conv3d_bn_non(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))  

    def forward(self, x):
        # Encoding path
        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)
        z4 = self.layer4_E(x)
        x = self.down(z4)
        z5 = self.layer5_E(x)
        x = self.down(z5)

        x = self.center(x)
        # side output 5
        so5_add = self.so5_conv1(x)

        x = self.up(self.conv5(x))
        x = x + z5
        x = self.layer5_D(x)

        # side output 4
        so4_add = self.so4_conv1(x) + self.up(so5_add)

        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

        # side output 3
        so3_add = self.so3_conv1(x) + self.up(so4_add)
        so3 = self.so3_fconv(so3_add)
        so3 = F.sigmoid(so3)

        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)
    
        # side output 2
        so2_add =  self.so2_conv1(x) + self.up(so3_add)
        so2 = self.so2_fconv(so2_add)
        so2 = F.sigmoid(so2)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        # side output 1
        so1_add =  self.so1_conv1(x) + self.up(so2_add)
        so1 = self.so1_fconv(so1_add)
        so1 = F.sigmoid(so1)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = x + self.up(so1_add)
        so0 = self.so0_fconv(x)
        so0 = F.sigmoid(so0)

        x = self.map_conv1(x)
        x = self.map_fconv(x)
        if self.activation == 'sigmoid':
            x = 2.0 * (torch.sigmoid(x) - 0.5)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 

        return x, so0, so1, so2, so3

class unet_3d_sk_v2(nn.Module):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,128,256,256], activation='sigmoid'):
        super(unet_3d_sk_v2, self).__init__()
        self.activation = activation
        print('final activation function: '+self.activation)

        # Encoding Path
        self.layer1_E = residual_block_2d(in_num, filters[0], projection=True)
        self.layer2_E = residual_block_2d(filters[0], filters[1], projection=True)
        self.layer3_E = residual_block_3d(filters[1], filters[2], projection=True)
        self.layer4_E = residual_block_3d(filters[2], filters[3], projection=True)
        self.layer5_E = bottleneck_dilated(filters[3], filters[4], projection=True)

        # Center Block
        self.center = bottleneck_dilated(filters[4], filters[5], projection=True)

        # Decoding Path
        self.layer1_D = residual_block_2d(filters[0], filters[0], projection=True)
        self.layer2_D = residual_block_2d(filters[1], filters[1], projection=True)
        self.layer3_D = residual_block_3d(filters[2], filters[2], projection=True)
        self.layer4_D = residual_block_3d(filters[3], filters[3], projection=True)
        self.layer5_D = bottleneck_dilated(filters[4], filters[4], projection=True)

        # down & up sampling
        self.down = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)

        # convert to probability
        self.conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv2 = conv3d_bn_elu(filters[2], filters[1], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv4 = conv3d_bn_elu(filters[4], filters[3], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv5 = conv3d_bn_elu(filters[5], filters[4], kernel_size=(1,1,1), padding=(0,0,0))

        self.so0_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so1_conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so1_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so2_conv1 = conv3d_bn_elu(filters[2], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so2_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so3_conv1 = conv3d_bn_elu(filters[3], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so3_fconv = conv3d_bn_elu(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.so4_conv1 = conv3d_bn_elu(filters[4], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.so5_conv1 = conv3d_bn_elu(filters[5], filters[0], kernel_size=(3,3,3), padding=(1,1,1))

        self.map_conv1 = conv3d_bn_elu(filters[0], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.map_conv2 = conv3d_bn_elu(filters[0], filters[0], kernel_size=(1,3,3), padding=(0,1,1))
        self.map_conv3 = conv3d_bn_non(filters[0], out_num, kernel_size=(1,3,3), padding=(0,1,1)) 

    def forward(self, x):
        # Encoding path
        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)
        z4 = self.layer4_E(x)
        x = self.down(z4)
        z5 = self.layer5_E(x)
        x = self.down(z5)

        x = self.center(x)
        # side output 5
        so5_add = self.so5_conv1(x)

        x = self.up(self.conv5(x))
        x = x + z5
        x = self.layer5_D(x)

        # side output 4
        so4_add = self.so4_conv1(x) + self.up(so5_add)

        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

        # side output 3
        so3_add = self.so3_conv1(x) + self.up(so4_add)
        so3 = self.so3_fconv(so3_add)
        so3 = F.sigmoid(so3)

        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)
    
        # side output 2
        so2_add =  self.so2_conv1(x) + self.up(so3_add)
        so2 = self.so2_fconv(so2_add)
        so2 = F.sigmoid(so2)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        # side output 1
        so1_add =  self.so1_conv1(x) + self.up(so2_add)
        so1 = self.so1_fconv(so1_add)
        so1 = F.sigmoid(so1)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = x + self.up(so1_add)
        so0 = self.so0_fconv(x)
        so0 = F.sigmoid(so0)

        x = self.map_conv1(x)
        x = self.map_conv2(x)
        x = self.map_conv3(x)
        if self.activation == 'sigmoid':
            x = 2.0 * (torch.sigmoid(x) - 0.5)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 

        return x, so0, so1, so2, so3

              