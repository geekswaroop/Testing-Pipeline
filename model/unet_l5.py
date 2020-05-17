import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

def conv2d_pad(in_planes, out_planes, kernel_size=(3,3), stride=1, 
               dilation=(1,1), padding=(1,1), bias=False):
    # the size of the padding should be a 6-tuple        
    padding = tuple([x for x in padding for _ in range(2)][::-1])
    return  nn.Sequential(
                nn.ReplicationPad2d(padding),
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias))     

def conv2d_bn_non(in_planes, out_planes, kernel_size=(3,3), stride=1, 
                  dilation=(1,1), padding=(1,1), bias=False):
    return nn.Sequential(
            conv2d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm2d(out_planes))              

def conv2d_bn_elu(in_planes, out_planes, kernel_size=(3,3), stride=1, 
                  dilation=(1,1), padding=(1,1), bias=False):
    return nn.Sequential(
            conv2d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm2d(out_planes),
            nn.ELU(inplace=True))                                   

class residual_block_2d(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True):
        super(residual_block_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_bn_elu( in_planes, out_planes, kernel_size=(3,3), padding=(1,1)),
            conv2d_bn_non(out_planes, out_planes, kernel_size=(3,3), padding=(1,1))
        )
        self.projector = conv2d_bn_non(in_planes, out_planes, kernel_size=(1,1), padding=(0,0))
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
    def __init__(self, in_planes, out_planes, projection=True, dilate=2):
        super(bottleneck_dilated, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_bn_elu( in_planes, out_planes, kernel_size=(1,1), padding=(0,0)),
            conv2d_bn_elu(out_planes, out_planes, kernel_size=(3,3), dilation=(dilate,dilate), padding=(dilate,dilate)),
            conv2d_bn_non(out_planes, out_planes, kernel_size=(1,1), padding=(0,0))
        )
        self.projector = conv2d_bn_non(in_planes, out_planes, kernel_size=(1,1), padding=(0,0))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y        

class unet_L5(nn.Module):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,128,128], activation='sigmoid'):
        super(unet_L5, self).__init__()
        self.activation = activation
        print('final activation function: '+self.activation)

        # Encoding Path
        self.layer1_E = nn.Sequential(
            residual_block_2d(in_num, filters[0], projection=True),
            residual_block_2d(filters[0], filters[0], projection=False),
            residual_block_2d(filters[0], filters[0], projection=True)
            #SELayer(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )

        self.layer2_E = nn.Sequential(
            residual_block_2d(filters[0], filters[1], projection=True),
            residual_block_2d(filters[1], filters[1], projection=False),
            residual_block_2d(filters[1], filters[1], projection=True)
            #SELayer(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3_E = nn.Sequential(
            residual_block_2d(filters[1], filters[2], projection=True),
            residual_block_2d(filters[2], filters[2], projection=False),
            residual_block_2d(filters[2], filters[2], projection=True)
            #SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        self.layer4_E = nn.Sequential(
            bottleneck_dilated(filters[2], filters[3], projection=True, dilate=2),
            bottleneck_dilated(filters[3], filters[3], projection=False,dilate=2),
            bottleneck_dilated(filters[3], filters[3], projection=True, dilate=2)
            #SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        # Center Block
        self.center = nn.Sequential(
            bottleneck_dilated(filters[3], filters[4], projection=True, dilate=2),
            bottleneck_dilated(filters[4], filters[4], projection=False,dilate=2),
            bottleneck_dilated(filters[4], filters[4], projection=True, dilate=2)
            #SELayer(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # Decoding Path
        self.layer1_D = nn.Sequential(
            residual_block_2d(filters[0], filters[0], projection=True),
            residual_block_2d(filters[0], filters[0], projection=False),
            residual_block_2d(filters[0], filters[0], projection=True)
            #SELayer(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )

        self.layer2_D = nn.Sequential(
            residual_block_2d(filters[1], filters[1], projection=True),
            residual_block_2d(filters[1], filters[1], projection=False),
            residual_block_2d(filters[1], filters[1], projection=True)
            #SELayer(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3_D = nn.Sequential(
            residual_block_2d(filters[2], filters[2], projection=True),
            residual_block_2d(filters[2], filters[2], projection=False),
            residual_block_2d(filters[2], filters[2], projection=True)
            #SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        self.layer4_D = nn.Sequential(
            bottleneck_dilated(filters[3], filters[3], projection=True, dilate=2),
            bottleneck_dilated(filters[3], filters[3], projection=False,dilate=2),
            bottleneck_dilated(filters[3], filters[3], projection=True, dilate=2)
            #SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        # down & up sampling
        self.down = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.up = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True)

        # convert to probability
        self.conv1 = conv2d_bn_elu(filters[1], filters[0], kernel_size=(1,1), padding=(0,0))
        self.conv2 = conv2d_bn_elu(filters[2], filters[1], kernel_size=(1,1), padding=(0,0))
        self.conv3 = conv2d_bn_elu(filters[3], filters[2], kernel_size=(1,1), padding=(0,0))
        self.conv4 = conv2d_bn_elu(filters[4], filters[3], kernel_size=(1,1), padding=(0,0))
        self.fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):

        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)
        z4 = self.layer4_E(x)
        x = self.down(z4)

        x = self.center(x)

        # Decoding Path
        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

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

class unet_L5_sk(unet_L5):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,128,128], activation='sigmoid'):
        super().__init__(in_num, out_num, filters, activation)

        self.map_conv1 = conv2d_bn_elu(filters[0], filters[0], kernel_size=(3,3), padding=(1,1))
        self.map_fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

        self.so1_conv1 = conv2d_bn_elu(filters[1], filters[0], kernel_size=(3,3), padding=(1,1))
        self.so1_fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

        self.so2_conv1 = conv2d_bn_elu(filters[2], filters[0], kernel_size=(3,3), padding=(1,1))
        self.so2_fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

        self.so3_conv1 = conv2d_bn_elu(filters[3], filters[0], kernel_size=(3,3), padding=(1,1))
        self.so3_fconv = conv2d_bn_non(filters[0], out_num, kernel_size=(3,3), padding=(1,1))

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

        x = self.center(x)

        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

        # side output 3
        so3_add =  self.so3_conv1(x)
        so3 = self.so3_fconv(so3_add)
        so3 = torch.sigmoid(so3)

        # Decoding Path
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

        # side output 0
        so0 = self.fconv(x)
        so0 = torch.sigmoid(so0)

        # energy map
        x = self.map_conv1(x)
        x = self.map_fconv(x)
        if self.activation == 'sigmoid':
            x = 2.0 * (torch.sigmoid(x) - 0.5)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 

        return x, so0, so1, so2, so3 

class unet_L5_sk_v2(unet_L5_sk):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,128,128], activation='sigmoid'):
        super().__init__(in_num, out_num, filters, activation)
        self.center_conv = conv2d_bn_elu(filters[4], filters[0], kernel_size=(3,3), padding=(1,1))

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

        x = self.center(x)
        so4_add = self.center_conv(x)

        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

        # side output 3
        so3_add =  self.so3_conv1(x) + self.up(so4_add)
        so3 = self.so3_fconv(so3_add)
        so3 = torch.sigmoid(so3)

        # Decoding Path
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

        # side output 0
        so0 = self.fconv(x)
        so0 = torch.sigmoid(so0)

        # energy map
        x = self.map_conv1(x)
        x = self.map_fconv(x)
        if self.activation == 'sigmoid':
            x = 2.0 * (torch.sigmoid(x) - 0.5)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 

        return x, so0, so1, so2, so3 

class unet_L6(unet_L5_sk_v2):
    def __init__(self, in_num=1, out_num=1, filters=[32,64,128,128,128,128], activation='sigmoid'):
        super().__init__(in_num, out_num, filters, activation)

        self.layer5_E = nn.Sequential(
            bottleneck_dilated(filters[3], filters[4], projection=True, dilate=2),
            bottleneck_dilated(filters[4], filters[4], projection=False,dilate=2),
            bottleneck_dilated(filters[4], filters[4], projection=True, dilate=2)
            #SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        self.layer5_D = nn.Sequential(
            bottleneck_dilated(filters[4], filters[4], projection=True, dilate=2),
            bottleneck_dilated(filters[4], filters[4], projection=False,dilate=2),
            bottleneck_dilated(filters[4], filters[4], projection=True, dilate=2)
            #SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )

        self.conv5 = conv2d_bn_elu(filters[5], filters[4], kernel_size=(1,1), padding=(0,0))
        self.l5_conv = conv2d_bn_elu(filters[4], filters[0], kernel_size=(3,3), padding=(1,1))

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
        center_add = self.center_conv(x)

        x = self.up(self.conv5(x))
        x = x + z5
        x = self.layer5_D(x)
        so4_add = self.l5_conv(x) + self.up(center_add)

        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

        # side output 3
        so3_add =  self.so3_conv1(x) + self.up(so4_add)
        so3 = self.so3_fconv(so3_add)
        so3 = torch.sigmoid(so3)

        # Decoding Path
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

        # side output 0
        so0 = self.fconv(x)
        so0 = torch.sigmoid(so0)

        # energy map
        x = self.map_conv1(x)
        x = self.map_fconv(x)
        if self.activation == 'sigmoid':
            x = 2.0 * (torch.sigmoid(x) - 0.5)
        elif self.activation == 'tanh':
            x = torch.tanh(x) 

        return x, so0, so1, so2, so3 
        