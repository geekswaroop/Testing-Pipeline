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

class smooth_module(nn.Module):
    def __init__(self, num_planes):
        super(smooth_module, self).__init__()
        self.conv1 = conv3d_bn_elu(num_planes, num_planes, kernel_size=(3,1,1), padding=(1,0,0))
        #self.conv2 = conv3d_bn_elu(num_planes, num_planes, kernel_size=(3,1,1), dilation=(2,1,1), padding=(2,0,0))
        self.conv3 = conv3d_bn_elu(num_planes, num_planes, kernel_size=(3,3,3), padding=(1,1,1))
        #self.conv4 = conv3d_bn_elu(num_planes, num_planes, kernel_size=(3,3,3), dilation=(2,2,2), padding=(2,2,2))
        self.conv5 = nn.Sequential(
            conv3d_bn_elu(num_planes, num_planes, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1)),
            conv3d_bn_elu(num_planes, num_planes, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.Upsample(scale_factor=(2,1,1), mode='trilinear', align_corners=True)
        )

    def forward(self, x):
        y1 = self.conv1(x)
        #y2 = self.conv2(x)
        y3 = self.conv3(x)
        #y4 = self.conv4(x)
        y5 = self.conv5(x)
        #y = y1+y2+y3+y4+y5
        y = y1 + y3 + y5
        return y  

class smooth_3d(nn.Module):
    def __init__(self, out_num=1, filters=[32,32,32,32,32,32], activation='sigmoid'):
        super(smooth_3d, self).__init__()
        self.activation = activation
        print('final activation function: '+self.activation)
        self.smooth5 = smooth_module(filters[5])
        self.smooth4 = smooth_module(filters[4])
        self.smooth3 = smooth_module(filters[3])
        self.smooth2 = smooth_module(filters[2])
        self.smooth1 = smooth_module(filters[1])
        self.smooth0 = smooth_module(filters[0])

        self.smooth_map = smooth_module(filters[0])

        self.fconv3 = conv3d_bn_non(filters[3], out_num, kernel_size=(3,3,3), padding=(1,1,1))
        self.fconv2 = conv3d_bn_non(filters[2], out_num, kernel_size=(3,3,3), padding=(1,1,1))
        self.fconv1 = conv3d_bn_non(filters[1], out_num, kernel_size=(3,3,3), padding=(1,1,1))
        self.fconv0 = conv3d_bn_non(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.map_conv1 = conv3d_bn_elu(filters[0], filters[0], kernel_size=(3,3,3), padding=(1,1,1))
        self.fconv_map = conv3d_bn_non(filters[0], out_num, kernel_size=(3,3,3), padding=(1,1,1))

        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, so0_add, so1_add, so2_add, so3_add, so4_add, so5_add, map_pre):
        so5_smt = self.smooth5(so5_add)
        so4_smt = self.smooth4(so4_add) + self.up(so5_smt)
        so3_smt = self.smooth3(so3_add) + self.up(so4_smt)
        so2_smt = self.smooth2(so2_add) + self.up(so3_smt)
        so1_smt = self.smooth1(so1_add) + self.up(so2_smt)
        so0_smt = self.smooth0(so0_add) + self.up(so1_smt)

        so3_out = torch.sigmoid(self.fconv3(so3_smt))
        so2_out = torch.sigmoid(self.fconv3(so2_smt))
        so1_out = torch.sigmoid(self.fconv3(so1_smt))
        so0_out = torch.sigmoid(self.fconv3(so0_smt))

        map_smt = self.smooth_map(map_pre) + self.map_conv1(so0_smt)
        map_out = self.fconv_map(map_smt)
        if self.activation == 'sigmoid':
            map_out = 2.0 * (torch.sigmoid(map_out) - 0.5)
        elif self.activation == 'tanh':
            map_out = torch.tanh(map_out) 

        return map_out, so0_out, so1_out, so2_out, so3_out