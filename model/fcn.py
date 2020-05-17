import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

def conv2d_pad(in_planes, out_planes, kernel_size=(3, 3), stride=1,
               dilation=(1, 1), padding=(1, 1), bias=False):
    # the size of the padding should be a 6-tuple
    padding = tuple([x for x in padding for _ in range(2)][::-1])
    return nn.Sequential(
        nn.ReplicationPad2d(padding),
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=0, dilation=dilation, bias=bias))

def conv2d_bn_non(in_planes, out_planes, kernel_size=(3, 3), stride=1,
                  dilation=(1, 1), padding=(1, 1), bias=False):
    return nn.Sequential(
        conv2d_pad(in_planes, out_planes, kernel_size, stride,
                   dilation, padding, bias),
        SynchronizedBatchNorm2d(out_planes))

def conv2d_bn_elu(in_planes, out_planes, kernel_size=(3, 3), stride=1,
                  dilation=(1, 1), padding=(1, 1), bias=False):
    return nn.Sequential(
        conv2d_pad(in_planes, out_planes, kernel_size, stride,
                   dilation, padding, bias),
        SynchronizedBatchNorm2d(out_planes),
        nn.ELU(inplace=True))

class conv_fusion_block(nn.Module):
    def __init__(self, in_planes, out_planes, up_scale=2):
        super().__init__()
        self.conv1 = conv2d_bn_elu(in_planes,  out_planes, kernel_size=(3,3), padding=(1,1))
        self.conv2 = conv2d_bn_elu(out_planes, out_planes, kernel_size=(3,3), dilation=(2,2), padding=(2,2))
        self.conv3 = conv2d_bn_elu(out_planes, out_planes, kernel_size=(3,3), dilation=(4,4), padding=(4,4))

        self.reduce1 = conv2d_bn_elu(out_planes, 128, kernel_size=(3,3), padding=(1,1))
        self.reduce2 = conv2d_bn_elu(out_planes, 128, kernel_size=(3,3), padding=(1,1))
        self.reduce3 = conv2d_bn_elu(out_planes, 128, kernel_size=(3,3), padding=(1,1))

        self.conv_up1 = conv2d_bn_elu(128, 64, kernel_size=(3,3), padding=(1,1))
        self.conv_up2 = conv2d_bn_elu(64 , 64, kernel_size=(3,3), padding=(1,1))
        self.conv_up3 = conv2d_bn_elu(64 , 64, kernel_size=(1,1), padding=(0,0))

        self.up = nn.Upsample(scale_factor=(up_scale,up_scale), mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = x
        y1 = self.reduce1(y1)

        x = self.conv2(x)
        y2 = x
        y2 = self.reduce1(y2)

        x = self.conv3(x)
        y3 = x
        y3 = self.reduce1(y3)

        y = y1 + y2 + y3
        y = self.conv_up1(y)
        y = self.up(y)
        y = self.conv_up2(y)
        y = self.conv_up3(y)

        return x, y

class fcn(nn.Module):
    def __init__(self, in_num=1, out_num=1, filters=[64, 128, 256, 512, 512], activation='sigmoid'):
        super().__init__()
        self.activation = activation
        print('final activation function: '+self.activation)

        # Encoding Path
        self.block1 = nn.Sequential(
            conv2d_bn_elu(in_num, filters[0], kernel_size=(3,3), padding=(1,1)),
            conv2d_bn_elu(filters[0], filters[0], kernel_size=(3,3), padding=(1,1)))
        self.block2 = conv_fusion_block(in_planes=64,  out_planes=128, up_scale=2)
        self.block3 = conv_fusion_block(in_planes=128, out_planes=256, up_scale=4)
        self.block4 = conv_fusion_block(in_planes=256, out_planes=512, up_scale=8)
        self.block5 = conv_fusion_block(in_planes=512, out_planes=512, up_scale=16)

        # down & up sampling
        self.down = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fconv1 = conv2d_bn_elu(256, 128, kernel_size=(3,3), padding=(1,1))
        self.fconv2 = conv2d_bn_non(128, out_num, kernel_size=(3,3), padding=(1,1))

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(
                    m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.block1(x)
        x = self.down(x)
        x, y1 = self.block2(x)
        x = self.down(x)
        x, y2 = self.block3(x)
        x = self.down(x)
        x, y3 = self.block4(x)
        x = self.down(x)
        x, y4 = self.block5(x)

        z = torch.cat((y1,y2,y3,y4), 1)
        z = self.fconv1(z)
        z = self.fconv2(z)
        z = 2.0 * (torch.sigmoid(z) - 0.5)
        return z