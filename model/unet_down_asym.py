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


class residual_block_2d(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True):
        super(residual_block_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_bn_elu(in_planes, out_planes,
                          kernel_size=(3, 3), padding=(1, 1)),
            conv2d_bn_non(out_planes, out_planes,
                          kernel_size=(3, 3), padding=(1, 1))
        )
        self.projector = conv2d_bn_non(
            in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0))
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
            conv2d_bn_elu(in_planes, out_planes,
                          kernel_size=(1, 1), padding=(0, 0)),
            conv2d_bn_elu(out_planes, out_planes, kernel_size=(
                3, 3), dilation=(dilate, dilate), padding=(dilate, dilate)),
            conv2d_bn_non(out_planes, out_planes,
                          kernel_size=(1, 1), padding=(0, 0))
        )
        self.projector = conv2d_bn_non(
            in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y


class unet_down_asym(nn.Module):
    def __init__(self, in_num=1, out_num=1, filters=[32, 64, 128, 128, 128, 128], activation='sigmoid'):
        super().__init__()
        self.activation = activation
        print('final activation function: '+self.activation)

        # Encoding Path
        self.layer1_E = nn.Sequential(
            residual_block_2d(filters[0], filters[0], projection=True),
            residual_block_2d(filters[0], filters[0], projection=False),
            residual_block_2d(filters[0], filters[0], projection=False)
        )

        self.layer2_E = nn.Sequential(
            residual_block_2d(filters[0], filters[1], projection=True),
            residual_block_2d(filters[1], filters[1], projection=False),
            residual_block_2d(filters[1], filters[1], projection=False)
        )

        self.layer3_E = nn.Sequential(
            residual_block_2d(filters[1], filters[2], projection=True),
            residual_block_2d(filters[2], filters[2], projection=False),
            residual_block_2d(filters[2], filters[2], projection=False)
        )

        self.layer4_E = nn.Sequential(
            bottleneck_dilated(filters[2], filters[3],
                               projection=True, dilate=2),
            bottleneck_dilated(filters[3], filters[3],
                               projection=False, dilate=2),
            bottleneck_dilated(filters[3], filters[3],
                               projection=False, dilate=2)
        )
        self.layer5_E = nn.Sequential(
            bottleneck_dilated(filters[3], filters[4],
                               projection=True, dilate=2),
            bottleneck_dilated(filters[4], filters[4],
                               projection=False, dilate=2),
            bottleneck_dilated(filters[4], filters[4],
                               projection=False, dilate=2)
        )

        # Center Block
        self.center = nn.Sequential(
            bottleneck_dilated(filters[4], filters[5],
                               projection=True, dilate=2),
            bottleneck_dilated(filters[5], filters[5],
                               projection=False, dilate=2),
            bottleneck_dilated(filters[5], filters[5],
                               projection=False, dilate=2)
        )

        # Decoding Path
        self.layer1_D = residual_block_2d(filters[0], filters[0], projection=False)
        self.layer2_D = residual_block_2d(filters[1], filters[1], projection=False)
        self.layer3_D = residual_block_2d(filters[2], filters[2], projection=False)

        self.layer4_D = bottleneck_dilated(filters[3], filters[3], projection=False, dilate=2)
        self.layer5_D = bottleneck_dilated(filters[4], filters[4], projection=False, dilate=2)

        # down & up sampling
        self.down = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.up = nn.Upsample(scale_factor=(
            2, 2), mode='bilinear', align_corners=True)

        self.iconv = conv2d_bn_elu(
            in_num,  filters[0], kernel_size=(3, 3), padding=(1, 1))

        self.lconv1 = conv2d_bn_elu(
            filters[0], filters[1], kernel_size=(1, 1), padding=(0, 0))
        self.lconv2 = conv2d_bn_elu(
            filters[0], filters[2], kernel_size=(1, 1), padding=(0, 0))
        self.lconv3 = conv2d_bn_elu(
            filters[0], filters[3], kernel_size=(1, 1), padding=(0, 0))
        self.lconv4 = conv2d_bn_elu(
            filters[0], filters[4], kernel_size=(1, 1), padding=(0, 0))

        self.rconv1 = conv2d_bn_elu(
            filters[1], filters[0], kernel_size=(1, 1), padding=(0, 0))
        self.rconv2 = conv2d_bn_elu(
            filters[2], filters[1], kernel_size=(1, 1), padding=(0, 0))
        self.rconv3 = conv2d_bn_elu(
            filters[3], filters[2], kernel_size=(1, 1), padding=(0, 0))
        self.rconv4 = conv2d_bn_elu(
            filters[4], filters[3], kernel_size=(1, 1), padding=(0, 0))
        self.rconv5 = conv2d_bn_elu(
            filters[5], filters[4], kernel_size=(1, 1), padding=(0, 0))
        self.fconv = conv2d_bn_non(
            filters[0], out_num, kernel_size=(3, 3), padding=(1, 1))

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(
                    m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):

        x = self.iconv(x)
        z0 = x
        z1 = self.layer1_E(x)
        x = self.down(z1) + self.down(z0)
        z2 = self.layer2_E(x)
        x = self.down(z2) + self.lconv1(self.down(self.down(z0)))
        z3 = self.layer3_E(x)
        x = self.down(z3) + self.lconv2(self.down(self.down(self.down(z0))))
        z4 = self.layer4_E(x)
        x = self.down(z4) + self.lconv3(self.down(self.down(self.down(self.down(z0)))))
        z5 = self.layer5_E(x)
        x = self.down(z5) + self.lconv4(self.down(self.down(self.down(self.down(self.down(z0))))))

        x = self.center(x)

        # Decoding Path
        x = self.up(self.rconv5(x))
        x = x + z5
        x = self.layer5_D(x)

        x = self.up(self.rconv4(x))
        x = x + z4
        x = self.layer4_D(x)

        x = self.up(self.rconv3(x))
        x = x + z3
        x = self.layer3_D(x)

        x = self.up(self.rconv2(x))
        x = x + z2
        x = self.layer2_D(x)

        x = self.up(self.rconv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = self.fconv(x)
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        return x

class unet_down_asym_sk(unet_down_asym):
    def __init__(self, in_num=1, out_num=1, filters=[32, 64, 128, 128, 128, 128],
                 activation='sigmoid'):
        super().__init__(in_num, out_num, filters, activation)

        self.map_conv1 = conv2d_bn_elu(
            filters[0], filters[0], kernel_size=(3, 3), padding=(1, 1))
        self.map_fconv = conv2d_bn_non(
            filters[0], out_num, kernel_size=(3, 3), padding=(1, 1))

        self.so1_conv1 = conv2d_bn_elu(
            filters[1], filters[0], kernel_size=(3, 3), padding=(1, 1))
        self.so1_fconv = conv2d_bn_non(
            filters[0], out_num, kernel_size=(3, 3), padding=(1, 1))

        self.so2_conv1 = conv2d_bn_elu(
            filters[2], filters[0], kernel_size=(3, 3), padding=(1, 1))
        self.so2_fconv = conv2d_bn_non(
            filters[0], out_num, kernel_size=(3, 3), padding=(1, 1))

        self.so3_conv1 = conv2d_bn_elu(
            filters[3], filters[0], kernel_size=(3, 3), padding=(1, 1))
        self.so3_fconv = conv2d_bn_non(
            filters[0], out_num, kernel_size=(3, 3), padding=(1, 1))

        # no supervision
        self.so4_conv1 = conv2d_bn_elu(
            filters[4], filters[0], kernel_size=(3, 3), padding=(1, 1))
        self.so5_conv1 = conv2d_bn_elu(
            filters[5], filters[0], kernel_size=(3, 3), padding=(1, 1))

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(
                    m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # Encoding path
        x = self.iconv(x)
        z0 = x

        y0 = self.down(z0)
        y1 = self.lconv1(self.down(self.down(z0)))
        y2 = self.lconv2(self.down(self.down(self.down(z0))))
        y3 = self.lconv3(self.down(self.down(self.down(self.down(z0)))))
        y4 = self.lconv4(self.down(self.down(self.down(self.down(self.down(z0))))))

        z1 = self.layer1_E(x)
        x = self.down(z1) + y0
        z2 = self.layer2_E(x)
        x = self.down(z2) + y1
        z3 = self.layer3_E(x)
        x = self.down(z3) + y2
        z4 = self.layer4_E(x)
        x = self.down(z4) + y3
        z5 = self.layer5_E(x)
        x = self.down(z5) + y4

        x = self.center(x)
        # side output 5 (no supervision)
        so5_add = self.so5_conv1(x)

        x = self.up(self.rconv5(x))
        x = x + z5
        x = self.layer5_D(x)

        # side output 4 (no supervision)
        so4_add = self.so4_conv1(x) + self.up(so5_add)

        x = self.up(self.rconv4(x))
        x = x + z4
        x = self.layer4_D(x)

        # side output 3
        so3_add = self.so3_conv1(x) + self.up(so4_add)
        so3 = self.so3_fconv(so3_add)
        so3 = F.sigmoid(so3)

        x = self.up(self.rconv3(x))
        x = x + z3
        x = self.layer3_D(x)

        # side output 2
        so2_add = self.so2_conv1(x) + self.up(so3_add)
        so2 = self.so2_fconv(so2_add)
        so2 = F.sigmoid(so2)

        x = self.up(self.rconv2(x))
        x = x + z2
        x = self.layer2_D(x)

        # side output 1
        so1_add = self.so1_conv1(x) + self.up(so2_add)
        so1 = self.so1_fconv(so1_add)
        so1 = F.sigmoid(so1)

        x = self.up(self.rconv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = x + self.up(so1_add)
        so0_add = x

        # side output 0
        so0 = self.fconv(so0_add)
        so0 = F.sigmoid(so0)

        # energy map
        x = self.map_conv1(x)
        x = self.map_fconv(x)
        if self.activation == 'sigmoid':
            x = 2.0 * (F.sigmoid(x) - 0.5)
        elif self.activation == 'tanh':
            x = torch.tanh(x)

        return x, so0, so1, so2, so3