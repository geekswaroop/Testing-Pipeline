import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# synchronized batch normalization
from libs.sync import SynchronizedBatchNorm1d
from libs.sync import SynchronizedBatchNorm2d
from libs.sync import SynchronizedBatchNorm3d

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1, 1)

class _E(nn.Module):
    def __init__(self, num_filter=32, in_num=1, latent_dim=512, bias=True, training=True):
        super(_E, self).__init__()
        self.latent_dim = latent_dim
        self.training = training

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_num, num_filter, kernel_size=(1,4,4), stride=(1,2,2), bias=bias, padding=(0,1,1)),
            SynchronizedBatchNorm3d(num_filter),
            nn.ELU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(num_filter, num_filter*2, kernel_size=(1,4,4), stride=(1,2,2), bias=bias, padding=(0,1,1)),
            SynchronizedBatchNorm3d(num_filter*2),
            nn.ELU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(num_filter*2, num_filter*4, kernel_size=(3,4,4), stride=(1,2,2), bias=bias, padding=(1,1,1)),
            SynchronizedBatchNorm3d(num_filter*4),
            nn.ELU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(num_filter*4, num_filter*8, kernel_size=(3,4,4), stride=(1,2,2), bias=bias, padding=(1,1,1)),
            SynchronizedBatchNorm3d(num_filter*8),
            nn.ELU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(num_filter*8, num_filter*16, kernel_size=4, stride=2, bias=bias, padding=1),
            SynchronizedBatchNorm3d(num_filter*16),
            nn.ELU(inplace=True)
        )
        # predict both mean and variance
        self.f_conv1 = nn.Sequential(
            nn.Conv3d(num_filter*16, self.latent_dim, kernel_size=4, stride=1, bias=bias, padding=0),
            Flatten()
        )
        self.f_conv2 = nn.Sequential(
            nn.Conv3d(num_filter*16, self.latent_dim, kernel_size=4, stride=1, bias=bias, padding=0),
            Flatten()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = x
        #print(out.size())  # torch.Size([b, 1, 8, 128, 128])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([b, 32, 8, 64, 64])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([b, 64, 8, 32, 32])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([b, 128, 8, 16, 16])
        out = self.layer4(out)
        #print(out.size())  # torch.Size([b, 256, 8, 8, 8])
        out = self.layer5(out)
        #print(out.size())  # torch.Size([b, 512, 4, 4, 4])

        mu = self.f_conv1(out)
        logvar = self.f_conv2(out)
        #print(out.size())  # torch.Size([b, latent_dim])

        return mu, logvar

class _G(nn.Module):
    def __init__(self, num_filter=32, out_num=1, latent_dim=512, bias=True):
        super(_G, self).__init__()
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(self.latent_dim, num_filter*16, kernel_size=4, stride=1, bias=bias, padding=0),
            SynchronizedBatchNorm3d(num_filter*16),
            nn.ELU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(num_filter*16, num_filter*8, kernel_size=4, stride=2, bias=bias, padding=1),
            SynchronizedBatchNorm3d(num_filter*8),
            nn.ELU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(num_filter*8, num_filter*4, kernel_size=(3,4,4), stride=(1,2,2), bias=bias, padding=(1,1,1)),
            SynchronizedBatchNorm3d(num_filter*4),
            nn.ELU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(num_filter*4, num_filter*2, kernel_size=(3,4,4), stride=(1,2,2), bias=bias, padding=(1,1,1)),
            SynchronizedBatchNorm3d(num_filter*2),
            nn.ELU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(num_filter*2, num_filter, kernel_size=(1,4,4), stride=(1,2,2), bias=bias, padding=(0,1,1)),
            SynchronizedBatchNorm3d(num_filter),
            nn.ELU(inplace=True)
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose3d(num_filter, out_num, kernel_size=(1,4,4), stride=(1,2,2), bias=bias, padding=(0,1,1))
            #SynchronizedBatchNorm3d(out_num)
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm3d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        out = x.view(-1, self.latent_dim, 1, 1, 1)
        #print(out.size())  # torch.Size([b, latent_dim, 1, 1, 1])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([b, 512, 4, 4, 4])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([b, 256, 8, 8, 8])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([b, 128, 16, 16, 16])
        out = self.layer4(out)
        #print(out.size())  # torch.Size([b, 64, 32, 32, 32])
        out = self.layer5(out)
        #print(out.size())  # torch.Size([b, 1, 64, 64, 64])
        out = self.layer6(out)
        
        return torch.sigmoid(out)

class VAE(nn.Module):
    def __init__(self, 
                 num_filter=32, 
                 in_num=1,
                 out_num=1,
                 latent_dim=512):

        super(VAE, self).__init__()
        self.encoder = _E(num_filter, in_num,  latent_dim)
        self.decoder = _G(num_filter, out_num, latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu  

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar      


class classifier1(torch.nn.Module):
    def __init__(self, dim=1024, reduction = 4):
        super(classifier1, self).__init__()  
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            SynchronizedBatchNorm1d(dim // reduction),
            nn.ELU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(dim // reduction, dim // reduction),
            SynchronizedBatchNorm1d(dim // reduction),
            nn.ELU(inplace=True))    
        self.fc3 = nn.Linear(dim // reduction, 1)

    def forward(self, x):
        print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

class classifier2(torch.nn.Module):
    def __init__(self, dim=512, reduction=4):
        super(classifier2, self).__init__()  
        # shared layer
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            SynchronizedBatchNorm1d(dim // reduction),
            nn.ELU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(dim // reduction, dim // reduction // reduction),
            SynchronizedBatchNorm1d(dim // reduction // reduction),
            nn.ELU(inplace=True))   

        # merge layer
        self.bil = nn.Bilinear(dim // reduction // reduction, # in1_features
                               dim // reduction // reduction, # in2_features
                               dim // reduction // reduction) # out_features
        self.fc3 = nn.Sequential(
            SynchronizedBatchNorm1d(dim // reduction // reduction),
            nn.ELU(inplace=True))           
        self.fc4 = nn.Sequential(
            nn.Linear(dim // reduction // reduction, 
                      dim // reduction // reduction),
            SynchronizedBatchNorm1d(dim // reduction // reduction),
            nn.ELU(inplace=True))

        # final layer
        self.fc5 = nn.Linear(dim // reduction // reduction, 1)   

    def forward_once(self, x):
        x = self.fc1(x)   
        x = self.fc2(x) 
        return x

    def forward(self, x1, x2): # x = (x1, x2)
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)

        out = self.bil(x1, x2)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = torch.sigmoid(out)
        return out    

# symmetry regularization of matrix in bilinear layer
# for name, param in model.named_parameters():
#     if name=='bil.weight':
#         bi_matrix = param # (out_features x in1_features x in2_features)
# sym_loss = (bi_matrix - bi_matrix.transpose(2,1))**2
# sym_loss = sym_loss.mean()

# class _D(torch.nn.Module):
#     def __init__(self, args, in_num=1):
#         super(_D, self).__init__()
#         self.args = args
#         self.cube_len = args.cube_len

#         self.layer1 = nn.Sequential(
#             nn.Conv3d(1, self.cube_len, kernel_size=(1,4,4), stride=(1,2,2), bias=args.bias, padding=(0,1,1)),
#             SynchronizedBatchNorm3d(self.cube_len),
#             nn.ELU(inplace=True)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=(1,4,4), stride=(1,2,2), bias=args.bias, padding=(0,1,1)),
#             SynchronizedBatchNorm3d(self.cube_len*2),
#             nn.ELU(inplace=True)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=(1,4,4), stride=(1,2,2), bias=args.bias, padding=(0,1,1)),
#             SynchronizedBatchNorm3d(self.cube_len*4),
#             nn.ELU(inplace=True)
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=args.bias, padding=1),
#             SynchronizedBatchNorm3d(self.cube_len*8),
#             nn.ELU(inplace=True)
#         )
#         self.layer5 = nn.Sequential(
#             nn.Conv3d(self.cube_len*8, self.cube_len*16, kernel_size=4, stride=2, bias=args.bias, padding=1),
#             SynchronizedBatchNorm3d(self.cube_len*8),
#             nn.ELU(inplace=True)
#         )
#         self.layer6 = nn.Sequential(
#             nn.Conv3d(self.cube_len*16, 1, kernel_size=4, stride=2, bias=args.bias, padding=0),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         out = x
#         #print(out.size())  # torch.Size([b, in_num, 16, 128, 128])
#         out = self.layer1(out)
#         #print(out.size())  # torch.Size([b, 32, 16, 64, 64])
#         out = self.layer2(out)
#         #print(out.size())  # torch.Size([b, 64, 16, 32, 32])
#         out = self.layer3(out)
#         #print(out.size())  # torch.Size([b, 128, 16, 16, 16])
#         out = self.layer4(out)
#         #print(out.size())  # torch.Size([b, 256, 8, 8, 8])
#         out = self.layer5(out)
#         #print(out.size())  # torch.Size([b, 512, 4, 4, 4])
#         out = self.layer6(out)
#         #print(out.size())  # torch.Size([b, 1, 1, 1, 1])

#         return out
