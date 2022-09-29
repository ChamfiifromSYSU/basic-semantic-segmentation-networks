
import torch
import torch.nn as nn 
import numpy as np
from torch.nn import functional as F 

class AttentionGate(nn.Module):
    def __init__(self,F_g, F_i, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            
            nn.UpsamplingBilinear2d(scale_factor=2), # 维度匹配（论文中没说具体匹配的方案）
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_i, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        gl = self.W_g(g)
        xl = self.W_x(x)
        psi = self.relu(gl + xl)
        psi = self.psi(psi)

        return x * psi

class Convx2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convx2, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),            
        )
    def forward(self,x):
        return self.conv2(x)

class AttentionUnet(nn.Module):
    def __init__(self):
        super(AttentionUnet, self).__init__()
        self.conv1 = Convx2(3,64)
        self.conv2 = Convx2(64,128)
        self.conv3 = Convx2(128,256)
        self.conv4 = Convx2(256,512)

        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(512,256,kernel_size=3, stride=1, padding=1)

        self.AG1 = AttentionGate(512,256,64)
        self.AG2 = AttentionGate(128,128,64)
        self.AG3 = AttentionGate(64,64,64)

        self.conv33 = Convx2(512,128)
        self.conv22 = Convx2(256,64)
        self.conv11 = Convx2(128,1)

    def forward(self,x):
        # encoding
        x_x3 = self.conv1(x)
        print('x3.size(), ',x_x3.size())
        x = self.maxpool(x_x3)
        print('size, ',x.size())
        x_x2 = self.conv2(x)
        x = self.maxpool(x_x2)
        x_x1 = self.conv3(x)
        x = self.maxpool(x_x1)
        x_g1 = self.conv4(x)
        x = self.up(x_g1)
        x = self.conv(x)

        x_AG1 = self.AG1(x_g1,x_x1)
        x = torch.cat([x_AG1,x],dim=1)

        x_g2 = self.conv33(x)
        x_AG2 = self.AG2(x_g2,x_x2)

        x = self.up(x_g2)
        x = torch.cat([x_AG2,x],dim=1)

        x_g3 = self.conv22(x)
        x_AG3 = self.AG3(x_g3,x_x3)

        x = self.up(x_g3)
        x = torch.cat([x_AG3,x],dim=1)

        output = self.conv11(x)

        return output


