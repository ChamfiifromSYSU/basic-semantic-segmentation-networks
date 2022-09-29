# 导入相关库

import os
import time
import utils
from tqdm import tqdm
from PIL import Image
import numpy as np
import scipy.misc

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


#%% 我自己的版本
class Convx2(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Convx2, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),            
        )

    def forward(self,x):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),                     
        )

    def forward(self,x):
        return self.up(x)



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.d1 = Convx2(3,64)
        self.d2 = Convx2(64,128)
        self.d3 = Convx2(128,256)
        self.d4 = Convx2(256,512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        self.u1 = UpSample(1024,512)
        self.u11 = Convx2(1024,512)
        self.u2 = UpSample(512,256)
        self.u22 = Convx2(512,256)
        self.u3 = UpSample(256,128)
        self.u33 = Convx2(256,128)
        self.u4 = UpSample(128,64)
        self.u44 = Convx2(128,64)

        self.final = nn.Conv2d(64,2,kernel_size=1)


    def forward(self,x):
        x1 = self.d1(x)  # cat用x1
        x11 = self.maxpool(x1)
        x2 = self.d2(x11)
        x22 = self.maxpool(x2)
        x3 = self.d3(x22)
        x33 = self.maxpool(x3)
        x4 = self.d4(x33)
        x44 = self.maxpool(x4)
        x5 = self.conv1(x44)
        x6 = self.conv2(x5)

        y1 = self.u1(x6)
        y11 = torch.cat([x4,y1],dim=1)
        y111 = self.u11(y11)
        y2 = self.u2(y111)
        y22 = torch.cat([x3,y2],dim=1)
        y222 = self.u22(y22)
        y3 = self.u3(y222)
        y33 = torch.cat([x2,y3],dim=1)
        y333 = self.u33(y33)
        y4 = self.u4(y333)
        y44 = torch.cat([x1,y4],dim=1)
        y444 = self.u44(y44)

        z = self.final(y444)
        return z



        





