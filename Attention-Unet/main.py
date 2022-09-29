# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


import model
import dataset 
import utils

DATAPATH = 'F:/DataCollection/GDGQ/testData2/'

trainLoader = dataset.getDataLoader("train", data_path=DATAPATH, BatchSize=1, shuffle=True, num_workers=0, pinMem=False)
valLoader = dataset.getDataLoader("val", data_path=DATAPATH, BatchSize=1, shuffle=False, num_workers=0, pinMem=False)
testLoader = dataset.getDataLoader("test", data_path=DATAPATH, BatchSize=1, shuffle=False, num_workers=0, pinMem=False)

# 网络、损失函数
net = model.AttentionUnet()

crt = nn.MultiLabelSoftMarginLoss()
        # 训练
utils.train(net, trainLoader, testLoader, crt, 0.0001, epoch=1, device=torch.device('cuda:0'), device_ids=(0,),from0=True)

