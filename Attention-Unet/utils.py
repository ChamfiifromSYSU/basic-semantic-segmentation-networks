import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
import torch.optim as optim


def randTile(inputs, num, size, randseed):
    tiles = []
    for b_num in range(inputs.shape[0]):
        for i in range(num):
            h, w = int(randseed[i, 0]), int(randseed[i, 1])
            tiles.append(inputs[b_num, :, h:h+size, w:w+size])
    return torch.stack(tiles, dim=0)


def oneHot(x, N=2, use_torch=False):
    if use_torch:
        shape = list(x.shape)
        x = x.reshape(-1)
        ones = torch.eye(N)
        ones = ones.index_select(0, x)
        shape.append(N)
        return ones.reshape(shape)
    else:
        shape = list(x.shape)
        x = list(x.reshape(-1))
        ones = np.eye(N)[x]
        shape.append(N)
        return ones.reshape(shape)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets[:, 0, :, :])
    
class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, label):
        logits = logits.long()
        return torch.mean(-1.4 * label * (logits + 0.0000001).log() - (1 - label) * np.log(1 - logits + 0.0000001).log())

def kmoment(x, k):
    return np.sum((x)**k) / np.size(x)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def train(model, data_loader,test_loader, criteria, lr_base, epoch, device=torch.device('cuda:0'), device_ids=(0,), from0=False):
    # model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_base)
    
    loss_func = criteria

    for epoch in range(epoch):
        running_loss = 0.0
        for i,data in enumerate(data_loader):
            inputs, labels = data            
            print('\n input.size() is: ',inputs.size())
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = loss_func(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3r' % (epoch+1,i+1,running_loss))
            
    print('Finish training')

    correct = 0
    total = 0
    acc = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            images, labels = data
            #images, labels = images.to(device), labels.to(device)
            #print('\n images.size is :',images.size())
            #print('\n labels.size is :',labels.size())
            outputs = model(images)
            labels = labels.float()
            predicted = (outputs >= 0.5 ).float()
            acc += torch.mean((predicted==labels).float())
            #_,predicted = torch.max(outputs.data,1)
            #total += labels.size(0)
            #print(predicted.size())
            #correct += (predicted==labels).long().sum().item()

    print('Accuracy : %4f' % (acc/(epoch+1)))
                    
    