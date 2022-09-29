import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import os
import cv2

Default_DataPath = 'E:\PyLearning/data/GDGQ/testData2'

class Crop_train(data.Dataset):
    def __init__(self,path):
        self.img_list = []
        self.label_list = []

        for item in os.scandir(path + 'train/image'):
            self.img_list.append(item.path)
        for item in self.img_list:
            self.label_list.append(item.replace('image','label').replace('.jpg','.png'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index],3)
        img = img / np.max(img)
        label = (cv2.imread(self.label_list[index],2) ==1)
        label = label.astype(int)
        return (torch.as_tensor(img.transpose(2,0,1),dtype=torch.float),
                torch.as_tensor(label[np.newaxis,:,:],dtype= torch.float))

class Crop_VT(data.Dataset):
    def __init__(self,path,dataset='val'):
        self.img_list = []
        self.label_list = []

        for item in os.scandir(path + dataset + '/image'):
            self.img_list.append(item.path)
        for item in self.img_list:
            self.label_list.append(item.replace('image','label').replace('.jpg','.png'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index], 3)
        img = img / np.max(img)
        label = (cv2.imread(self.label_list[index], 2) == 1)
        label = label.astype(int)
        img_t = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float)
        label_t = torch.as_tensor(label[np.newaxis, :, :], dtype=torch.float)

        return (img_t,label_t)

def getDataLoader(data_set='train',data_path=Default_DataPath, BatchSize=4, shuffle=False, num_workers=0,pinMem=True):
    if data_set == "train":
        data_set = Crop_train(data_path)
    elif data_set == "val":
        data_set = Crop_VT(data_path, dataset='val')
    elif data_set == "test":
        data_set = Crop_VT(data_path, dataset='test')
    else:
        print("%s is Invalid Input for arg \"data_set\" " % data_set)
    return data.DataLoader(data_set, batch_size=BatchSize, shuffle=shuffle, num_workers=num_workers, pin_memory=pinMem)
