import pandas as pd
from osgeo import gdal
from PIL import Image
import matplotlib.image as mpimg
from osgeo import ogr
import subprocess
from tensorflow import keras
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import pathlib
import tensorflow as tf
import torch
import os
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import random


def get_allfile(path):  # 获取所有文件
    all_file = []
    for f in os.listdir(path):  #listdir返回文件中所有目录
        f_name = os.path.join(path, f)
        all_file.append(f_name)
    return all_file

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name


#read the GeoTIFF file
def read_img(dataset_path):
    dataset = gdal.Open(dataset_path)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    del dataset
    temp = np.where(np.isnan(im_data),0,im_data)
    temp = np.transpose(temp,(1,2,0))
    final = np.resize(temp,(224,224,9))
    final = np.transpose(final,(2,0,1))
    return final

class MyDataset(torch.utils.data.Dataset): #Create my dataset which inherits from torch.utils.data.Dataset
    def __init__(self,txt, level, my_dict, transform=None, target_transform=None):
        super(MyDataset,self).__init__()
        path=txt
        imgs = []
        for line in path:
            words = line.split('/')
            if len(words)< 6 or '.tif' not in line:
                continue
            label = my_dict[words[level+1]]
            imgs.append((line,int(label)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.imgs[index]
        image = read_img(img)
        if self.transform is not None:
            image = torch.tensor(image)
        return image,label

    def __len__(self):
        return len(self.imgs)

class  VGG(nn.Module):
    def __init__(self,num_classes=40):
        super(VGG,self).__init__()
        layers=[]
        in_dim=9
        out_dim=64
        for i in range(13):
            layers+=[nn.Conv2d(in_dim,out_dim,3,1,1),nn.ReLU(inplace=True)]
            in_dim=out_dim
            if i==1 or i==3 or i==6 or i==9 or i==12:
                layers+=[nn.MaxPool2d(2,2)]
                if i!=9:
                    out_dim*=2
        self.features=nn.Sequential(*layers)
        self.classifier=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

def train(level=1):
    file_list = listdir('data/single_tree',[])
    family = []
    genus = []
    spices = []
    for f in file_list:
        a = f.split('/')
        if len(a) != 6:
            continue
        family.append(a[2])
        genus.append(a[3])
        spices.append(a[4])
    value = []
    temp = []
    x = list(set(family))
    for i in range(len(x)):
        temp.append([x[i],str(i)])
    family_dict = dict(temp)
    f_class = len(x)
    value = []
    temp = []
    x = list(set(genus))
    for i in range(len(x)):
        temp.append([x[i],str(i)])
    genus_dict = dict(temp)
    g_class = len(x)
    value = []
    temp = []
    x = list(set(spices))
    for i in range(len(x)):
        temp.append([x[i],str(i)])
    spices_dict = dict(temp)
    s_class = len(x)
    level = 1
    if level == 1:
        train_data=MyDataset(txt=file_list,level = level, my_dict = family_dict, transform='yes')
        test_data=MyDataset(txt=file_list,level = level, my_dict = family_dict, transform='yes')
        net = VGG(num_classes=f_class)
    elif level == 2:
        train_data=MyDataset(txt=file_list,level = level, my_dict = genus_dict, transform='yes')
        test_data=MyDataset(txt=file_list,level = level, my_dict = genus_dict, transform='yes')
        net = VGG(num_classes=g_class)
    else:
        train_data=MyDataset(txt=file_list,level = level, my_dict = spices_dict, transform='yes')
        test_data=MyDataset(txt=file_list,level = level, my_dict = spices_dict, transform='yes')
        net = VGG(num_classes=s_class)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=16)

    net = net.float()
    criterion =torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.005 )

    acc_test = []
    acc_train = []
    loss_train = []
    loss_test = []
    hign_acc = 0.
    for epoch in range(50):
        net.train()
        for i, data in enumerate(train_loader,0):
            img, label = data
            optimizer.zero_grad()
            output = net(img)
            prediction = torch.max(F.softmax(output), 1)[1]
            pred = prediction.data.numpy().squeeze()
            x = label.data.numpy().squeeze()
            acc_now = pred-x
            temp = np.sum(acc_now == 0)
            acc_train.append(temp/len(label))
            loss_contrastive = criterion(output,label)
            loss_contrastive.backward()
            optimizer.step()
            loss_train.append(loss_contrastive.item())
            print("train times: {}\nEpoch number {}\n Current loss {}\n Current accuracy {}\n".format(i,epoch,loss_contrastive.item(),temp/len(label)))
            if temp/len(label) > hign_acc:
                torch.save(net,'model_'+str(level)+'.pth')
                hign_acc = temp/len(label)
        else:
            net.eval()
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for i, data in enumerate(test_loader,0):
                   img, label= data
                   out = net(img)
                   prediction = torch.max(F.softmax(out), 1)[1]
                   x = label.data.numpy().squeeze()
                   pred = prediction.data.numpy().squeeze()
                   acc_now = pred-x
                   temp = np.sum(acc_now == 0)
                   acc.append(temp/len(label))
                   loss_contrastive = criterion(out,label)
                   loss_test.append(loss_contrastive.item())

train(level=1)
