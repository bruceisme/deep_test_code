%matplotlib inline
from cProfile import label
from operator import index
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from torch import nn 
from sklearn.model_selection import train_test_split,KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import ttach as tta 
from d2l import torch as d2l


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


train_transform = transforms.Compose([
                    # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。 然后，缩放图像以创建224 x 224的新图像
                    transforms.RandomResizedCrop(128, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.Totnesor(),
                    transforms.Normalize([0.485, 0.456,0.406],[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(128),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class LeavesSet(data.Dataset):
    #构建数据集
    def __init__(self, images_path, images_label, transform=None, train=True):
        self.imgs=[os.path.join('./data/classify-leaves',"".join(image_path)) for image_path in images_path]

        if train:
            self.train=True
            self.labels=images_label
        else:
            self.train=False
        
        self.transform=transform
    
    def __getitem__(self, index):
        image_path = self.imgs[index]
        pil_img=Image.open(image_path)
        if self.transform:
            transform=self.transform
        else:
            #如果没有额外的transform要求，默认将模型转为224维
            transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        
        data = transform(pil_img)

        if self.train:
            image_label=self.lanbels[index]
            return data, image_label
        else:
            return data
    
    def __len__(self):
        return len(self.imgs)

def load_data_leaves(train_transform=None, test_transform=None):
    train_data = pd.read_csv('./data/classify-leaves/train.csv')
    test_data = pd.read_csv('./data/classify-leaves/test.csv')

    labelencoder = LabelEncoder()
    labelencoder.fit(train_data['label'])
    train_data['label']=labelencoder.transform(train_data['label'])
    label_map = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    label_inv_map = {v:k for k,v in label_map.items()}

    train_dataSet = LeavesSet(train_data['image'], train_data['label'], transform=train_transform, train=True)
    test_dataSet = LeavesSet(test_data['image'], images_label=0, transform=test_transform, train=False)

    return (train_dataSet, test_dataSet, label_map, label_inv_map)

train_dataset, test_dataset, label_map, label_inv_map = load_data_leaves(train_transform, test_transform)

class Residual(nn.Module):  
    #stride为2时，特征高宽减半
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def create_net():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    return nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(512, 10)
)

