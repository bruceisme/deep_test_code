from ast import increment_lineno
from multiprocessing import reduction
import matplotlib
from sklearn.utils import shuffle
import torch


%matplotlib inline
import os
import torchvision
from torch import nn
from d2l import torch as d2l

#以热狗数据集为例
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

#数据增强
#归一化
normalize=torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

#训练数据处理，随机裁剪224x224大小的图片
train_augs=torchvision.transforms.Compose([
    #裁剪
    torchvision.transforms.RandomResizedCrop(224),
    #左右翻转
    torchvision.transforms.RandomHorizontalFlip(),
    #转为张量
    torchvision.transforms.ToTensor(),
    normalize])

#测试集将图像缩放到256x256,然后裁剪中央224x224
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

#下载预训练模型
finetune_net = torchvision.models.resnet18(pretrained=True)
#更改全连接层结构，分类类别设为为两类
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
#初始化新分类器的参数
nn.init.xavier_uniform_(finetune_net.fc.weightn)

#如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter=torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
                                            batch_size=batch_size,
                                            shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
                                            batch_size=batch_size)

    devices = d2l.try_all_gpus()
    #reduction = 'elementwise_mean'/默认时,返回N个样本的loss求平均之后的返回
    #reduction = 'sum'时，指对n和样本的loss求和
    #reduction = 'None'时，指直接返回n分样本的loss
    loss=nn.CrossEntropyLoss(reduction='None')
    if param_group:
        #对特征提取层和输出层的模型使用不同的学习率，通过SGD实现
        params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"] ]
        trainer = torch.optim.SGD([{'params':params_1x},
                                    {'params':net.fc.parameters(), 'lr':learning_rate*10}],
                                lr=learning_rate,
                                weight_decay=0.001)
    else:
        #对所有层参数使用统一的学习率训练
        trainer=torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)

    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


if __name__ == "__main__":
    train_fine_tuning(finetune_net, 5e-5)



