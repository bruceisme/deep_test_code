import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers=[]
    for _ in range(num_convs):
        #VGG块中卷积核大小固定为3，填充为1
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        #第一个块后，所有的VGG块的输入输出通道数都为out_channels
        in_channels=out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

#VGG块的设定(卷积核数，输出通道数)
conv_arch = ((1,64), (1, 128), (2,256), (2,512), (2, 512))

#VGG11
def vgg(conv_arch):
    conv_blks=[]
    in_channels = 1
    #卷积
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        #下个输入通道数等于上个块的输出数
        in_channels=out_channels
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

net = vgg(conv_arch)

def train():
    lr=0.1
    num_epochs=10
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

if __name__ == "__main__":
    train()


