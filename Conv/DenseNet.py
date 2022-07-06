import torch
from torch import nn
from d2l import torch as d2l

def conv_block(input_channels, num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels),
    nn.ReLU(),
    nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    #输出通道数:input+out*num_convs
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer=[]
        for i in range(num_convs):
            #下一层的输入是上一层输出和原输入的维度和
            layer.append(conv_block(num_channels*i+input_channels, num_channels))
            self.net=nn.Sequential(*layer)
    
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X=torch.cat((X,Y), dim=1)
        return X

def transition_block(input_channels, num_channels):
    #修改特征的通道数
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )


def create_net():
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4,4,4,4]
    blks=[]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs,num_channels, growth_rate))
        num_channels +=num_convs*growth_rate

        if i != len(num_convs_in_dense_blocks)-1:
            blks.append(transition_block(num_channels, num_channels//2))
            num_channels=num_channels//2

    b1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))



    return nn.Sequential(b1,
                        *blks,
                        nn.BatchNorm2d(num_channels),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool1d((1,1)),
                        nn.Flatten(),
                        nn.Linear(num_channels,10))



def train():
    net = create_net()
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


            
