import torch
from torch import nn
from d2l import torch as d2l

#mvoing_mean 方差；moving_var均值
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    #判断是test还是train
    if not torch.is_grad_enabled():
        #test
        X_hat = (X-moving_mean)/torch.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4)
        #判断输入维度是2D（全连接）还是4D（卷积）
        if len(X.shape)==2:
            mean = X.mean(dim=0)    #对特征进行归一化
            var = ((X-mean)**2).mean(dim=0)
        else:
            #卷积要在通道维上进行计算axis=1
            mean = X.mean(dim=(0,2,3), keepdim=True)
            var = ((X-mean)**2).mean(dim=(0,2,3), keepdim=True)
        X_hat = (X-mean)/torch.sqrt(var+eps)
        moving_mean=momentum*moving_mean+(1.0-momentum)*mean
        moving_var = momentum*moving_var+(1.0-momentum)*var

    Y=gamma*X_hat +beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape=(1, num_features)
        else:
            shape = (1, num_features,1,1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        #保证数据在同一个device，方便计算
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y

#在构建模型时使用BatchNorm,LeMet
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), 
    BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    BatchNorm(16, num_dims=4),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*4*4, 120),
    BatchNorm(120, num_dims=2),nn.Sigmoid(),
    nn.Linear(120, 84),
    BatchNorm(84, num_dims=2),nn.Sigmoid(),
    nn.Linear(84, 10)
)
