from regex import W
from sklearn.utils import shuffle
from sympy import root
import torch
import torchvision
from torch.utils import data
from IPythonn import display
from d2l import torch as d2l
from torchvision import transforms

batch_size = 256
num_inputs=784
num_outputs =10

def get_dataloader_workers():
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.Totensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train=torchvision.dataser.FashionMNIST(
        root="../data", train=True, transform = trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data",train=False, download=True
    )
    return (data.Dataloader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers())
            data.Dataloader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp/partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])),W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

class Accmulator:
    def _init_(self, n):
        self.data=[0.0]*n
    
    def add(self, *args):
        self.data=[a + float(b) for a, b in zip(self.data, args)]
    
    def reser(self):
        self.data=[0.0]*len(self.data)

    def _getitem_(self, idx):
        return self.date[idx]

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric=Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0]/metric[1]


lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def train():
    #加载数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    #初始化参数
    w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    loss=cross_entropy()

    if isinstance(net, torch.nn.Module):
        net.train()
    
    metric=Accmulator(3)
    for X, y in train_iter:
        y_hat=net(X)
        l=loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    return metric[0]/metric[2], metric[1]/metric[2]
    

    evaluate_accuracy(net, test_iter)


    

