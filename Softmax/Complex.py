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
num_epochs=10
lr = 0.1
w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

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


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Moudle):
        net.train()
    metric = Accmulator(3)
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

    return metric[0]/metric[2],metric[1]/metric[2]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, loss, num_epochs, updater):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

if __name__ == "__main__":
    train_ch3(net, cross_entropy, num_epochs, updater)

