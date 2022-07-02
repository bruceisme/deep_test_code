from audioop import avg
from matplotlib.pyplot import legend, xlabel
from regex import F
import torch
from torch import avg_pool1d, nn
from d2l import torch as d2l


class Reshape(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = torch.nn.Sequential(
    #Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  #28-5+4+1=28
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84), 
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

net2 = torch.nn.Sequential(
    #Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  #28-5+4+1=28
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120),
    nn.ReLU(),
    nn.Linear(120, 84), 
    nn.ReLU(),
    nn.Linear(84, 10)
)
def get_shape():
    #查看各层的输出形状
    X=torch.rand(size=(1,1,28,28), dtype=torch.float32)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__, 'shape:\t', X.shape)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Moudle):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X=[x.to(device) for x in X]
            else:
                X=X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0]/metric[1]

def init__weight(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def train_ch6(net, num_epochs, lr, device, batch_size):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    net.apply(init__weight)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters, lr=lr)
    loss = nn.CrossEntropyLoss()
    #绘制图像
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches=d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):
        #计数器设置为3个变量
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            X, y=X.to(device), y.to(device)
            #梯度清零
            optimizer.zero_grad()
            #计算
            y_hat = net(X)
            #计算loss
            l = loss(y_hat, y)
            #反向传播
            l.backward()
            #优化
            optimizer.step()
            with torch.no_grad():
                #X.shape[0]?batch?
                metric.add(l*X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0]/ metric[2]
            train_acc = metric[1]/ metric[2]

            if (i+1)%(num_batches//5) == 0 or i==num_batches-1:
                animator.add(epoch+(i+1)/num_batches,(train_l, train_acc, None))
        
        #测试
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'loss{train_l:.3f}, train acc{train_acc:.3f}, '
                f'test acc{test_acc:.3f}')
        print(f'{metric[2]*num_epochs / timer.sum():.1f} examples/sec'
                f'on{str(device)}')

if __name__ == "__main__":
    lr=0.9
    num_epochs = 10
    batch_size=256

    train_ch6(net=net,num_epochs=num_epochs,lr=lr, device=d2l.try_gpu(),batch_size=batch_size)
