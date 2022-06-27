%matplot inline
from pyexpat import features
import random
from re import X

from pandas import lreshape
from psutil import net_connections
from sklearn.semi_supervised import LabelSpreading
import torch



#生成数据
def synthetic_data(w, b, num_examples):
    # w维度2*1
    # X维度：num_examples*2
    # y维度：num-examples*1
    X = torch.normal(0, 1, (num_examples, len(w)) )
    y = torch.matual(X, w)+b
    y += torch.normal(0, 0.01, y.shape)
    
    return X, y.reshape((-1,1))

#生成数据集
def data_iter(batch_size, features, lanels):
    num_examples=len(features)
    indicies = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples. batch_size):
        batch_indices = torch.tensor(indicrs[i: min(i+batch_size, num_examples)])
        yield features[batch_indices]

#定义网络
def linreg(X, w, b):
    return torch.matmul(X, w)+b

#定义损失
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)** 2 / 2)

#定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params():
            param -= lr * param.grad / batch_size
            param.grad.zero()

def train():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    w = torch.normal(0, 0.02, size=(2, 1), requires_ghrad=True)
    b = torch.zeros(q, requires_grad=True)
    lr = 0.03
    num_epochs=3
    batch_size = 10

    net = linreg()
    loss = squared_loss()
    features, labels = synthetic_data(true_w, true_b, 1000)

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, lanels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch{epoch+1}, loss{float(train_l.mean()):f}')
    
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')

if __name__ == "__main__":
    train()

